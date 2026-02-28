// InferenceEngine.swift — Load exported weights and run seq2seq inference.
//
// Reads the model.json + weights.bin files produced by export_weights.py
// and builds the neural network layers for pure-Swift inference.

import Foundation

// MARK: - Special Token Indices

private let PAD_IDX = 0
private let SOS_IDX = 1
private let EOS_IDX = 2

// MARK: - Inference Engine

/// Loads the exported model and runs character-level seq2seq inference.
///
/// This is an internal type — the public API is ``Conjugator``.
final class InferenceEngine: @unchecked Sendable {

    // ── Vocabulary ───────────────────────────────────────────────────────

    let charToIdx: [Character: Int]
    let idxToChar: [Int: Character]
    let modeToIdx: [String: Int]
    let tenseToIdx: [String: Int]
    let personToIdx: [String: Int]

    // ── Metadata ─────────────────────────────────────────────────────────

    let exceptions: [String: String]
    let etreVerbs: Set<String>
    let pronoVerbs: Set<String>
    let hAspire: Set<String>
    let knownVerbs: Set<String>
    let invariablePPVerbs: Set<String>
    let impersonalVerbs: Set<String>
    let thirdPersonOnlyVerbs: Set<String>

    // ── Layers ───────────────────────────────────────────────────────────

    private let encoder: EncoderLayer
    private let attention: AttentionLayer
    private let decoder: DecoderLayer
    private let bridge: BridgeLayer

    // MARK: - Loading

    /// Load the model from a directory containing `model.json` and `weights.bin`.
    ///
    /// - Parameter directory: URL to the model directory.
    /// - Throws: If files are missing or malformed.
    init(modelDirectory directory: URL) throws {
        let jsonURL = directory.appendingPathComponent("model.json")
        let binURL = directory.appendingPathComponent("weights.bin")

        // Load JSON metadata
        let jsonData = try Data(contentsOf: jsonURL)
        guard let root = try JSONSerialization.jsonObject(with: jsonData) as? [String: Any] else {
            throw ConjugationError.modelLoadFailed(path: directory.path)
        }

        guard let hp = root["hyperparams"] as? [String: Int],
              let vocabDict = root["vocab"] as? [String: Any],
              let manifestDict = root["weight_manifest"] as? [String: Any] else {
            throw ConjugationError.modelLoadFailed(path: directory.path)
        }

        let vocabSize = hp["vocab_size"]!
        let embDim = hp["emb_dim"]!
        let hiddenDim = hp["hidden_dim"]!
        let condDim = hp["cond_dim"]!
        _ = vocabSize  // suppress unused warning

        // ── Parse vocabulary ─────────────────────────────────────────────
        let c2i = vocabDict["char_to_idx"] as! [String: Int]
        var charToIdx = [Character: Int]()
        for (k, v) in c2i {
            if let ch = k.first, k.count == 1 {
                charToIdx[ch] = v
            }
        }
        self.charToIdx = charToIdx

        let i2c = vocabDict["idx_to_char"] as! [String: String]
        var idxToChar = [Int: Character]()
        for (k, v) in i2c {
            if let idx = Int(k), let ch = v.first, v.count == 1 {
                idxToChar[idx] = ch
            }
        }
        self.idxToChar = idxToChar

        self.modeToIdx = vocabDict["mode_to_idx"] as! [String: Int]
        self.tenseToIdx = vocabDict["tense_to_idx"] as! [String: Int]
        self.personToIdx = vocabDict["person_to_idx"] as! [String: Int]

        // ── Parse metadata sets ──────────────────────────────────────────
        self.exceptions = (root["exceptions"] as? [String: String]) ?? [:]
        self.etreVerbs = Set((root["etre_verbs"] as? [String]) ?? [])
        self.pronoVerbs = Set((root["prono_verbs"] as? [String]) ?? [])
        self.hAspire = Set((root["h_aspire"] as? [String]) ?? [])
        self.knownVerbs = Set((root["known_verbs"] as? [String]) ?? [])
        self.invariablePPVerbs = Set((root["invariable_pp_verbs"] as? [String]) ?? [])
        self.impersonalVerbs = Set((root["impersonal_verbs"] as? [String]) ?? [])
        self.thirdPersonOnlyVerbs = Set((root["third_person_only_verbs"] as? [String]) ?? [])

        // ── Load binary weights ──────────────────────────────────────────
        let binData = try Data(contentsOf: binURL)

        /// Helper to load a tensor from the manifest.
        func loadTensor(_ name: String) -> Tensor {
            let info = manifestDict[name] as! [String: Any]
            let shape = info["shape"] as! [Int]
            let offset = info["offset"] as! Int
            let count = info["count"] as! Int
            return binData.withUnsafeBytes { rawBuf in
                let ptr = rawBuf.baseAddress!.advanced(by: offset)
                return Tensor(from: ptr, count: count, shape: shape)
            }
        }

        // ── Build Encoder ────────────────────────────────────────────────
        let fwdGRU = GRUCell(
            hiddenDim: hiddenDim,
            weightIH: loadTensor("encoder.rnn.weight_ih_l0"),
            weightHH: loadTensor("encoder.rnn.weight_hh_l0"),
            biasIH: loadTensor("encoder.rnn.bias_ih_l0"),
            biasHH: loadTensor("encoder.rnn.bias_hh_l0")
        )
        let bwdGRU = GRUCell(
            hiddenDim: hiddenDim,
            weightIH: loadTensor("encoder.rnn.weight_ih_l0_reverse"),
            weightHH: loadTensor("encoder.rnn.weight_hh_l0_reverse"),
            biasIH: loadTensor("encoder.rnn.bias_ih_l0_reverse"),
            biasHH: loadTensor("encoder.rnn.bias_hh_l0_reverse")
        )
        self.encoder = EncoderLayer(
            embedding: loadTensor("encoder.embedding.weight"),
            forward: fwdGRU,
            backward: bwdGRU,
            hiddenDim: hiddenDim
        )

        // ── Build Attention ──────────────────────────────────────────────
        self.attention = AttentionLayer(
            attnWeight: loadTensor("attention.attn.weight"),
            vWeight: loadTensor("attention.v.weight"),
            decHiddenDim: hiddenDim
        )

        // ── Build Decoder ────────────────────────────────────────────────
        let decGRU = GRUCell(
            hiddenDim: hiddenDim,
            weightIH: loadTensor("decoder.rnn.weight_ih_l0"),
            weightHH: loadTensor("decoder.rnn.weight_hh_l0"),
            biasIH: loadTensor("decoder.rnn.bias_ih_l0"),
            biasHH: loadTensor("decoder.rnn.bias_hh_l0")
        )
        self.decoder = DecoderLayer(
            embedding: loadTensor("decoder.embedding.weight"),
            gru: decGRU,
            fcWeight: loadTensor("decoder.fc.weight"),
            fcBias: loadTensor("decoder.fc.bias"),
            embDim: embDim
        )

        // ── Build Bridge ─────────────────────────────────────────────────
        self.bridge = BridgeLayer(
            weight: loadTensor("bridge.weight"),
            bias: loadTensor("bridge.bias"),
            modeEmb: loadTensor("mode_emb.weight"),
            tenseEmb: loadTensor("tense_emb.weight"),
            personEmb: loadTensor("person_emb.weight"),
            condDim: condDim
        )
    }

    // MARK: - Prediction

    /// Run a single neural prediction: infinitive + mode/tense/person → conjugated form.
    ///
    /// Checks the exception table first, then falls back to greedy decoding.
    ///
    /// - Returns: The predicted string, or `nil` if any character is out of vocabulary.
    func predict(
        infinitive: String,
        mode: String,
        tense: String,
        person: String
    ) -> String? {
        // Check exception table
        let excKey = "\(infinitive)|\(mode)|\(tense)|\(person)"
        if let exc = exceptions[excKey] {
            return exc
        }

        // Encode characters
        var charIds = [Int]()
        for ch in infinitive {
            guard let idx = charToIdx[ch] else { return nil }
            charIds.append(idx)
        }
        charIds.append(EOS_IDX)

        let modeIdx = modeToIdx[mode] ?? 0
        let tenseIdx = tenseToIdx[tense] ?? 0
        let personIdx = personToIdx[person] ?? 0

        // Encode
        let (encOutputs, encHidden) = encoder.encode(charIds)

        // Bridge → initial decoder hidden
        var decHidden = bridge.initDecoder(
            encoderHidden: encHidden,
            modeIdx: modeIdx,
            tenseIdx: tenseIdx,
            personIdx: personIdx
        )

        // Greedy decode
        var tokenIdx = SOS_IDX
        var result = [Character]()
        let maxLen = 50

        for _ in 0..<maxLen {
            let (context, _) = attention.attend(
                hidden: decHidden,
                encoderOutputs: encOutputs
            )
            let (logits, newHidden) = decoder.step(
                tokenIdx: tokenIdx,
                hidden: decHidden,
                context: context
            )
            decHidden = newHidden
            tokenIdx = logits.argmax()
            if tokenIdx == EOS_IDX { break }
            if let ch = idxToChar[tokenIdx] {
                result.append(ch)
            }
        }

        return String(result)
    }
}
