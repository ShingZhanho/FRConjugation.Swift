// Layers.swift — Neural network layer implementations for seq2seq inference.
//
// These match the PyTorch architecture in french_conjugation_model.py exactly:
//   Encoder  — Embedding + BiGRU
//   Attention — Bahdanau (additive) attention
//   Decoder  — Embedding + GRU + FC
//   Bridge   — Linear + tanh
//
// All layers operate on `Tensor` and use no external ML framework.

import Foundation

// MARK: - GRU Cell

/// A single GRU cell step: given input x and previous hidden state h,
/// produce the next hidden state.
///
/// PyTorch GRU equations:
///   r = σ(W_ir @ x + b_ir + W_hr @ h + b_hr)
///   z = σ(W_iz @ x + b_iz + W_hz @ h + b_hz)
///   n = tanh(W_in @ x + b_in + r * (W_hn @ h + b_hn))
///   h' = (1 - z) * n + z * h
///
/// Weight layout: [W_ir; W_iz; W_in] concatenated along rows (3*hidden × input).
struct GRUCell {
    let hiddenDim: Int
    /// [3*hidden, input]
    let weightIH: Tensor
    /// [3*hidden, hidden]
    let weightHH: Tensor
    /// [3*hidden]
    let biasIH: Tensor
    /// [3*hidden]
    let biasHH: Tensor

    /// Run one GRU step.
    ///
    /// - Parameters:
    ///   - x: Input vector [inputDim].
    ///   - h: Previous hidden state [hiddenDim].
    /// - Returns: New hidden state [hiddenDim].
    func step(x: Tensor, h: Tensor) -> Tensor {
        // gate_x = W_ih @ x + b_ih   shape: [3*hidden]
        let gateX = weightIH.matvec(x).add(biasIH)
        // gate_h = W_hh @ h + b_hh   shape: [3*hidden]
        let gateH = weightHH.matvec(h).add(biasHH)

        let hd = hiddenDim

        // Split into r, z, n gates
        let rxSlice = Array(gateX.data[0..<hd])
        let zxSlice = Array(gateX.data[hd..<2*hd])
        let nxSlice = Array(gateX.data[2*hd..<3*hd])

        let rhSlice = Array(gateH.data[0..<hd])
        let zhSlice = Array(gateH.data[hd..<2*hd])
        let nhSlice = Array(gateH.data[2*hd..<3*hd])

        let rx = Tensor.vector(rxSlice)
        let zx = Tensor.vector(zxSlice)
        let nx = Tensor.vector(nxSlice)
        let rh = Tensor.vector(rhSlice)
        let zh = Tensor.vector(zhSlice)
        let nh = Tensor.vector(nhSlice)

        // r = sigmoid(rx + rh)
        let r = rx.add(rh).sigmoid()
        // z = sigmoid(zx + zh)
        let z = zx.add(zh).sigmoid()
        // n = tanh(nx + r * nh)
        let n = nx.add(r.mul(nh)).tanh()

        // h' = (1 - z) * n + z * h
        let ones = Tensor(data: [Float](repeating: 1.0, count: hd), shape: [hd])
        let oneMinusZ = ones.sub(z)
        return oneMinusZ.mul(n).add(z.mul(h))
    }
}

// MARK: - Encoder

/// Bidirectional GRU encoder.
///
/// Input: sequence of char indices → Embedding → BiGRU
/// Output: encoder outputs [seqLen, hidden*2], final hidden [hidden*2]
struct EncoderLayer {
    let embedding: Tensor     // [vocabSize, embDim]
    let forward: GRUCell      // forward GRU
    let backward: GRUCell     // backward GRU
    let hiddenDim: Int

    /// Encode a sequence of character indices.
    ///
    /// - Parameter charIndices: Array of integer char indices (no batch dim).
    /// - Returns: (outputs, hidden) where outputs is [seqLen, 2*hidden]
    ///           and hidden is [2*hidden].
    func encode(_ charIndices: [Int]) -> (outputs: Tensor, hidden: Tensor) {
        let seqLen = charIndices.count
        let embDim = embedding.shape[1]

        // Look up embeddings
        var embeddings = [[Float]]()
        embeddings.reserveCapacity(seqLen)
        for idx in charIndices {
            embeddings.append(Array(embedding.data[idx * embDim..<(idx + 1) * embDim]))
        }

        // Forward pass
        var hFwd = Tensor.zeros([hiddenDim])
        var fwdOutputs = [[Float]]()
        fwdOutputs.reserveCapacity(seqLen)
        for t in 0..<seqLen {
            let x = Tensor.vector(embeddings[t])
            hFwd = forward.step(x: x, h: hFwd)
            fwdOutputs.append(hFwd.data)
        }

        // Backward pass
        var hBwd = Tensor.zeros([hiddenDim])
        var bwdOutputs = [[Float]](repeating: [], count: seqLen)
        for t in stride(from: seqLen - 1, through: 0, by: -1) {
            let x = Tensor.vector(embeddings[t])
            hBwd = backward.step(x: x, h: hBwd)
            bwdOutputs[t] = hBwd.data
        }

        // Concatenate forward and backward outputs → [seqLen, 2*hidden]
        var outputData = [Float]()
        outputData.reserveCapacity(seqLen * hiddenDim * 2)
        for t in 0..<seqLen {
            outputData.append(contentsOf: fwdOutputs[t])
            outputData.append(contentsOf: bwdOutputs[t])
        }
        let outputs = Tensor(data: outputData, shape: [seqLen, hiddenDim * 2])

        // Final hidden: cat(hFwd, hBwd) — hBwd is the state after
        // processing the first token (index 0) from right to left.
        let hidden = Tensor.cat(
            Tensor.vector(fwdOutputs[seqLen - 1]),
            Tensor.vector(bwdOutputs[0])
        )

        return (outputs, hidden)
    }
}

// MARK: - Attention

/// Bahdanau (additive) attention.
///
/// energy = v @ tanh(W @ [dec_hidden ; enc_output_t])
/// weights = softmax(energy)
/// context = weights @ encoder_outputs
struct AttentionLayer {
    /// [decHidden, encDim + decHidden] — W_attn
    let attnWeight: Tensor
    /// [1, decHidden] — v
    let vWeight: Tensor
    let decHiddenDim: Int

    /// Compute context vector.
    ///
    /// - Parameters:
    ///   - hidden: Decoder hidden state [decHidden].
    ///   - encoderOutputs: Encoder outputs [seqLen, encDim].
    /// - Returns: (context [encDim], weights [seqLen]).
    func attend(hidden: Tensor, encoderOutputs: Tensor) -> (context: Tensor, weights: Tensor) {
        let seqLen = encoderOutputs.shape[0]
        let encDim = encoderOutputs.shape[1]

        // For each encoder timestep, compute energy
        var scores = [Float](repeating: 0, count: seqLen)
        let vRow = Array(vWeight.data)  // [decHidden]

        for t in 0..<seqLen {
            // Concatenate [hidden, enc_output_t]
            let encSlice = Array(encoderOutputs.data[t * encDim..<(t + 1) * encDim])
            let concat = Tensor.cat(Tensor.vector(hidden.data), Tensor.vector(encSlice))
            // energy_t = tanh(W_attn @ concat)
            let energy = attnWeight.matvec(concat).tanh()
            // score_t = v @ energy (dot product)
            var dotResult: Float = 0
            for i in 0..<decHiddenDim {
                dotResult += vRow[i] * energy.data[i]
            }
            scores[t] = dotResult
        }

        // Softmax over scores
        let weights = Tensor.vector(scores).softmax()

        // Context = weighted sum of encoder outputs
        var contextData = [Float](repeating: 0, count: encDim)
        for t in 0..<seqLen {
            let w = weights.data[t]
            let base = t * encDim
            for i in 0..<encDim {
                contextData[i] += w * encoderOutputs.data[base + i]
            }
        }

        return (Tensor.vector(contextData), weights)
    }
}

// MARK: - Decoder

/// Single-step GRU decoder with attention context.
///
/// Input: previous token index + hidden + context
/// Output: logits [vocabSize] + new hidden
struct DecoderLayer {
    let embedding: Tensor     // [vocabSize, embDim]
    let gru: GRUCell
    let fcWeight: Tensor      // [vocabSize, hiddenDim + encDim + embDim]
    let fcBias: Tensor        // [vocabSize]
    let embDim: Int

    /// Run one decoder step.
    ///
    /// - Parameters:
    ///   - tokenIdx: The input token index.
    ///   - hidden: Previous hidden state [hiddenDim].
    ///   - context: Attention context vector [encDim].
    /// - Returns: (logits [vocabSize], newHidden [hiddenDim]).
    func step(tokenIdx: Int, hidden: Tensor, context: Tensor) -> (logits: Tensor, hidden: Tensor) {
        // Embedding lookup
        let emb = Tensor.vector(Array(embedding.data[tokenIdx * embDim..<(tokenIdx + 1) * embDim]))

        // GRU input: cat(emb, context)
        let gruInput = Tensor.cat(emb, context)
        let newHidden = gru.step(x: gruInput, h: hidden)

        // FC: cat(hidden, context, emb) → logits
        let fcInput = Tensor.cat([newHidden, context, emb])
        let logits = fcWeight.matvec(fcInput).add(fcBias)

        return (logits, newHidden)
    }
}

// MARK: - Bridge

/// Bridge layer: maps encoder hidden + conditioning embeddings → decoder
/// initial hidden state.
///
/// h_dec0 = tanh(W_bridge @ [enc_hidden ; mode_emb ; tense_emb ; person_emb] + b_bridge)
struct BridgeLayer {
    let weight: Tensor    // [hiddenDim, encHidden + 3*condDim]
    let bias: Tensor      // [hiddenDim]
    let modeEmb: Tensor   // [nModes, condDim]
    let tenseEmb: Tensor  // [nTenses, condDim]
    let personEmb: Tensor // [nPersons, condDim]
    let condDim: Int

    /// Compute decoder initial hidden state.
    func initDecoder(
        encoderHidden: Tensor,
        modeIdx: Int,
        tenseIdx: Int,
        personIdx: Int
    ) -> Tensor {
        let mSlice = Array(modeEmb.data[modeIdx * condDim..<(modeIdx + 1) * condDim])
        let tSlice = Array(tenseEmb.data[tenseIdx * condDim..<(tenseIdx + 1) * condDim])
        let pSlice = Array(personEmb.data[personIdx * condDim..<(personIdx + 1) * condDim])

        let input = Tensor.cat([
            encoderHidden,
            Tensor.vector(mSlice),
            Tensor.vector(tSlice),
            Tensor.vector(pSlice),
        ])

        return weight.matvec(input).add(bias).tanh()
    }
}
