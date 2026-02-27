/**
 * ConjugationModel.swift — Swift wrapper for the French verb conjugation model.
 *
 * Uses the Objective-C FRConjugation class (bridged via Bridging-Header.h),
 * which itself calls the C/LibTorch library.
 *
 * Usage:
 *     let model = try ConjugationModel(directory: "/path/to/model_files")
 *     let form = model.conjugate("parler", mode: "indicatif", tense: "present", person: "1s")
 *     // → "parle"
 */

import Foundation

public enum ConjugationError: Error, LocalizedError {
    case modelLoadFailed(directory: String)

    public var errorDescription: String? {
        switch self {
        case .modelLoadFailed(let dir):
            return "Failed to load conjugation model from directory: \(dir)"
        }
    }
}

/// A French verb conjugation engine backed by a seq2seq neural network.
public final class ConjugationModel {

    private let objcModel: FRConjugation

    // MARK: - Initialization

    /// Load the model from a directory containing the exported files.
    ///
    /// Expects: conjugation_encoder.pt, conjugation_bridge.pt,
    ///          conjugation_attention.pt, conjugation_decoder.pt,
    ///          conjugation_meta.json
    ///
    /// - Parameter directory: Path to the model directory.
    /// - Throws: `ConjugationError.modelLoadFailed` if loading fails.
    public init(directory: String) throws {
        guard let model = FRConjugation(modelDirectory: directory) else {
            throw ConjugationError.modelLoadFailed(directory: directory)
        }
        self.objcModel = model
    }

    // MARK: - Properties

    /// The number of verbs known to the model.
    public var verbCount: Int {
        return Int(objcModel.verbCount)
    }

    // MARK: - Query

    /// Whether the model knows this verb.
    public func hasVerb(_ infinitive: String) -> Bool {
        return objcModel.hasVerb(infinitive)
    }

    /// Whether the verb begins with an aspirate h.
    public func isHAspire(_ infinitive: String) -> Bool {
        return objcModel.isHAspire(infinitive)
    }

    /// Auxiliary verbs used for compound tenses.
    /// Returns e.g. ["avoir"], ["être"], ["avoir", "pronominal"].
    public func auxiliary(for infinitive: String) -> [String] {
        return objcModel.auxiliaryForVerb(infinitive) as [String]
    }

    // MARK: - Conjugation

    /// Conjugate a single form.
    ///
    /// - Parameters:
    ///   - infinitive: The verb infinitive (e.g. "parler").
    ///   - mode: Grammatical mode (e.g. "indicatif", "ind").
    ///   - tense: Tense (e.g. "present", "passe_compose").
    ///   - person: Person (e.g. "1s", "je", "3sf").
    /// - Returns: The conjugated form, or nil if not available.
    public func conjugate(
        _ infinitive: String,
        mode: String,
        tense: String,
        person: String
    ) -> String? {
        return objcModel.conjugate(infinitive, mode: mode, tense: tense, person: person)
    }

    /// Get a participle form.
    ///
    /// - Parameters:
    ///   - infinitive: The verb infinitive.
    ///   - forme: "present", "passe_sm", "passe_sf", "passe_pm", "passe_pf".
    /// - Returns: The participle, or nil if not available.
    public func participle(
        _ infinitive: String,
        forme: String = "passe_sm"
    ) -> String? {
        return objcModel.participle(infinitive, forme: forme)
    }
}
