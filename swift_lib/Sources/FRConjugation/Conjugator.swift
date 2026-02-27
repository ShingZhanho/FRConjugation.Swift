// Conjugator.swift — Idiomatic Swift interface to the French verb conjugation model.

import CFRConjugation
import Foundation

/// A French verb conjugation engine backed by a character-level seq2seq
/// neural network with Bahdanau attention.
///
/// ## Overview
///
/// `Conjugator` loads a pre-trained model from disk and exposes a fully
/// typed Swift API for conjugating French verbs.  Every mode, tense, and
/// person is represented by an enum — no raw strings required.
///
/// ```swift
/// let c = try Conjugator(modelDirectory: "/path/to/model")
///
/// // Single form
/// c.conjugate("aller", mode: .indicatif, tense: .present, person: .firstPersonSingular)
/// // → "vais"
///
/// // Full paradigm for a tense
/// let forms = c.conjugate("finir", mode: .indicatif, tense: .imparfait)
/// for (person, form) in forms.sorted(by: { $0.key.rawValue < $1.key.rawValue }) {
///     print("\(person.pronoun) \(form)")
/// }
///
/// // Participle
/// c.participle("partir", form: .passeFemininPlural)
/// // → "parties"
/// ```
///
/// ## Thread Safety
///
/// Each `Conjugator` instance is **not** thread-safe.  If you need
/// concurrent access, create one instance per thread or protect access
/// with a lock.
///
/// ## Model Directory
///
/// The directory must contain the files exported by `c_wrapper/export_model.py`:
/// - `conjugation_encoder.pt`
/// - `conjugation_bridge.pt`
/// - `conjugation_attention.pt`
/// - `conjugation_decoder.pt`
/// - `conjugation_meta.json`
public final class Conjugator {

    // MARK: - Private State

    private let handle: OpaquePointer

    /// Internal buffer size for C function output.
    private static let bufferSize = 512

    // MARK: - Initialization

    /// Load the conjugation model from a directory path.
    ///
    /// - Parameter path: Absolute path to the directory containing the
    ///   exported model files.
    /// - Throws: ``ConjugationError/modelLoadFailed(path:)`` if loading fails.
    public init(modelDirectory path: String) throws {
        guard let h = fr_conjugation_load(path) else {
            throw ConjugationError.modelLoadFailed(path: path)
        }
        self.handle = h
    }

    /// Load the conjugation model from a directory URL.
    ///
    /// - Parameter url: File URL to the model directory.
    /// - Throws: ``ConjugationError/modelLoadFailed(path:)`` if loading fails.
    public convenience init(modelDirectory url: URL) throws {
        try self.init(modelDirectory: url.path)
    }

    deinit {
        fr_conjugation_free(handle)
    }

    // MARK: - Properties

    /// The number of verbs the model recognises.
    public var verbCount: Int {
        Int(fr_conjugation_verb_count(handle))
    }

    // MARK: - Queries

    /// Whether the model recognises this infinitive.
    ///
    ///     conjugator.hasVerb("parler")  // true
    ///     conjugator.hasVerb("xyzzy")   // false
    public func hasVerb(_ infinitive: String) -> Bool {
        fr_conjugation_has_verb(handle, infinitive)
    }

    /// Whether a verb beginning with *h* is h-aspiré (no elision/liaison).
    ///
    ///     conjugator.isHAspire("haïr")    // true
    ///     conjugator.isHAspire("habiter")  // false
    public func isHAspire(_ infinitive: String) -> Bool {
        fr_conjugation_is_h_aspire(handle, infinitive)
    }

    /// Auxiliary verb information for compound tenses.
    ///
    ///     let aux = conjugator.auxiliary(for: "aller")
    ///     // Auxiliary(avoir: false, etre: true, pronominal: true)
    public func auxiliary(for infinitive: String) -> Auxiliary {
        var buf = [CChar](repeating: 0, count: Self.bufferSize)
        let n = fr_conjugation_auxiliary(handle, infinitive, &buf, buf.count)
        guard n >= 0 else {
            return Auxiliary(avoir: false, etre: false, pronominal: false)
        }
        let raw = String(cString: buf).lowercased()
        let parts = Set(raw.split(separator: ",").map {
            $0.trimmingCharacters(in: .whitespaces)
        })
        return Auxiliary(
            avoir: parts.contains("avoir"),
            etre: parts.contains("être") || parts.contains("etre"),
            pronominal: parts.contains("pronominal")
        )
    }

    // MARK: - Conjugation

    /// Conjugate a single form.
    ///
    ///     conjugator.conjugate("aller",
    ///         mode: .indicatif, tense: .present, person: .firstPersonPlural)
    ///     // → "allons"
    ///
    /// - Returns: The conjugated form, or `nil` if the combination is
    ///   invalid or the verb is unknown.
    public func conjugate(
        _ infinitive: String,
        mode: Mode,
        tense: Tense,
        person: Person
    ) -> String? {
        var buf = [CChar](repeating: 0, count: Self.bufferSize)
        let n = fr_conjugation_conjugate(
            handle, infinitive,
            mode.rawValue, tense.rawValue, person.rawValue,
            &buf, buf.count
        )
        guard n >= 0 else { return nil }
        return String(cString: buf)
    }

    /// Conjugate all persons for a given mode and tense.
    ///
    /// For impératif, only the valid persons (tu, nous, vous) are included.
    ///
    ///     let forms = conjugator.conjugate("finir", mode: .indicatif, tense: .present)
    ///     // [.firstPersonSingular: "finis", .secondPersonSingular: "finis", ...]
    ///
    /// - Returns: A dictionary mapping each person to its conjugated form.
    ///   Persons that produce no result are omitted.
    public func conjugate(
        _ infinitive: String,
        mode: Mode,
        tense: Tense
    ) -> [Person: String] {
        let persons: [Person] = (mode == .imperatif)
            ? [.secondPersonSingular, .firstPersonPlural, .secondPersonPlural]
            : Person.allCases

        var result: [Person: String] = [:]
        result.reserveCapacity(persons.count)
        for p in persons {
            if let form = conjugate(infinitive, mode: mode, tense: tense, person: p) {
                result[p] = form
            }
        }
        return result
    }

    // MARK: - Participles

    /// Get a participle form.
    ///
    ///     conjugator.participle("partir")
    ///     // → "parti" (default: passé masculin singulier)
    ///
    ///     conjugator.participle("partir", form: .passeFemininPlural)
    ///     // → "parties"
    ///
    /// - Parameters:
    ///   - infinitive: The verb infinitive.
    ///   - form: Which participle form to retrieve (default: `.passeMasculinSingular`).
    /// - Returns: The participle, or `nil` if unavailable.
    public func participle(
        _ infinitive: String,
        form: ParticipleForm = .passeMasculinSingular
    ) -> String? {
        var buf = [CChar](repeating: 0, count: Self.bufferSize)
        let n = fr_conjugation_get_participle(
            handle, infinitive, form.rawValue, &buf, buf.count
        )
        guard n >= 0 else { return nil }
        return String(cString: buf)
    }
}
