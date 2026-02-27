// Conjugator.swift — Idiomatic Swift interface to the French verb conjugation model.

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
/// The directory must contain the files exported by `export_weights.py`:
/// - `model.json`  — vocabulary, metadata, weight manifest
/// - `weights.bin` — raw float32 weight data
public final class Conjugator {

    // MARK: - Private State

    private let engine: InferenceEngine

    // MARK: - Linguistic Constants

    /// Persons valid for impersonal verbs (il/elle only).
    private static let impersonalPersons: Set<String> = ["3sm", "3sf"]

    /// Persons valid for third-person-only verbs (il/elle/ils/elles).
    private static let thirdPersonOnlyPersons: Set<String> = ["3sm", "3sf", "3pm", "3pf"]

    /// Simple tenses that the model predicts directly.
    private static let simpleTenses: Set<String> = [
        "indicatif|present", "indicatif|imparfait",
        "indicatif|passe_simple", "indicatif|futur_simple",
        "conditionnel|present",
        "subjonctif|present", "subjonctif|imparfait",
        "imperatif|present",
    ]

    /// Compound tense → (aux mode, aux tense) mapping.
    private static let compoundTenseMap: [String: (String, String)] = [
        "indicatif|passe_compose": ("indicatif", "present"),
        "indicatif|plus_que_parfait": ("indicatif", "imparfait"),
        "indicatif|passe_anterieur": ("indicatif", "passe_simple"),
        "indicatif|futur_anterieur": ("indicatif", "futur_simple"),
        "conditionnel|passe": ("conditionnel", "present"),
        "subjonctif|passe": ("subjonctif", "present"),
        "subjonctif|plus_que_parfait": ("subjonctif", "imparfait"),
        "imperatif|passe": ("imperatif", "present"),
    ]

    /// Person → participle form mapping for être-auxiliary agreement.
    private static let ppFormeEtre: [String: String] = [
        "1s": "passe_sm", "2s": "passe_sm",
        "3sm": "passe_sm", "3sf": "passe_sf",
        "1p": "passe_pm", "2p": "passe_pm",
        "3pm": "passe_pm", "3pf": "passe_pf",
    ]

    // MARK: - Initialization

    /// Load the conjugation model from a directory path.
    ///
    /// - Parameter path: Absolute path to the directory containing
    ///   `model.json` and `weights.bin`.
    /// - Throws: ``ConjugationError/modelLoadFailed(path:)`` if loading fails.
    public init(modelDirectory path: String) throws {
        let url = URL(fileURLWithPath: path, isDirectory: true)
        self.engine = try InferenceEngine(modelDirectory: url)
    }

    /// Load the conjugation model from a directory URL.
    ///
    /// - Parameter url: File URL to the model directory.
    /// - Throws: ``ConjugationError/modelLoadFailed(path:)`` if loading fails.
    public convenience init(modelDirectory url: URL) throws {
        try self.init(modelDirectory: url.path)
    }

    /// Load the conjugation model from the bundled resources.
    ///
    /// This initializer uses the `model.json` and `weights.bin` files
    /// that are embedded in the Swift package / framework resources.
    ///
    /// - Throws: ``ConjugationError/modelLoadFailed(path:)`` if the
    ///   bundled resources cannot be found.
    public convenience init() throws {
        let bundle = Self.resourceBundle
        guard let jsonURL = bundle.url(forResource: "model", withExtension: "json"),
              let _ = bundle.url(forResource: "weights", withExtension: "bin") else {
            throw ConjugationError.modelLoadFailed(path: "bundled resources")
        }
        let dir = jsonURL.deletingLastPathComponent()
        try self.init(modelDirectory: dir)
    }

    /// Locate the resource bundle for both SPM and Xcode framework builds.
    private static var resourceBundle: Bundle {
        #if SWIFT_PACKAGE
        return Bundle.module
        #else
        return Bundle(for: Conjugator.self)
        #endif
    }

    // MARK: - Properties

    /// The number of verbs the model recognises.
    public var verbCount: Int {
        engine.knownVerbs.count
    }

    // MARK: - Queries

    /// Whether the model recognises this infinitive.
    ///
    ///     conjugator.hasVerb("parler")  // true
    ///     conjugator.hasVerb("xyzzy")   // false
    public func hasVerb(_ infinitive: String) -> Bool {
        engine.knownVerbs.contains(infinitive)
    }

    /// Whether a verb beginning with *h* is h-aspiré (no elision/liaison).
    ///
    ///     conjugator.isHAspire("haïr")    // true
    ///     conjugator.isHAspire("habiter")  // false
    public func isHAspire(_ infinitive: String) -> Bool {
        engine.hAspire.contains(infinitive)
    }

    /// Whether a verb is impersonal (only conjugates for il/elle).
    ///
    /// Impersonal verbs like *falloir* and weather verbs (*neiger*, *bruiner*, …)
    /// only have third-person singular forms.
    ///
    ///     conjugator.isImpersonal("falloir")  // true
    ///     conjugator.isImpersonal("neiger")   // true
    ///     conjugator.isImpersonal("pleuvoir") // false  (has plural forms too)
    ///     conjugator.isImpersonal("parler")   // false
    public func isImpersonal(_ infinitive: String) -> Bool {
        engine.impersonalVerbs.contains(infinitive)
    }

    /// Whether a verb only conjugates in the third person (il/elle/ils/elles).
    ///
    /// Some verbs like *pleuvoir* (figurative: "les coups pleuvent") and
    /// *advenir* accept both singular and plural third-person subjects but
    /// no first- or second-person forms.
    ///
    ///     conjugator.isThirdPersonOnly("pleuvoir") // true
    ///     conjugator.isThirdPersonOnly("advenir")  // true
    ///     conjugator.isThirdPersonOnly("falloir")  // false  (use isImpersonal)
    ///     conjugator.isThirdPersonOnly("parler")   // false
    public func isThirdPersonOnly(_ infinitive: String) -> Bool {
        engine.thirdPersonOnlyVerbs.contains(infinitive)
    }

    /// Whether a verb is defective — either impersonal or third-person-only.
    ///
    /// This is the union of ``isImpersonal(_:)`` and ``isThirdPersonOnly(_:)``.
    public func isDefective(_ infinitive: String) -> Bool {
        isImpersonal(infinitive) || isThirdPersonOnly(infinitive)
    }

    /// Auxiliary verb information for compound tenses.
    ///
    ///     let aux = conjugator.auxiliary(for: "aller")
    ///     // Auxiliary(avoir: false, etre: true, pronominal: true)
    public func auxiliary(for infinitive: String) -> Auxiliary {
        let usesEtre = engine.etreVerbs.contains(infinitive)
        return Auxiliary(
            avoir: !usesEtre,
            etre: usesEtre,
            pronominal: engine.pronoVerbs.contains(infinitive)
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
        singleForm(infinitive, mode: mode.rawValue, tense: tense.rawValue, person: person.rawValue)
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
    ///     conjugator.participle("parler")
    ///     // → "parlé" (default: passé masculin singulier)
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
        getParticiple(infinitive, forme: form.rawValue)
    }

    // MARK: - Private Helpers

    /// Direct neural prediction wrapper.
    private func predict(_ infinitive: String, mode: String, tense: String, person: String) -> String? {
        engine.predict(infinitive: infinitive, mode: mode, tense: tense, person: person)
    }

    /// Retrieve a participle form, redirecting invariable PP verbs.
    private func getParticiple(_ infinitive: String, forme: String) -> String? {
        var f = forme
        if f == "passe_sf" || f == "passe_pm" || f == "passe_pf" {
            if engine.invariablePPVerbs.contains(infinitive) {
                f = "passe_sm"
            }
        }
        return predict(infinitive, mode: "participe", tense: f, person: "-")
    }

    /// Check whether the person is valid for this verb (impersonal / defective filtering).
    private func isPersonAllowed(_ infinitive: String, person: String) -> Bool {
        if engine.impersonalVerbs.contains(infinitive) {
            return Self.impersonalPersons.contains(person)
        }
        if engine.thirdPersonOnlyVerbs.contains(infinitive) {
            return Self.thirdPersonOnlyPersons.contains(person)
        }
        return true
    }

    /// Route a single form through simple/compound/participle logic.
    private func singleForm(_ infinitive: String, mode: String, tense: String, person: String) -> String? {
        if mode == "participe" {
            return getParticiple(infinitive, forme: tense)
        }

        // Block invalid persons for impersonal / defective verbs
        guard isPersonAllowed(infinitive, person: person) else {
            return nil
        }

        let key = "\(mode)|\(tense)"

        if Self.simpleTenses.contains(key) {
            return predict(infinitive, mode: mode, tense: tense, person: person)
        }

        if let (auxMode, auxTense) = Self.compoundTenseMap[key] {
            return conjugateCompound(
                infinitive, mode: mode,
                auxMode: auxMode, auxTense: auxTense, person: person
            )
        }

        return nil
    }

    /// Build a compound tense: aux conjugation + past participle.
    private func conjugateCompound(
        _ infinitive: String,
        mode: String,
        auxMode: String,
        auxTense: String,
        person: String
    ) -> String? {
        let auxVerb = engine.etreVerbs.contains(infinitive) ? "être" : "avoir"
        guard let auxForm = predict(auxVerb, mode: auxMode, tense: auxTense, person: person) else {
            return nil
        }
        let ppForme: String
        if auxVerb == "être" {
            ppForme = Self.ppFormeEtre[person] ?? "passe_sm"
        } else {
            ppForme = "passe_sm"
        }
        guard let pp = getParticiple(infinitive, forme: ppForme) else {
            return nil
        }
        return "\(auxForm) \(pp)"
    }
}
