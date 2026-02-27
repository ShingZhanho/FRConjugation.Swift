// Types.swift — Enums and value types for the FRConjugation library.

import Foundation

// MARK: - Mode

/// French grammatical mode (mood).
public enum Mode: String, CaseIterable, Sendable {
    case indicatif
    case subjonctif
    case conditionnel
    case imperatif
    case participe
}

// MARK: - Tense

/// French verb tense.
///
/// Not every tense is valid in every mode. Invalid combinations return `nil`.
///
/// ```
/// Indicatif:     .present  .imparfait  .passeSimple  .futurSimple
///                .passeCompose  .plusQueParfait  .passeAnterieur  .futurAnterieur
/// Subjonctif:    .present  .imparfait  .passe  .plusQueParfait
/// Conditionnel:  .present  .passe
/// Impératif:     .present  .passe
/// Participe:     .present  (passé forms via participle() method)
/// ```
public enum Tense: String, CaseIterable, Sendable {
    // Simple tenses
    case present        = "present"
    case imparfait      = "imparfait"
    case passeSimple    = "passe_simple"
    case futurSimple    = "futur_simple"

    // Compound tenses
    case passeCompose   = "passe_compose"
    case plusQueParfait  = "plus_que_parfait"
    case passeAnterieur = "passe_anterieur"
    case futurAnterieur = "futur_anterieur"

    // Generic "passé" — used in conditionnel, subjonctif, impératif
    case passe          = "passe"
}

// MARK: - Person

/// Grammatical person (with gender for third person).
///
/// French conjugation distinguishes masculine/feminine in the third person,
/// particularly for compound tenses with être auxiliary (past participle
/// agreement).
public enum Person: String, CaseIterable, Sendable, Hashable {
    case firstPersonSingular  = "1s"
    case secondPersonSingular = "2s"
    case thirdPersonMasculineSingular = "3sm"
    case thirdPersonFeminineSingular  = "3sf"
    case firstPersonPlural    = "1p"
    case secondPersonPlural   = "2p"
    case thirdPersonMasculinePlural   = "3pm"
    case thirdPersonFemininePlural    = "3pf"

    /// The French subject pronoun for this person.
    ///
    ///     Person.firstPersonSingular.pronoun  // "je"
    ///     Person.thirdPersonFemininePlural.pronoun  // "elles"
    public var pronoun: String {
        switch self {
        case .firstPersonSingular:  return "je"
        case .secondPersonSingular: return "tu"
        case .thirdPersonMasculineSingular: return "il"
        case .thirdPersonFeminineSingular:  return "elle"
        case .firstPersonPlural:    return "nous"
        case .secondPersonPlural:   return "vous"
        case .thirdPersonMasculinePlural:   return "ils"
        case .thirdPersonFemininePlural:    return "elles"
        }
    }

    /// Short label (e.g. "1s", "3pf").
    public var shortLabel: String { rawValue }
}

// MARK: - Participle Form

/// Past/present participle forms.
///
/// French past participles agree in gender and number with the subject
/// when the auxiliary is "être".
public enum ParticipleForm: String, CaseIterable, Sendable {
    case present                = "present"
    case passeMasculinSingular  = "passe_sm"
    case passeFemininSingular   = "passe_sf"
    case passeMasculinPlural    = "passe_pm"
    case passeFemininPlural     = "passe_pf"
}

// MARK: - Auxiliary

/// Auxiliary verb information for compound tenses.
///
///     let aux = conjugator.auxiliary(for: "aller")
///     aux.etre        // true — "aller" uses être
///     aux.avoir       // false
///     aux.pronominal  // true — "s'en aller" exists
public struct Auxiliary: Equatable, Hashable, Sendable, CustomStringConvertible {

    /// Uses "avoir" as auxiliary.
    public let avoir: Bool

    /// Uses "être" as auxiliary.
    public let etre: Bool

    /// Has a pronominal (reflexive) form.
    public let pronominal: Bool

    public var description: String {
        var parts: [String] = []
        if avoir { parts.append("avoir") }
        if etre  { parts.append("être") }
        if pronominal { parts.append("pronominal") }
        return parts.isEmpty ? "none" : parts.joined(separator: ", ")
    }
}

// MARK: - Errors

/// Errors thrown by ``Conjugator``.
public enum ConjugationError: Error, LocalizedError {
    /// The model directory could not be loaded.
    case modelLoadFailed(path: String)

    public var errorDescription: String? {
        switch self {
        case .modelLoadFailed(let path):
            return "Failed to load conjugation model from: \(path)"
        }
    }
}
