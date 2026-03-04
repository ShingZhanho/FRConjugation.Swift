// Types.swift — Enums and value types for the FRConjugation library.

import Foundation

// MARK: - Voice

/// French grammatical voice.
///
/// The database distinguishes five voices.  The vast majority of verbs
/// use ``activeAvoir`` (transitive / intransitive with *avoir*).
///
///     Voice.activeAvoir      // il a mangé
///     Voice.activeEtre       // il est allé
///     Voice.active           // (rare) impersonal active
///     Voice.passive          // il est mangé
///     Voice.pronominal       // il se lave
public enum Voice: String, CaseIterable, Sendable, Hashable {
    case activeAvoir  = "voix_active_avoir"
    case activeEtre   = "voix_active_etre"
    case active       = "voix_active"
    case passive      = "voix_passive"
    case pronominal   = "voix_prono"
}

// MARK: - Mode

/// French grammatical mode (mood).
public enum Mode: String, CaseIterable, Sendable, Hashable {
    case indicatif
    case subjonctif
    case conditionnel
    case imperatif
    case participe
}

// MARK: - Tense

/// French verb tense.
///
/// Not every tense is valid in every mode or voice.  Invalid
/// combinations return `nil`.  Use ``Conjugator/tenses(_:voice:mode:)``
/// to discover valid tenses for a given verb, voice and mode.
///
/// ## Simple tenses
/// `.present`, `.imparfait`, `.passeSimple`, `.futurSimple`
///
/// ## Compound tenses
/// `.passeCompose`, `.plusQueParfait`, `.passeAnterieur`,
/// `.futurAnterieur`, `.passe`
///
/// ## Participle sub-forms (mode: `.participe`)
/// `.passeMasculinSingulier`, `.passeFemininSingulier`,
/// `.passeMasculinPluriel`,   `.passeFemininPluriel`,
/// `.passeCompoundMasculinSingulier`, `.passeCompoundFemininSingulier`,
/// `.passeCompoundMasculinPluriel`, `.passeCompoundFemininPluriel`
public enum Tense: String, CaseIterable, Sendable, Hashable {
    // Simple
    case present        = "present"
    case imparfait      = "imparfait"
    case passeSimple    = "passe_simple"
    case futurSimple    = "futur_simple"

    // Compound
    case passeCompose       = "passe_compose"
    case plusQueParfait      = "plus_que_parfait"
    case passeAnterieur     = "passe_anterieur"
    case futurAnterieur     = "futur_anterieur"

    // Generic passé (conditionnel, subjonctif, impératif)
    case passe = "passe"

    // Participle sub-forms (simple)
    case passeMasculinSingulier = "passe_sm"
    case passeFemininSingulier  = "passe_sf"
    case passeMasculinPluriel   = "passe_pm"
    case passeFemininPluriel    = "passe_pf"

    // Participle sub-forms (compound)
    case passeCompoundMasculinSingulier = "passe_compound_sm"
    case passeCompoundFemininSingulier  = "passe_compound_sf"
    case passeCompoundMasculinPluriel   = "passe_compound_pm"
    case passeCompoundFemininPluriel    = "passe_compound_pf"
}

// MARK: - Person

/// Grammatical person with gender.
///
/// French conjugation distinguishes masculine and feminine forms for all
/// persons (due to participle agreement in compound tenses).
///
/// The person keys follow the pattern `{number}{plurality}{gender}`:
/// - Number: `1`, `2`, `3`
/// - Plurality: `s` (singular), `p` (plural)
/// - Gender: `m` (masculine), `f` (feminine)
public enum Person: String, CaseIterable, Sendable, Hashable {
    case firstSingularMasculine   = "1sm"
    case firstSingularFeminine    = "1sf"
    case secondSingularMasculine  = "2sm"
    case secondSingularFeminine   = "2sf"
    case thirdSingularMasculine   = "3sm"
    case thirdSingularFeminine    = "3sf"
    case firstPluralMasculine     = "1pm"
    case firstPluralFeminine      = "1pf"
    case secondPluralMasculine    = "2pm"
    case secondPluralFeminine     = "2pf"
    case thirdPluralMasculine     = "3pm"
    case thirdPluralFeminine      = "3pf"

    /// The French subject pronoun for this person.
    ///
    ///     Person.firstSingularMasculine.pronoun  // "je"
    ///     Person.thirdPluralFeminine.pronoun      // "elles"
    public var pronoun: String {
        switch self {
        case .firstSingularMasculine, .firstSingularFeminine:   return "je"
        case .secondSingularMasculine, .secondSingularFeminine: return "tu"
        case .thirdSingularMasculine:  return "il"
        case .thirdSingularFeminine:   return "elle"
        case .firstPluralMasculine, .firstPluralFeminine:       return "nous"
        case .secondPluralMasculine, .secondPluralFeminine:     return "vous"
        case .thirdPluralMasculine:    return "ils"
        case .thirdPluralFeminine:     return "elles"
        }
    }

    /// Short label (e.g. "1sm", "3pf").
    public var shortLabel: String { rawValue }
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
