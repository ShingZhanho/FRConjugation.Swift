// ConjugationTests.swift -- Unit tests for the FRConjugation Swift library.
//
// The model is now bundled as Swift package resources (model.json + weights.bin).
// No external model directory or C library required.
//
// Run:  cd swift_lib && swift test

import XCTest
@testable import FRConjugation

final class ConjugationTests: XCTestCase {

    /// Resolve model directory: environment override or Resources/ via #filePath.
    static var modelDir: String? {
        if let env = ProcessInfo.processInfo.environment["MODEL_DIR"] {
            return env
        }
        let swiftLib = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()  // FRConjugationTests/
            .deletingLastPathComponent()  // Tests/
            .deletingLastPathComponent()  // swift_lib/
        let candidate = swiftLib
            .appendingPathComponent("Sources")
            .appendingPathComponent("FRConjugation")
            .appendingPathComponent("Resources")
        if FileManager.default.fileExists(atPath: candidate.appendingPathComponent("model.json").path) {
            return candidate.path
        }
        return nil
    }

    static var conjugator: Conjugator?

    override class func setUp() {
        super.setUp()
        do {
            conjugator = try Conjugator()
            return
        } catch {
            // Bundle.module may not work during development builds -- fall back
        }
        guard let dir = modelDir else {
            print("[WARNING]  Could not find model files -- skipping tests")
            return
        }
        do {
            conjugator = try Conjugator(modelDirectory: dir)
        } catch {
            XCTFail("Failed to load model: \(error)")
        }
    }

    private var c: Conjugator {
        get throws {
            guard let conj = Self.conjugator else {
                throw XCTSkip("Model not available")
            }
            return conj
        }
    }

    // MARK: - Basic Queries

    func testVerbCount() throws {
        let conj = try c
        XCTAssertGreaterThan(conj.verbCount, 6000)
    }

    func testAllVerbs() throws {
        let conj = try c
        let verbs = conj.allVerbs
        XCTAssertEqual(verbs.count, conj.verbCount)
        // Must be sorted
        XCTAssertEqual(verbs, verbs.sorted())
        // Spot-check known verbs
        XCTAssertTrue(verbs.contains("parler"))
        XCTAssertTrue(verbs.contains("être"))
        XCTAssertFalse(verbs.contains("xyzzy"))
    }

    func testHasVerb() throws {
        let conj = try c
        XCTAssertTrue(conj.hasVerb("parler"))
        XCTAssertTrue(conj.hasVerb("être"))
        XCTAssertFalse(conj.hasVerb("xyzzy"))
    }

    func testHAspire() throws {
        let conj = try c
        XCTAssertTrue(conj.isHAspire("haïr"))
        XCTAssertFalse(conj.isHAspire("habiter"))
    }

    // MARK: - Structure Queries

    func testVoices() throws {
        let conj = try c
        let v = conj.voices("parler")
        XCTAssertTrue(v.contains(.activeAvoir))
        XCTAssertFalse(v.isEmpty)
    }

    func testModes() throws {
        let conj = try c
        let m = conj.modes("parler", voice: .activeAvoir)
        XCTAssertTrue(m.contains(.indicatif))
        XCTAssertTrue(m.contains(.subjonctif))
        XCTAssertTrue(m.contains(.conditionnel))
    }

    func testTenses() throws {
        let conj = try c
        let t = conj.tenses("parler", voice: .activeAvoir, mode: .indicatif)
        XCTAssertTrue(t.contains(.present))
        XCTAssertTrue(t.contains(.imparfait))
        XCTAssertTrue(t.contains(.passeCompose))
    }

    func testPersons() throws {
        let conj = try c
        let p = conj.persons("parler", voice: .activeAvoir, mode: .indicatif, tense: .present)
        XCTAssertTrue(p.contains(.firstSingularMasculine))
        XCTAssertTrue(p.contains(.thirdPluralFeminine))
    }

    func testUnknownVerbStructure() throws {
        let conj = try c
        XCTAssertTrue(conj.voices("xyzzy").isEmpty)
        XCTAssertTrue(conj.modes("xyzzy", voice: .activeAvoir).isEmpty)
    }

    // MARK: - Indicatif Présent (Active Avoir)

    func testIndicatifPresent() throws {
        let conj = try c
        XCTAssertEqual(
            conj.conjugate("parler", voice: .activeAvoir, mode: .indicatif,
                           tense: .present, person: .firstSingularMasculine),
            "parle"
        )
        XCTAssertEqual(
            conj.conjugate("finir", voice: .activeAvoir, mode: .indicatif,
                           tense: .present, person: .firstPluralMasculine),
            "finissons"
        )
        XCTAssertEqual(
            conj.conjugate("être", voice: .activeAvoir, mode: .indicatif,
                           tense: .present, person: .thirdPluralMasculine),
            "sont"
        )
    }

    // MARK: - Full Paradigm

    func testFullParadigm() throws {
        let conj = try c
        let forms = conj.conjugate("avoir", voice: .activeAvoir, mode: .indicatif, tense: .present)
        XCTAssertEqual(forms[.firstSingularMasculine], "ai")
        XCTAssertEqual(forms[.secondSingularMasculine], "as")
        XCTAssertEqual(forms[.firstPluralMasculine], "avons")
        XCTAssertEqual(forms[.secondPluralMasculine], "avez")
        XCTAssertGreaterThanOrEqual(forms.count, 6)
    }

    // MARK: - Other Tenses

    func testImparfait() throws {
        let conj = try c
        XCTAssertEqual(
            conj.conjugate("parler", voice: .activeAvoir, mode: .indicatif,
                           tense: .imparfait, person: .firstSingularMasculine),
            "parlais"
        )
    }

    func testFuturSimple() throws {
        let conj = try c
        XCTAssertEqual(
            conj.conjugate("aller", voice: .activeEtre, mode: .indicatif,
                           tense: .futurSimple, person: .firstSingularMasculine),
            "irai"
        )
    }

    func testConditionnelPresent() throws {
        let conj = try c
        XCTAssertEqual(
            conj.conjugate("vouloir", voice: .activeAvoir, mode: .conditionnel,
                           tense: .present, person: .firstSingularMasculine),
            "voudrais"
        )
    }

    func testSubjonctifPresent() throws {
        let conj = try c
        XCTAssertEqual(
            conj.conjugate("faire", voice: .activeAvoir, mode: .subjonctif,
                           tense: .present, person: .firstSingularMasculine),
            "fasse"
        )
    }

    // MARK: - Impératif

    func testImperatif() throws {
        let conj = try c
        let forms = conj.conjugate("parler", voice: .activeAvoir, mode: .imperatif, tense: .present)
        XCTAssertEqual(forms[.secondSingularMasculine], "parle")
        XCTAssertEqual(forms[.firstPluralMasculine], "parlons")
        XCTAssertEqual(forms[.secondPluralMasculine], "parlez")
    }

    // MARK: - Compound Tenses (Directly Predicted)

    func testPasseCompose() throws {
        let conj = try c
        // "aller" uses être -> agreement
        XCTAssertEqual(
            conj.conjugate("aller", voice: .activeEtre, mode: .indicatif, tense: .passeCompose,
                           person: .thirdSingularFeminine),
            "est allée"
        )
        // "parler" uses avoir -> no agreement
        XCTAssertEqual(
            conj.conjugate("parler", voice: .activeAvoir, mode: .indicatif, tense: .passeCompose,
                           person: .firstSingularMasculine),
            "ai parlé"
        )
    }

    // MARK: - Participles

    func testParticiple() throws {
        let conj = try c
        XCTAssertEqual(
            conj.participle("parler", voice: .activeAvoir, tense: .passeMasculinSingulier),
            "parlé"
        )
        XCTAssertEqual(
            conj.participle("partir", voice: .activeAvoir, tense: .passeFemininPluriel),
            "parties"
        )
        XCTAssertEqual(
            conj.participle("finir", voice: .activeAvoir, tense: .present),
            "finissant"
        )
    }

    func testParticiples() throws {
        let conj = try c
        let parts = conj.participles("parler", voice: .activeAvoir)
        XCTAssertFalse(parts.isEmpty)
        XCTAssertEqual(parts[.present], "parlant")
        XCTAssertEqual(parts[.passeMasculinSingulier], "parlé")
    }

    // MARK: - Defective Verbs (via verb_structure)

    func testFalloirOnlyThirdSingular() throws {
        let conj = try c
        // "falloir" should conjugate via verb_structure
        XCTAssertNotNil(conj.conjugate("falloir", voice: .activeAvoir, mode: .indicatif,
                                        tense: .present, person: .thirdSingularMasculine))
        XCTAssertEqual(
            conj.conjugate("falloir", voice: .activeAvoir, mode: .indicatif,
                           tense: .present, person: .thirdSingularMasculine),
            "faut"
        )
        // Invalid persons return nil (via verb_structure validation)
        XCTAssertNil(conj.conjugate("falloir", voice: .activeAvoir, mode: .indicatif,
                                     tense: .present, person: .firstSingularMasculine))
        XCTAssertNil(conj.conjugate("falloir", voice: .activeAvoir, mode: .indicatif,
                                     tense: .present, person: .secondPluralMasculine))
    }

    // MARK: - Voice Variants

    func testAllerActiveEtre() throws {
        let conj = try c
        XCTAssertEqual(
            conj.conjugate("aller", voice: .activeEtre, mode: .indicatif,
                           tense: .present, person: .firstSingularMasculine),
            "vais"
        )
    }

    func testPronoVerb() throws {
        let conj = try c
        // "laver" in pronominal voice
        let v = conj.voices("laver")
        if v.contains(.pronominal) {
            let form = conj.conjugate("laver", voice: .pronominal, mode: .indicatif,
                                       tense: .present, person: .firstSingularMasculine)
            XCTAssertNotNil(form)
        }
    }

    // MARK: - Third Person Singular Neutral (3sn -- reciprocal verbs)

    func testThirdSingularNeutral() throws {
        let conj = try c
        let form = conj.conjugate("entraider", voice: .pronominal, mode: .indicatif,
                                   tense: .present, person: .thirdSingularNeutral)
        XCTAssertEqual(form, "s'entraide")
    }

    func testThirdSingularNeutralInPersons() throws {
        let conj = try c
        let persons = conj.persons("entraider", voice: .pronominal, mode: .indicatif, tense: .present)
        XCTAssertTrue(persons.contains(.thirdSingularNeutral))
    }

    func testThirdSingularNeutralPronoun() {
        XCTAssertEqual(Person.thirdSingularNeutral.pronoun, "on")
    }

    // MARK: - Gendered Present Participles (passive voice)

    func testGenderedPresentParticiple() throws {
        let conj = try c
        XCTAssertEqual(
            conj.participle("aimer", voice: .passive, tense: .presentMasculinSingulier),
            "étant aimé"
        )
        XCTAssertEqual(
            conj.participle("aimer", voice: .passive, tense: .presentFemininSingulier),
            "étant aimée"
        )
        XCTAssertEqual(
            conj.participle("aimer", voice: .passive, tense: .presentMasculinPluriel),
            "étant aimés"
        )
        XCTAssertEqual(
            conj.participle("aimer", voice: .passive, tense: .presentFemininPluriel),
            "étant aimées"
        )
    }

    func testGenderedPresentParticipleInTenses() throws {
        let conj = try c
        let tenses = conj.tenses("aimer", voice: .passive, mode: .participe)
        XCTAssertTrue(tenses.contains(.presentMasculinSingulier))
        XCTAssertTrue(tenses.contains(.presentFemininSingulier))
        XCTAssertTrue(tenses.contains(.presentMasculinPluriel))
        XCTAssertTrue(tenses.contains(.presentFemininPluriel))
    }

    // MARK: - Aggregate Conjugation

    func testConjugateModeTenses() throws {
        let conj = try c
        let allTenses = conj.conjugate("parler", voice: .activeAvoir, mode: .indicatif)
        XCTAssertFalse(allTenses.isEmpty)
        XCTAssertNotNil(allTenses[.present])
        XCTAssertNotNil(allTenses[.imparfait])
    }

    func testConjugateVoice() throws {
        let conj = try c
        let allModes = conj.conjugate("parler", voice: .activeAvoir)
        XCTAssertFalse(allModes.isEmpty)
        XCTAssertNotNil(allModes[.indicatif])
    }

    func testConjugateAll() throws {
        let conj = try c
        let all = conj.conjugate("parler")
        XCTAssertNotNil(all)
        XCTAssertFalse(all!.isEmpty)
    }

    // MARK: - Invalid Input

    func testUnknownVerb() throws {
        let conj = try c
        XCTAssertFalse(conj.hasVerb("xyzzy"))
        // conjugate returns nil for unknown verb (verb_structure lookup fails)
        let result = conj.conjugate("xyzzy", voice: .activeAvoir, mode: .indicatif,
                                     tense: .present, person: .firstSingularMasculine)
        XCTAssertNil(result)
    }

    func testInvalidCombination() throws {
        let conj = try c
        // Invalid voice for a verb should return nil
        let result = conj.conjugate("falloir", voice: .passive, mode: .indicatif,
                                     tense: .present, person: .firstSingularMasculine)
        XCTAssertNil(result)
    }

    // MARK: - LRU Cache

    func testCacheDefaultCapacity() throws {
        let conj = try c
        XCTAssertEqual(conj.cacheCapacity, Conjugator.defaultCacheSize)
    }

    func testCacheCustomCapacity() throws {
        let conj = try Conjugator(cacheSize: 128)
        XCTAssertEqual(conj.cacheCapacity, 128)
        XCTAssertEqual(conj.cacheCount, 0)
    }

    func testCacheDisabled() throws {
        let conj = try Conjugator(cacheSize: 0)
        XCTAssertEqual(conj.cacheCapacity, 0)
        // Conjugation should still work
        XCTAssertEqual(
            conj.conjugate("parler", voice: .activeAvoir, mode: .indicatif,
                           tense: .present, person: .firstSingularMasculine),
            "parle"
        )
        XCTAssertEqual(conj.cacheCount, 0)
    }

    func testCachePopulatesOnConjugate() throws {
        let conj = try Conjugator(cacheSize: 16)
        XCTAssertEqual(conj.cacheCount, 0)

        // First call populates the cache
        _ = conj.conjugate("parler", voice: .activeAvoir, mode: .indicatif,
                           tense: .present, person: .firstSingularMasculine)
        XCTAssertEqual(conj.cacheCount, 1)

        // Same verb, different form: still 1 verb in cache
        _ = conj.conjugate("parler", voice: .activeAvoir, mode: .indicatif,
                           tense: .imparfait, person: .firstSingularMasculine)
        XCTAssertEqual(conj.cacheCount, 1)

        // Different verb: now 2
        _ = conj.conjugate("finir", voice: .activeAvoir, mode: .indicatif,
                           tense: .present, person: .firstSingularMasculine)
        XCTAssertEqual(conj.cacheCount, 2)
    }

    func testCacheHitReturnsSameResult() throws {
        let conj = try Conjugator(cacheSize: 16)
        let first = conj.conjugate("aller", voice: .activeEtre, mode: .indicatif,
                                    tense: .present, person: .firstSingularMasculine)
        let second = conj.conjugate("aller", voice: .activeEtre, mode: .indicatif,
                                     tense: .present, person: .firstSingularMasculine)
        XCTAssertEqual(first, second)
        XCTAssertEqual(first, "vais")
    }

    func testCacheEvictsLRU() throws {
        let conj = try Conjugator(cacheSize: 2)

        // Fill cache: parler, finir
        _ = conj.conjugate("parler", voice: .activeAvoir, mode: .indicatif,
                           tense: .present, person: .firstSingularMasculine)
        _ = conj.conjugate("finir", voice: .activeAvoir, mode: .indicatif,
                           tense: .present, person: .firstSingularMasculine)
        XCTAssertEqual(conj.cacheCount, 2)

        // Adding a 3rd verb should evict the LRU ("parler")
        _ = conj.conjugate("avoir", voice: .activeAvoir, mode: .indicatif,
                           tense: .present, person: .firstSingularMasculine)
        XCTAssertEqual(conj.cacheCount, 2)
    }

    func testClearCache() throws {
        let conj = try Conjugator(cacheSize: 16)

        _ = conj.conjugate("parler", voice: .activeAvoir, mode: .indicatif,
                           tense: .present, person: .firstSingularMasculine)
        XCTAssertGreaterThan(conj.cacheCount, 0)

        conj.clearCache()
        XCTAssertEqual(conj.cacheCount, 0)
    }

    func testCachePopulatesOnParticiple() throws {
        let conj = try Conjugator(cacheSize: 16)
        _ = conj.participle("parler", voice: .activeAvoir, tense: .passeMasculinSingulier)
        XCTAssertEqual(conj.cacheCount, 1)
    }

    func testSharedInstanceCacheSize() throws {
        // Reset the shared singleton so we can configure it
        Conjugator._resetShared()

        let shared = Conjugator.getShared(cacheSize: 32)
        XCTAssertEqual(shared.cacheCapacity, 32)

        // Second call ignores the parameter
        let shared2 = Conjugator.getShared(cacheSize: 999)
        XCTAssertTrue(shared === shared2)
        XCTAssertEqual(shared2.cacheCapacity, 32)

        // Reset so other tests get default
        Conjugator._resetShared()
    }

    // MARK: - Variant Forms (Spelling Alternatives)

    func testConjugateReturnsPrimaryForm() throws {
        let conj = try c
        // "abréger" futur_simple 1sm has two variants: abrégerai;abrègerai
        let form = conj.conjugate("abréger", voice: .activeAvoir, mode: .indicatif,
                                   tense: .futurSimple, person: .firstSingularMasculine)
        XCTAssertEqual(form, "abrégerai")
    }

    func testConjugateAlternativeReturnsSecondForm() throws {
        let conj = try c
        let alt = conj.conjugateAlternative("abréger", voice: .activeAvoir, mode: .indicatif,
                                             tense: .futurSimple, person: .firstSingularMasculine)
        XCTAssertEqual(alt, "abrègerai")
    }

    func testHasAlternativeFormTrue() throws {
        let conj = try c
        XCTAssertTrue(conj.hasAlternativeForm("abréger", voice: .activeAvoir, mode: .indicatif,
                                               tense: .futurSimple, person: .firstSingularMasculine))
    }

    func testHasAlternativeFormFalse() throws {
        let conj = try c
        // "parler" has no variant forms
        XCTAssertFalse(conj.hasAlternativeForm("parler", voice: .activeAvoir, mode: .indicatif,
                                                tense: .present, person: .firstSingularMasculine))
    }

    func testAlternativeFallsBackToDefault() throws {
        let conj = try c
        // "parler" has no alternatives -- both methods should return the same
        let primary = conj.conjugate("parler", voice: .activeAvoir, mode: .indicatif,
                                      tense: .present, person: .firstSingularMasculine)
        let alt = conj.conjugateAlternative("parler", voice: .activeAvoir, mode: .indicatif,
                                             tense: .present, person: .firstSingularMasculine)
        XCTAssertEqual(primary, alt)
        XCTAssertEqual(primary, "parle")
    }

    func testAlternativeInvalidCombinationReturnsNil() throws {
        let conj = try c
        XCTAssertNil(conj.conjugateAlternative("xyzzy", voice: .activeAvoir, mode: .indicatif,
                                                tense: .present, person: .firstSingularMasculine))
    }

    func testBatchConjugateReturnsPrimaryForms() throws {
        let conj = try c
        let forms = conj.conjugate("abréger", voice: .activeAvoir, mode: .indicatif, tense: .futurSimple)
        // Batch should return primary (first) form
        XCTAssertEqual(forms[.firstSingularMasculine], "abrégerai")
    }

    func testConditionnelAlternativeForm() throws {
        let conj = try c
        // abréger conditionnel present 1sm: abrégerais;abrègerais
        let primary = conj.conjugate("abréger", voice: .activeAvoir, mode: .conditionnel,
                                      tense: .present, person: .firstSingularMasculine)
        let alt = conj.conjugateAlternative("abréger", voice: .activeAvoir, mode: .conditionnel,
                                             tense: .present, person: .firstSingularMasculine)
        XCTAssertEqual(primary, "abrégerais")
        XCTAssertEqual(alt, "abrègerais")
    }

    // MARK: - getPronoun

    func testGetPronounBasicConsonant() throws {
        let conj = try c
        // "parle" starts with consonant -- no elision
        let pronoun = conj.getPronoun("parler", voice: .activeAvoir, mode: .indicatif,
                                       tense: .present, person: .firstSingularMasculine)
        XCTAssertEqual(pronoun, "je ")
    }

    func testGetPronounJeElisionVowel() throws {
        let conj = try c
        // "aime" starts with vowel -- je -> j'
        let pronoun = conj.getPronoun("aimer", voice: .activeAvoir, mode: .indicatif,
                                       tense: .present, person: .firstSingularMasculine)
        XCTAssertEqual(pronoun, "j'")
    }

    func testGetPronounJeElisionCompound() throws {
        let conj = try c
        // passe compose "ai parle" starts with vowel 'a' -- je -> j'
        let pronoun = conj.getPronoun("parler", voice: .activeAvoir, mode: .indicatif,
                                       tense: .passeCompose, person: .firstSingularMasculine)
        XCTAssertEqual(pronoun, "j'")
    }

    func testGetPronounHAspireNoElision() throws {
        let conj = try c
        // "hair" is h-aspire -- no elision
        XCTAssertTrue(conj.isHAspire("ha\u{00EF}r"))
        let pronoun = conj.getPronoun("ha\u{00EF}r", voice: .activeAvoir, mode: .indicatif,
                                       tense: .present, person: .firstSingularMasculine)
        XCTAssertEqual(pronoun, "je ")
    }

    func testGetPronounHMuetElision() throws {
        let conj = try c
        // "habiter" is h-muet -- elision
        XCTAssertFalse(conj.isHAspire("habiter"))
        let pronoun = conj.getPronoun("habiter", voice: .activeAvoir, mode: .indicatif,
                                       tense: .present, person: .firstSingularMasculine)
        XCTAssertEqual(pronoun, "j'")
    }

    func testGetPronounThirdPersonNoElision() throws {
        let conj = try c
        // "il" does not elide regardless of form
        let pronoun = conj.getPronoun("aimer", voice: .activeAvoir, mode: .indicatif,
                                       tense: .present, person: .thirdSingularMasculine)
        XCTAssertEqual(pronoun, "il ")
    }

    func testGetPronounSubjonctifQue() throws {
        let conj = try c
        // subjonctif: "que je parle"
        let pronoun = conj.getPronoun("parler", voice: .activeAvoir, mode: .subjonctif,
                                       tense: .present, person: .firstSingularMasculine)
        XCTAssertEqual(pronoun, "que je ")
    }

    func testGetPronounSubjonctifQueElision() throws {
        let conj = try c
        // subjonctif je + vowel: "que j'aime"
        let pronoun = conj.getPronoun("aimer", voice: .activeAvoir, mode: .subjonctif,
                                       tense: .present, person: .firstSingularMasculine)
        XCTAssertEqual(pronoun, "que j'")
    }

    func testGetPronounSubjonctifQuApostrophe() throws {
        let conj = try c
        // subjonctif 3sm: "qu'il" (il starts with vowel)
        let pronoun = conj.getPronoun("parler", voice: .activeAvoir, mode: .subjonctif,
                                       tense: .present, person: .thirdSingularMasculine)
        XCTAssertEqual(pronoun, "qu'il ")
    }

    func testGetPronounSubjonctifQuElles() throws {
        let conj = try c
        // subjonctif 3pf: "qu'elles"
        let pronoun = conj.getPronoun("parler", voice: .activeAvoir, mode: .subjonctif,
                                       tense: .present, person: .thirdPluralFeminine)
        XCTAssertEqual(pronoun, "qu'elles ")
    }

    func testGetPronounSubjonctifQueNous() throws {
        let conj = try c
        // subjonctif 1pm: "que nous" (nous starts with consonant)
        let pronoun = conj.getPronoun("parler", voice: .activeAvoir, mode: .subjonctif,
                                       tense: .present, person: .firstPluralMasculine)
        XCTAssertEqual(pronoun, "que nous ")
    }

    func testGetPronounImperatifNil() throws {
        let conj = try c
        let pronoun = conj.getPronoun("parler", voice: .activeAvoir, mode: .imperatif,
                                       tense: .present, person: .secondSingularMasculine)
        XCTAssertNil(pronoun)
    }

    func testGetPronounParticipeNil() throws {
        let conj = try c
        let pronoun = conj.getPronoun("parler", voice: .activeAvoir, mode: .participe,
                                       tense: .present, person: .firstSingularMasculine)
        XCTAssertNil(pronoun)
    }

    func testGetPronounUnknownVerbNil() throws {
        let conj = try c
        let pronoun = conj.getPronoun("zzzzz", voice: .activeAvoir, mode: .indicatif,
                                       tense: .present, person: .firstSingularMasculine)
        XCTAssertNil(pronoun)
    }

    func testGetPronounOtherPersons() throws {
        let conj = try c
        XCTAssertEqual(conj.getPronoun("parler", voice: .activeAvoir, mode: .indicatif,
                                        tense: .present, person: .secondSingularMasculine), "tu ")
        XCTAssertEqual(conj.getPronoun("parler", voice: .activeAvoir, mode: .indicatif,
                                        tense: .present, person: .firstPluralMasculine), "nous ")
        XCTAssertEqual(conj.getPronoun("parler", voice: .activeAvoir, mode: .indicatif,
                                        tense: .present, person: .secondPluralMasculine), "vous ")
        XCTAssertEqual(conj.getPronoun("parler", voice: .activeAvoir, mode: .indicatif,
                                        tense: .present, person: .thirdPluralFeminine), "elles ")
        // 3sn (on) only exists for pronominal voice
        XCTAssertEqual(conj.getPronoun("entraider", voice: .pronominal, mode: .indicatif,
                                        tense: .present, person: .thirdSingularNeutral), "on ")
    }

    // MARK: - conjugateWithPronoun

    func testConjugateWithPronounBasic() throws {
        let conj = try c
        let result = conj.conjugateWithPronoun("parler", voice: .activeAvoir, mode: .indicatif,
                                                tense: .present, person: .firstSingularMasculine)
        XCTAssertEqual(result, "je parle")
    }

    func testConjugateWithPronounElision() throws {
        let conj = try c
        let result = conj.conjugateWithPronoun("aimer", voice: .activeAvoir, mode: .indicatif,
                                                tense: .present, person: .firstSingularMasculine)
        XCTAssertEqual(result, "j'aime")
    }

    func testConjugateWithPronounCompound() throws {
        let conj = try c
        let result = conj.conjugateWithPronoun("parler", voice: .activeAvoir, mode: .indicatif,
                                                tense: .passeCompose, person: .firstSingularMasculine)
        XCTAssertEqual(result, "j'ai parl\u{00E9}")
    }

    func testConjugateWithPronounSubjonctif() throws {
        let conj = try c
        let result = conj.conjugateWithPronoun("parler", voice: .activeAvoir, mode: .subjonctif,
                                                tense: .present, person: .thirdSingularMasculine)
        XCTAssertEqual(result, "qu'il parle")
    }

    func testConjugateWithPronounSubjonctifJeElision() throws {
        let conj = try c
        let result = conj.conjugateWithPronoun("aimer", voice: .activeAvoir, mode: .subjonctif,
                                                tense: .present, person: .firstSingularMasculine)
        XCTAssertEqual(result, "que j'aime")
    }

    func testConjugateWithPronounImperatifBareForm() throws {
        let conj = try c
        let result = conj.conjugateWithPronoun("parler", voice: .activeAvoir, mode: .imperatif,
                                                tense: .present, person: .secondSingularMasculine)
        XCTAssertEqual(result, "parle")
    }

    func testConjugateWithPronounUnknownNil() throws {
        let conj = try c
        let result = conj.conjugateWithPronoun("zzzzz", voice: .activeAvoir, mode: .indicatif,
                                                tense: .present, person: .firstSingularMasculine)
        XCTAssertNil(result)
    }

    func testConjugateWithPronounHAspire() throws {
        let conj = try c
        let result = conj.conjugateWithPronoun("ha\u{00EF}r", voice: .activeAvoir, mode: .indicatif,
                                                tense: .present, person: .firstSingularMasculine)
        XCTAssertEqual(result, "je hais")
    }

    func testConjugateWithPronounHMuet() throws {
        let conj = try c
        let result = conj.conjugateWithPronoun("habiter", voice: .activeAvoir, mode: .indicatif,
                                                tense: .present, person: .firstSingularMasculine)
        XCTAssertEqual(result, "j'habite")
    }

    // MARK: - conjugateAlternativeWithPronoun

    func testConjugateAlternativeWithPronounBasic() throws {
        let conj = try c
        let result = conj.conjugateAlternativeWithPronoun("abr\u{00E9}ger", voice: .activeAvoir,
                                                           mode: .indicatif, tense: .futurSimple,
                                                           person: .firstSingularMasculine)
        XCTAssertEqual(result, "j'abr\u{00E8}gerai")
    }

    func testConjugateAlternativeWithPronounFallback() throws {
        let conj = try c
        // No alternative for "parler" -- should fall back to primary form
        let result = conj.conjugateAlternativeWithPronoun("parler", voice: .activeAvoir,
                                                           mode: .indicatif, tense: .present,
                                                           person: .firstSingularMasculine)
        XCTAssertEqual(result, "je parle")
    }

    func testConjugateAlternativeWithPronounImperatif() throws {
        let conj = try c
        let result = conj.conjugateAlternativeWithPronoun("parler", voice: .activeAvoir,
                                                           mode: .imperatif, tense: .present,
                                                           person: .secondSingularMasculine)
        XCTAssertEqual(result, "parle")
    }

    func testConjugateAlternativeWithPronounUnknownNil() throws {
        let conj = try c
        let result = conj.conjugateAlternativeWithPronoun("zzzzz", voice: .activeAvoir,
                                                           mode: .indicatif, tense: .present,
                                                           person: .firstSingularMasculine)
        XCTAssertNil(result)
    }

    // MARK: - Homonym Support

    func testHasHomonymsReturnsFalseForNormalVerb() throws {
        let conj = try c
        XCTAssertFalse(conj.hasHomonyms("parler"))
    }

    func testHomonymCountReturnsOneForNormalVerb() throws {
        let conj = try c
        XCTAssertEqual(conj.homonymCount("parler"), 1)
    }

    func testHomonymIndicesReturnsEmptyForNormalVerb() throws {
        let conj = try c
        XCTAssertTrue(conj.homonymIndices("parler").isEmpty)
    }

    func testHomonymIndexNilDefaultsToNormalBehaviour() throws {
        let conj = try c
        // Passing homonymIndex: nil to a non-homonym verb works normally
        let form = conj.conjugate("parler", voice: .activeAvoir, mode: .indicatif,
                                   tense: .present, person: .firstSingularMasculine,
                                   homonymIndex: nil)
        XCTAssertEqual(form, "parle")
    }

    func testAllVerbsExcludesSuffixedHomonyms() throws {
        let conj = try c
        let verbs = conj.allVerbs
        // No verb in allVerbs should end with _1, _2, etc.
        for verb in verbs {
            if let underscoreRange = verb.range(of: "_", options: .backwards) {
                let suffix = verb[verb.index(after: underscoreRange.lowerBound)...]
                XCTAssertNil(Int(suffix), "allVerbs should not contain suffixed homonym '\(verb)'")
            }
        }
    }

    func testHasVerbRecognisesHomonymBaseName() throws {
        let conj = try c
        // If the model has any homonym verbs, the base name should be recognised
        let verbs = conj.allVerbs
        for verb in verbs {
            XCTAssertTrue(conj.hasVerb(verb),
                          "hasVerb should recognise '\(verb)' from allVerbs")
        }
    }

    func testVerbCountMatchesAllVerbsCount() throws {
        let conj = try c
        XCTAssertEqual(conj.verbCount, conj.allVerbs.count)
    }
}
