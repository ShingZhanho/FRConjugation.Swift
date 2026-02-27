// ConjugationTests.swift — Unit tests for the FRConjugation Swift library.
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
        // Walk from #filePath → Sources/FRConjugation/Resources/
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
        // Try bundled resources first, fall back to file path
        do {
            conjugator = try Conjugator()
            return
        } catch {
            // Bundle.module may not work during development builds — fall back
        }
        guard let dir = modelDir else {
            print("⚠️  Could not find model files — skipping tests")
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

    func testAuxiliary() throws {
        let conj = try c
        let avoirAux = conj.auxiliary(for: "parler")
        XCTAssertTrue(avoirAux.avoir)
        XCTAssertFalse(avoirAux.etre)

        let etreAux = conj.auxiliary(for: "aller")
        XCTAssertTrue(etreAux.etre)
    }

    // MARK: - Indicatif Présent

    func testIndicatifPresent() throws {
        let conj = try c
        XCTAssertEqual(
            conj.conjugate("parler", mode: .indicatif, tense: .present, person: .firstPersonSingular),
            "parle"
        )
        XCTAssertEqual(
            conj.conjugate("finir", mode: .indicatif, tense: .present, person: .firstPersonPlural),
            "finissons"
        )
        XCTAssertEqual(
            conj.conjugate("aller", mode: .indicatif, tense: .present, person: .firstPersonSingular),
            "vais"
        )
        XCTAssertEqual(
            conj.conjugate("être", mode: .indicatif, tense: .present, person: .thirdPersonMasculinePlural),
            "sont"
        )
    }

    // MARK: - Full Paradigm

    func testFullParadigm() throws {
        let conj = try c
        let forms = conj.conjugate("avoir", mode: .indicatif, tense: .present)
        XCTAssertEqual(forms[.firstPersonSingular], "ai")
        XCTAssertEqual(forms[.secondPersonSingular], "as")
        XCTAssertEqual(forms[.firstPersonPlural], "avons")
        XCTAssertEqual(forms[.secondPersonPlural], "avez")
        XCTAssertGreaterThanOrEqual(forms.count, 6)
    }

    // MARK: - Other Tenses

    func testImparfait() throws {
        let conj = try c
        XCTAssertEqual(
            conj.conjugate("parler", mode: .indicatif, tense: .imparfait, person: .firstPersonSingular),
            "parlais"
        )
    }

    func testFuturSimple() throws {
        let conj = try c
        XCTAssertEqual(
            conj.conjugate("aller", mode: .indicatif, tense: .futurSimple, person: .firstPersonSingular),
            "irai"
        )
    }

    func testConditionnelPresent() throws {
        let conj = try c
        XCTAssertEqual(
            conj.conjugate("vouloir", mode: .conditionnel, tense: .present, person: .firstPersonSingular),
            "voudrais"
        )
    }

    func testSubjonctifPresent() throws {
        let conj = try c
        XCTAssertEqual(
            conj.conjugate("faire", mode: .subjonctif, tense: .present, person: .firstPersonSingular),
            "fasse"
        )
    }

    // MARK: - Impératif

    func testImperatif() throws {
        let conj = try c
        let forms = conj.conjugate("parler", mode: .imperatif, tense: .present)
        XCTAssertEqual(forms.count, 3)
        XCTAssertEqual(forms[.secondPersonSingular], "parle")
        XCTAssertEqual(forms[.firstPersonPlural], "parlons")
        XCTAssertEqual(forms[.secondPersonPlural], "parlez")
    }

    // MARK: - Compound Tenses

    func testPasseCompose() throws {
        let conj = try c
        // "aller" uses être → agreement
        XCTAssertEqual(
            conj.conjugate("aller", mode: .indicatif, tense: .passeCompose,
                           person: .thirdPersonFeminineSingular),
            "est allée"
        )
        // "parler" uses avoir → no agreement
        XCTAssertEqual(
            conj.conjugate("parler", mode: .indicatif, tense: .passeCompose,
                           person: .firstPersonSingular),
            "ai parlé"
        )
    }

    // MARK: - Participles

    func testParticiple() throws {
        let conj = try c
        XCTAssertEqual(conj.participle("parler"), "parlé")
        XCTAssertEqual(conj.participle("partir", form: .passeFemininPlural), "parties")
        XCTAssertEqual(conj.participle("finir", form: .present), "finissant")
    }

    // MARK: - Impersonal / Defective Verbs

    func testImpersonalFlags() throws {
        let conj = try c
        XCTAssertTrue(conj.isImpersonal("falloir"))
        XCTAssertTrue(conj.isImpersonal("neiger"))
        XCTAssertFalse(conj.isImpersonal("pleuvoir"))  // pleuvoir is third-person-only
        XCTAssertFalse(conj.isImpersonal("parler"))

        XCTAssertTrue(conj.isThirdPersonOnly("pleuvoir"))
        XCTAssertTrue(conj.isThirdPersonOnly("advenir"))
        XCTAssertFalse(conj.isThirdPersonOnly("falloir"))
        XCTAssertFalse(conj.isThirdPersonOnly("parler"))

        XCTAssertTrue(conj.isDefective("falloir"))
        XCTAssertTrue(conj.isDefective("pleuvoir"))
        XCTAssertFalse(conj.isDefective("parler"))
    }

    func testFalloirOnlyIl() throws {
        let conj = try c
        // "falloir" should only conjugate for il/elle
        XCTAssertNotNil(conj.conjugate("falloir", mode: .indicatif, tense: .present,
                                        person: .thirdPersonMasculineSingular))
        XCTAssertEqual(
            conj.conjugate("falloir", mode: .indicatif, tense: .present,
                           person: .thirdPersonMasculineSingular),
            "faut"
        )
        // Other persons should return nil
        XCTAssertNil(conj.conjugate("falloir", mode: .indicatif, tense: .present,
                                     person: .firstPersonSingular))
        XCTAssertNil(conj.conjugate("falloir", mode: .indicatif, tense: .present,
                                     person: .secondPersonSingular))
        XCTAssertNil(conj.conjugate("falloir", mode: .indicatif, tense: .present,
                                     person: .firstPersonPlural))
        XCTAssertNil(conj.conjugate("falloir", mode: .indicatif, tense: .present,
                                     person: .secondPersonPlural))
        XCTAssertNil(conj.conjugate("falloir", mode: .indicatif, tense: .present,
                                     person: .thirdPersonMasculinePlural))

        // Full paradigm should only have il/elle
        let forms = conj.conjugate("falloir", mode: .indicatif, tense: .present)
        XCTAssertEqual(forms.count, 2)  // 3sm + 3sf only
        XCTAssertNotNil(forms[.thirdPersonMasculineSingular])
        XCTAssertNotNil(forms[.thirdPersonFeminineSingular])
        XCTAssertNil(forms[.firstPersonSingular])

        // No impératif forms
        let imperatifForms = conj.conjugate("falloir", mode: .imperatif, tense: .present)
        XCTAssertTrue(imperatifForms.isEmpty)
    }

    func testPleuvoirThirdPersonOnly() throws {
        let conj = try c
        // "pleuvoir" should conjugate for 3rd person only (singular + plural)
        XCTAssertNotNil(conj.conjugate("pleuvoir", mode: .indicatif, tense: .present,
                                        person: .thirdPersonMasculineSingular))
        XCTAssertNotNil(conj.conjugate("pleuvoir", mode: .indicatif, tense: .present,
                                        person: .thirdPersonMasculinePlural))
        // Non-3rd-person should return nil
        XCTAssertNil(conj.conjugate("pleuvoir", mode: .indicatif, tense: .present,
                                     person: .firstPersonSingular))
        XCTAssertNil(conj.conjugate("pleuvoir", mode: .indicatif, tense: .present,
                                     person: .secondPersonPlural))

        // Full paradigm should only have 3rd-person entries
        let forms = conj.conjugate("pleuvoir", mode: .indicatif, tense: .present)
        XCTAssertEqual(forms.count, 4)  // 3sm, 3sf, 3pm, 3pf
        XCTAssertNil(forms[.firstPersonSingular])
        XCTAssertNil(forms[.secondPersonSingular])
        XCTAssertNil(forms[.firstPersonPlural])
        XCTAssertNil(forms[.secondPersonPlural])
    }

    func testImpersonalParticiplesStillWork() throws {
        let conj = try c
        // Participles should not be affected by impersonal filtering
        let pp = conj.participle("falloir", form: .passeMasculinSingular)
        XCTAssertEqual(pp, "fallu")
    }

    func testImpersonalCompoundTense() throws {
        let conj = try c
        // Compound tenses should also be filtered
        XCTAssertNotNil(conj.conjugate("falloir", mode: .indicatif, tense: .passeCompose,
                                        person: .thirdPersonMasculineSingular))
        XCTAssertNil(conj.conjugate("falloir", mode: .indicatif, tense: .passeCompose,
                                     person: .firstPersonSingular))
    }

    // MARK: - Invalid Input

    func testUnknownVerb() throws {
        let conj = try c
        // The neural model will attempt to conjugate unknown input.
        // hasVerb() is the correct way to check if a verb is known.
        XCTAssertFalse(conj.hasVerb("xyzzy"))
        // conjugate still returns something (the model guesses) — not nil.
        let result = conj.conjugate("xyzzy", mode: .indicatif, tense: .present,
                                     person: .firstPersonSingular)
        // Just verify it doesn't crash; value is unimportant.
        _ = result
    }

    func testInvalidCombination() throws {
        let conj = try c
        // Participe mode with a person → should return nil or the participle
        let result = conj.conjugate("parler", mode: .participe, tense: .present,
                                     person: .firstPersonSingular)
        // Not asserting specific behavior — just shouldn't crash
        _ = result
    }
}
