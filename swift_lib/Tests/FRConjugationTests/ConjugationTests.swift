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
        // "aller" uses être → agreement
        XCTAssertEqual(
            conj.conjugate("aller", voice: .activeEtre, mode: .indicatif, tense: .passeCompose,
                           person: .thirdSingularFeminine),
            "est allée"
        )
        // "parler" uses avoir → no agreement
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
}
