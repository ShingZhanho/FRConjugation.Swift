/**
 * main.swift — Example / smoke test for the Swift conjugation wrapper.
 *
 * Build & run after setting up the Xcode project with all dependencies.
 */

import Foundation

let modelDir = CommandLine.arguments.count > 1
    ? CommandLine.arguments[1]
    : "."

do {
    let model = try ConjugationModel(directory: modelDir)
    print("Model loaded — \(model.verbCount) verbs\n")

    let tests: [(String, String, String, String, String)] = [
        ("parler",  "indicatif", "present",       "1s",  "parle"),
        ("finir",   "indicatif", "present",       "1s",  "finis"),
        ("aller",   "indicatif", "present",       "1s",  "vais"),
        ("être",    "indicatif", "present",       "1s",  "suis"),
        ("avoir",   "indicatif", "present",       "1s",  "ai"),
        ("parler",  "indicatif", "passe_compose", "1s",  "ai parlé"),
        ("aller",   "indicatif", "passe_compose", "3sf", "est allée"),
    ]

    var passed = 0, failed = 0

    for (verb, mode, tense, person, expected) in tests {
        if let result = model.conjugate(verb, mode: mode, tense: tense, person: person),
           result == expected {
            print("  [PASS] \(verb) \(mode).\(tense).\(person) → \(result)")
            passed += 1
        } else {
            let got = model.conjugate(verb, mode: mode, tense: tense, person: person) ?? "nil"
            print("  [FAIL] \(verb) \(mode).\(tense).\(person) → \(got) (expected \(expected))")
            failed += 1
        }
    }

    // Participles
    if let pp = model.participle("parler", forme: "present"), pp == "parlant" {
        print("  [PASS] parler participe present → \(pp)")
        passed += 1
    } else {
        print("  [FAIL] parler participe present")
        failed += 1
    }

    // Helpers
    print("")
    print("  has_verb(parler):    \(model.hasVerb("parler"))")
    print("  has_verb(xyzfake):   \(model.hasVerb("xyzfake"))")
    print("  auxiliary(parler):   \(model.auxiliary(for: "parler"))")
    print("  auxiliary(aller):    \(model.auxiliary(for: "aller"))")
    print("  isHAspire(hurler):   \(model.isHAspire("hurler"))")

    print("\n========================================")
    print("  \(passed)/\(passed + failed) tests passed")
    print("========================================")

} catch {
    print("Error: \(error.localizedDescription)")
}
