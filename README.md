# FRConjugation.Swift

A **pure Swift** package for conjugating French verbs, powered by a character-level seq2seq
neural network with Bahdanau attention.

Covers **6,298 verbs** across **5 voices**, all modes, tenses, and
13 gender-explicit persons -- **2,553,802 conjugated forms** at **100% accuracy**.

**Zero external dependencies.** Uses Apple's Accelerate framework for fast matrix operations.
No LibTorch, no CoreML, no Python runtime needed.

```swift
import FRConjugation

let fr = Conjugator.getShared()  // loads bundled model from package resources

fr.conjugate("aller", voice: .activeEtre, mode: .indicatif,
             tense: .present, person: .firstSingularMasculine)
// -> "vais"

fr.conjugate("partir", voice: .activeEtre, mode: .indicatif,
             tense: .passeCompose, person: .thirdSingularFeminine)
// -> "est partie"

fr.participle("prendre", voice: .activeAvoir, tense: .passeFemininPluriel)
// -> "prises"
```

---

## Features

- **Fully typed API** -- `Voice`, `Mode`, `Tense`, and `Person` enums.
  No raw strings.
- **Five grammatical voices** -- active-avoir, active-etre, active,
  passive, and pronominal.
- **Simple & compound tenses** -- all 17 tenses predicted directly by the
  neural model (no rule-based composition).
- **Participles** -- present participle, 4 gendered present participle
  forms, 4 simple past participle forms, and 4 compound past participle
  forms.
- **13 gender-explicit persons** -- masculine/feminine distinction for
  every person, plus a neutral third-person singular (`3sn`) for
  reciprocal verbs.
- **Variant forms** -- some forms have alternative spellings (e.g. 1990
  reform variants separated by `;` in the database).  The primary form
  is returned by default; dedicated methods expose the alternative.
- **Structure queries** -- discover available voices, modes, tenses, and
  persons for any verb dynamically.
- **Verb listing** -- `allVerbs` returns a sorted list of all 6,298
  supported verb infinitives.
- **LRU cache** -- configurable per-instance verb cache for repeated lookups.
- **1990 reform** -- query whether a verb has reform spellings and get
  the variant form.
- **H-aspire** -- correctly detects aspirate-h verbs.
- **Pure Swift** -- no C library, no LibTorch, no external ML framework.
- **Lightweight** -- ~6 MB model, character-level neural network with
  Accelerate-backed inference.
- **Thread-safe** -- all public methods are synchronised; `Conjugator`
  conforms to `Sendable`.
- **App Store ready** -- no dynamic linking concerns, no large framework bundles.

---

## Requirements

| Requirement | Version |
|:------------|:--------|
| Swift       | >= 5.6  |
| macOS       | >= 10.15 |
| iOS         | >= 13   |

No other dependencies.

---

## Installation

Add the package dependency in your `Package.swift`:

```swift
dependencies: [
  .package(url: "https://github.com/ShingZhanho/FRConjugation.Swift.git", from: "4.0.0"),
]
```

Or in Xcode: **File > Add Package Dependencies** and enter the repository URL.

---

## API

### `Conjugator`

```swift
// Shared singleton (recommended) -- sync
let fr = Conjugator.getShared()

// Shared singleton -- async (won't block the main thread)
let fr = try await Conjugator.getShared()

// Configure cache size on first call (subsequent calls ignore the parameter)
let fr = Conjugator.getShared(cacheSize: 128)

// Async factory (creates a new instance, non-blocking)
let fr = try await Conjugator.load()

// Load from bundled resources (new instance each time)
let fr = try Conjugator()

// Custom cache size (default: 64 verbs; pass 0 to disable)
let fr = try Conjugator(cacheSize: 256)

// Or load from a custom directory containing model.json + weights.bin
let fr = try Conjugator(modelDirectory: "/path/to/model")
let fr = try Conjugator(modelDirectory: modelURL)

// Async load from a custom directory
let fr = try await Conjugator.load(modelDirectory: "/path/to/model")
```

#### Conjugation

```swift
// Single form
fr.conjugate("parler", voice: .activeAvoir, mode: .indicatif,
             tense: .present, person: .firstSingularMasculine)
// -> "parle"

// All persons for a voice + mode + tense
let forms: [Person: String] = fr.conjugate("avoir", voice: .activeAvoir,
                                           mode: .indicatif, tense: .present)
// [.firstSingularMasculine: "ai", .secondSingularMasculine: "as", ...]

// All tenses and persons for a voice + mode
let indic: [Tense: [Person: String]] = fr.conjugate("finir",
    voice: .activeAvoir, mode: .indicatif)

// All modes, tenses and persons for a voice
let all: [Mode: [Tense: [Person: String]]] = fr.conjugate("aller",
    voice: .activeEtre)

// Everything for a verb (all voices)
let full: [Voice: [Mode: [Tense: [Person: String]]]]? = fr.conjugate("battre")
```

#### Variant (Alternative) Forms

Some conjugated forms have alternative spellings (e.g. 1990 reform
variants).  The standard `conjugate` and `participle` methods always
return the primary form.  Use the following methods to access variants:

```swift
// Check if a specific form has an alternative
fr.hasAlternativeForm("abreger", voice: .activeAvoir, mode: .indicatif,
                      tense: .present, person: .firstSingularMasculine)
// -> true

// Get the alternative form
fr.conjugateAlternative("abreger", voice: .activeAvoir, mode: .indicatif,
                        tense: .present, person: .firstSingularMasculine)
// -> "abrege" (1990 reform spelling)

// Participle variants
fr.hasAlternativeParticiple("verb", voice: .activeAvoir, tense: .present)
fr.participleAlternative("verb", voice: .activeAvoir, tense: .present)
```

All variant methods have async overloads.

#### Participles

```swift
fr.participle("parler", voice: .activeAvoir, tense: .present)
// -> "parlant"

fr.participle("partir", voice: .activeEtre, tense: .passeFemininPluriel)
// -> "parties"

// All participle forms for a voice
let parts: [Tense: String] = fr.participles("prendre", voice: .activeAvoir)
// [.present: "prenant", .passeMasculinSingulier: "pris",
//  .passeFemininSingulier: "prise", ...]
```

#### Structure Queries

```swift
fr.voices("aller")
// -> [.activeEtre, .pronominal]

fr.modes("aller", voice: .activeEtre)
// -> [.indicatif, .subjonctif, .conditionnel, .imperatif, .participe]

fr.tenses("aller", voice: .activeEtre, mode: .indicatif)
// -> [.present, .imparfait, .passeSimple, .futurSimple, ...]

fr.persons("aller", voice: .activeEtre, mode: .indicatif, tense: .present)
// -> [.firstSingularMasculine, .secondSingularMasculine, ...]
```

#### Other Queries

```swift
fr.hasVerb("parler")            // true
fr.isHAspire("hair")            // true
fr.is1990Reform("ceder")        // true
fr.reformVariante("ceder")      // Optional("ceder")
fr.verbCount                    // 6298
fr.allVerbs                     // ["abaisser", "abandonner", ..., "zozoter"]
```

#### Caching

```swift
fr.cacheCapacity    // 64 (default)
fr.cacheCount       // number of verbs currently cached
fr.clearCache()     // evict all entries
```

The LRU cache is measured in **verbs** -- all forms for the same verb
share a single cache slot.  Pass `cacheSize: 0` at init to disable.

### Enums

| Enum | Cases |
|:-----|:------|
| `Voice` | `.activeAvoir` `.activeEtre` `.active` `.passive` `.pronominal` |
| `Mode` | `.indicatif` `.subjonctif` `.conditionnel` `.imperatif` `.participe` |
| `Tense` | `.present` `.imparfait` `.passeSimple` `.futurSimple` `.passeCompose` `.plusQueParfait` `.passeAnterieur` `.futurAnterieur` `.passe` `.passeMasculinSingulier` `.passeFemininSingulier` `.passeMasculinPluriel` `.passeFemininPluriel` `.passeCompoundMasculinSingulier` `.passeCompoundFemininSingulier` `.passeCompoundMasculinPluriel` `.passeCompoundFemininPluriel` `.presentMasculinSingulier` `.presentFemininSingulier` `.presentMasculinPluriel` `.presentFemininPluriel` |
| `Person` | `.firstSingularMasculine` `.firstSingularFeminine` `.secondSingularMasculine` `.secondSingularFeminine` `.thirdSingularMasculine` `.thirdSingularFeminine` `.thirdSingularNeutral` `.firstPluralMasculine` `.firstPluralFeminine` `.secondPluralMasculine` `.secondPluralFeminine` `.thirdPluralMasculine` `.thirdPluralFeminine` |

Each `Person` case has a `.pronoun` property (`"je"`, `"tu"`, `"il"`, ...) and a `.shortLabel` (`"1sm"`, `"3pf"`, ...).

---

## Breaking Changes in v4.0.0

This release adds new enum cases, which is a **source-breaking change**
for any code that uses exhaustive `switch` statements over `Person` or
`Tense`.

**New `Person` case:**
- `.thirdSingularNeutral` (`"3sn"`) -- the pronoun *on* in reciprocal
  verbs.

**New `Tense` cases** (gendered present participles, passive voice):
- `.presentMasculinSingulier` (`"present_sm"`)
- `.presentFemininSingulier` (`"present_sf"`)
- `.presentMasculinPluriel` (`"present_pm"`)
- `.presentFemininPluriel` (`"present_pf"`)

Add a `default` branch (or handle the new cases) to fix any exhaustive
`switch` compiler errors.

---

## Building

### 1. Get the dataset

The `verbs.db` SQLite database is a release artefact of
[**ShingZhanho/verbe-conjugaison-academie-francaise**](https://github.com/ShingZhanho/verbe-conjugaison-academie-francaise).
Download it and place it in `python_model/`.

### 2. Train the model (or use a pre-trained checkpoint)

```bash
cd python_model
python3 train_model.py              # -> conjugation_model.pt
python3 full_test_model.py          # -> full_test_errors.json
python3 build_final_model.py        # -> conjugation_model_final.pt (100%)
```

See [python_model/README.md](python_model/README.md) for details.

### 3. Export weights to portable format

```bash
python3 python_model/export_weights.py
```

Produces `model.json` and `weights.bin` in `swift_lib/Sources/FRConjugation/Resources/`.

### 4. Build & test the Swift package

```bash
cd swift_lib
swift build
swift test
```

No LibTorch, no C library, no linker flags.

---

## Repository Structure

```
.
+-- swift_lib/                       The Swift Package (FRConjugation)
|   +-- Package.swift
|   +-- Sources/FRConjugation/
|   |   +-- Conjugator.swift         Main API + LRU-cached conjugation
|   |   +-- VerbCache.swift          LRU cache (verb-keyed, O(1))
|   |   +-- InferenceEngine.swift    Model loader + greedy decoder
|   |   +-- Layers.swift             GRU, attention, encoder, decoder, bridge
|   |   +-- Tensor.swift             Accelerate-backed dense tensor
|   |   +-- Types.swift              Voice, Mode, Tense, Person enums
|   |   +-- Resources/
|   |       +-- model.json           Vocabulary, metadata, weight manifest
|   |       +-- weights.bin          Raw float32 weight data (~6 MB)
|   +-- Tests/
|       +-- FRConjugationTests/      Unit tests
|
+-- python_model/                    ML model training & Python API
    +-- french_conjugation_model.py  Python conjugation module
    +-- train_model.py               Training script
    +-- build_final_model.py         Exception-table builder
    +-- export_weights.py            Export to portable format
    +-- test_model.py                Unit tests
    +-- full_test_model.py           Full-DB validation (2,553,802 forms)
    +-- README.md                    Python component docs
```

> Model files (`*.pt`), `verbs.db`, and build artefacts are git-ignored.
> Exported weights (`model.json`, `weights.bin`) are committed in Resources/.

---

## Model Architecture

| Component | Detail |
|:----------|:-------|
| Type | Character-level seq2seq with Bahdanau attention |
| Encoder | Bidirectional GRU, 256 hidden, 64-dim char embeddings |
| Decoder | GRU with attention over encoder states |
| Conditioning | Voice + mode + tense + person embeddings (32-dim each) |
| Bridge | Linear + tanh: encoder hidden + 4 conditioning embeddings -> decoder initial state |
| Exception table | 771 hard-coded corrections embedded in model metadata |
| Parameters | ~1,540,000 |
| Accuracy | **100%** on 2,553,802 forms across 6,298 verbs (5 voices) |
| Model size | ~6 MB (weights.bin) + ~474 KB (model.json) |

---

## Data Source

The training data (`verbs.db`) is a release artefact of
[**ShingZhanho/verbe-conjugaison-academie-francaise**](https://github.com/ShingZhanho/verbe-conjugaison-academie-francaise)
-- a comprehensive French verb conjugation dataset scraped from the
dictionaries of the Academie francaise.

---

## Licence

This project is provided as-is for personal and educational use.
