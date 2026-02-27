# FRConjugation.Swift

A **pure Swift** package for conjugating French verbs, powered by a character-level seq2seq
neural network with Bahdanau attention.

Covers **6,288 verbs** across all standard modes, tenses, and persons — with **100% accuracy**.

**Zero external dependencies.** Uses Apple's Accelerate framework for fast matrix operations.
No LibTorch, no CoreML, no Python runtime needed.

```swift
import FRConjugation

let fr = try Conjugator()  // loads bundled model from package resources

fr.conjugate("aller", mode: .indicatif, tense: .present, person: .firstPersonSingular)
// → "vais"

fr.conjugate("partir", mode: .indicatif, tense: .passeCompose, person: .thirdPersonFeminineSingular)
// → "est partie"

fr.participle("prendre", form: .passeFemininPlural)
// → "prises"
```

---

## Features

- **Fully typed API** — `Mode`, `Tense`, `Person`, and `ParticipleForm` enums.
  No raw strings.
- **Simple & compound tenses** — automatically composes auxiliary (avoir/être) +
  past participle with gender/number agreement.
- **Participles** — present participle and all 4 past participle agreement forms.
- **Pronominal verbs** — reflexive prefix handled automatically.
- **H-aspiré** — correctly detects aspirate-h verbs.
- **Pure Swift** — no C library, no LibTorch, no external ML framework.
- **Lightweight** — ~6 MB model, character-level neural network with Accelerate-backed inference.
- **App Store ready** — no dynamic linking concerns, no large framework bundles.

---

## Requirements

| Requirement | Version |
|:------------|:--------|
| Swift       | ≥ 5.9   |
| macOS       | ≥ 12    |
| iOS         | ≥ 15    |

No other dependencies.

---

## Installation

Add the package dependency in your `Package.swift`:

```swift
dependencies: [
    .package(url: "https://github.com/ShingZhanho/FRConjugation.Swift.git", from: "1.0.0"),
]
```

Or in Xcode: **File → Add Package Dependencies** and enter the repository URL.

---

## API

### `Conjugator`

```swift
// Load from bundled resources (recommended)
let fr = try Conjugator()

// Or load from a custom directory containing model.json + weights.bin
let fr = try Conjugator(modelDirectory: "/path/to/model")
let fr = try Conjugator(modelDirectory: modelURL)
```

#### Conjugation

```swift
// Single form
fr.conjugate("parler", mode: .indicatif, tense: .present, person: .firstPersonSingular)
// → "parle"

// All persons for a mode + tense
let forms: [Person: String] = fr.conjugate("avoir", mode: .indicatif, tense: .present)
// [.firstPersonSingular: "ai", .secondPersonSingular: "as", ...]
```

#### Participles

```swift
fr.participle("finir")                                   // → "fini"   (default: passé masc. sing.)
fr.participle("finir", form: .present)                   // → "finissant"
fr.participle("partir", form: .passeFemininPlural)       // → "parties"
```

#### Queries

```swift
fr.hasVerb("parler")            // true
fr.isHAspire("haïr")            // true
fr.verbCount                    // 6132

let aux = fr.auxiliary(for: "aller")
aux.etre            // true
aux.avoir           // false
aux.pronominal      // true
```

### Enums

| Enum | Cases |
|:-----|:------|
| `Mode` | `.indicatif` `.subjonctif` `.conditionnel` `.imperatif` `.participe` |
| `Tense` | `.present` `.imparfait` `.passeSimple` `.futurSimple` `.passeCompose` `.plusQueParfait` `.passeAnterieur` `.futurAnterieur` `.passe` |
| `Person` | `.firstPersonSingular` `.secondPersonSingular` `.thirdPersonMasculineSingular` `.thirdPersonFeminineSingular` `.firstPersonPlural` `.secondPersonPlural` `.thirdPersonMasculinePlural` `.thirdPersonFemininePlural` |
| `ParticipleForm` | `.present` `.passeMasculinSingular` `.passeFemininSingular` `.passeMasculinPlural` `.passeFemininPlural` |

Each `Person` case also has a `.pronoun` property (`"je"`, `"tu"`, `"il"`, …) and a `.shortLabel` (`"1s"`, `"2s"`, …).

---

## Building

### 1. Get the dataset

The `verbs.db` SQLite database is a release artefact of
[**ShingZhanho/verbe-conjugaison-academie-francaise**](https://github.com/ShingZhanho/verbe-conjugaison-academie-francaise).
Download it and place it in `python_model/`.

### 2. Train the model (or use a pre-trained checkpoint)

```bash
cd python_model
python3 train_model.py              # → conjugation_model.pt
python3 full_test_model.py          # → full_test_errors.json
python3 build_final_model.py        # → conjugation_model_final.pt (100%)
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

That's it — no LibTorch, no C library, no linker flags.

---

## Repository Structure

```
.
├── swift_lib/                       ★ The Swift Package (FRConjugation)
│   ├── Package.swift
│   ├── Sources/FRConjugation/
│   │   ├── Conjugator.swift         Main API
│   │   ├── InferenceEngine.swift    Model loader + greedy decoder
│   │   ├── Layers.swift             GRU, attention, encoder, decoder, bridge
│   │   ├── Tensor.swift             Accelerate-backed dense tensor
│   │   ├── Types.swift              Enums & value types
│   │   └── Resources/
│   │       ├── model.json           Vocabulary, metadata, weight manifest
│   │       └── weights.bin          Raw float32 weight data (~6 MB)
│   └── Tests/
│       └── FRConjugationTests/      15 unit tests
│
├── python_model/                    ML model training & Python API
│   ├── french_conjugation_model.py  Python conjugation module
│   ├── train_model.py               Training script
│   ├── build_final_model.py         Exception-table builder
│   ├── export_weights.py            Export to portable format
│   ├── test_model.py                31 unit tests
│   ├── full_test_model.py           Full-DB validation (390,546 forms)
│   └── README.md                    Python component docs
│
├── c_wrapper/                       C/C++ library (LibTorch runtime, legacy)
│   ├── conjugation.{h,cpp}          C API
│   ├── CMakeLists.txt               CMake build
│   ├── export_model.py              PyTorch → traced components
│   └── README.md                    C component docs
│
├── objc_wrapper/                    Objective-C wrapper (legacy)
│   └── README.md
│
└── swift_wrapper/                   Legacy string-based Swift wrapper
    └── README.md
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
| Conditioning | Mode + tense + person embeddings (32-dim each) |
| Bridge | Linear + tanh: encoder hidden + conditioning → decoder initial state |
| Exception table | 195 hard-coded corrections embedded in model metadata |
| Parameters | 1,528,073 |
| Accuracy | **100%** on 390,546 forms across 6,288 verbs |
| Model size | ~6 MB (weights.bin) + ~120 KB (model.json) |

---

## Data Source

The training data (`verbs.db`) is a release artefact of
[**ShingZhanho/verbe-conjugaison-academie-francaise**](https://github.com/ShingZhanho/verbe-conjugaison-academie-francaise)
— a comprehensive French verb conjugation dataset scraped from the
dictionaries of the Académie française.

---

## Licence

This project is provided as-is for personal and educational use.
