# FRConjugation.Swift

A **Swift package** for conjugating French verbs, powered by a character-level seq2seq
neural network with Bahdanau attention.

Covers **6,288 verbs** across all standard modes, tenses, and persons — with **100% accuracy**.

```swift
import FRConjugation

let fr = try Conjugator(modelDirectory: modelPath)

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
- **Lightweight** — ~6 MB model, character-level neural network (no dictionary lookup at runtime).

---

## Requirements

| Dependency | Version | Notes |
|:-----------|:--------|:------|
| Swift | ≥ 5.9 | |
| LibTorch (C++) | 2.x | [pytorch.org](https://pytorch.org/get-started/locally/) → C++ / LibTorch |
| `libfrconjugation` | — | Built from `c_wrapper/` (see [Building](#building)) |
| macOS | ≥ 12 | or iOS ≥ 15 |

---

## Installation

Add the package dependency in your `Package.swift`:

```swift
dependencies: [
    .package(url: "https://github.com/ShingZhanho/FRConjugation.Swift.git", from: "1.0.0"),
]
```

Or in Xcode: **File → Add Package Dependencies** and enter the repository URL.

> **Note:** At link time you must also provide the paths to `libfrconjugation` and
> LibTorch. See [Building](#building) for details.

---

## API

### `Conjugator`

```swift
let fr = try Conjugator(modelDirectory: "/path/to/model")
let fr = try Conjugator(modelDirectory: modelURL)
```

Loads the model from a directory containing the exported files
(`conjugation_encoder.pt`, `conjugation_bridge.pt`,
`conjugation_attention.pt`, `conjugation_decoder.pt`,
`conjugation_meta.json`).

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

The Swift package wraps a C library (`libfrconjugation`) that uses LibTorch under
the hood. Below is the full build pipeline.

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

### 3. Export to LibTorch traced components

```bash
cd c_wrapper
python3 export_model.py
```

Produces `conjugation_{encoder,bridge,attention,decoder}.pt` and
`conjugation_meta.json`. See [c_wrapper/README.md](c_wrapper/README.md).

### 4. Build the C library

```bash
cd c_wrapper
mkdir build && cd build
cmake -DCMAKE_PREFIX_PATH=<path-to-libtorch> -DCMAKE_BUILD_TYPE=Release ..
make -j$(sysctl -n hw.ncpu)
```

This produces `libfrconjugation.dylib` and `libfrconjugation_static.a`.

### 5. Build & test the Swift package

```bash
cd swift_lib
LIBTORCH=<path-to-libtorch>

swift build \
  -Xlinker -L../c_wrapper/build \
  -Xlinker -L"$LIBTORCH/lib" \
  -Xlinker -rpath -Xlinker ../c_wrapper/build \
  -Xlinker -rpath -Xlinker "$LIBTORCH/lib" \
  -Xlinker -lc10 -Xlinker -ltorch -Xlinker -ltorch_cpu

swift test \
  -Xlinker -L../c_wrapper/build \
  -Xlinker -L"$LIBTORCH/lib" \
  -Xlinker -rpath -Xlinker ../c_wrapper/build \
  -Xlinker -rpath -Xlinker "$LIBTORCH/lib" \
  -Xlinker -lc10 -Xlinker -ltorch -Xlinker -ltorch_cpu
```

---

## Repository Structure

```
.
├── swift_lib/                       ★ The Swift Package (FRConjugation)
│   ├── Package.swift
│   ├── Sources/
│   │   ├── CFRConjugation/          C bridge module
│   │   └── FRConjugation/
│   │       ├── Conjugator.swift     Main API
│   │       └── Types.swift          Enums & value types
│   └── Tests/
│       └── FRConjugationTests/      15 unit tests
│
├── python_model/                    ML model training & Python API
│   ├── french_conjugation_model.py  Python conjugation module
│   ├── train_model.py               Training script
│   ├── build_final_model.py         Exception-table builder
│   ├── test_model.py                31 unit tests
│   ├── full_test_model.py           Full-DB validation (390,546 forms)
│   └── README.md                    Python component docs
│
├── c_wrapper/                       C/C++ library (LibTorch runtime)
│   ├── conjugation.{h,cpp}          C API
│   ├── CMakeLists.txt               CMake build
│   ├── export_model.py              PyTorch → traced components
│   ├── test_conjugation.cpp         18 smoke tests
│   └── README.md                    C component docs
│
├── objc_wrapper/                    Objective-C wrapper
│   ├── FRConjugation.{h,m}
│   └── README.md
│
└── swift_wrapper/                   Legacy string-based Swift wrapper
    ├── ConjugationModel.swift
    └── README.md
```

> Model files (`*.pt`), `verbs.db`, and build artefacts are git-ignored.
> They are produced by the build pipeline or attached to releases.

---

## Model Architecture

| Component | Detail |
|:----------|:-------|
| Type | Character-level seq2seq with Bahdanau attention |
| Encoder | Bidirectional GRU, 256 hidden, 64-dim char embeddings |
| Decoder | GRU with attention over encoder states |
| Conditioning | Mode + tense + person embeddings (32-dim each) |
| Exception table | 195 hard-coded corrections embedded in checkpoint |
| Accuracy | **100%** on 390,546 forms across 6,288 verbs |
| Model size | ~6 MB |

---

## Data Source

The training data (`verbs.db`) is a release artefact of
[**ShingZhanho/verbe-conjugaison-academie-francaise**](https://github.com/ShingZhanho/verbe-conjugaison-academie-francaise)
— a comprehensive French verb conjugation dataset scraped from the
dictionaries of the Académie française.

---

## Licence

This project is provided as-is for personal and educational use.
