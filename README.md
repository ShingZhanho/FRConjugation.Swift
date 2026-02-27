# French Verb Conjugation — ML Model & Cross-Language Wrappers

A lightweight, character-level seq2seq neural network that conjugates **6,288 French verbs** across all standard modes, tenses, and persons — with 100% accuracy.

Includes ready-to-use wrappers for **Python**, **C/C++**, **Objective-C**, and **Swift**.

---

## Repository Structure

```
.
├── README.md                        ← You are here
├── WRAPPERS.md                      ← C / ObjC / Swift build & API guide
│
├── python_model/                    ← Python module, training, & tests
│   ├── french_conjugation_model.py  ← Reusable conjugation module (import this)
│   ├── train_model.py               ← Training script (reads verbs.db)
│   ├── build_final_model.py         ← Builds exception table → final model
│   ├── test_model.py                ← 31 unit tests
│   ├── full_test_model.py           ← Full-DB test (390,546 forms)
│   ├── USAGE.md                     ← Detailed Python API documentation
│   ├── conjugation_model_final.pt   ← Final model (ML + exceptions) ★
│   ├── conjugation_model.pt         ← ML-only model (no exceptions)
│   └── verbs.db                     ← SQLite source database (6,288 verbs)
│
├── c_wrapper/                       ← C/C++ library (LibTorch)
│   ├── export_model.py              ← PyTorch → 4 traced components + JSON
│   ├── conjugation.h                ← Public C API
│   ├── conjugation.cpp              ← C++ implementation
│   ├── CMakeLists.txt               ← CMake build configuration
│   ├── test_conjugation.cpp         ← 18 C smoke tests
│   ├── conjugation_encoder.pt       ← Traced encoder      (generated)
│   ├── conjugation_bridge.pt        ← Traced bridge        (generated)
│   ├── conjugation_attention.pt     ← Traced attention     (generated)
│   ├── conjugation_decoder.pt       ← Traced decoder       (generated)
│   └── conjugation_meta.json        ← Vocab + exceptions   (generated)
│
├── objc_wrapper/                    ← Objective-C wrapper
│   ├── FRConjugation.h
│   └── FRConjugation.m
│
├── swift_lib/                       ← Swift Package (idiomatic, typed API) ★
│   ├── Package.swift
│   ├── Sources/
│   │   ├── CFRConjugation/          ← C bridge module
│   │   └── FRConjugation/
│   │       ├── Conjugator.swift      ← Main Conjugator class
│   │       └── Types.swift           ← Mode / Tense / Person enums
│   └── Tests/
│       └── FRConjugationTests/
│           └── ConjugationTests.swift
│
└── swift_wrapper/                   ← Legacy Swift wrapper (string-based, ObjC bridge)
    ├── ConjugationModel.swift
    ├── Bridging-Header.h
    └── main.swift
```

> **Note:** Model files (`*.pt`), `verbs.db`, and build artifacts are excluded from git via `.gitignore` because of their size. See [Getting the Model Files](#getting-the-model-files) below.

---

## Model Architecture

| Component | Detail |
|:----------|:-------|
| **Type** | Character-level seq2seq with Bahdanau attention |
| **Encoder** | Bidirectional GRU (256 hidden, 64-dim char embeddings) |
| **Decoder** | GRU with attention over encoder states |
| **Conditioning** | Mode, tense, and person embeddings (32-dim each) injected into bridge + decoder |
| **Exception table** | 195 hard-coded corrections for edge cases (embedded in checkpoint) |
| **Accuracy** | 100% on all 390,546 forms across 6,288 verbs |
| **Model size** | ~6 MB (PyTorch checkpoint) |

### Capabilities

- **Simple tenses**: All modes × tenses × persons (indicatif, subjonctif, conditionnel, impératif)
- **Compound tenses**: Automatically composes auxiliary (avoir/être) + past participle with agreement
- **Participles**: Present participle, past participle (4 agreement forms: `sm`, `sf`, `pm`, `pf`)
- **Pronominal verbs**: Detects and handles reflexive prefixing (`se laver` → `je me lave`)
- **H-aspiré**: Correctly avoids elision before aspirate-h verbs

---

## Getting the Model Files

The binary files are too large for git. To get them, either:

### Option A — Train from scratch

```bash
cd python_model

# 1. Train the ML model (requires verbs.db + PyTorch)
python3 train_model.py
# → produces conjugation_model.pt (~6 MB)

# 2. Run full test to find errors
python3 full_test_model.py
# → produces full_test_errors.json

# 3. Build final model with exception table
python3 build_final_model.py
# → produces conjugation_model_final.pt (100% accuracy)
```

### Option B — Copy pre-built files

If you already have the model files, place them in the right directories:

```
python_model/conjugation_model_final.pt
python_model/conjugation_model.pt        (optional, ML-only)
python_model/verbs.db                    (needed for training/full test)
```

For the C wrapper, generate the traced components:

```bash
cd c_wrapper
python3 export_model.py
```

---

## Quick Start (Python)

```python
from french_conjugation_model import ConjugationModel

model = ConjugationModel()  # loads conjugation_model_final.pt from same directory

# Single conjugation
model.conjugate("parler", mode="indicatif", tense="present", person="1s")
# → "parle"

# Compound tense
model.conjugate("aller", mode="indicatif", tense="passe_compose", person="3s")
# → "est allé"

# All forms for a verb
model.conjugate("finir")
# → { "indicatif": { "present": { "1s": "finis", ... }, ... }, ... }

# Past participle with agreement
model.get_participle("prendre", "passe_sf")
# → "prise"

# Auxiliary
model.auxiliary("aller")
# → ["être"]
```

See [python_model/USAGE.md](python_model/USAGE.md) for the complete API reference (modes, tenses, persons, aliases).

---

## Quick Start (C / Objective-C / Swift)

The model can be used from C, Objective-C, and Swift via LibTorch. The pipeline:

```
Python (PyTorch)  →  export_model.py  →  4 traced .pt files + JSON metadata
                                              ↓
                          C++ / C library  (LibTorch)
                                              ↓
                          Objective-C wrapper
                                              ↓
                          Swift wrapper
```

### Build the C library

```bash
cd c_wrapper

# 1. Export traced models (if not already done)
python3 export_model.py

# 2. Build with CMake
mkdir build && cd build
cmake -DCMAKE_PREFIX_PATH=<path-to-libtorch> -DCMAKE_BUILD_TYPE=Release ..
make -j$(sysctl -n hw.ncpu)

# 3. Run smoke tests
./test_conjugation ..
```

### C usage

```c
#include "conjugation.h"

FRConjugationModel *model = fr_conjugation_load("path/to/c_wrapper");
char buf[256];
fr_conjugation_conjugate(model, "parler", "indicatif", "present", "1s", buf, sizeof(buf));
// buf → "parle"
fr_conjugation_free(model);
```

### Swift usage (recommended: swift_lib)

```swift
import FRConjugation

let conjugator = try Conjugator(modelDirectory: "path/to/c_wrapper")

// Fully typed — no raw strings
conjugator.conjugate("aller",
    mode: .indicatif, tense: .present, person: .firstPersonSingular)
// → "vais"

// Full paradigm
let forms = conjugator.conjugate("finir", mode: .indicatif, tense: .present)
for (person, form) in forms {
    print("\(person.pronoun) \(form)")
}

// Participle with agreement
conjugator.participle("partir", form: .passeFemininPlural)
// → "parties"
```

See [WRAPPERS.md](WRAPPERS.md) for complete build instructions and API reference.

---

## Testing

### Python unit tests (31 tests)

```bash
cd python_model
python3 test_model.py
```

### Full database test (390,546 forms)

```bash
cd python_model
python3 full_test_model.py
```

### C smoke test (18 tests)

```bash
cd c_wrapper/build
./test_conjugation ..
```

### Swift package tests (15 tests)

```bash
cd swift_lib
swift test \
  -Xlinker -L../c_wrapper/build \
  -Xlinker -L"$LIBTORCH/lib" \
  -Xlinker -rpath -Xlinker ../c_wrapper/build \
  -Xlinker -rpath -Xlinker "$LIBTORCH/lib" \
  -Xlinker -lc10 -Xlinker -ltorch -Xlinker -ltorch_cpu
```

---

## Requirements

| Component | Requirement |
|:----------|:------------|
| Python model | Python 3.10+, PyTorch 2.x |
| Training | + `tqdm` (optional, for progress bars) |
| C library | CMake ≥ 3.18, LibTorch 2.x, C++17 compiler |
| ObjC wrapper | Xcode ≥ 14, the built C library |
| Swift package | Swift 5.9+, the built C library |

---

## License

This project is provided as-is for personal and educational use.
