# c_wrapper — C/C++ Library (LibTorch)

A C API for French verb conjugation, backed by LibTorch traced model components.

This library is the runtime engine that powers the Swift package
([`swift_lib/`](../swift_lib/)) and the Objective-C wrapper
([`objc_wrapper/`](../objc_wrapper/)).

## Overview

| File | Purpose |
|:-----|:--------|
| `conjugation.h` | Public C API header |
| `conjugation.cpp` | C++ implementation (LibTorch inference) |
| `CMakeLists.txt` | CMake build configuration |
| `test_conjugation.cpp` | 18 smoke tests |
| `export_model.py` | Exports PyTorch checkpoint → 4 traced `.pt` files + JSON |

### Generated files (by `export_model.py`)

| File | Size | Contents |
|:-----|:-----|:---------|
| `conjugation_encoder.pt` | ~2 MB | Traced bidirectional GRU encoder |
| `conjugation_bridge.pt` | ~0.6 MB | Traced conditioning bridge |
| `conjugation_attention.pt` | ~0.8 MB | Traced Bahdanau attention |
| `conjugation_decoder.pt` | ~2.7 MB | Traced decoder step |
| `conjugation_meta.json` | ~162 KB | Vocabulary, exceptions, verb sets |

## Requirements

| Dependency | Version | Notes |
|:-----------|:--------|:------|
| LibTorch (C++) | 2.x | [pytorch.org](https://pytorch.org/get-started/locally/) → C++ / LibTorch |
| CMake | ≥ 3.18 | |
| C++ compiler | C++17 | Clang / GCC |
| nlohmann/json | ≥ 3.11 | Auto-downloaded by CMake |

## Export

Before building, export the PyTorch model to traced components:

```bash
python3 export_model.py                                    # uses ../python_model/conjugation_model_final.pt
python3 export_model.py /path/to/conjugation_model_final.pt  # or specify explicitly
```

The script also verifies that the traced pipeline reproduces the original model's output.

## Build

```bash
mkdir build && cd build
cmake -DCMAKE_PREFIX_PATH=<path-to-libtorch> -DCMAKE_BUILD_TYPE=Release ..
make -j$(sysctl -n hw.ncpu)
```

Produces:
- `libfrconjugation.dylib` — shared library
- `libfrconjugation_static.a` — static library
- `test_conjugation` — smoke test binary

## Test

```bash
cd build
./test_conjugation ..      # model_dir = parent where the .pt files live
```

Expected:
```
Verb count: 6132
— Simple tenses —
  [PASS] parler ind.present 1s → parle
  ...
  18/18 tests passed
```

## C API

```c
#include "conjugation.h"

// Load
FRConjugationModel *model = fr_conjugation_load("/path/to/model_dir");

// Query
int count = fr_conjugation_verb_count(model);
bool known = fr_conjugation_has_verb(model, "parler");

// Conjugate
char buf[256];
int n = fr_conjugation_conjugate(model, "parler", "indicatif", "present", "1s", buf, sizeof(buf));
// buf → "parle", n → 5

// Participle
fr_conjugation_get_participle(model, "finir", "passe_sm", buf, sizeof(buf));
// buf → "fini"

// Auxiliary
fr_conjugation_auxiliary(model, "aller", buf, sizeof(buf));
// buf → "être,pronominal"

// Free
fr_conjugation_free(model);
```

All buffer-writing functions return bytes written (excluding NUL), or -1 on error.
