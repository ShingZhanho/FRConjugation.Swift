# C / Objective-C / Swift Wrappers

Use the PyTorch conjugation model from C, Objective-C, or Swift via LibTorch.

```
Python (PyTorch)  →  export_model.py  →  4 traced .pt files + JSON
                                              ↓
                          C++ / C library  (LibTorch)     ← c_wrapper/
                                              ↓
                          Objective-C wrapper              ← objc_wrapper/
                                              ↓
                          Swift wrapper                    ← swift_wrapper/
```

---

## Prerequisites

| Dependency      | Version  | Notes |
|:----------------|:---------|:------|
| Python + PyTorch| 2.x      | Only for the export step |
| LibTorch (C++)  | 2.x      | https://pytorch.org/get-started/locally/ → C++/Libtorch |
| CMake           | ≥ 3.18   | Build system for the C library |
| Xcode           | ≥ 14     | For ObjC / Swift targets |
| nlohmann/json   | ≥ 3.11   | Auto-downloaded by CMake, or place `json.hpp` in `c_wrapper/` |

---

## Step 1 — Export the Model

```bash
cd c_wrapper
python3 export_model.py                 # defaults to ../python_model/conjugation_model_final.pt
python3 export_model.py /path/to/model  # or specify a checkpoint
```

Produces five files inside `c_wrapper/`:

| File | Size | Contents |
|:-----|:-----|:---------|
| `conjugation_encoder.pt` | ~2 MB | Traced bidirectional GRU encoder |
| `conjugation_bridge.pt` | ~0.6 MB | Traced conditioning bridge |
| `conjugation_attention.pt` | ~0.8 MB | Traced Bahdanau attention |
| `conjugation_decoder.pt` | ~2.7 MB | Traced decoder step |
| `conjugation_meta.json` | ~162 KB | Vocabulary, exceptions, verb sets |

The script also verifies that the traced pipeline reproduces the original model's output.

---

## Step 2 — Build the C Library

```bash
# Download LibTorch (CPU, macOS) and unzip to e.g. ~/libtorch

cd c_wrapper
mkdir build && cd build
cmake -DCMAKE_PREFIX_PATH=$HOME/libtorch -DCMAKE_BUILD_TYPE=Release ..
make -j$(sysctl -n hw.ncpu)
```

Produces:
- `libfrconjugation.dylib` — shared library
- `libfrconjugation_static.a` — static library
- `test_conjugation` — smoke test

### Test it

```bash
./test_conjugation ..      # model_dir = parent where the .pt files live
```

Expected:
```
Verb count: 6132
— Simple tenses —
  [PASS] parler ind.present 1s → parle
  ...
  17/17 tests passed
```

---

## Step 3 — Objective-C

Add to your Xcode project:
- `c_wrapper/conjugation.h`
- `objc_wrapper/FRConjugation.h` + `FRConjugation.m`
- Link `libfrconjugation.dylib` (or `.a`)
- Add LibTorch headers + libs to search paths

```objc
#import "FRConjugation.h"

FRConjugation *model = [[FRConjugation alloc]
    initWithModelDirectory:@"/path/to/c_wrapper"];

NSString *form = [model conjugate:@"parler"
                             mode:@"indicatif"
                            tense:@"present"
                           person:@"1s"];
// → @"parle"

NSString *pp = [model participle:@"finir" forme:@"passe_sm"];
// → @"fini"

NSArray *aux = [model auxiliaryForVerb:@"aller"];
// → @[@"être"]
```

---

## Step 4 — Swift

1. Add the ObjC files from Step 3
2. Set **Objective-C Bridging Header** → `swift_wrapper/Bridging-Header.h`
3. Add `swift_wrapper/ConjugationModel.swift` to your target

```swift
let model = try ConjugationModel(directory: "/path/to/c_wrapper")

model.conjugate("parler", mode: "indicatif", tense: "present", person: "1s")
// → "parle"

model.conjugate("aller", mode: "indicatif", tense: "passe_compose", person: "3sf")
// → "est allée"

model.participle("prendre", forme: "passe_sf")
// → "prise"

model.hasVerb("parler")         // true
model.auxiliary(for: "aller")   // ["être"]
model.isHAspire("hurler")      // true
model.verbCount                 // 6132
```

---

## API Reference

### C (`conjugation.h`)

```c
FRConjugationModel *fr_conjugation_load(const char *model_dir);
void                fr_conjugation_free(FRConjugationModel *model);
int   fr_conjugation_verb_count(const FRConjugationModel *model);
bool  fr_conjugation_has_verb(const FRConjugationModel *model, const char *infinitive);
bool  fr_conjugation_is_h_aspire(const FRConjugationModel *model, const char *infinitive);
int   fr_conjugation_auxiliary(const FRConjugationModel *model, const char *infinitive,
                               char *out_buf, size_t buf_size);
int   fr_conjugation_conjugate(const FRConjugationModel *model,
                               const char *infinitive, const char *mode,
                               const char *tense, const char *person,
                               char *out_buf, size_t buf_size);
int   fr_conjugation_get_participle(const FRConjugationModel *model,
                                    const char *infinitive, const char *forme,
                                    char *out_buf, size_t buf_size);
```

All buffer-writing functions return bytes written (excl. NUL) or -1 on error.

### Objective-C (`FRConjugation`)

| Method | Returns |
|:-------|:--------|
| `-initWithModelDirectory:` | `FRConjugation?` |
| `-conjugate:mode:tense:person:` | `NSString?` |
| `-participle:forme:` | `NSString?` |
| `-hasVerb:` | `BOOL` |
| `-auxiliaryForVerb:` | `NSArray<NSString*>*` |
| `-isHAspire:` | `BOOL` |
| `.verbCount` | `NSInteger` |

### Swift (`ConjugationModel`)

| Method / Property | Returns |
|:------------------|:--------|
| `init(directory:) throws` | — |
| `conjugate(_:mode:tense:person:)` | `String?` |
| `participle(_:forme:)` | `String?` |
| `hasVerb(_:)` | `Bool` |
| `auxiliary(for:)` | `[String]` |
| `isHAspire(_:)` | `Bool` |
| `verbCount` | `Int` |

---

## File Reference

```
c_wrapper/
  export_model.py             Export PyTorch → 4 traced .pt + JSON
  conjugation.h               Public C API header
  conjugation.cpp             C++ implementation (LibTorch)
  CMakeLists.txt              Build config
  test_conjugation.cpp        C smoke test
  conjugation_encoder.pt      ← generated
  conjugation_bridge.pt       ← generated
  conjugation_attention.pt    ← generated
  conjugation_decoder.pt      ← generated
  conjugation_meta.json       ← generated

objc_wrapper/
  FRConjugation.h             ObjC class interface
  FRConjugation.m             ObjC implementation

swift_wrapper/
  ConjugationModel.swift      Swift class
  Bridging-Header.h           ObjC → Swift bridge
  main.swift                  Swift smoke test
```
