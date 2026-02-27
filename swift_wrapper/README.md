# swift_wrapper — Legacy String-Based Swift Wrapper

A simple Swift wrapper that bridges to the Objective-C `FRConjugation` class.

> **Note:** For new projects, use the Swift package in
> [`swift_lib/`](../swift_lib/) instead — it provides a fully typed API
> with `Mode`, `Tense`, `Person`, and `ParticipleForm` enums.

## Files

| File | Purpose |
|:-----|:--------|
| `ConjugationModel.swift` | Swift class (string-based parameters) |
| `Bridging-Header.h` | ObjC → Swift bridging header |
| `main.swift` | Smoke test |

## Requirements

- Built C library from [`c_wrapper/`](../c_wrapper/)
- ObjC wrapper from [`objc_wrapper/`](../objc_wrapper/)
- Xcode ≥ 14

## Integration

1. Add the ObjC files from `objc_wrapper/`
2. Set **Objective-C Bridging Header** to `swift_wrapper/Bridging-Header.h`
3. Add `ConjugationModel.swift` to your target
4. Link `libfrconjugation` and LibTorch

## Usage

```swift
let model = try ConjugationModel(directory: "/path/to/model_dir")

model.conjugate("parler", mode: "indicatif", tense: "present", person: "1s")
// → "parle"

model.participle("finir", forme: "passe_sm")
// → "fini"

model.hasVerb("parler")         // true
model.auxiliary(for: "aller")   // ["être", "pronominal"]
model.isHAspire("haïr")        // true
model.verbCount                 // 6132
```
