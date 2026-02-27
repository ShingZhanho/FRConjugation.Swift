# objc_wrapper — Objective-C Wrapper

An Objective-C class (`FRConjugation`) wrapping the C library for use in
Objective-C and mixed ObjC/Swift projects.

## Files

| File | Purpose |
|:-----|:--------|
| `FRConjugation.h` | Class interface |
| `FRConjugation.m` | Implementation (calls C API via `conjugation.h`) |

## Requirements

- The built C library (`libfrconjugation`) from [`c_wrapper/`](../c_wrapper/)
- LibTorch headers and libraries on the search paths
- Xcode ≥ 14

## Integration

Add to your Xcode project:
1. `FRConjugation.h` and `FRConjugation.m`
2. `c_wrapper/conjugation.h`
3. Link against `libfrconjugation.dylib` (or `.a`) and LibTorch

## Usage

```objc
#import "FRConjugation.h"

FRConjugation *model = [[FRConjugation alloc]
    initWithModelDirectory:@"/path/to/model_dir"];

NSString *form = [model conjugate:@"parler"
                             mode:@"indicatif"
                            tense:@"present"
                           person:@"1s"];
// → @"parle"

NSString *pp = [model participle:@"finir" forme:@"passe_sm"];
// → @"fini"

NSArray *aux = [model auxiliaryForVerb:@"aller"];
// → @[@"être", @"pronominal"]

BOOL known = [model hasVerb:@"parler"];       // YES
BOOL aspirate = [model isHAspire:@"haïr"];    // YES
NSInteger count = model.verbCount;             // 6132
```

## API

| Method | Returns |
|:-------|:--------|
| `-initWithModelDirectory:` | `FRConjugation?` |
| `-conjugate:mode:tense:person:` | `NSString?` |
| `-participle:forme:` | `NSString?` |
| `-hasVerb:` | `BOOL` |
| `-auxiliaryForVerb:` | `NSArray<NSString*>*` |
| `-isHAspire:` | `BOOL` |
| `.verbCount` | `NSInteger` |

> For a fully typed Swift API, use the Swift package in
> [`swift_lib/`](../swift_lib/) instead.
