# French Conjugation Model — Usage Guide

A lightweight, ML-based French verb conjugation engine.  Given a verb
infinitive and optional voice/mode/tense/person parameters, it returns
the conjugated form(s).

## Requirements

- **Python 3.10+**
- **PyTorch 2.x** (`pip install torch`)
- Model file: `conjugation_model_final.pt` (or `conjugation_model.pt`)

Both the model file and `french_conjugation_model.py` must be in the same directory (or you can pass the model path explicitly).

---

## Quick Start

```python
from french_conjugation_model import ConjugationModel

model = ConjugationModel("conjugation_model_final.pt")

# Single form (voice is required when specifying mode/tense/person)
model.conjugate("parler", voice="active_avoir", mode="indicatif",
                tense="present", person="1sm")
# → "parle"

# All conjugations for a verb (all voices)
model.conjugate("finir")
# → { "voix_active_avoir": { "indicatif": { "present": { "1sm": "finis", ... }, ... }, ... } }
```

### Singleton Helper

If you prefer a shared instance across your application:

```python
from french_conjugation_model import get_model

model = get_model("conjugation_model_final.pt")
```

Calling `get_model()` again returns the same cached instance.

---

## API Reference

### `ConjugationModel(model_path=None)`

Loads the trained model from disk.

| Parameter    | Type            | Default                         | Description                        |
|:-------------|:----------------|:--------------------------------|:-----------------------------------|
| `model_path` | `str` or `None` | `conjugation_model_final.pt`    | Path to the `.pt` checkpoint file. |

---

### `model.conjugate(infinitive, *, voice=None, mode=None, tense=None, person=None)`

Main conjugation method.  Returns a single string, a nested dictionary, or `None`.

Parameters are **layered** — specifying a lower layer requires all upper
layers to be present:

| Specified                                  | Returns |
|:-------------------------------------------|:--------|
| `voice` + `mode` + `tense` + `person`     | `str` — the conjugated form |
| `voice` + `mode` + `tense`                | `dict` — `{ person: form }` |
| `voice` + `mode`                           | `dict` — `{ tense: { person: form } }` |
| `voice`                                    | `dict` — `{ mode: { tense: { person: form } } }` |
| Nothing                                    | `dict` — `{ voice: { mode: { tense: { person: form } } } }` |
| Verb unknown                               | `None` |

Specifying `person` without `tense`, or `tense` without `mode`, or
`mode` without `voice` raises `ValueError`.

#### Examples

```python
# Single form
model.conjugate("aller", voice="active_etre", mode="indicatif",
                tense="present", person="1sm")
# → "vais"

# All persons for a tense
model.conjugate("venir", voice="active_etre", mode="indicatif",
                tense="passe_compose")
# → { "1sm": "suis venu", "1sf": "suis venue", "3sf": "est venue", ... }

# All tenses in indicatif
model.conjugate("finir", voice="active_avoir", mode="indicatif")
# → { "present": {"1sm": "finis", ...}, "imparfait": {...}, ... }

# All modes for a voice
model.conjugate("aller", voice="active_etre")
# → { "indicatif": { ... }, "subjonctif": { ... }, ... }

# Everything
model.conjugate("battre")
# → { "voix_active_avoir": { ... }, "voix_prono": { ... } }
```

---

### `model.voices(infinitive)`

List available voices for a verb.

```python
model.voices("aller")
# → ["voix_active_etre", "voix_prono"]

model.voices("manger")
# → ["voix_active_avoir", "voix_passive", "voix_prono"]
```

### `model.modes(infinitive, voice)`

List available modes for a verb in a given voice.

```python
model.modes("parler", "active_avoir")
# → ["conditionnel", "imperatif", "indicatif", "participe", "subjonctif"]
```

### `model.tenses(infinitive, voice, mode)`

List available tenses for a verb in a given voice and mode.

```python
model.tenses("parler", "active_avoir", "indicatif")
# → ["futur_anterieur", "futur_simple", "imparfait", "passe_anterieur",
#    "passe_compose", "passe_simple", "plus_que_parfait", "present"]
```

### `model.persons(infinitive, voice, mode, tense)`

List available person keys for a specific combination.

```python
model.persons("parler", "active_avoir", "indicatif", "present")
# → ["1sm", "1sf", "2sm", "2sf", "3sm", "3sf", "1pm", "1pf", "2pm", "2pf", "3pm", "3pf"]

model.persons("falloir", "voix_active", "indicatif", "present")
# → ["3sm"]
```

---

### `model.has_verb(infinitive) → bool`

Check whether a verb is in the model's vocabulary.

```python
model.has_verb("parler")   # → True
model.has_verb("xyzfake")  # → False
```

---

### `model.is_h_aspire(infinitive) → bool`

Returns whether the verb begins with an aspirate *h*.

```python
model.is_h_aspire("hurler")   # → True
model.is_h_aspire("habiter")  # → False
```

---

### `model.is_1990_reform(infinitive) → bool`

Returns whether the verb has 1990 orthographic reform spelling changes.

```python
model.is_1990_reform("céder")  # → True
```

### `model.reform_variante(infinitive) → str | None`

Returns the 1990 reform variant spelling, or `None`.

```python
model.reform_variante("céder")  # → "cèder"
```

---

### `model.verb_count → int`

The number of verbs known to the model.

```python
model.verb_count  # → 6298
```

---

### `model.verbs(prefix=None) → list[str]`

A sorted list of all known verb infinitives.  Optionally filter by prefix.

```python
model.verbs()              # all 6,298 verbs
model.verbs("par")         # ["paraître", "pardonner", "parer", "parfaire", "parier", "parler", ...]
```

---

## Accepted Values

### Voices

| Canonical             | Aliases                          |
|:----------------------|:---------------------------------|
| `voix_active_avoir`   | `active_avoir`                   |
| `voix_active_etre`    | `active_etre`                    |
| `voix_active`         | `active`                         |
| `voix_passive`        | `passive`                        |
| `voix_prono`          | `prono`, `pronominal`            |

### Modes

| Canonical         | Aliases                          |
|:------------------|:---------------------------------|
| `indicatif`       | `ind`                            |
| `subjonctif`      | `sub`                            |
| `conditionnel`    | `cond`                           |
| `imperatif`       | `imp`                            |
| `participe`       | `part`                           |

### Tenses

**Simple tenses:**

| Mode            | Tense                | Aliases                      |
|:----------------|:---------------------|:-----------------------------|
| `indicatif`     | `present`            |                              |
| `indicatif`     | `imparfait`          |                              |
| `indicatif`     | `passe_simple`       |                              |
| `indicatif`     | `futur_simple`       | `futur`                      |
| `conditionnel`  | `present`            |                              |
| `subjonctif`    | `present`            |                              |
| `subjonctif`    | `imparfait`          |                              |
| `imperatif`     | `present`            |                              |

**Compound tenses** (predicted directly by the neural model):

| Mode            | Tense                  | Aliases                      |
|:----------------|:-----------------------|:-----------------------------|
| `indicatif`     | `passe_compose`        |                              |
| `indicatif`     | `plus_que_parfait`     |                              |
| `indicatif`     | `passe_anterieur`      |                              |
| `indicatif`     | `futur_anterieur`      |                              |
| `conditionnel`  | `passe`                |                              |
| `subjonctif`    | `passe`                |                              |
| `subjonctif`    | `plus_que_parfait`     |                              |
| `imperatif`     | `passe`                |                              |

**Participle sub-forms** (used with `mode="participe"`):

| Tense                  | Meaning                                          |
|:-----------------------|:-------------------------------------------------|
| `present`              | Present participle (*parlant*)                   |
| `passe_sm`             | Past participle, masculine singular              |
| `passe_sf`             | Past participle, feminine singular               |
| `passe_pm`             | Past participle, masculine plural                |
| `passe_pf`             | Past participle, feminine plural                 |
| `passe_compound_sm`    | Compound past participle, masculine singular     |
| `passe_compound_sf`    | Compound past participle, feminine singular      |
| `passe_compound_pm`    | Compound past participle, masculine plural       |
| `passe_compound_pf`    | Compound past participle, feminine plural        |

### Persons

| Canonical | Aliases          | Meaning                        |
|:----------|:-----------------|:-------------------------------|
| `1sm`     | `je`             | First person singular masc.    |
| `1sf`     |                  | First person singular fem.     |
| `2sm`     | `tu`             | Second person singular masc.   |
| `2sf`     |                  | Second person singular fem.    |
| `3sm`     | `il`, `on`       | Third person singular masc.    |
| `3sf`     | `elle`           | Third person singular fem.     |
| `1pm`     | `nous`           | First person plural masc.      |
| `1pf`     |                  | First person plural fem.       |
| `2pm`     | `vous`           | Second person plural masc.     |
| `2pf`     |                  | Second person plural fem.      |
| `3pm`     | `ils`            | Third person plural masc.      |
| `3pf`     | `elles`          | Third person plural fem.       |
| `-`       |                  | No person (participles)        |

> **Imperatif** only uses `2sm`/`2sf`, `1pm`/`1pf`, `2pm`/`2pf`.

---

## Command-Line Usage

The module can also be run directly from the command line:

```bash
# Full conjugation table
python french_conjugation_model.py parler

# Specific form (verb voice mode tense person)
python french_conjugation_model.py aller active_etre indicatif present 1sm
```

Arguments: `verb [voice] [mode] [tense] [person]`

---

## Architecture Summary

The model is a character-level sequence-to-sequence neural network with
Bahdanau attention, built in PyTorch:

- **Encoder:** Bidirectional GRU over the input verb's characters
- **Decoder:** GRU with attention, conditioned on voice/mode/tense/person
  via four learned embeddings (32-dim each)
- **Bridge:** Linear + tanh combining encoder final hidden + 4 conditioning
  embeddings into the decoder initial hidden state
- **Exception table:** A small lookup (2,329 entries) embedded in the
  checkpoint to guarantee 100% accuracy on the training vocabulary

The model has **1,538,795 parameters**, is approximately **6 MB**, and
covers **6,298 French verbs** across **5 voices** — **2,559,057
conjugated forms** at **100% accuracy**.

---

## File Overview

| File                             | Purpose                                                    |
|:---------------------------------|:-----------------------------------------------------------|
| `french_conjugation_model.py`    | Module to load and use the model (import this)             |
| `conjugation_model_final.pt`     | Final model checkpoint (ML weights + exception table)      |
| `conjugation_model.pt`           | ML-only model checkpoint (no exception table, ~99.91%)     |
| `train_model.py`                 | Training script (produces `conjugation_model.pt`)          |
| `build_final_model.py`           | Builds the final model by adding the exception table       |
| `export_weights.py`              | Export weights to portable format for Swift package         |
| `test_model.py`                  | Unit tests                                                 |
| `full_test_model.py`             | Full dataset test against `verbs.db`                       |
| `verbs.db`                       | SQLite database of 6,298 French verbs (source of truth)    |
