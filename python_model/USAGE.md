# French Conjugation Model — Usage Guide

A lightweight, ML-based French verb conjugation engine. Given a verb infinitive and optional mode/tense/person parameters, it returns the conjugated form(s).

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

# Single form
model.conjugate("parler", mode="indicatif", tense="present", person="1s")
# → "parle"

# All conjugations for a verb
model.conjugate("finir")
# → { "indicatif": { "present": { "1s": "finis", ... }, ... }, ... }
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

| Parameter    | Type            | Default                  | Description                        |
|:-------------|:----------------|:-------------------------|:-----------------------------------|
| `model_path` | `str` or `None` | `conjugation_model.pt`   | Path to the `.pt` checkpoint file. |

---

### `model.conjugate(infinitive, *, mode=None, tense=None, person=None)`

Main conjugation method. Returns a single string, a nested dictionary, or `None`.

| Parameter    | Type            | Description                                      |
|:-------------|:----------------|:-------------------------------------------------|
| `infinitive` | `str`           | Verb infinitive (e.g. `"parler"`, `"être"`)      |
| `mode`       | `str` or `None` | Grammatical mode (see tables below)               |
| `tense`      | `str` or `None` | Tense within the mode (see tables below)          |
| `person`     | `str` or `None` | Grammatical person (see tables below)             |

**Return type depends on how many parameters are specified:**

| Specified             | Returns                                                     |
|:----------------------|:------------------------------------------------------------|
| All three             | `str` — the conjugated form                                 |
| One or two omitted    | `dict` — nested `{ mode: { tense: { person: form } } }`    |
| Verb unknown          | `None`                                                      |

#### Examples

```python
# Single form
model.conjugate("aller", mode="indicatif", tense="present", person="1s")
# → "vais"

# All tenses in indicatif
model.conjugate("finir", mode="indicatif")
# → { "present": {"1s": "finis", ...}, "imparfait": {...}, ... }

# A specific tense across all persons
model.conjugate("venir", mode="indicatif", tense="passe_compose")
# → { "1s": "suis venu", "3sf": "est venue", ... }

# A specific person across all modes and tenses
model.conjugate("avoir", person="1s")
# → { "indicatif": { "present": {"1s": "ai"}, ... }, ... }
```

---

### `model.get_participle(infinitive, forme="passe_sm")`

Returns a participle form.

| Parameter    | Type  | Default      | Description                                               |
|:-------------|:------|:-------------|:----------------------------------------------------------|
| `infinitive` | `str` | —            | Verb infinitive                                           |
| `forme`      | `str` | `"passe_sm"` | One of: `present`, `passe_sm`, `passe_sf`, `passe_pm`, `passe_pf` |

```python
model.get_participle("parler", "present")    # → "parlant"
model.get_participle("finir")                # → "fini"
model.get_participle("prendre", "passe_sf")  # → "prise"
```

> **Note:** For intransitive verbs conjugated with *avoir* (e.g. *courir*, *dormir*), the past participle is invariable — `passe_sf`, `passe_pm`, and `passe_pf` return the same form as `passe_sm`.

---

### `model.has_verb(infinitive) → bool`

Check whether a verb is in the model's vocabulary.

```python
model.has_verb("parler")   # → True
model.has_verb("xyzfake")  # → False
```

---

### `model.auxiliary(infinitive) → list[str]`

Returns the auxiliary verb(s) used to form compound tenses.

```python
model.auxiliary("parler")  # → ["avoir"]
model.auxiliary("aller")   # → ["être"]
model.auxiliary("battre")  # → ["avoir", "pronominal"]
```

Possible values in the list: `"avoir"`, `"être"`, `"pronominal"`.

---

### `model.is_h_aspire(infinitive) → bool`

Returns whether the verb begins with an aspirate *h*.

```python
model.is_h_aspire("hurler")   # → True
model.is_h_aspire("habiter")  # → False
```

---

### `model.verb_count → int`

The number of verbs known to the model.

```python
model.verb_count  # → 6132
```

---

### `model.verbs → list[str]`

A sorted list of all known verb infinitives.

---

## Accepted Values

### Modes

| Canonical         | Aliases                          |
|:------------------|:---------------------------------|
| `indicatif`       | `ind`                            |
| `subjonctif`      | `sub`                            |
| `conditionnel`    | `cond`                           |
| `imperatif`       | `imp`, `impératif`               |
| `participe`       | `part`                           |

### Tenses

**Simple tenses** (directly predicted by the neural model):

| Mode            | Tense                | Aliases                      |
|:----------------|:---------------------|:-----------------------------|
| `indicatif`     | `present`            | `présent`                    |
| `indicatif`     | `imparfait`          |                              |
| `indicatif`     | `passe_simple`       | `passé_simple`               |
| `indicatif`     | `futur_simple`       | `futur`                      |
| `conditionnel`  | `present`            | `présent`                    |
| `subjonctif`    | `present`            | `présent`                    |
| `subjonctif`    | `imparfait`          |                              |
| `imperatif`     | `present`            | `présent`                    |

**Compound tenses** (composed automatically from auxiliary + past participle):

| Mode            | Tense                  | Aliases                      |
|:----------------|:-----------------------|:-----------------------------|
| `indicatif`     | `passe_compose`        | `passé_composé`              |
| `indicatif`     | `plus_que_parfait`     |                              |
| `indicatif`     | `passe_anterieur`      | `passé_antérieur`            |
| `indicatif`     | `futur_anterieur`      | `futur_antérieur`            |
| `conditionnel`  | `passe`                | `passé`                      |
| `subjonctif`    | `passe`                | `passé`                      |
| `subjonctif`    | `plus_que_parfait`     |                              |
| `imperatif`     | `passe`                | `passé`                      |

**Participle formes** (used with `get_participle` or `mode="participe"`):

| Forme       | Meaning                              |
|:------------|:-------------------------------------|
| `present`   | Present participle (*parlant*)        |
| `passe_sm`  | Past participle, masculine singular   |
| `passe_sf`  | Past participle, feminine singular    |
| `passe_pm`  | Past participle, masculine plural     |
| `passe_pf`  | Past participle, feminine plural      |

### Persons

| Canonical | Aliases          | Meaning                       |
|:----------|:-----------------|:------------------------------|
| `1s`      | `je`             | First person singular         |
| `2s`      | `tu`             | Second person singular        |
| `3sm`     | `il`, `on`, `3s` | Third person singular masc.   |
| `3sf`     | `elle`           | Third person singular fem.    |
| `1p`      | `nous`           | First person plural           |
| `2p`      | `vous`           | Second person plural          |
| `3pm`     | `ils`, `3p`      | Third person plural masc.     |
| `3pf`     | `elles`          | Third person plural fem.      |

> **Imperatif** only uses `2s`, `1p`, `2p`.

---

## Command-Line Usage

The module can also be run directly from the command line:

```bash
# Full conjugation table
python french_conjugation_model.py parler

# Specific form
python french_conjugation_model.py aller indicatif present 1s
```

Arguments: `verb [mode] [tense] [person]`

---

## Architecture Summary

The model is a character-level sequence-to-sequence neural network with Bahdanau attention, built in PyTorch:

- **Encoder:** Bidirectional GRU over the input verb's characters
- **Decoder:** GRU with attention, conditioned on mode/tense/person via learned embeddings
- **Exception table:** A small lookup (195 entries) embedded in the checkpoint to guarantee 100% accuracy on the training vocabulary

The model file is approximately **6 MB** and covers **6,132 French verbs** across all simple and compound tenses — **390,546 conjugated forms** at **100% accuracy**.

---

## File Overview

| File                             | Purpose                                                    |
|:---------------------------------|:-----------------------------------------------------------|
| `french_conjugation_model.py`    | Module to load and use the model (import this)             |
| `conjugation_model_final.pt`     | Final model checkpoint (ML weights + exception table)      |
| `conjugation_model.pt`           | ML-only model checkpoint (no exception table, 99.95%)      |
| `train_model.py`                 | Training script (produces `conjugation_model.pt`)          |
| `build_final_model.py`           | Builds the final model by adding the exception table       |
| `test_model.py`                  | Unit tests (31 tests)                                      |
| `full_test_model.py`             | Full dataset test against `verbs.db`                       |
| `verbs.db`                       | SQLite database of 6,288 French verbs (source of truth)    |
