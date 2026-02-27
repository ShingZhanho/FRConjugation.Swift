# python_model — ML Training & Python API

This directory contains the PyTorch seq2seq model for French verb conjugation:
training scripts, evaluation tools, and a reusable Python module.

## Overview

| File | Purpose |
|:-----|:--------|
| `french_conjugation_model.py` | Reusable module — `import` this |
| `train_model.py` | Training script (reads `verbs.db`) |
| `build_final_model.py` | Embeds exception table → `conjugation_model_final.pt` |
| `test_model.py` | 31 unit tests |
| `full_test_model.py` | Full-DB validation (390,546 forms) |
| `USAGE.md` | Detailed Python API reference (modes, tenses, persons, aliases) |
| `conjugation_model_final.pt` | Final checkpoint — ML + 195 exceptions (100%) |
| `conjugation_model.pt` | ML-only checkpoint (99.95%) |
| `verbs.db` | SQLite database (6,288 verbs) |

## Data Source

`verbs.db` is a release artefact of
[**ShingZhanho/verbe-conjugaison-academie-francaise**](https://github.com/ShingZhanho/verbe-conjugaison-academie-francaise).
Download it and place it in this directory before training.

## Requirements

- Python 3.10+
- PyTorch 2.x (`pip install torch`)
- `tqdm` (optional, for progress bars during training)

## Training Pipeline

```bash
# 1. Train the neural model
python3 train_model.py
# → conjugation_model.pt (~6 MB, ~99.95% accuracy)

# 2. Run full database test to identify remaining errors
python3 full_test_model.py
# → full_test_errors.json (195 errors)

# 3. Build the final model with an exception table for 100% accuracy
python3 build_final_model.py
# → conjugation_model_final.pt
```

## Quick Start

```python
from french_conjugation_model import ConjugationModel

model = ConjugationModel()  # loads conjugation_model_final.pt

model.conjugate("parler", mode="indicatif", tense="present", person="1s")
# → "parle"

model.conjugate("aller", mode="indicatif", tense="passe_compose", person="3sf")
# → "est allée"

model.conjugate("finir")
# → { "indicatif": { "present": { "1s": "finis", ... }, ... }, ... }

model.get_participle("prendre", "passe_sf")
# → "prise"

model.auxiliary("aller")    # → ["être"]
model.has_verb("parler")    # → True
model.is_h_aspire("haïr")  # → True
```

See [USAGE.md](USAGE.md) for the complete API reference including all accepted
mode, tense, and person values with aliases.

## Testing

```bash
# Unit tests (31 tests)
python3 test_model.py

# Full database validation (390,546 forms)
python3 full_test_model.py [path/to/model.pt]
```

## Architecture

| Component | Detail |
|:----------|:-------|
| Encoder | Bidirectional GRU, 256 hidden, 64-dim char embeddings |
| Attention | Bahdanau (additive) |
| Decoder | GRU conditioned on mode + tense + person embeddings (32-dim each) |
| Exception table | 195 hard-coded corrections for 100% accuracy |
| Size | ~6 MB |
