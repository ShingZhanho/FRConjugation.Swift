# python_model — ML Training & Python API

This directory contains the PyTorch seq2seq model for French verb conjugation:
training scripts, evaluation tools, and a reusable Python module.

## Overview

| File | Purpose |
|:-----|:--------|
| `french_conjugation_model.py` | Reusable module — `import` this |
| `train_model.py` | Training script (reads `verbs.db`) |
| `build_final_model.py` | Embeds exception table → `conjugation_model_final.pt` |
| `test_model.py` | Unit tests |
| `full_test_model.py` | Full-DB validation (2,559,057 forms) |
| `export_weights.py` | Export to portable format for the Swift package |
| `USAGE.md` | Detailed Python API reference (voices, modes, tenses, persons, aliases) |
| `conjugation_model_final.pt` | Final checkpoint — ML + 2,329 exceptions (100%) |
| `conjugation_model.pt` | ML-only checkpoint (~99.91%) |
| `verbs.db` | SQLite database (6,298 verbs, 5 voices, 2,559,057 forms) |

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
# → conjugation_model.pt (~6 MB, ~99.91% accuracy)

# 2. Run full database test to identify remaining errors
python3 full_test_model.py
# → full_test_errors.json (2,329 errors)

# 3. Build the final model with an exception table for 100% accuracy
python3 build_final_model.py
# → conjugation_model_final.pt
```

## Quick Start

```python
from french_conjugation_model import ConjugationModel

model = ConjugationModel()  # loads conjugation_model_final.pt

# Single form (voice is required)
model.conjugate("parler", voice="active_avoir", mode="indicatif",
                tense="present", person="1sm")
# → "parle"

# All forms for a voice
model.conjugate("aller", voice="active_etre")
# → { "indicatif": { "present": { "1sm": "vais", ... }, ... }, ... }

# Full conjugation (all voices)
model.conjugate("finir")
# → { "voix_active_avoir": { ... }, ... }

# Discover voices
model.voices("aller")
# → ["voix_active_etre", "voix_prono"]

model.has_verb("parler")    # → True
model.is_h_aspire("haïr")  # → True
```

See [USAGE.md](USAGE.md) for the complete API reference including all accepted
voice, mode, tense, and person values with aliases.

## Testing

```bash
# Unit tests
python3 test_model.py

# Full database validation (2,559,057 forms)
python3 full_test_model.py [path/to/model.pt]
```

## Architecture

| Component | Detail |
|:----------|:-------|
| Encoder | Bidirectional GRU, 256 hidden, 64-dim char embeddings |
| Attention | Bahdanau (additive) |
| Decoder | GRU conditioned on voice + mode + tense + person embeddings (32-dim each) |
| Bridge | Linear + tanh: encoder hidden + 4 conditioning embeddings → decoder initial hidden |
| Parameters | 1,538,795 |
| Exception table | 2,329 hard-coded corrections for 100% accuracy |
| Accuracy | **100%** on 2,559,057 forms across 6,298 verbs (5 voices) |
| Size | ~6 MB |
