#!/usr/bin/env python3
"""
train_model.py — Train a character-level seq2seq model for French verb conjugation.

Data source : verbs.db  (SQLite – sole source of truth)
Output      : conjugation_model.pt  (model weights + vocabularies + metadata)

The model learns to map  (infinitive, mode, tense, person) → conjugated form
for all simple (single-word) tenses and participles.  Compound tenses are
derived at inference time by composing model predictions (auxiliary + participle).
"""

import os
import random
import sqlite3
import sys
import time

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

_CI = os.environ.get("GH_ACTIONS") == "1"
if not _CI:
    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = None
else:
    tqdm = None

from french_conjugation_model import (
    EOS_IDX,
    PAD_IDX,
    SOS_IDX,
    Seq2SeqModel,
)

# ─── Paths ────────────────────────────────────────────────────────────────────

_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(_DIR, "verbs.db")
MODEL_PATH = os.path.join(_DIR, "conjugation_model.pt")
LOG_PATH = os.path.join(_DIR, "training_log.txt")

# ─── Hyperparameters ────────────────────────────────────────────────────────────────

EMB_DIM = 64
HIDDEN_DIM = 256
COND_DIM = 32
DROPOUT = 0.1
BATCH_SIZE = 512
EPOCHS = 25
LR = 1e-3
WEIGHT_DECAY = 0          # dropout alone provides regularisation
TEACHER_FORCING_START = 1.0
TEACHER_FORCING_MIN = 0.1  # let model practise autoregressive generation
EARLY_STOP_PATIENCE = 7

# Tiered oversampling (total representation multiplier)
OVERSAMPLE_TIER1 = 40   # être, avoir — critical auxiliaries used in every compound tense
OVERSAMPLE_TIER2 = 15   # aller, faire, pouvoir, vouloir, venir, etc. — highly irregular
OVERSAMPLE_TIER3 = 5    # être-only intransitives (mourir, naître, …)

# Key verbs to spot-check with greedy decode each epoch
_SPOT_CHECK = [
    ("être",  "indicatif", "present", "1s", "suis"),
    ("avoir", "indicatif", "present", "1s", "ai"),
    ("aller", "indicatif", "present", "1s", "vais"),
    ("faire", "indicatif", "present", "1s", "fais"),
    ("venir", "indicatif", "present", "1s", "viens"),
    ("être",  "subjonctif", "present", "1s", "sois"),
    ("avoir", "indicatif", "present", "3sm", "a"),
    ("pouvoir", "indicatif", "present", "2s", "peux"),
]

# ─── Data loading ─────────────────────────────────────────────────────────────


def load_training_data():
    """Extract single-word conjugations + simple participles from verbs.db."""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # Verbs
    cur.execute("SELECT id, infinitif, h_aspire FROM verbes")
    verb_rows = cur.fetchall()
    id_to_inf = {r[0]: r[1] for r in verb_rows}
    h_aspire = {r[1] for r in verb_rows if r[2]}

    # Être verbs
    cur.execute("SELECT DISTINCT verbe_id FROM conjugaisons WHERE voix = 'voix_active_etre'")
    etre_verbs = {id_to_inf[r[0]] for r in cur.fetchall() if r[0] in id_to_inf}

    # Pronominal verbs
    cur.execute("SELECT DISTINCT verbe_id FROM conjugaisons WHERE voix = 'voix_prono'")
    prono_verbs = {id_to_inf[r[0]] for r in cur.fetchall() if r[0] in id_to_inf}

    # ── Single-word conjugations (simple tenses) ──
    cur.execute(
        """
        SELECT v.infinitif, c.mode, c.temps, c.personne, c.conjugaison
        FROM conjugaisons c
        JOIN verbes v ON c.verbe_id = v.id
        WHERE c.voix IN ('voix_active_avoir', 'voix_active_etre')
          AND c.conjugaison NOT LIKE '%% %%'
        """
    )
    examples: list[tuple[str, str, str, str, str]] = []
    seen: set[tuple[str, str, str, str]] = set()
    for inf, mode, tense, person, form in cur.fetchall():
        # Take first variant if semicolon-separated
        form = form.split(";")[0]
        key = (inf, mode, tense, person)
        if key not in seen:
            seen.add(key)
            examples.append((inf, mode, tense, person, form))

    # ── Simple participles ──
    # For intransitive verbs the DB stores the invariable (masculine singular)
    # form for all agreement slots (passe_sm == passe_sf == passe_pm == passe_pf).
    # Training on those duplicate-valued slots confuses the model — it can't
    # distinguish "same form because intransitive" from "should inflect".
    # Fix: detect invariable verbs and only keep passe_sm for them.

    cur.execute(
        """
        SELECT v.infinitif, p.forme, p.participe
        FROM participes p
        JOIN verbes v ON p.verbe_id = v.id
        WHERE p.voix IN ('voix_active_avoir', 'voix_active_etre')
          AND p.participe NOT LIKE '%% %%'
        ORDER BY v.infinitif, p.forme
        """
    )
    # First pass: collect all participles per verb
    pp_by_verb: dict[str, dict[str, str]] = {}
    for inf, forme, participe in cur.fetchall():
        participe = participe.split(";")[0]
        pp_by_verb.setdefault(inf, {})[forme] = participe

    _AGREEMENT_FORMES = {"passe_sf", "passe_pm", "passe_pf"}

    pseen: set[tuple[str, str]] = set()
    n_skipped_invariable = 0
    for inf, formes in pp_by_verb.items():
        sm_val = formes.get("passe_sm")
        # Check if this verb has an invariable PP (all passe_* identical to sm)
        is_invariable = sm_val is not None and all(
            formes.get(f) == sm_val for f in _AGREEMENT_FORMES if f in formes
        )
        for forme, participe in formes.items():
            # Skip agreement variants for invariable verbs
            if is_invariable and forme in _AGREEMENT_FORMES:
                n_skipped_invariable += 1
                continue
            key2 = (inf, forme)
            if key2 not in pseen:
                pseen.add(key2)
                examples.append((inf, "participe", forme, "-", participe))
    # Collect the set of invariable verbs for the checkpoint
    invariable_pp_verbs = sorted(
        inf for inf, formes in pp_by_verb.items()
        if formes.get("passe_sm") is not None
        and all(formes.get(f) == formes["passe_sm"]
                for f in _AGREEMENT_FORMES if f in formes)
    )
    print(f"   Skipped {n_skipped_invariable} invariable PP agreement entries "
          f"({len(invariable_pp_verbs)} verbs)")

    conn.close()

    # ── Oversample underrepresented verbs ──
    # Être-only verbs (aller, mourir, naître, …) and auxiliary verbs themselves
    # (être, avoir) are vastly outnumbered by regular verbs.  Repeat their
    # examples so the model sees them often enough to memorize their irregular
    # forms.
    avoir_verb_infs = {e[0] for e in examples} - etre_verbs
    etre_only = etre_verbs - avoir_verb_infs  # verbs with NO avoir entries

    tier1 = {"être", "avoir"}
    tier2 = {"aller", "faire", "pouvoir", "vouloir", "savoir", "devoir",
             "voir", "venir", "dire", "prendre"}
    tier3 = etre_only - tier1 - tier2  # remaining être-only verbs

    extra: list[tuple[str, str, str, str, str]] = []
    for ex in examples:
        verb = ex[0]
        if verb in tier1:
            extra.extend([ex] * (OVERSAMPLE_TIER1 - 1))
        elif verb in tier2:
            extra.extend([ex] * (OVERSAMPLE_TIER2 - 1))
        elif verb in tier3:
            extra.extend([ex] * (OVERSAMPLE_TIER3 - 1))
    examples.extend(extra)
    print(f"   Oversampling tiers: T1({OVERSAMPLE_TIER1}x)={len(tier1)} verbs, "
          f"T2({OVERSAMPLE_TIER2}x)={len(tier2)} verbs, "
          f"T3({OVERSAMPLE_TIER3}x)={len(tier3)} verbs")
    print(f"   After oversampling irregular verbs: {len(examples):,}")

    return examples, sorted(etre_verbs), sorted(prono_verbs), sorted(h_aspire), invariable_pp_verbs


# ─── Vocabularies ─────────────────────────────────────────────────────────────


def build_vocabularies(examples):
    chars: set[str] = set()
    modes: set[str] = set()
    tenses: set[str] = set()
    persons: set[str] = set()

    for inf, mode, tense, person, form in examples:
        chars.update(inf)
        chars.update(form)
        modes.add(mode)
        tenses.add(tense)
        persons.add(person)

    sorted_chars = sorted(chars)
    char_to_idx = {c: i + 3 for i, c in enumerate(sorted_chars)}
    idx_to_char = {i + 3: c for i, c in enumerate(sorted_chars)}
    idx_to_char[PAD_IDX] = "<PAD>"
    idx_to_char[SOS_IDX] = "<SOS>"
    idx_to_char[EOS_IDX] = "<EOS>"

    sorted_modes = sorted(modes)
    sorted_tenses = sorted(tenses)
    sorted_persons = sorted(persons)

    mode_to_idx = {m: i + 1 for i, m in enumerate(sorted_modes)}
    tense_to_idx = {t: i + 1 for i, t in enumerate(sorted_tenses)}
    person_to_idx = {p: i + 1 for i, p in enumerate(sorted_persons)}

    return {
        "char_to_idx": char_to_idx,
        "idx_to_char": idx_to_char,
        "mode_to_idx": mode_to_idx,
        "tense_to_idx": tense_to_idx,
        "person_to_idx": person_to_idx,
        "vocab_size": len(sorted_chars) + 3,
        "n_modes": len(sorted_modes) + 1,
        "n_tenses": len(sorted_tenses) + 1,
        "n_persons": len(sorted_persons) + 1,
    }


# ─── Dataset ──────────────────────────────────────────────────────────────────


class ConjugationDataset(Dataset):
    def __init__(self, examples, vocab):
        self.examples = examples
        self.c2i = vocab["char_to_idx"]
        self.m2i = vocab["mode_to_idx"]
        self.t2i = vocab["tense_to_idx"]
        self.p2i = vocab["person_to_idx"]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        inf, mode, tense, person, form = self.examples[idx]
        src = torch.tensor([self.c2i[c] for c in inf] + [EOS_IDX], dtype=torch.long)
        tgt = torch.tensor(
            [SOS_IDX] + [self.c2i[c] for c in form] + [EOS_IDX], dtype=torch.long
        )
        return (
            src,
            tgt,
            torch.tensor(self.m2i.get(mode, 0), dtype=torch.long),
            torch.tensor(self.t2i.get(tense, 0), dtype=torch.long),
            torch.tensor(self.p2i.get(person, 0), dtype=torch.long),
        )


def collate_fn(batch):
    srcs, tgts, ms, ts, ps = zip(*batch)
    return (
        pad_sequence(srcs, batch_first=True, padding_value=PAD_IDX),
        pad_sequence(tgts, batch_first=True, padding_value=PAD_IDX),
        torch.stack(ms),
        torch.stack(ts),
        torch.stack(ps),
    )


# ─── Training ─────────────────────────────────────────────────────────────────


def evaluate_greedy(model, examples, vocab, limit=None):
    """Evaluate with greedy decoding.  Returns (correct, total, errors)."""
    model.eval()
    c2i = vocab["char_to_idx"]
    i2c = vocab["idx_to_char"]
    m2i = vocab["mode_to_idx"]
    t2i = vocab["tense_to_idx"]
    p2i = vocab["person_to_idx"]

    correct = 0
    total = 0
    errors: list[tuple] = []

    subset = examples if limit is None else examples[:limit]
    for inf, mode, tense, person, expected in subset:
        src = torch.tensor([[c2i[c] for c in inf] + [EOS_IDX]], dtype=torch.long)
        mi = torch.tensor([m2i.get(mode, 0)], dtype=torch.long)
        ti = torch.tensor([t2i.get(tense, 0)], dtype=torch.long)
        pi = torch.tensor([p2i.get(person, 0)], dtype=torch.long)

        out_ids = model.predict(src, mi, ti, pi)
        predicted = "".join(i2c.get(i, "") for i in out_ids)

        total += 1
        if predicted == expected:
            correct += 1
        else:
            errors.append((inf, f"{mode}.{tense}.{person}", expected, predicted))

    return correct, total, errors


def setup_logging():
    """Duplicate all output to both stdout and training_log.txt."""
    log_file = open(LOG_PATH, "w", encoding="utf-8")

    class Tee:
        def __init__(self, *streams):
            self.streams = streams

        def write(self, data):
            for s in self.streams:
                s.write(data)
                s.flush()

        def flush(self):
            for s in self.streams:
                s.flush()

    sys.stdout = Tee(sys.__stdout__, log_file)
    sys.stderr = Tee(sys.__stderr__, log_file)
    return log_file


def _spot_check(model, vocab):
    """Quick greedy decode of key irregular forms.  Returns (n_ok, n_total, details)."""
    c2i = vocab["char_to_idx"]
    i2c = vocab["idx_to_char"]
    m2i = vocab["mode_to_idx"]
    t2i = vocab["tense_to_idx"]
    p2i = vocab["person_to_idx"]
    model.eval()
    ok = 0
    details: list[str] = []
    for inf, mode, tense, person, expected in _SPOT_CHECK:
        src = torch.tensor([[c2i[c] for c in inf] + [EOS_IDX]], dtype=torch.long)
        mi = torch.tensor([m2i.get(mode, 0)], dtype=torch.long)
        ti = torch.tensor([t2i.get(tense, 0)], dtype=torch.long)
        pi = torch.tensor([p2i.get(person, 0)], dtype=torch.long)
        out_ids = model.predict(src, mi, ti, pi)
        pred = "".join(i2c.get(i, "") for i in out_ids)
        status = "✓" if pred == expected else "✗"
        if pred == expected:
            ok += 1
        details.append(f"{status} {inf}({person})={pred} (exp {expected})")
    return ok, len(_SPOT_CHECK), details


def train():
    log_file = setup_logging()

    print("=" * 60)
    print("  French Verb Conjugation — ML Model Training")
    print("=" * 60)

    # 1. Load data
    print("\n1. Loading data from verbs.db …")
    examples, etre_verbs, prono_verbs, h_aspire, invariable_pp_verbs = load_training_data()
    print(f"   Training examples (with oversampling): {len(examples):,}")

    # 2. Vocabularies (built from unique examples only)
    print("2. Building vocabularies …")
    unique_examples = list({(inf, m, t, p, f): (inf, m, t, p, f)
                            for inf, m, t, p, f in examples}.values())
    vocab = build_vocabularies(unique_examples)
    known_verbs = sorted({inf for inf, *_ in unique_examples})
    print(f"   Char vocab : {vocab['vocab_size']} tokens")
    print(f"   Modes      : {vocab['n_modes'] - 1}")
    print(f"   Tenses     : {vocab['n_tenses'] - 1}")
    print(f"   Persons    : {vocab['n_persons'] - 1}")

    # 3. Train/val split
    random.seed(42)
    random.shuffle(examples)
    split = int(0.97 * len(examples))
    train_ex = examples[:split]
    val_ex = examples[split:]
    print(f"   Train: {len(train_ex):,}  |  Val: {len(val_ex):,}")

    train_loader = DataLoader(
        ConjugationDataset(train_ex, vocab),
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
    )
    val_loader = DataLoader(
        ConjugationDataset(val_ex, vocab),
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )

    # 4. Model
    print("3. Building seq2seq model …")
    model = Seq2SeqModel(
        vocab_size=vocab["vocab_size"],
        emb_dim=EMB_DIM,
        hidden_dim=HIDDEN_DIM,
        cond_dim=COND_DIM,
        n_modes=vocab["n_modes"],
        n_tenses=vocab["n_tenses"],
        n_persons=vocab["n_persons"],
        dropout=DROPOUT,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"   Parameters : {n_params:,}")
    print(f"   Dropout    : {DROPOUT}")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR,
                                   weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=3, factor=0.5, min_lr=1e-5
    )

    # 5. Training loop
    print(f"\n4. Training (up to {EPOCHS} epochs, early-stop patience={EARLY_STOP_PATIENCE}) …")
    best_val_acc = 0.0
    best_state = None
    epochs_no_improve = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0
        t0 = time.time()

        # Linear decay from TEACHER_FORCING_START to TEACHER_FORCING_MIN
        progress = (epoch - 1) / max(EPOCHS - 1, 1)
        tf_ratio = TEACHER_FORCING_START - (TEACHER_FORCING_START - TEACHER_FORCING_MIN) * progress

        loader_iter = train_loader
        if tqdm is not None:
            loader_iter = tqdm(train_loader, desc=f"   Epoch {epoch:2d}/{EPOCHS}",
                               leave=False, file=sys.__stdout__, ncols=80)
        for src, tgt, ms, ts, ps in loader_iter:
            optimizer.zero_grad()
            out = model(src, tgt, ms, ts, ps, tf_ratio)
            loss = criterion(out.reshape(-1, out.size(-1)), tgt[:, 1:].reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
            if tqdm is not None:
                loader_iter.set_postfix(loss=f"{total_loss / n_batches:.4f}")
            elif n_batches % 100 == 0:
                print(f"   Epoch {epoch:2d}/{EPOCHS}  batch {n_batches}  loss={total_loss / n_batches:.4f}")
        if tqdm is not None:
            loader_iter.close()

        avg_loss = total_loss / n_batches
        elapsed = time.time() - t0

        # Quick validation (full-sequence match with TF=0)
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for src, tgt, ms, ts, ps in val_loader:
                out = model(src, tgt, ms, ts, ps, 0.0)
                preds = out.argmax(-1)
                target = tgt[:, 1:]
                for i in range(preds.size(0)):
                    t_seq = []
                    for j in range(target.size(1)):
                        ti = target[i, j].item()
                        if ti in (EOS_IDX, PAD_IDX):
                            break
                        t_seq.append(ti)
                    p_seq = []
                    for j in range(preds.size(1)):
                        pi = preds[i, j].item()
                        if pi in (EOS_IDX, PAD_IDX):
                            break
                        p_seq.append(pi)
                    if p_seq == t_seq:
                        val_correct += 1
                    val_total += 1

        val_acc = val_correct / val_total * 100 if val_total else 0
        scheduler.step(val_acc)  # schedule on validation accuracy
        lr_now = optimizer.param_groups[0]["lr"]

        # Spot-check key irregular verbs with greedy decoding
        sc_ok, sc_n, sc_details = _spot_check(model, vocab)

        print(
            f"   Epoch {epoch:2d}/{EPOCHS}  loss={avg_loss:.4f}  "
            f"val_acc={val_acc:.2f}%  spot={sc_ok}/{sc_n}  "
            f"tf={tf_ratio:.2f}  lr={lr_now:.1e}  time={elapsed:.0f}s"
        )
        for d in sc_details:
            print(f"     {d}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= EARLY_STOP_PATIENCE:
                print(f"\n   Early stopping after {epoch} epochs (no improvement for {EARLY_STOP_PATIENCE}).")
                break

    # 6. Restore best weights
    if best_state is None:
        print("   WARNING: no best state saved — using final weights.")
    else:
        model.load_state_dict(best_state)

    # 7. Final greedy-decoding evaluation
    print(f"\n5. Final evaluation (greedy decoding on {len(val_ex):,} val examples) …")
    correct, total, errors = evaluate_greedy(model, val_ex, vocab)
    final_acc = correct / total * 100
    print(f"   Greedy accuracy: {final_acc:.2f}% ({correct:,}/{total:,})")
    if errors:
        print("   Sample errors:")
        for inf, slot, exp, pred in errors[:15]:
            print(f"     {inf} [{slot}]: expected '{exp}', got '{pred}'")

    # 8. Save
    print(f"\n6. Saving to {MODEL_PATH} …")
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "vocab": vocab,
        "etre_verbs": etre_verbs,
        "prono_verbs": prono_verbs,
        "h_aspire": h_aspire,
        "known_verbs": known_verbs,
        "invariable_pp_verbs": invariable_pp_verbs,
        "hyperparams": {
            "emb_dim": EMB_DIM,
            "hidden_dim": HIDDEN_DIM,
            "cond_dim": COND_DIM,
            "dropout": DROPOUT,
        },
        "accuracy": final_acc,
    }
    torch.save(checkpoint, MODEL_PATH)
    size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)
    print(f"   Model size : {size_mb:.1f} MB")
    print(f"   Best val accuracy  : {best_val_acc:.2f}%")
    print(f"   Greedy accuracy    : {final_acc:.2f}%")
    print(f"\nDone!  Log saved to {LOG_PATH}")


if __name__ == "__main__":
    train()
