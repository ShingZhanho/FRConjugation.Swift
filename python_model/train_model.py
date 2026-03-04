#!/usr/bin/env python3
"""
train_model.py -- Train a character-level seq2seq model for French verb
conjugation.

Data source: verbs.db (SQLite -- sole source of truth)
Output:      conjugation_model.pt (model weights + vocabularies + metadata)

The model learns to map (infinitive, voice, mode, tense, person) to a
conjugated form. All voices, tenses (simple and compound), and participle
forms from the database are included in training data.
"""

import json
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

# -- Device ----------------------------------------------------------------

if torch.backends.mps.is_available():
    DEVICE = torch.device("cpu")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

# -- Paths -----------------------------------------------------------------

_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(_DIR, "verbs.db")
MODEL_PATH = os.path.join(_DIR, "conjugation_model.pt")
RESUME_PATH = os.path.join(_DIR, "training_resume.pt")
LOG_PATH = os.path.join(_DIR, "training_log.txt")

# -- Hyperparameters -------------------------------------------------------

EMB_DIM = 64
HIDDEN_DIM = 256
COND_DIM = 32
DROPOUT = 0.15
BATCH_SIZE = 1024
EPOCHS = 25
LR = 1.5e-3
WEIGHT_DECAY = 0
TEACHER_FORCING_START = 1.0
TEACHER_FORCING_MIN = 0.1
EARLY_STOP_PATIENCE = 7

# Oversampling multipliers for rare categories
OVERSAMPLE_VOIX_ACTIVE = 40   # only ~49 rows (defective verbs)
OVERSAMPLE_ETRE_VERBS = 10    # ~8K rows vs 500K for avoir

# Downsample multipliers for very large categories (passive ~840K, prono ~280K)
DOWNSAMPLE_PASSIVE = 0.15   # keep 15% of passive rows
DOWNSAMPLE_PRONO = 0.25     # keep 25% of pronominal rows

# Spot-check forms each epoch
_SPOT_CHECK = [
    ("\u00eatre",  "voix_active_avoir", "indicatif", "pr\u00e9sent", "1sm", "suis"),
    ("avoir", "voix_active_avoir", "indicatif", "pr\u00e9sent", "1sm", "ai"),
    ("aller", "voix_active_etre",  "indicatif", "pr\u00e9sent", "1sm", "vais"),
    ("faire", "voix_active_avoir", "indicatif", "pr\u00e9sent", "1sm", "fais"),
    ("venir", "voix_active_etre",  "indicatif", "pr\u00e9sent", "1sm", "viens"),
    ("\u00eatre",  "voix_active_avoir", "subjonctif", "pr\u00e9sent", "1sm", "sois"),
    ("aimer", "voix_passive",      "indicatif", "pr\u00e9sent", "1sm", "suis aim\u00e9"),
    ("laver", "voix_prono",        "indicatif", "pr\u00e9sent", "1sm", "me lave"),
]


# -- Data loading ----------------------------------------------------------


def _expand_person_key(merged_key):
    """Split '1sm;1sf' into ['1sm', '1sf']."""
    return merged_key.split(";")


def load_training_data():
    """Load all conjugations and participles from verbs.db.

    Merged person keys (like '1sm;1sf') are expanded into separate
    training examples, each mapping to the same conjugated form.

    Returns (examples, metadata_dict).
    """
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # -- verb metadata --
    cur.execute("SELECT id, infinitif, h_aspire, rectification_1990, "
                "rectification_1990_variante FROM verbes")
    verb_rows = cur.fetchall()
    id_to_inf = {r[0]: r[1] for r in verb_rows}
    h_aspire = sorted({r[1] for r in verb_rows if r[2]})
    reform_1990_verbs = sorted({r[1] for r in verb_rows if r[3]})
    reform_variantes = {r[1]: r[4] for r in verb_rows if r[4]}

    # -- conjugations --
    # Load all rows from all voices
    cur.execute("""
        SELECT v.infinitif, c.voix, c.mode, c.temps, c.personne, c.conjugaison
        FROM conjugaisons c
        JOIN verbes v ON c.verbe_id = v.id
    """)

    # example format: (infinitive, voice, mode, tense, person, form)
    examples = []
    # structure tracks what slots exist per verb
    verb_structure = {}
    seen = set()

    for inf, voice, mode, tense, person_merged, form in cur.fetchall():
        # take first variant if semicolon-separated (reform variants)
        form = form.split(";")[0].strip()

        # expand merged person keys into individual training examples
        for person in _expand_person_key(person_merged):
            key = (inf, voice, mode, tense, person)
            if key in seen:
                continue
            seen.add(key)
            examples.append((inf, voice, mode, tense, person, form))

            # record structure
            vs = verb_structure.setdefault(inf, {})
            ms = vs.setdefault(voice, {})
            ts = ms.setdefault(mode, {})
            ps = ts.setdefault(tense, [])
            if person not in ps:
                ps.append(person)

    n_conj = len(examples)
    print(f"   Conjugation examples (expanded): {n_conj:,}")

    # -- participles --
    cur.execute("""
        SELECT v.infinitif, p.voix, p.forme, p.participe
        FROM participes p
        JOIN verbes v ON p.verbe_id = v.id
    """)

    part_seen = set()
    for inf, voice, forme, participe in cur.fetchall():
        participe = participe.split(";")[0].strip()
        key = (inf, voice, forme)
        if key in part_seen:
            continue
        part_seen.add(key)
        # participles use mode="participe", tense=<forme>, person="-"
        examples.append((inf, voice, "participe", forme, "-", participe))

        # record structure
        vs = verb_structure.setdefault(inf, {})
        ms = vs.setdefault(voice, {})
        ts = ms.setdefault("participe", {})
        ps = ts.setdefault(forme, [])
        if "-" not in ps:
            ps.append("-")

    n_part = len(examples) - n_conj
    print(f"   Participle examples: {n_part:,}")

    conn.close()

    # -- downsampling large categories --
    # passive (~840K expanded) and pronominal (~280K) dominate the dataset
    # Downsample them to keep training time manageable
    rng = random.Random(42)
    filtered = []
    for ex in examples:
        voice = ex[1]
        if voice == "voix_passive":
            if rng.random() < DOWNSAMPLE_PASSIVE:
                filtered.append(ex)
        elif voice == "voix_prono":
            if rng.random() < DOWNSAMPLE_PRONO:
                filtered.append(ex)
        else:
            filtered.append(ex)
    examples = filtered

    # -- oversampling rare categories --
    # voix_active is extremely rare (~49 rows) -- upsample heavily
    # voix_active_etre is also relatively rare -- upsample
    extra = []
    for ex in examples:
        voice = ex[1]
        if voice == "voix_active":
            extra.extend([ex] * (OVERSAMPLE_VOIX_ACTIVE - 1))
        elif voice == "voix_active_etre":
            extra.extend([ex] * (OVERSAMPLE_ETRE_VERBS - 1))
    examples.extend(extra)

    known_verbs = sorted(verb_structure.keys())
    print(f"   Total examples (with oversampling): {len(examples):,}")
    print(f"   Known verbs: {len(known_verbs):,}")

    metadata = {
        "h_aspire": h_aspire,
        "reform_1990_verbs": reform_1990_verbs,
        "reform_variantes": reform_variantes,
        "known_verbs": known_verbs,
        "verb_structure": verb_structure,
    }
    return examples, metadata


# -- Vocabularies ----------------------------------------------------------


def build_vocabularies(examples):
    chars = set()
    voices = set()
    modes = set()
    tenses = set()
    persons = set()

    for inf, voice, mode, tense, person, form in examples:
        chars.update(inf)
        chars.update(form)
        voices.add(voice)
        modes.add(mode)
        tenses.add(tense)
        persons.add(person)

    sorted_chars = sorted(chars)
    char_to_idx = {c: i + 3 for i, c in enumerate(sorted_chars)}
    idx_to_char = {i + 3: c for i, c in enumerate(sorted_chars)}
    idx_to_char[PAD_IDX] = "<PAD>"
    idx_to_char[SOS_IDX] = "<SOS>"
    idx_to_char[EOS_IDX] = "<EOS>"

    sorted_voices = sorted(voices)
    sorted_modes = sorted(modes)
    sorted_tenses = sorted(tenses)
    sorted_persons = sorted(persons)

    voice_to_idx = {v: i + 1 for i, v in enumerate(sorted_voices)}
    mode_to_idx = {m: i + 1 for i, m in enumerate(sorted_modes)}
    tense_to_idx = {t: i + 1 for i, t in enumerate(sorted_tenses)}
    person_to_idx = {p: i + 1 for i, p in enumerate(sorted_persons)}

    return {
        "char_to_idx": char_to_idx,
        "idx_to_char": idx_to_char,
        "voice_to_idx": voice_to_idx,
        "mode_to_idx": mode_to_idx,
        "tense_to_idx": tense_to_idx,
        "person_to_idx": person_to_idx,
        "vocab_size": len(sorted_chars) + 3,
        "n_voices": len(sorted_voices) + 1,
        "n_modes": len(sorted_modes) + 1,
        "n_tenses": len(sorted_tenses) + 1,
        "n_persons": len(sorted_persons) + 1,
    }


# -- Dataset ---------------------------------------------------------------


class ConjugationDataset(Dataset):
    def __init__(self, examples, vocab):
        self.examples = examples
        self.c2i = vocab["char_to_idx"]
        self.v2i = vocab["voice_to_idx"]
        self.m2i = vocab["mode_to_idx"]
        self.t2i = vocab["tense_to_idx"]
        self.p2i = vocab["person_to_idx"]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        inf, voice, mode, tense, person, form = self.examples[idx]
        src = torch.tensor([self.c2i[c] for c in inf] + [EOS_IDX],
                           dtype=torch.long)
        tgt = torch.tensor(
            [SOS_IDX] + [self.c2i[c] for c in form] + [EOS_IDX],
            dtype=torch.long,
        )
        return (
            src, tgt,
            torch.tensor(self.v2i.get(voice, 0), dtype=torch.long),
            torch.tensor(self.m2i.get(mode, 0), dtype=torch.long),
            torch.tensor(self.t2i.get(tense, 0), dtype=torch.long),
            torch.tensor(self.p2i.get(person, 0), dtype=torch.long),
        )


def collate_fn(batch):
    srcs, tgts, vs, ms, ts, ps = zip(*batch)
    return (
        pad_sequence(srcs, batch_first=True, padding_value=PAD_IDX),
        pad_sequence(tgts, batch_first=True, padding_value=PAD_IDX),
        torch.stack(vs),
        torch.stack(ms),
        torch.stack(ts),
        torch.stack(ps),
    )


# -- Evaluation ------------------------------------------------------------


def evaluate_greedy(model, examples, vocab, limit=None):
    """Evaluate with greedy decoding. Returns (correct, total, errors)."""
    model.eval()
    c2i = vocab["char_to_idx"]
    i2c = vocab["idx_to_char"]
    v2i = vocab["voice_to_idx"]
    m2i = vocab["mode_to_idx"]
    t2i = vocab["tense_to_idx"]
    p2i = vocab["person_to_idx"]

    correct = 0
    total = 0
    errors = []

    subset = examples if limit is None else examples[:limit]
    for inf, voice, mode, tense, person, expected in subset:
        src = torch.tensor([[c2i[c] for c in inf] + [EOS_IDX]],
                           dtype=torch.long).to(DEVICE)
        vi = torch.tensor([v2i.get(voice, 0)], dtype=torch.long).to(DEVICE)
        mi = torch.tensor([m2i.get(mode, 0)], dtype=torch.long).to(DEVICE)
        ti = torch.tensor([t2i.get(tense, 0)], dtype=torch.long).to(DEVICE)
        pi = torch.tensor([p2i.get(person, 0)], dtype=torch.long).to(DEVICE)

        out_ids = model.predict(src, vi, mi, ti, pi)
        predicted = "".join(i2c.get(i, "") for i in out_ids)

        total += 1
        if predicted == expected:
            correct += 1
        else:
            errors.append((inf, f"{voice}.{mode}.{tense}.{person}",
                           expected, predicted))

    return correct, total, errors


# -- Training --------------------------------------------------------------


def setup_logging():
    """Tee stdout/stderr to both terminal and log file."""
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
    c2i = vocab["char_to_idx"]
    i2c = vocab["idx_to_char"]
    v2i = vocab["voice_to_idx"]
    m2i = vocab["mode_to_idx"]
    t2i = vocab["tense_to_idx"]
    p2i = vocab["person_to_idx"]
    model.eval()
    ok = 0
    details = []
    for inf, voice, mode, tense, person, expected in _SPOT_CHECK:
        src = torch.tensor([[c2i.get(c, 0) for c in inf] + [EOS_IDX]],
                           dtype=torch.long).to(DEVICE)
        vi = torch.tensor([v2i.get(voice, 0)], dtype=torch.long).to(DEVICE)
        mi = torch.tensor([m2i.get(mode, 0)], dtype=torch.long).to(DEVICE)
        ti = torch.tensor([t2i.get(tense, 0)], dtype=torch.long).to(DEVICE)
        pi = torch.tensor([p2i.get(person, 0)], dtype=torch.long).to(DEVICE)
        out_ids = model.predict(src, vi, mi, ti, pi)
        pred = "".join(i2c.get(i, "") for i in out_ids)
        match = pred == expected
        tag = " OK " if match else "MISS"
        if match:
            ok += 1
        details.append(f"[{tag}] {inf}({voice[:10]},{person})={pred}"
                       f" (exp {expected})")
    return ok, len(_SPOT_CHECK), details


def _optimizer_state_to_cpu(optimizer):
    """Return a copy of optimizer.state_dict() with all tensors on CPU."""
    sd = optimizer.state_dict()
    cpu_state = []
    for group_state in sd["state"].values():
        gs = {}
        for k, v in group_state.items():
            gs[k] = v.cpu() if isinstance(v, torch.Tensor) else v
        cpu_state.append(gs)
    return {
        "state": dict(zip(sd["state"].keys(), cpu_state)),
        "param_groups": sd["param_groups"],
    }


def _save_resume_checkpoint(epoch, model, optimizer, scheduler,
                            best_val_acc, best_state, epochs_no_improve):
    """Save everything needed to resume training after interruption.

    Does NOT save vocab or metadata -- those are rebuilt deterministically
    from verbs.db on every run, so duplicating them here wastes memory.
    """
    torch.save({
        "epoch": epoch,
        "model_state_dict": {k: v.cpu().clone()
                             for k, v in model.state_dict().items()},
        "optimizer_state_dict": _optimizer_state_to_cpu(optimizer),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_val_acc": best_val_acc,
        "best_state": best_state,  # already on CPU
        "epochs_no_improve": epochs_no_improve,
        "hyperparams": {
            "emb_dim": EMB_DIM,
            "hidden_dim": HIDDEN_DIM,
            "cond_dim": COND_DIM,
            "dropout": DROPOUT,
        },
    }, RESUME_PATH)


def _load_resume_checkpoint():
    """Load resume checkpoint if it exists. Returns dict or None."""
    if not os.path.isfile(RESUME_PATH):
        return None
    print(f"   Found resume checkpoint: {RESUME_PATH}")
    cp = torch.load(RESUME_PATH, map_location="cpu", weights_only=False)
    # quick sanity check: hyperparams must match current settings
    hp = cp.get("hyperparams", {})
    if (hp.get("emb_dim") != EMB_DIM or hp.get("hidden_dim") != HIDDEN_DIM
            or hp.get("cond_dim") != COND_DIM):
        print("   WARNING: hyperparams changed since checkpoint -- "
              "starting fresh.")
        return None
    print(f"   Resuming from epoch {cp['epoch']} "
          f"(best_val_acc={cp['best_val_acc']:.2f}%)")
    return cp


def train():
    fresh = "--fresh" in sys.argv
    log_file = setup_logging()

    print("=" * 60)
    print("  French Verb Conjugation -- Model Training (v3)")
    print("  All voices, expanded person keys, compound tenses")
    print("=" * 60)

    # check for resume checkpoint before loading data
    resume_cp = None if fresh else _load_resume_checkpoint()
    if fresh and os.path.isfile(RESUME_PATH):
        os.remove(RESUME_PATH)
        print("   Removed stale resume checkpoint (--fresh).")

    # 1. Load data
    print("\n1. Loading data from verbs.db ...")
    examples, metadata = load_training_data()

    # 2. Vocabularies -- rebuild from data (deterministic)
    print("2. Building vocabularies ...")
    unique_examples = list({
        (inf, v, m, t, p, f): (inf, v, m, t, p, f)
        for inf, v, m, t, p, f in examples
    }.values())
    vocab = build_vocabularies(unique_examples)
    del unique_examples  # free memory
    print(f"   Char vocab : {vocab['vocab_size']} tokens")
    print(f"   Voices     : {vocab['n_voices'] - 1}")
    print(f"   Modes      : {vocab['n_modes'] - 1}")
    print(f"   Tenses     : {vocab['n_tenses'] - 1}")
    print(f"   Persons    : {vocab['n_persons'] - 1}")

    # 3. Train/val split (deterministic with seed=42)
    random.seed(42)
    random.shuffle(examples)
    split = int(0.97 * len(examples))
    train_ex = examples[:split]
    val_ex = examples[split:]
    print(f"   Train: {len(train_ex):,}  |  Val: {len(val_ex):,}")
    del examples  # train_ex and val_ex hold the actual refs now

    train_loader = DataLoader(
        ConjugationDataset(train_ex, vocab),
        batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=collate_fn, num_workers=0,
    )
    val_loader = DataLoader(
        ConjugationDataset(val_ex, vocab),
        batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=collate_fn, num_workers=0,
    )

    # 4. Model
    print("3. Building seq2seq model ...")
    print(f"   Device     : {DEVICE}")
    model = Seq2SeqModel(
        vocab_size=vocab["vocab_size"],
        emb_dim=EMB_DIM,
        hidden_dim=HIDDEN_DIM,
        cond_dim=COND_DIM,
        n_voices=vocab["n_voices"],
        n_modes=vocab["n_modes"],
        n_tenses=vocab["n_tenses"],
        n_persons=vocab["n_persons"],
        dropout=DROPOUT,
    ).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"   Parameters : {n_params:,}")
    print(f"   Dropout    : {DROPOUT}")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR,
                                 weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=3, factor=0.5, min_lr=1e-5,
    )

    # -- Restore state if resuming --
    start_epoch = 1
    best_val_acc = 0.0
    best_state = None
    epochs_no_improve = 0

    if resume_cp is not None:
        model.load_state_dict(resume_cp["model_state_dict"])
        model.to(DEVICE)
        optimizer.load_state_dict(resume_cp["optimizer_state_dict"])
        # move optimizer tensors to device
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(DEVICE)
        scheduler.load_state_dict(resume_cp["scheduler_state_dict"])
        start_epoch = resume_cp["epoch"] + 1
        best_val_acc = resume_cp["best_val_acc"]
        best_state = resume_cp["best_state"]
        epochs_no_improve = resume_cp["epochs_no_improve"]
        del resume_cp  # free the loaded checkpoint (~66 MB on disk,
                       # much more in memory)
        import gc; gc.collect()
        print(f"   Restored state: start_epoch={start_epoch}, "
              f"best_val_acc={best_val_acc:.2f}%, "
              f"epochs_no_improve={epochs_no_improve}")

    # 5. Training loop
    print(f"\n4. Training (epochs {start_epoch}..{EPOCHS}, "
          f"early-stop patience={EARLY_STOP_PATIENCE}) ...")

    for epoch in range(start_epoch, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0
        t0 = time.time()

        progress = (epoch - 1) / max(EPOCHS - 1, 1)
        tf_ratio = (TEACHER_FORCING_START
                    - (TEACHER_FORCING_START - TEACHER_FORCING_MIN) * progress)

        loader_iter = train_loader
        if tqdm is not None:
            loader_iter = tqdm(train_loader,
                               desc=f"   Epoch {epoch:2d}/{EPOCHS}",
                               leave=False, file=sys.__stdout__, ncols=80)

        for src, tgt, vs, ms, ts, ps in loader_iter:
            src = src.to(DEVICE)
            tgt = tgt.to(DEVICE)
            vs = vs.to(DEVICE)
            ms = ms.to(DEVICE)
            ts = ts.to(DEVICE)
            ps = ps.to(DEVICE)
            optimizer.zero_grad()
            out = model(src, tgt, vs, ms, ts, ps, tf_ratio)
            loss = criterion(out.reshape(-1, out.size(-1)),
                             tgt[:, 1:].reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
            if tqdm is not None:
                loader_iter.set_postfix(loss=f"{total_loss / n_batches:.4f}")
            elif n_batches % 200 == 0:
                print(f"   Epoch {epoch:2d}/{EPOCHS}  batch {n_batches}  "
                      f"loss={total_loss / n_batches:.4f}")
            # periodically release MPS cached memory
            if DEVICE.type == "mps" and n_batches % 50 == 0:
                torch.mps.empty_cache()

        if tqdm is not None:
            loader_iter.close()

        avg_loss = total_loss / n_batches
        elapsed = time.time() - t0

        # release MPS cached memory before validation
        if DEVICE.type == "mps":
            torch.mps.empty_cache()

        # validation (token match with TF=0)
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for src, tgt, vs, ms, ts, ps in val_loader:
                src = src.to(DEVICE)
                tgt = tgt.to(DEVICE)
                vs = vs.to(DEVICE)
                ms = ms.to(DEVICE)
                ts = ts.to(DEVICE)
                ps = ps.to(DEVICE)
                out = model(src, tgt, vs, ms, ts, ps, 0.0)
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
        scheduler.step(val_acc)
        lr_now = optimizer.param_groups[0]["lr"]

        sc_ok, sc_n, sc_details = _spot_check(model, vocab)

        print(f"   Epoch {epoch:2d}/{EPOCHS}  loss={avg_loss:.4f}  "
              f"val_acc={val_acc:.2f}%  spot={sc_ok}/{sc_n}  "
              f"tf={tf_ratio:.2f}  lr={lr_now:.1e}  time={elapsed:.0f}s")
        for d in sc_details:
            print(f"     {d}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= EARLY_STOP_PATIENCE:
                print(f"\n   Early stopping after {epoch} epochs "
                      f"(no improvement for {EARLY_STOP_PATIENCE}).")
                break

        # release MPS cache before checkpoint save
        if DEVICE.type == "mps":
            torch.mps.empty_cache()

        # save resume checkpoint after each epoch
        _save_resume_checkpoint(
            epoch, model, optimizer, scheduler,
            best_val_acc, best_state, epochs_no_improve,
        )
        print(f"   (resume checkpoint saved)")

    # 6. Restore best
    if best_state is None:
        print("   WARNING: no best state saved -- using final weights.")
    else:
        model.load_state_dict(best_state)
        model.to(DEVICE)

    # 7. Final greedy evaluation on validation set
    print(f"\n5. Final evaluation (greedy on {len(val_ex):,} val examples) ...")
    correct, total, errors = evaluate_greedy(model, val_ex, vocab)
    final_acc = correct / total * 100
    print(f"   Greedy accuracy: {final_acc:.2f}% ({correct:,}/{total:,})")
    if errors:
        print("   Sample errors:")
        for inf, slot, exp, pred in errors[:20]:
            print(f"     {inf} [{slot}]: '{exp}' -> got '{pred}'")

    # 8. Save (move model to CPU first for portable checkpoint)
    model.cpu()
    print(f"\n6. Saving to {MODEL_PATH} ...")
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "vocab": vocab,
        "hyperparams": {
            "emb_dim": EMB_DIM,
            "hidden_dim": HIDDEN_DIM,
            "cond_dim": COND_DIM,
            "dropout": DROPOUT,
        },
        "accuracy": final_acc,
        # metadata
        "known_verbs": metadata["known_verbs"],
        "h_aspire": metadata["h_aspire"],
        "reform_1990_verbs": metadata["reform_1990_verbs"],
        "reform_variantes": metadata["reform_variantes"],
    }

    # Compress verb_structure: deduplicate identical templates
    _vs = metadata["verb_structure"]
    _tmpl_map = {}  # canonical JSON -> template_id
    _verb_tid = {}  # verb -> template_id
    for _verb, _struct in _vs.items():
        _key = json.dumps(_struct, sort_keys=True)
        if _key not in _tmpl_map:
            _tmpl_map[_key] = len(_tmpl_map)
        _verb_tid[_verb] = _tmpl_map[_key]
    _templates = [None] * len(_tmpl_map)
    for _key, _tid in _tmpl_map.items():
        _templates[_tid] = json.loads(_key)
    checkpoint["verb_structure_templates"] = _templates
    checkpoint["verb_structure_ids"] = _verb_tid
    print(f"   verb_structure: {len(_vs)} verbs -> {len(_templates)} unique templates")

    torch.save(checkpoint, MODEL_PATH)
    size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)
    print(f"   Model size : {size_mb:.1f} MB")
    print(f"   Best val accuracy  : {best_val_acc:.2f}%")
    print(f"   Greedy accuracy    : {final_acc:.2f}%")

    # clean up resume checkpoint -- training completed
    if os.path.isfile(RESUME_PATH):
        os.remove(RESUME_PATH)
        print("   Removed resume checkpoint (training complete).")

    print(f"\nDone. Log saved to {LOG_PATH}")


if __name__ == "__main__":
    train()
