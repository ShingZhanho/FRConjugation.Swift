#!/usr/bin/env python3
"""
full_test_model.py -- Test the conjugation model against the entire verbs.db.

Tests every conjugation and participle row (with merged person keys
expanded). Outputs all errors to stdout and full_test_errors.json.

Uses multiprocessing to parallelise inference across CPU cores.

Usage:
    python3 full_test_model.py [model_path]

Default model: conjugation_model.pt in the same directory.
"""

import json
import multiprocessing as mp
import os
import sqlite3
import sys
import time

DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(DIR, "verbs.db")
ERRORS_PATH = os.path.join(DIR, "full_test_errors.json")

model_path = sys.argv[1] if len(sys.argv) > 1 else None


def _expand_person_key(merged_key):
    return merged_key.split(";")


def load_ground_truth():
    """Load every conjugation + participle from verbs.db, expanding merged
    person keys."""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # conjugations
    cur.execute("""
        SELECT v.infinitif, c.voix, c.mode, c.temps, c.personne, c.conjugaison
        FROM conjugaisons c
        JOIN verbes v ON c.verbe_id = v.id
    """)
    examples = []
    seen = set()
    for inf, voice, mode, tense, person_merged, form in cur.fetchall():
        form = form.strip()
        for person in _expand_person_key(person_merged):
            key = (inf, voice, mode, tense, person)
            if key in seen:
                continue
            seen.add(key)
            examples.append((inf, voice, mode, tense, person, form))

    n_conj = len(examples)

    # participles
    cur.execute("""
        SELECT v.infinitif, p.voix, p.forme, p.participe
        FROM participes p
        JOIN verbes v ON p.verbe_id = v.id
    """)
    pseen = set()
    for inf, voice, forme, participe in cur.fetchall():
        participe = participe.strip()
        key = (inf, voice, forme)
        if key in pseen:
            continue
        pseen.add(key)
        examples.append((inf, voice, "participe", forme, "-", participe))

    conn.close()
    print(f"   Conjugations: {n_conj:,}")
    print(f"   Participles:  {len(examples) - n_conj:,}")
    return examples


def _worker_init(mp_path):
    """Initialise a per-worker model (called once per process)."""
    global _worker_model
    import torch
    torch.set_num_threads(1)  # avoid over-subscription
    from french_conjugation_model import ConjugationModel
    _worker_model = ConjugationModel(mp_path)


def _worker_test_chunk(chunk):
    """Test a chunk of (inf, voice, mode, tense, person, expected) tuples.
    Returns (correct, errors_list)."""
    correct = 0
    errors = []
    for inf, voice, mode, tense, person, expected in chunk:
        predicted = _worker_model.conjugate(
            inf, voice=voice, mode=mode, tense=tense, person=person,
        )
        if predicted == expected:
            correct += 1
        else:
            errors.append({
                "infinitive": inf,
                "voice": voice,
                "mode": mode,
                "tense": tense,
                "person": person,
                "expected": expected,
                "predicted": predicted,
            })
    return correct, errors


def main():
    print("=" * 60)
    print("  Full Model Test -- vs complete verbs.db")
    print("=" * 60)

    # load ground truth
    print("\nLoading ground truth from verbs.db ...")
    examples = load_ground_truth()
    total = len(examples)
    print(f"   Total forms: {total:,}")

    # determine worker count
    n_workers = min(mp.cpu_count(), 8)
    print(f"\nTesting with {n_workers} worker processes ...")

    # split into many small sub-chunks for progress reporting
    sub_chunk_size = 5000
    chunks = [examples[i:i + sub_chunk_size]
              for i in range(0, total, sub_chunk_size)]
    n_chunks = len(chunks)

    t0 = time.time()
    correct = 0
    errors = []
    done = 0

    with mp.Pool(processes=n_workers,
                 initializer=_worker_init,
                 initargs=(model_path,)) as pool:
        for c_correct, c_errors in pool.imap_unordered(
                _worker_test_chunk, chunks):
            correct += c_correct
            errors.extend(c_errors)
            done += 1
            pct = done / n_chunks * 100
            elapsed_so_far = time.time() - t0
            rate = (correct + len(errors)) / elapsed_so_far if elapsed_so_far > 0 else 0
            print(f"\r   Progress: {pct:5.1f}%  "
                  f"({correct + len(errors):,}/{total:,})  "
                  f"{rate:.0f} forms/s",
                  end="", flush=True)

    print()  # newline after progress
    elapsed = time.time() - t0

    acc = correct / total * 100 if total else 0
    n_err = len(errors)

    print(f"\n{'=' * 60}")
    print(f"  Results: {correct:,}/{total:,} correct  ({acc:.4f}%)")
    print(f"  Errors : {n_err:,}")
    print(f"  Time   : {elapsed:.1f}s  ({total / elapsed:.0f} forms/s)")
    print(f"{'=' * 60}")

    if errors:
        by_verb = {}
        for e in errors:
            by_verb.setdefault(e["infinitive"], []).append(e)

        print(f"\n  Errors by verb ({len(by_verb)} verbs):")
        for verb in sorted(by_verb)[:50]:  # show first 50 verbs
            verb_errors = by_verb[verb]
            print(f"\n    {verb} ({len(verb_errors)} errors):")
            for e in sorted(verb_errors,
                            key=lambda x: (x["voice"], x["mode"],
                                           x["tense"], x["person"]))[:10]:
                print(f"      {e['voice']}.{e['mode']}.{e['tense']}"
                      f".{e['person']}: "
                      f"'{e['expected']}' -> got '{e['predicted']}'")

    # save errors
    output = {
        "total": total,
        "correct": correct,
        "accuracy": round(acc, 4),
        "n_errors": n_err,
        "errors": errors,
    }
    with open(ERRORS_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n  Errors saved to {ERRORS_PATH}")


if __name__ == "__main__":
    main()
