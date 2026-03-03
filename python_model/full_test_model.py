#!/usr/bin/env python3
"""
full_test_model.py -- Test the conjugation model against the entire verbs.db.

Tests every conjugation and participle row (with merged person keys
expanded). Outputs all errors to stdout and full_test_errors.json.

Usage:
    python3 full_test_model.py [model_path]

Default model: conjugation_model.pt in the same directory.
"""

import json
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
        form = form.split(";")[0].strip()
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
        participe = participe.split(";")[0].strip()
        key = (inf, voice, forme)
        if key in pseen:
            continue
        pseen.add(key)
        examples.append((inf, voice, "participe", forme, "-", participe))

    conn.close()
    print(f"   Conjugations: {n_conj:,}")
    print(f"   Participles:  {len(examples) - n_conj:,}")
    return examples


def main():
    from french_conjugation_model import ConjugationModel

    print("=" * 60)
    print("  Full Model Test -- vs complete verbs.db")
    print("=" * 60)

    # load model
    print(f"\nLoading model ... ", end="", flush=True)
    t0 = time.time()
    model = ConjugationModel(model_path)
    print(f"done ({time.time() - t0:.1f}s)  {model}")

    # load ground truth
    print("Loading ground truth from verbs.db ...")
    examples = load_ground_truth()
    print(f"   Total forms: {len(examples):,}")

    # test
    print(f"\nTesting all forms ...")
    correct = 0
    total = 0
    errors = []

    _ci = os.environ.get("GH_ACTIONS") == "1"
    try:
        if _ci:
            raise ImportError
        from tqdm import tqdm
        iterator = tqdm(examples, desc="   Testing", ncols=80)
    except ImportError:
        iterator = examples

    for inf, voice, mode, tense, person, expected in iterator:
        total += 1
        predicted = model.conjugate(
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

    acc = correct / total * 100 if total else 0
    n_err = len(errors)

    print(f"\n{'=' * 60}")
    print(f"  Results: {correct:,}/{total:,} correct  ({acc:.4f}%)")
    print(f"  Errors : {n_err:,}")
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
