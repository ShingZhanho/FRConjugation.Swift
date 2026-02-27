#!/usr/bin/env python3
"""
full_test_model.py — Test the conjugation model against the COMPLETE verbs.db dataset.

Tests every single-word conjugation and simple participle.
Outputs all errors to both stdout and full_test_errors.json.

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

# ─── Allow specifying a model path as CLI arg ────────────────────────────────

model_path = sys.argv[1] if len(sys.argv) > 1 else None


def load_ground_truth():
    """Load every single-word conjugation + simple participle from verbs.db."""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # ── Single-word conjugations (simple tenses) ──
    cur.execute(
        """
        SELECT v.infinitif, c.mode, c.temps, c.personne, c.conjugaison
        FROM conjugaisons c
        JOIN verbes v ON c.verbe_id = v.id
        WHERE c.voix IN ('voix_active_avoir', 'voix_active_etre')
          AND c.conjugaison NOT LIKE '% %'
        """
    )
    examples: list[tuple[str, str, str, str, str]] = []
    seen: set[tuple[str, str, str, str]] = set()
    for inf, mode, tense, person, form in cur.fetchall():
        form = form.split(";")[0]
        key = (inf, mode, tense, person)
        if key not in seen:
            seen.add(key)
            examples.append((inf, mode, tense, person, form))

    # ── Simple participles ──
    cur.execute(
        """
        SELECT v.infinitif, p.forme, p.participe
        FROM participes p
        JOIN verbes v ON p.verbe_id = v.id
        WHERE p.voix IN ('voix_active_avoir', 'voix_active_etre')
          AND p.participe NOT LIKE '% %'
        """
    )
    pseen: set[tuple[str, str]] = set()
    for inf, forme, participe in cur.fetchall():
        participe = participe.split(";")[0]
        key2 = (inf, forme)
        if key2 not in pseen:
            pseen.add(key2)
            examples.append((inf, "participe", forme, "-", participe))

    conn.close()
    return examples


def main():
    from french_conjugation_model import ConjugationModel

    print("=" * 60)
    print("  Full Model Test — vs complete verbs.db")
    print("=" * 60)

    # Load model
    print(f"\nLoading model … ", end="", flush=True)
    t0 = time.time()
    model = ConjugationModel(model_path)
    print(f"done ({time.time() - t0:.1f}s)  {model}")

    # Load ground truth
    print("Loading ground truth from verbs.db … ", end="", flush=True)
    examples = load_ground_truth()
    print(f"{len(examples):,} forms")

    # Test every form
    print(f"\nTesting all forms …")
    correct = 0
    total = 0
    errors: list[dict] = []

    _ci = os.environ.get("GH_ACTIONS") == "1"
    try:
        if _ci:
            raise ImportError
        from tqdm import tqdm
        iterator = tqdm(examples, desc="   Testing", ncols=80)
    except ImportError:
        iterator = examples

    for inf, mode, tense, person, expected in iterator:
        total += 1
        if mode == "participe":
            predicted = model.get_participle(inf, tense)
        else:
            predicted = model.conjugate(inf, mode=mode, tense=tense, person=person)

        if predicted == expected:
            correct += 1
        else:
            errors.append({
                "infinitive": inf,
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
        # Group by verb for readability
        by_verb: dict[str, list[dict]] = {}
        for e in errors:
            by_verb.setdefault(e["infinitive"], []).append(e)

        print(f"\n  Errors by verb ({len(by_verb)} verbs):")
        for verb in sorted(by_verb):
            verb_errors = by_verb[verb]
            print(f"\n    {verb} ({len(verb_errors)} errors):")
            for e in sorted(verb_errors, key=lambda x: (x["mode"], x["tense"], x["person"])):
                print(
                    f"      {e['mode']}.{e['tense']}.{e['person']}: "
                    f"expected '{e['expected']}', got '{e['predicted']}'"
                )

    # Save errors to JSON
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
