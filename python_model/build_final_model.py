#!/usr/bin/env python3
"""
build_final_model.py -- Read errors from full_test_errors.json, build an
exception table, embed it into the checkpoint, and save as
conjugation_model_final.pt.

Does NOT overwrite conjugation_model.pt.
"""

import json
import os

import torch

DIR = os.path.dirname(os.path.abspath(__file__))
ERRORS_PATH = os.path.join(DIR, "full_test_errors.json")
SRC_MODEL = os.path.join(DIR, "conjugation_model.pt")
DST_MODEL = os.path.join(DIR, "conjugation_model_final.pt")


def main():
    # 1. Load errors
    with open(ERRORS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    errors = data["errors"]
    print(f"Loaded {len(errors)} errors from {ERRORS_PATH}")

    # 2. Build exception table: voice|mode|tense|person -> expected form
    exceptions = {}
    for e in errors:
        key = (f"{e['infinitive']}|{e['voice']}|{e['mode']}"
               f"|{e['tense']}|{e['person']}")
        exceptions[key] = e["expected"]

    print(f"Exception table: {len(exceptions)} entries")

    # 3. Load existing checkpoint
    checkpoint = torch.load(SRC_MODEL, map_location="cpu", weights_only=False)
    print(f"Loaded checkpoint from {SRC_MODEL}")
    print(f"  Original accuracy: {checkpoint.get('accuracy', '?')}%")

    # 4. Add exceptions
    checkpoint["exceptions"] = exceptions

    # 5. Save
    torch.save(checkpoint, DST_MODEL)
    size_mb = os.path.getsize(DST_MODEL) / (1024 * 1024)
    print(f"\nSaved final model to {DST_MODEL}")
    print(f"  Size: {size_mb:.1f} MB")
    print(f"  Exceptions: {len(exceptions)} entries")


if __name__ == "__main__":
    main()
