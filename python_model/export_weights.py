#!/usr/bin/env python3
"""
export_weights.py -- Export trained model weights to a portable binary format.

Produces two files in the output directory:
  - model.json  -- metadata, vocabulary, exceptions, verb sets, weight manifest
  - weights.bin -- raw float32 weight data (little-endian)

These files are consumed by the pure-Swift inference engine in FRConjugation.
"""

from __future__ import annotations

import json
import os
import sys
from collections import OrderedDict

import numpy as np
import torch


def export(checkpoint_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    cp = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # -- weight manifest and binary blob --
    state_dict = cp["model_state_dict"]
    manifest = {}
    offset = 0
    bin_path = os.path.join(output_dir, "weights.bin")

    with open(bin_path, "wb") as f:
        for name, tensor in state_dict.items():
            arr = tensor.numpy().astype(np.float32).flatten()
            size = arr.nbytes
            manifest[name] = {
                "shape": list(tensor.shape),
                "offset": offset,
                "count": int(arr.size),
            }
            f.write(arr.tobytes())
            offset += size

    print(f"Wrote {offset:,} bytes to weights.bin ({len(manifest)} tensors)")

    # -- JSON metadata --
    vocab = cp["vocab"]
    hp = cp["hyperparams"]

    meta = {
        "format_version": 2,
        "hyperparams": {
            "vocab_size": vocab["vocab_size"],
            "emb_dim": hp["emb_dim"],
            "hidden_dim": hp["hidden_dim"],
            "cond_dim": hp["cond_dim"],
            "n_voices": vocab["n_voices"],
            "n_modes": vocab["n_modes"],
            "n_tenses": vocab["n_tenses"],
            "n_persons": vocab["n_persons"],
        },
        "vocab": {
            "char_to_idx": vocab["char_to_idx"],
            "idx_to_char": {str(k): v
                            for k, v in vocab["idx_to_char"].items()},
            "voice_to_idx": vocab["voice_to_idx"],
            "mode_to_idx": vocab["mode_to_idx"],
            "tense_to_idx": vocab["tense_to_idx"],
            "person_to_idx": vocab["person_to_idx"],
        },
        "exceptions": cp.get("exceptions", {}),
        "known_verbs": sorted(cp.get("known_verbs", [])),
        "h_aspire": sorted(cp.get("h_aspire", [])),
        "reform_1990_verbs": sorted(cp.get("reform_1990_verbs", [])),
        "reform_variantes": cp.get("reform_variantes", {}),
        "verb_structure_templates": cp.get("verb_structure_templates", []),
        "verb_structure_ids": cp.get("verb_structure_ids", {}),
        "weight_manifest": manifest,
    }

    json_path = os.path.join(output_dir, "model.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, separators=(",", ":"))

    json_size = os.path.getsize(json_path)
    print(f"Wrote {json_size:,} bytes to model.json")
    print(f"Total model size: {(offset + json_size) / 1024:.1f} KB")


if __name__ == "__main__":
    checkpoint = (sys.argv[1] if len(sys.argv) > 1
                  else "python_model/conjugation_model_final.pt")
    outdir = (sys.argv[2] if len(sys.argv) > 2
              else "swift_lib/Sources/FRConjugation/Resources")
    export(checkpoint, outdir)
