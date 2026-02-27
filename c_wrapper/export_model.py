#!/usr/bin/env python3
"""
export_model.py — Export the PyTorch conjugation model for use with LibTorch (C++).

Strategy: Trace each component separately (Encoder, Bridge, Attention, Decoder step)
since the full predict loop with variable-length output can't be cleanly TorchScript'd.
The greedy decode loop is reimplemented in C++.

Produces:
  1. conjugation_encoder.pt      — Traced encoder
  2. conjugation_bridge.pt       — Traced bridge (init decoder hidden)
  3. conjugation_attention.pt    — Traced attention step
  4. conjugation_decoder.pt      — Traced single decoder step
  5. conjugation_meta.json       — Vocabulary, exceptions, verb metadata

Usage:
    python3 export_model.py [path/to/conjugation_model_final.pt]
"""

import json
import os
import sys

import torch
import torch.nn as nn

DIR = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(DIR)
PYTHON_MODEL_DIR = os.path.join(PARENT, "python_model")
sys.path.insert(0, PYTHON_MODEL_DIR)

from french_conjugation_model import (
    Seq2SeqModel, PAD_IDX, SOS_IDX, EOS_IDX,
)

DEFAULT_MODEL = os.path.join(PYTHON_MODEL_DIR, "conjugation_model_final.pt")


# ── Traceable wrapper modules ────────────────────────────────────────

class TracedEncoder(nn.Module):
    """src (Long[1, seq_len]) → enc_outputs (Float[1, seq_len, H*2]), enc_hidden (Float[1, H*2])"""
    def __init__(self, encoder):
        super().__init__()
        self.embedding = encoder.embedding
        self.dropout = encoder.dropout
        self.rnn = encoder.rnn

    def forward(self, src: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embedded = self.dropout(self.embedding(src))
        outputs, hidden = self.rnn(embedded)
        enc_hidden = torch.cat([hidden[0], hidden[1]], dim=-1)
        return outputs, enc_hidden


class TracedBridge(nn.Module):
    """enc_hidden(Float[1, H*2]), mode/tense/person (Long[1]) → dec_hidden (Float[1, 1, H])"""
    def __init__(self, model):
        super().__init__()
        self.mode_emb = model.mode_emb
        self.tense_emb = model.tense_emb
        self.person_emb = model.person_emb
        self.bridge = model.bridge
        self.bridge_dropout = model.bridge_dropout

    def forward(
        self,
        enc_hidden: torch.Tensor,
        mode_idx: torch.Tensor,
        tense_idx: torch.Tensor,
        person_idx: torch.Tensor,
    ) -> torch.Tensor:
        cond = torch.cat([
            enc_hidden,
            self.mode_emb(mode_idx),
            self.tense_emb(tense_idx),
            self.person_emb(person_idx),
        ], dim=-1)
        return torch.tanh(self.bridge_dropout(self.bridge(cond))).unsqueeze(0)


class TracedAttention(nn.Module):
    """hidden(Float[1, H]), enc_outputs(Float[1, seq, H*2]), mask(Float[1, seq]) → context(Float[1, H*2])"""
    def __init__(self, attention):
        super().__init__()
        self.attn = attention.attn
        self.v = attention.v

    def forward(
        self,
        hidden: torch.Tensor,
        encoder_outputs: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        src_len = encoder_outputs.size(1)
        h = hidden.unsqueeze(1).expand(-1, src_len, -1)
        energy = torch.tanh(self.attn(torch.cat([h, encoder_outputs], dim=-1)))
        scores = self.v(energy).squeeze(-1)
        scores = scores.masked_fill(mask == 0, -1e9)
        weights = torch.softmax(scores, dim=-1)
        context = torch.bmm(weights.unsqueeze(1), encoder_outputs).squeeze(1)
        return context


class TracedDecoderStep(nn.Module):
    """token(Long[1]), hidden(Float[1,1,H]), context(Float[1,H*2]) → logits(Float[1,V]), new_hidden(Float[1,1,H])"""
    def __init__(self, decoder):
        super().__init__()
        self.embedding = decoder.embedding
        self.dropout = decoder.dropout
        self.rnn = decoder.rnn
        self.fc = decoder.fc

    def forward(
        self,
        token: torch.Tensor,
        hidden: torch.Tensor,
        context: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        emb = self.dropout(self.embedding(token))
        rnn_input = torch.cat([emb, context], dim=-1).unsqueeze(1)
        output, new_hidden = self.rnn(rnn_input, hidden)
        output = output.squeeze(1)
        prediction = self.fc(torch.cat([output, context, emb], dim=-1))
        return prediction, new_hidden


# ── Main ─────────────────────────────────────────────────────────────

def main():
    model_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL
    print(f"Loading checkpoint: {model_path}")
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)

    hp = checkpoint["hyperparams"]
    vocab = checkpoint["vocab"]
    HIDDEN = hp["hidden_dim"]
    EMB = hp["emb_dim"]
    COND = hp["cond_dim"]
    VOCAB_SIZE = vocab["vocab_size"]

    model = Seq2SeqModel(
        vocab_size=VOCAB_SIZE,
        emb_dim=EMB,
        hidden_dim=HIDDEN,
        cond_dim=COND,
        n_modes=vocab["n_modes"],
        n_tenses=vocab["n_tenses"],
        n_persons=vocab["n_persons"],
        dropout=hp.get("dropout", 0.0),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # ── Trace each component ──────────────────────────────────────

    # 1. Encoder
    enc_mod = TracedEncoder(model.encoder)
    enc_mod.eval()
    example_src = torch.tensor([[5, 6, 7, 8, 9, 10, EOS_IDX]], dtype=torch.long)
    traced_encoder = torch.jit.trace(enc_mod, (example_src,))
    enc_path = os.path.join(DIR, "conjugation_encoder.pt")
    traced_encoder.save(enc_path)
    print(f"  Saved: {enc_path} ({os.path.getsize(enc_path)/1024:.0f} KB)")

    # Get example outputs for downstream tracing
    with torch.no_grad():
        enc_outputs, enc_hidden = enc_mod(example_src)

    # 2. Bridge
    bridge_mod = TracedBridge(model)
    bridge_mod.eval()
    example_mode = torch.tensor([1], dtype=torch.long)
    example_tense = torch.tensor([1], dtype=torch.long)
    example_person = torch.tensor([1], dtype=torch.long)
    traced_bridge = torch.jit.trace(
        bridge_mod, (enc_hidden, example_mode, example_tense, example_person)
    )
    bridge_path = os.path.join(DIR, "conjugation_bridge.pt")
    traced_bridge.save(bridge_path)
    print(f"  Saved: {bridge_path} ({os.path.getsize(bridge_path)/1024:.0f} KB)")

    with torch.no_grad():
        dec_hidden = bridge_mod(enc_hidden, example_mode, example_tense, example_person)

    # 3. Attention
    attn_mod = TracedAttention(model.attention)
    attn_mod.eval()
    src_mask = (example_src != PAD_IDX).float()
    traced_attn = torch.jit.trace(
        attn_mod, (dec_hidden.squeeze(0), enc_outputs, src_mask)
    )
    attn_path = os.path.join(DIR, "conjugation_attention.pt")
    traced_attn.save(attn_path)
    print(f"  Saved: {attn_path} ({os.path.getsize(attn_path)/1024:.0f} KB)")

    with torch.no_grad():
        context = attn_mod(dec_hidden.squeeze(0), enc_outputs, src_mask)

    # 4. Decoder step
    dec_mod = TracedDecoderStep(model.decoder)
    dec_mod.eval()
    example_token = torch.tensor([SOS_IDX], dtype=torch.long)
    traced_dec = torch.jit.trace(dec_mod, (example_token, dec_hidden, context))
    dec_path = os.path.join(DIR, "conjugation_decoder.pt")
    traced_dec.save(dec_path)
    print(f"  Saved: {dec_path} ({os.path.getsize(dec_path)/1024:.0f} KB)")

    # ── Verify the traced pipeline matches the original ───────────

    print("\nVerifying traced pipeline vs original…")
    test_verbs = ["parler", "finir", "aller", "être", "avoir"]
    char_to_idx = vocab["char_to_idx"]
    idx_to_char = {int(k): v for k, v in vocab["idx_to_char"].items()}
    mode_to_idx = vocab["mode_to_idx"]
    tense_to_idx = vocab["tense_to_idx"]
    person_to_idx = vocab["person_to_idx"]

    mismatches = 0
    for verb in test_verbs:
        ids = [char_to_idx[c] for c in verb] + [EOS_IDX]
        src_t = torch.tensor([ids], dtype=torch.long)
        m_t = torch.tensor([mode_to_idx["indicatif"]], dtype=torch.long)
        t_t = torch.tensor([tense_to_idx["present"]], dtype=torch.long)
        p_t = torch.tensor([person_to_idx["1s"]], dtype=torch.long)

        # Original model
        orig_ids = model.predict(src_t, m_t, t_t, p_t)
        orig_str = "".join(idx_to_char.get(i, "") for i in orig_ids)

        # Traced pipeline
        with torch.no_grad():
            t_enc_out, t_enc_hid = traced_encoder(src_t)
            t_dec_hid = traced_bridge(t_enc_hid, m_t, t_t, p_t)
            t_mask = (src_t != PAD_IDX).float()
            dec_input = torch.tensor([SOS_IDX], dtype=torch.long)
            traced_ids = []
            for _ in range(50):
                t_ctx = traced_attn(t_dec_hid.squeeze(0), t_enc_out, t_mask)
                logits, t_dec_hid = traced_dec(dec_input, t_dec_hid, t_ctx)
                tok = logits.argmax(-1).item()
                if tok == EOS_IDX:
                    break
                traced_ids.append(tok)
                dec_input = torch.tensor([tok], dtype=torch.long)
        traced_str = "".join(idx_to_char.get(i, "") for i in traced_ids)

        match = "✓" if orig_str == traced_str else "✗ MISMATCH"
        if orig_str != traced_str:
            mismatches += 1
        print(f"  {verb}: original={orig_str}, traced={traced_str}  {match}")

    if mismatches:
        print(f"\n  WARNING: {mismatches} mismatches found!")
    else:
        print("\n  All verifications passed — traced output matches original.")

    # ── Export metadata JSON ──────────────────────────────────────

    meta = {
        "special_tokens": {"PAD": PAD_IDX, "SOS": SOS_IDX, "EOS": EOS_IDX},
        "hyperparams": {
            "hidden_dim": HIDDEN,
            "emb_dim": EMB,
            "cond_dim": COND,
            "vocab_size": VOCAB_SIZE,
        },
        "char_to_idx": vocab["char_to_idx"],
        "idx_to_char": {str(k): v for k, v in vocab["idx_to_char"].items()},
        "mode_to_idx": vocab["mode_to_idx"],
        "tense_to_idx": vocab["tense_to_idx"],
        "person_to_idx": vocab["person_to_idx"],
        "known_verbs": sorted(checkpoint.get("known_verbs", [])),
        "etre_verbs": sorted(checkpoint.get("etre_verbs", [])),
        "prono_verbs": sorted(checkpoint.get("prono_verbs", [])),
        "h_aspire": sorted(checkpoint.get("h_aspire", [])),
        "invariable_pp_verbs": sorted(checkpoint.get("invariable_pp_verbs", [])),
        "exceptions": checkpoint.get("exceptions", {}),
    }

    meta_path = os.path.join(DIR, "conjugation_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    size_kb = os.path.getsize(meta_path) / 1024
    print(f"\nSaved metadata: {meta_path} ({size_kb:.0f} KB)")

    total = sum(
        os.path.getsize(os.path.join(DIR, f))
        for f in ["conjugation_encoder.pt", "conjugation_bridge.pt",
                   "conjugation_attention.pt", "conjugation_decoder.pt",
                   "conjugation_meta.json"]
    )
    print(f"Total export size: {total/1024:.0f} KB ({total/(1024*1024):.1f} MB)")
    print("\nDone. Files ready for LibTorch C++ integration.")


if __name__ == "__main__":
    main()
