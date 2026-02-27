#!/usr/bin/env python3
"""
french_conjugation_model.py — Reusable French verb conjugation module (ML-based).

This module provides:
  • Seq2SeqModel  — Character-level encoder-decoder with attention
  • ConjugationModel — High-level API for conjugating French verbs

Quick start:
    from french_conjugation_model import ConjugationModel
    model = ConjugationModel()
    model.conjugate("parler", mode="indicatif", tense="present", person="1s")
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn

# ─── Special token indices ────────────────────────────────────────────────────

PAD_IDX = 0
SOS_IDX = 1
EOS_IDX = 2

# ─── Neural-network architecture ─────────────────────────────────────────────


class Encoder(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD_IDX)
        self.dropout = nn.Dropout(dropout)
        self.rnn = nn.GRU(emb_dim, hidden_dim, batch_first=True, bidirectional=True)

    def forward(self, src):
        embedded = self.dropout(self.embedding(src))
        outputs, hidden = self.rnn(embedded)
        # Concatenate forward/backward final states → (batch, hidden*2)
        hidden = torch.cat([hidden[0], hidden[1]], dim=-1)
        return outputs, hidden


class Attention(nn.Module):
    def __init__(self, enc_dim: int, dec_dim: int):
        super().__init__()
        self.attn = nn.Linear(enc_dim + dec_dim, dec_dim, bias=False)
        self.v = nn.Linear(dec_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs, mask=None):
        src_len = encoder_outputs.size(1)
        h = hidden.unsqueeze(1).expand(-1, src_len, -1)
        energy = torch.tanh(self.attn(torch.cat([h, encoder_outputs], dim=-1)))
        scores = self.v(energy).squeeze(-1)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        weights = torch.softmax(scores, dim=-1)
        context = torch.bmm(weights.unsqueeze(1), encoder_outputs).squeeze(1)
        return context, weights


class Decoder(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int, enc_dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD_IDX)
        self.dropout = nn.Dropout(dropout)
        self.rnn = nn.GRU(emb_dim + enc_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim + enc_dim + emb_dim, vocab_size)

    def forward(self, token, hidden, context):
        emb = self.dropout(self.embedding(token))
        rnn_input = torch.cat([emb, context], dim=-1).unsqueeze(1)
        output, hidden = self.rnn(rnn_input, hidden)
        output = output.squeeze(1)
        prediction = self.fc(torch.cat([output, context, emb], dim=-1))
        return prediction, hidden


class Seq2SeqModel(nn.Module):
    """Character-level encoder-decoder with Bahdanau attention."""

    def __init__(
        self,
        vocab_size: int,
        emb_dim: int,
        hidden_dim: int,
        cond_dim: int,
        n_modes: int,
        n_tenses: int,
        n_persons: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.encoder = Encoder(vocab_size, emb_dim, hidden_dim, dropout)
        self.attention = Attention(hidden_dim * 2, hidden_dim)
        self.decoder = Decoder(vocab_size, emb_dim, hidden_dim * 2, hidden_dim, dropout)

        # Conditioning embeddings for mode / tense / person
        self.mode_emb = nn.Embedding(n_modes, cond_dim, padding_idx=0)
        self.tense_emb = nn.Embedding(n_tenses, cond_dim, padding_idx=0)
        self.person_emb = nn.Embedding(n_persons, cond_dim, padding_idx=0)

        # Bridge: encoder hidden + conditioning → decoder initial hidden
        self.bridge = nn.Linear(hidden_dim * 2 + cond_dim * 3, hidden_dim)
        self.bridge_dropout = nn.Dropout(dropout)

    def _init_decoder(self, enc_hidden, mode_idx, tense_idx, person_idx):
        cond = torch.cat(
            [
                enc_hidden,
                self.mode_emb(mode_idx),
                self.tense_emb(tense_idx),
                self.person_emb(person_idx),
            ],
            dim=-1,
        )
        return torch.tanh(self.bridge_dropout(self.bridge(cond))).unsqueeze(0)

    def forward(self, src, tgt, mode_idx, tense_idx, person_idx, tf_ratio=0.5):
        batch_size, tgt_len = tgt.size()
        vocab_size = self.decoder.fc.out_features
        src_mask = (src != PAD_IDX).float()

        enc_outputs, enc_hidden = self.encoder(src)
        dec_hidden = self._init_decoder(enc_hidden, mode_idx, tense_idx, person_idx)

        outputs = torch.zeros(batch_size, tgt_len - 1, vocab_size, device=src.device)
        dec_input = tgt[:, 0]  # SOS

        for t in range(1, tgt_len):
            context, _ = self.attention(dec_hidden.squeeze(0), enc_outputs, src_mask)
            pred, dec_hidden = self.decoder(dec_input, dec_hidden, context)
            outputs[:, t - 1] = pred
            use_tf = torch.rand(1).item() < tf_ratio if self.training else False
            dec_input = tgt[:, t] if use_tf else pred.argmax(-1)

        return outputs

    @torch.no_grad()
    def predict(self, src, mode_idx, tense_idx, person_idx, max_len=50):
        """Greedy decode a single example."""
        self.eval()
        src_mask = (src != PAD_IDX).float()
        enc_outputs, enc_hidden = self.encoder(src)
        dec_hidden = self._init_decoder(enc_hidden, mode_idx, tense_idx, person_idx)

        dec_input = torch.tensor([SOS_IDX], device=src.device)
        result: list[int] = []
        for _ in range(max_len):
            context, _ = self.attention(dec_hidden.squeeze(0), enc_outputs, src_mask)
            pred, dec_hidden = self.decoder(dec_input, dec_hidden, context)
            token_id = pred.argmax(-1).item()
            if token_id == EOS_IDX:
                break
            result.append(token_id)
            dec_input = torch.tensor([token_id], device=src.device)
        return result


# ─── Alias maps ──────────────────────────────────────────────────────────────

_MODE_ALIASES: Dict[str, str] = {
    "indicatif": "indicatif", "ind": "indicatif",
    "subjonctif": "subjonctif", "sub": "subjonctif",
    "conditionnel": "conditionnel", "cond": "conditionnel",
    "imperatif": "imperatif", "impératif": "imperatif", "imp": "imperatif",
    "participe": "participe", "part": "participe",
}

_TENSE_ALIASES: Dict[str, str] = {
    "present": "present", "présent": "present",
    "imparfait": "imparfait",
    "passe_simple": "passe_simple", "passé_simple": "passe_simple",
    "futur_simple": "futur_simple", "futur": "futur_simple",
    "passe_compose": "passe_compose", "passé_composé": "passe_compose",
    "plus_que_parfait": "plus_que_parfait",
    "passe_anterieur": "passe_anterieur", "passé_antérieur": "passe_anterieur",
    "futur_anterieur": "futur_anterieur", "futur_antérieur": "futur_anterieur",
    "passe": "passe", "passé": "passe",
    # participle forme aliases
    "passe_sm": "passe_sm", "passe_sf": "passe_sf",
    "passe_pm": "passe_pm", "passe_pf": "passe_pf",
}

_PERSON_ALIASES: Dict[str, str] = {
    "1s": "1s", "2s": "2s", "3s": "3sm", "3sm": "3sm", "3sf": "3sf",
    "1p": "1p", "2p": "2p", "3p": "3pm", "3pm": "3pm", "3pf": "3pf",
    "je": "1s", "tu": "2s", "il": "3sm", "elle": "3sf", "on": "3sm",
    "nous": "1p", "vous": "2p", "ils": "3pm", "elles": "3pf",
}

# ─── Linguistic constants ────────────────────────────────────────────────────

_SIMPLE_TENSES = {
    ("indicatif", "present"), ("indicatif", "imparfait"),
    ("indicatif", "passe_simple"), ("indicatif", "futur_simple"),
    ("conditionnel", "present"),
    ("subjonctif", "present"), ("subjonctif", "imparfait"),
    ("imperatif", "present"),
}

_COMPOUND_TENSE_MAP: Dict[tuple, tuple] = {
    ("indicatif", "passe_compose"): ("indicatif", "present"),
    ("indicatif", "plus_que_parfait"): ("indicatif", "imparfait"),
    ("indicatif", "passe_anterieur"): ("indicatif", "passe_simple"),
    ("indicatif", "futur_anterieur"): ("indicatif", "futur_simple"),
    ("conditionnel", "passe"): ("conditionnel", "present"),
    ("subjonctif", "passe"): ("subjonctif", "present"),
    ("subjonctif", "plus_que_parfait"): ("subjonctif", "imparfait"),
    ("imperatif", "passe"): ("imperatif", "present"),
}

_IMPERATIF_PERSONS = ["2s", "1p", "2p"]
_ALL_PERSONS = ["1s", "2s", "3sm", "3sf", "1p", "2p", "3pm", "3pf"]

# Person → participle forme mapping for être-auxiliary agreement
_PP_FORME_ETRE: Dict[str, str] = {
    "1s": "passe_sm", "2s": "passe_sm",
    "3sm": "passe_sm", "3sf": "passe_sf",
    "1p": "passe_pm", "2p": "passe_pm",
    "3pm": "passe_pm", "3pf": "passe_pf",
}


# ─── High-level API ──────────────────────────────────────────────────────────

class ConjugationModel:
    """Load a trained seq2seq model and conjugate French verbs."""

    def __init__(self, model_path: Optional[str] = None) -> None:
        if model_path is None:
            model_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "conjugation_model_final.pt"
            )

        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)

        self._vocab: dict = checkpoint["vocab"]
        self._etre_verbs: set = set(checkpoint["etre_verbs"])
        self._prono_verbs: set = set(checkpoint.get("prono_verbs", []))
        self._h_aspire: set = set(checkpoint.get("h_aspire", []))
        self._known_verbs: set = set(checkpoint.get("known_verbs", []))
        self._invariable_pp: set = set(checkpoint.get("invariable_pp_verbs", []))
        self._exceptions: dict = checkpoint.get("exceptions", {})

        hp = checkpoint["hyperparams"]
        self._model = Seq2SeqModel(
            vocab_size=self._vocab["vocab_size"],
            emb_dim=hp["emb_dim"],
            hidden_dim=hp["hidden_dim"],
            cond_dim=hp["cond_dim"],
            n_modes=self._vocab["n_modes"],
            n_tenses=self._vocab["n_tenses"],
            n_persons=self._vocab["n_persons"],
            dropout=hp.get("dropout", 0.0),
        )
        self._model.load_state_dict(checkpoint["model_state_dict"])
        self._model.eval()

        self._char_to_idx: dict = self._vocab["char_to_idx"]
        self._idx_to_char: dict = {int(k): v for k, v in self._vocab["idx_to_char"].items()}
        self._mode_to_idx: dict = self._vocab["mode_to_idx"]
        self._tense_to_idx: dict = self._vocab["tense_to_idx"]
        self._person_to_idx: dict = self._vocab["person_to_idx"]

    # ── properties ────────────────────────────────────────────

    @property
    def verb_count(self) -> int:
        return len(self._known_verbs)

    @property
    def verbs(self) -> List[str]:
        return sorted(self._known_verbs)

    def has_verb(self, infinitive: str) -> bool:
        return infinitive in self._known_verbs

    def auxiliary(self, infinitive: str) -> List[str]:
        aux: list[str] = []
        if infinitive not in self._etre_verbs:
            aux.append("avoir")
        if infinitive in self._etre_verbs:
            aux.append("être")
        if infinitive in self._prono_verbs:
            aux.append("pronominal")
        return aux

    def is_h_aspire(self, infinitive: str) -> bool:
        return infinitive in self._h_aspire

    # ── core ML prediction ────────────────────────────────────

    def _predict(self, infinitive: str, mode: str, tense: str, person: str) -> Optional[str]:
        """Run the neural model to produce a single conjugated form.

        Checks the exception table first; falls back to the neural model.
        """
        # Exception table override
        exc_key = f"{infinitive}|{mode}|{tense}|{person}"
        if exc_key in self._exceptions:
            return self._exceptions[exc_key]

        char_ids: list[int] = []
        for c in infinitive:
            idx = self._char_to_idx.get(c)
            if idx is None:
                return None
            char_ids.append(idx)
        char_ids.append(EOS_IDX)

        src = torch.tensor([char_ids], dtype=torch.long)
        m = torch.tensor([self._mode_to_idx.get(mode, 0)], dtype=torch.long)
        t = torch.tensor([self._tense_to_idx.get(tense, 0)], dtype=torch.long)
        p = torch.tensor([self._person_to_idx.get(person, 0)], dtype=torch.long)

        out_ids = self._model.predict(src, m, t, p)
        return "".join(self._idx_to_char.get(i, "") for i in out_ids)

    # ── compound tense composition ────────────────────────────

    def _conjugate_compound(
        self, infinitive: str, mode: str, tense: str, person: str
    ) -> Optional[str]:
        aux_info = _COMPOUND_TENSE_MAP.get((mode, tense))
        if aux_info is None:
            return None
        aux_mode, aux_tense = aux_info
        aux_verb = "être" if infinitive in self._etre_verbs else "avoir"
        aux_form = self._predict(aux_verb, aux_mode, aux_tense, person)
        if not aux_form:
            return None
        pp_forme = (
            _PP_FORME_ETRE.get(person, "passe_sm") if aux_verb == "être" else "passe_sm"
        )
        pp = self._predict(infinitive, "participe", pp_forme, "-")
        if not pp:
            return None
        return f"{aux_form} {pp}"

    # ── main API ──────────────────────────────────────────────

    def conjugate(
        self,
        infinitive: str,
        *,
        mode: Optional[str] = None,
        tense: Optional[str] = None,
        person: Optional[str] = None,
    ) -> Union[str, Dict[str, Any], None]:
        """
        Conjugate a French verb.

        Parameters
        ----------
        infinitive : str
            Verb infinitive (e.g. "parler", "aller", "être").
        mode : str, optional
            indicatif | subjonctif | conditionnel | imperatif | participe
        tense : str, optional
            present | imparfait | passe_simple | futur_simple | passe_compose | …
        person : str, optional
            1s | 2s | 3sm | 3sf | 1p | 2p | 3pm | 3pf  (or je/tu/il/…)

        Returns
        -------
        str   — when mode + tense + person are all given (single form).
        dict  — when any axis is omitted (nested mode → tense → person → form).
        None  — when the combination is not available.
        """
        r_mode = _MODE_ALIASES.get(mode, mode) if mode else None
        r_tense = _TENSE_ALIASES.get(tense, tense) if tense else None
        r_person = _PERSON_ALIASES.get(person, person) if person else None

        # All specified → single form
        if r_mode and r_tense and r_person:
            return self._single_form(infinitive, r_mode, r_tense, r_person)

        # Build a filtered dict
        result: Dict[str, Any] = {}

        target_modes = (
            [r_mode] if r_mode else ["indicatif", "subjonctif", "conditionnel", "imperatif"]
        )
        for m in target_modes:
            mode_dict: Dict[str, Any] = {}
            for t in self._tenses_for_mode(m):
                if r_tense and t != r_tense:
                    continue
                persons = _IMPERATIF_PERSONS if m == "imperatif" else _ALL_PERSONS
                tense_dict: Dict[str, str] = {}
                for p in persons:
                    if r_person and p != r_person:
                        continue
                    form = self._single_form(infinitive, m, t, p)
                    if form:
                        tense_dict[p] = form
                if tense_dict:
                    mode_dict[t] = tense_dict
            if mode_dict:
                result[m] = mode_dict

        # Participles
        if not r_mode or r_mode == "participe":
            part: Dict[str, str] = {}
            for forme in ["present", "passe_sm", "passe_sf", "passe_pm", "passe_pf"]:
                if r_tense and forme != r_tense:
                    continue
                pp = self.get_participle(infinitive, forme)
                if pp:
                    part[forme] = pp
            if part:
                result["participe"] = part

        return result if result else None

    def _single_form(self, inf: str, mode: str, tense: str, person: str) -> Optional[str]:
        if mode == "participe":
            return self.get_participle(inf, tense)  # handles invariable PP redirection
        if (mode, tense) in _SIMPLE_TENSES:
            return self._predict(inf, mode, tense, person)
        if (mode, tense) in _COMPOUND_TENSE_MAP:
            return self._conjugate_compound(inf, mode, tense, person)
        return None

    @staticmethod
    def _tenses_for_mode(mode: str) -> List[str]:
        simple = [t for m, t in _SIMPLE_TENSES if m == mode]
        compound = [t for (m, t) in _COMPOUND_TENSE_MAP if m == mode]
        return simple + compound

    def get_participle(self, infinitive: str, forme: str = "passe_sm") -> Optional[str]:
        """Return a participle form (present, passe_sm, passe_sf, passe_pm, passe_pf).

        For invariable-PP verbs (intransitive avoir verbs), agreement
        forms (sf/pm/pf) are redirected to passe_sm since they are identical.
        """
        if forme in ("passe_sf", "passe_pm", "passe_pf") and infinitive in self._invariable_pp:
            forme = "passe_sm"
        return self._predict(infinitive, "participe", forme, "-")

    def __repr__(self) -> str:
        return f"<ConjugationModel verbs={self.verb_count}>"


# ─── Module-level singleton ──────────────────────────────────────────────────

_singleton: Optional[ConjugationModel] = None


def get_model(model_path: Optional[str] = None) -> ConjugationModel:
    """Return a lazily-loaded singleton model instance."""
    global _singleton
    if _singleton is None:
        _singleton = ConjugationModel(model_path)
    return _singleton


# ─── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    m = ConjugationModel()
    print(m)

    verb = sys.argv[1] if len(sys.argv) > 1 else "parler"
    mode_arg = sys.argv[2] if len(sys.argv) > 2 else None
    tense_arg = sys.argv[3] if len(sys.argv) > 3 else None
    person_arg = sys.argv[4] if len(sys.argv) > 4 else None

    result = m.conjugate(verb, mode=mode_arg, tense=tense_arg, person=person_arg)

    if isinstance(result, str):
        print(f"\n{verb} → {result}")
    elif isinstance(result, dict):
        print(f"\nConjugation of: {verb}")
        print(f"Auxiliary: {m.auxiliary(verb)}")
        for mk in sorted(result):
            print(f"\n  {mk}:")
            val = result[mk]
            if isinstance(val, dict):
                for tk in sorted(val):
                    tv = val[tk]
                    if isinstance(tv, dict):
                        print(f"    {tk}:")
                        for pk in ["1s", "2s", "3sm", "3sf", "1p", "2p", "3pm", "3pf"]:
                            if pk in tv:
                                print(f"      {pk:4s}  {tv[pk]}")
                    else:
                        print(f"    {tk}: {tv}")
            else:
                print(f"    {val}")
    else:
        print(f"Verb '{verb}' not found or could not be conjugated.")
