#!/usr/bin/env python3
"""
french_conjugation_model.py -- French verb conjugation module (ML-based).

Provides:
  - Seq2SeqModel     -- Character-level encoder-decoder with attention
  - ConjugationModel -- High-level API for conjugating French verbs

Quick start:
    from french_conjugation_model import ConjugationModel
    model = ConjugationModel()
    model.conjugate("parler", voice="voix_active_avoir",
                     mode="indicatif", tense="present", person="1sm")
"""

from __future__ import annotations

import bisect
import os
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn

# -- Special token indices -------------------------------------------------

PAD_IDX = 0
SOS_IDX = 1
EOS_IDX = 2

# -- Neural network architecture -------------------------------------------


class Encoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD_IDX)
        self.dropout = nn.Dropout(dropout)
        self.rnn = nn.GRU(emb_dim, hidden_dim, batch_first=True,
                          bidirectional=True)

    def forward(self, src):
        embedded = self.dropout(self.embedding(src))
        outputs, hidden = self.rnn(embedded)
        # concat forward/backward final states -> (batch, hidden*2)
        hidden = torch.cat([hidden[0], hidden[1]], dim=-1)
        return outputs, hidden


class Attention(nn.Module):
    def __init__(self, enc_dim, dec_dim):
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
    def __init__(self, vocab_size, emb_dim, enc_dim, hidden_dim, dropout=0.0):
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
    """Character-level encoder-decoder with Bahdanau attention.

    Conditioned on (voice, mode, tense, person) via learned embeddings
    that feed into the decoder's initial hidden state.
    """

    def __init__(
        self,
        vocab_size,
        emb_dim,
        hidden_dim,
        cond_dim,
        n_voices,
        n_modes,
        n_tenses,
        n_persons,
        dropout=0.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.encoder = Encoder(vocab_size, emb_dim, hidden_dim, dropout)
        self.attention = Attention(hidden_dim * 2, hidden_dim)
        self.decoder = Decoder(vocab_size, emb_dim, hidden_dim * 2,
                               hidden_dim, dropout)

        # conditioning embeddings
        self.voice_emb = nn.Embedding(n_voices, cond_dim, padding_idx=0)
        self.mode_emb = nn.Embedding(n_modes, cond_dim, padding_idx=0)
        self.tense_emb = nn.Embedding(n_tenses, cond_dim, padding_idx=0)
        self.person_emb = nn.Embedding(n_persons, cond_dim, padding_idx=0)

        # bridge: encoder hidden + 4 conditioning vectors -> decoder hidden
        self.bridge = nn.Linear(hidden_dim * 2 + cond_dim * 4, hidden_dim)
        self.bridge_dropout = nn.Dropout(dropout)

    def _init_decoder(self, enc_hidden, voice_idx, mode_idx, tense_idx,
                      person_idx):
        cond = torch.cat([
            enc_hidden,
            self.voice_emb(voice_idx),
            self.mode_emb(mode_idx),
            self.tense_emb(tense_idx),
            self.person_emb(person_idx),
        ], dim=-1)
        return torch.tanh(self.bridge_dropout(self.bridge(cond))).unsqueeze(0)

    def forward(self, src, tgt, voice_idx, mode_idx, tense_idx, person_idx,
                tf_ratio=0.5):
        batch_size, tgt_len = tgt.size()
        vocab_size = self.decoder.fc.out_features
        src_mask = (src != PAD_IDX).float()

        enc_outputs, enc_hidden = self.encoder(src)
        dec_hidden = self._init_decoder(enc_hidden, voice_idx, mode_idx,
                                        tense_idx, person_idx)

        outputs = torch.zeros(batch_size, tgt_len - 1, vocab_size,
                              device=src.device)
        dec_input = tgt[:, 0]  # SOS

        for t in range(1, tgt_len):
            context, _ = self.attention(dec_hidden.squeeze(0), enc_outputs,
                                        src_mask)
            pred, dec_hidden = self.decoder(dec_input, dec_hidden, context)
            outputs[:, t - 1] = pred
            use_tf = (torch.rand(1).item() < tf_ratio if self.training
                      else False)
            dec_input = tgt[:, t] if use_tf else pred.argmax(-1)

        return outputs

    @torch.no_grad()
    def predict(self, src, voice_idx, mode_idx, tense_idx, person_idx,
                max_len=80):
        """Greedy-decode a single example."""
        self.eval()
        src_mask = (src != PAD_IDX).float()
        enc_outputs, enc_hidden = self.encoder(src)
        dec_hidden = self._init_decoder(enc_hidden, voice_idx, mode_idx,
                                        tense_idx, person_idx)
        dec_input = torch.tensor([SOS_IDX], device=src.device)
        result = []
        for _ in range(max_len):
            context, _ = self.attention(dec_hidden.squeeze(0), enc_outputs,
                                        src_mask)
            pred, dec_hidden = self.decoder(dec_input, dec_hidden, context)
            token_id = pred.argmax(-1).item()
            if token_id == EOS_IDX:
                break
            result.append(token_id)
            dec_input = torch.tensor([token_id], device=src.device)
        return result


# -- Alias maps ------------------------------------------------------------

_VOICE_ALIASES = {
    "voix_active_avoir": "voix_active_avoir",
    "voix_active_etre": "voix_active_etre",
    "voix_active": "voix_active",
    "voix_passive": "voix_passive",
    "voix_prono": "voix_prono",
    "active_avoir": "voix_active_avoir",
    "active_etre": "voix_active_etre",
    "active": "voix_active",
    "passive": "voix_passive",
    "prono": "voix_prono",
    "pronominal": "voix_prono",
}

_MODE_ALIASES = {
    "indicatif": "indicatif", "ind": "indicatif",
    "subjonctif": "subjonctif", "sub": "subjonctif",
    "conditionnel": "conditionnel", "cond": "conditionnel",
    "imperatif": "imperatif", "imp": "imperatif",
    "participe": "participe", "part": "participe",
}

_TENSE_ALIASES = {
    "present": "present",
    "imparfait": "imparfait",
    "passe_simple": "passe_simple",
    "futur_simple": "futur_simple", "futur": "futur_simple",
    "passe_compose": "passe_compose",
    "plus_que_parfait": "plus_que_parfait",
    "passe_anterieur": "passe_anterieur",
    "futur_anterieur": "futur_anterieur",
    "passe": "passe",
    # participle formes
    "passe_sm": "passe_sm", "passe_sf": "passe_sf",
    "passe_pm": "passe_pm", "passe_pf": "passe_pf",
    "passe_compound_sm": "passe_compound_sm",
    "passe_compound_sf": "passe_compound_sf",
    "passe_compound_pm": "passe_compound_pm",
    "passe_compound_pf": "passe_compound_pf",
}

_PERSON_ALIASES = {
    "1sm": "1sm", "1sf": "1sf",
    "2sm": "2sm", "2sf": "2sf",
    "3sm": "3sm", "3sf": "3sf",
    "1pm": "1pm", "1pf": "1pf",
    "2pm": "2pm", "2pf": "2pf",
    "3pm": "3pm", "3pf": "3pf",
    "-": "-",
    # convenience aliases
    "je": "1sm", "tu": "2sm",
    "il": "3sm", "elle": "3sf", "on": "3sm",
    "nous": "1pm", "vous": "2pm",
    "ils": "3pm", "elles": "3pf",
}


# -- High-level API --------------------------------------------------------

class ConjugationModel:
    """Load a trained seq2seq model and conjugate French verbs.

    Parameters are layered: specifying a lower layer requires all upper
    layers to be present.

        conjugate(inf)                                                -> full dict
        conjugate(inf, voice=...)                                     -> mode dict
        conjugate(inf, voice=..., mode=...)                           -> tense dict
        conjugate(inf, voice=..., mode=..., tense=...)                -> person dict
        conjugate(inf, voice=..., mode=..., tense=..., person=...)    -> str
    """

    def __init__(self, model_path=None):
        if model_path is None:
            model_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "conjugation_model_final.pt",
            )

        checkpoint = torch.load(model_path, map_location="cpu",
                                weights_only=False)

        self._vocab = checkpoint["vocab"]
        self._exceptions = checkpoint.get("exceptions", {})

        # metadata
        self._known_verbs = sorted(checkpoint.get("known_verbs", []))
        self._h_aspire = set(checkpoint.get("h_aspire", []))
        self._reform_1990 = set(checkpoint.get("reform_1990_verbs", []))
        self._reform_variantes = checkpoint.get("reform_variantes", {})

        # structure: {verb: {voice: {mode: {tense: [person_keys]}}}}
        self._verb_structure = checkpoint.get("verb_structure", {})

        hp = checkpoint["hyperparams"]
        self._model = Seq2SeqModel(
            vocab_size=self._vocab["vocab_size"],
            emb_dim=hp["emb_dim"],
            hidden_dim=hp["hidden_dim"],
            cond_dim=hp["cond_dim"],
            n_voices=self._vocab["n_voices"],
            n_modes=self._vocab["n_modes"],
            n_tenses=self._vocab["n_tenses"],
            n_persons=self._vocab["n_persons"],
            dropout=hp.get("dropout", 0.0),
        )
        self._model.load_state_dict(checkpoint["model_state_dict"])
        self._model.eval()

        self._char_to_idx = self._vocab["char_to_idx"]
        self._idx_to_char = {int(k): v
                             for k, v in self._vocab["idx_to_char"].items()}
        self._voice_to_idx = self._vocab["voice_to_idx"]
        self._mode_to_idx = self._vocab["mode_to_idx"]
        self._tense_to_idx = self._vocab["tense_to_idx"]
        self._person_to_idx = self._vocab["person_to_idx"]

    # -- properties --------------------------------------------------------

    @property
    def verb_count(self):
        return len(self._known_verbs)

    def verbs(self, prefix=None):
        """Return the sorted list of known infinitives.

        If prefix is given, filter to verbs starting with that prefix
        using binary search.
        """
        if prefix is None:
            return list(self._known_verbs)
        lo = bisect.bisect_left(self._known_verbs, prefix)
        hi_prefix = prefix[:-1] + chr(ord(prefix[-1]) + 1)
        hi = bisect.bisect_left(self._known_verbs, hi_prefix)
        return self._known_verbs[lo:hi]

    def has_verb(self, infinitive):
        i = bisect.bisect_left(self._known_verbs, infinitive)
        return (i < len(self._known_verbs)
                and self._known_verbs[i] == infinitive)

    def is_h_aspire(self, infinitive):
        return infinitive in self._h_aspire

    def is_1990_reform(self, infinitive):
        return infinitive in self._reform_1990

    def reform_variante(self, infinitive):
        """Return the 1990 reform variant, or None."""
        return self._reform_variantes.get(infinitive)

    def voices(self, infinitive):
        """List available voices for a verb."""
        struct = self._verb_structure.get(infinitive, {})
        return sorted(struct.keys())

    def modes(self, infinitive, voice):
        voice = _VOICE_ALIASES.get(voice, voice)
        struct = self._verb_structure.get(infinitive, {})
        return sorted(struct.get(voice, {}).keys())

    def tenses(self, infinitive, voice, mode):
        voice = _VOICE_ALIASES.get(voice, voice)
        mode = _MODE_ALIASES.get(mode, mode)
        struct = self._verb_structure.get(infinitive, {})
        return sorted(struct.get(voice, {}).get(mode, {}).keys())

    def persons(self, infinitive, voice, mode, tense):
        voice = _VOICE_ALIASES.get(voice, voice)
        mode = _MODE_ALIASES.get(mode, mode)
        tense = _TENSE_ALIASES.get(tense, tense)
        struct = self._verb_structure.get(infinitive, {})
        return list(struct.get(voice, {}).get(mode, {}).get(tense, []))

    # -- core ML prediction ------------------------------------------------

    def _predict(self, infinitive, voice, mode, tense, person):
        """Run the neural model for a single form.

        Checks the exception table first, then falls back to the model.
        """
        exc_key = f"{infinitive}|{voice}|{mode}|{tense}|{person}"
        if exc_key in self._exceptions:
            return self._exceptions[exc_key]

        char_ids = []
        for c in infinitive:
            idx = self._char_to_idx.get(c)
            if idx is None:
                return None
            char_ids.append(idx)
        char_ids.append(EOS_IDX)

        src = torch.tensor([char_ids], dtype=torch.long)
        vi = torch.tensor([self._voice_to_idx.get(voice, 0)], dtype=torch.long)
        mi = torch.tensor([self._mode_to_idx.get(mode, 0)], dtype=torch.long)
        ti = torch.tensor([self._tense_to_idx.get(tense, 0)], dtype=torch.long)
        pi = torch.tensor([self._person_to_idx.get(person, 0)], dtype=torch.long)

        out_ids = self._model.predict(src, vi, mi, ti, pi)
        return "".join(self._idx_to_char.get(i, "") for i in out_ids)

    # -- main API ----------------------------------------------------------

    def conjugate(
        self,
        infinitive,
        *,
        voice=None,
        mode=None,
        tense=None,
        person=None,
    ):
        """Conjugate a French verb.

        Parameters are optional but layered: specifying a lower layer
        requires all upper layers to be given. For example, specifying
        mode without voice raises ValueError.

        Returns str when all parameters resolve to a single form,
        otherwise a nested dict of the remaining layers.
        Returns None if the verb or combination is not found.
        """
        r_voice = _VOICE_ALIASES.get(voice, voice) if voice else None
        r_mode = _MODE_ALIASES.get(mode, mode) if mode else None
        r_tense = _TENSE_ALIASES.get(tense, tense) if tense else None
        r_person = _PERSON_ALIASES.get(person, person) if person else None

        # enforce layering
        if r_person and not r_tense:
            raise ValueError("person requires tense to be specified")
        if r_tense and not r_mode:
            raise ValueError("tense requires mode to be specified")
        if r_mode and not r_voice:
            raise ValueError("mode requires voice to be specified")

        struct = self._verb_structure.get(infinitive)
        if struct is None:
            return None

        # all specified -> single form
        if r_voice and r_mode and r_tense and r_person:
            return self._single_form(infinitive, r_voice, r_mode, r_tense,
                                     r_person, struct)

        # partial spec -> nested dict
        if r_voice and r_mode and r_tense:
            return self._dict_persons(infinitive, r_voice, r_mode,
                                      r_tense, struct)
        if r_voice and r_mode:
            return self._dict_tenses(infinitive, r_voice, r_mode, struct)
        if r_voice:
            return self._dict_modes(infinitive, r_voice, struct)

        # nothing specified -> full dict
        return self._dict_voices(infinitive, struct)

    def _single_form(self, inf, voice, mode, tense, person, struct):
        available = struct.get(voice, {}).get(mode, {}).get(tense, [])
        if not available or person not in available:
            return None
        return self._predict(inf, voice, mode, tense, person)

    def _dict_persons(self, inf, voice, mode, tense, struct):
        persons = struct.get(voice, {}).get(mode, {}).get(tense, [])
        if not persons:
            return None
        result = {}
        for p in persons:
            form = self._predict(inf, voice, mode, tense, p)
            if form:
                result[p] = form
        return result or None

    def _dict_tenses(self, inf, voice, mode, struct):
        tenses = struct.get(voice, {}).get(mode, {})
        if not tenses:
            return None
        result = {}
        for t in sorted(tenses):
            d = self._dict_persons(inf, voice, mode, t, struct)
            if d:
                result[t] = d
        return result or None

    def _dict_modes(self, inf, voice, struct):
        modes = struct.get(voice, {})
        if not modes:
            return None
        result = {}
        for m in sorted(modes):
            d = self._dict_tenses(inf, voice, m, struct)
            if d:
                result[m] = d
        return result or None

    def _dict_voices(self, inf, struct):
        result = {}
        for v in sorted(struct):
            d = self._dict_modes(inf, v, struct)
            if d:
                result[v] = d
        return result or None

    def __repr__(self):
        return f"<ConjugationModel verbs={self.verb_count}>"


# -- Module-level singleton ------------------------------------------------

_singleton = None


def get_model(model_path=None):
    """Return a lazily-loaded singleton model instance."""
    global _singleton
    if _singleton is None:
        _singleton = ConjugationModel(model_path)
    return _singleton


# -- CLI -------------------------------------------------------------------

if __name__ == "__main__":
    import json
    import sys

    m = ConjugationModel()
    print(m)

    verb = sys.argv[1] if len(sys.argv) > 1 else "parler"
    voice_arg = sys.argv[2] if len(sys.argv) > 2 else None
    mode_arg = sys.argv[3] if len(sys.argv) > 3 else None
    tense_arg = sys.argv[4] if len(sys.argv) > 4 else None
    person_arg = sys.argv[5] if len(sys.argv) > 5 else None

    result = m.conjugate(verb, voice=voice_arg, mode=mode_arg,
                         tense=tense_arg, person=person_arg)

    if isinstance(result, str):
        print(f"\n{verb} -> {result}")
    elif isinstance(result, dict):
        print(f"\nConjugation of: {verb}")
        print(f"Voices: {m.voices(verb)}")
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(f"Verb '{verb}' not found or could not be conjugated.")
