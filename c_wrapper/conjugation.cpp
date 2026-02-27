/**
 * conjugation.cpp — C API implementation using LibTorch traced components.
 *
 * Loads four traced modules (encoder, bridge, attention, decoder-step)
 * plus a JSON metadata file, and implements the full conjugation logic
 * including the greedy decode loop, compound tenses, exception table,
 * invariable PP handling, and alias resolution.
 */

#include "conjugation.h"

#include <torch/script.h>
#include <torch/torch.h>

#include <algorithm>
#include <cstring>
#include <fstream>
#include <iostream>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

/* nlohmann/json — single-header JSON library.
   Download from https://github.com/nlohmann/json/releases → json.hpp
   and place next to this file or on the include path. */
#include "json.hpp"

using json = nlohmann::json;

/* ── Constants ──────────────────────────────────────────────────────── */

static constexpr int PAD_IDX = 0;
static constexpr int SOS_IDX = 1;
static constexpr int EOS_IDX = 2;
static constexpr int MAX_DECODE_LEN = 50;

/* ── Internal types ─────────────────────────────────────────────────── */

using StringPair = std::pair<std::string, std::string>;

struct CompoundInfo {
    std::string aux_mode;
    std::string aux_tense;
};

struct FRConjugationModel {
    /* Traced neural-network components */
    torch::jit::Module encoder;
    torch::jit::Module bridge;
    torch::jit::Module attention;
    torch::jit::Module decoder_step;

    /* Vocabulary */
    std::unordered_map<char32_t, int> char_to_idx;
    std::unordered_map<int, char32_t> idx_to_char;
    std::unordered_map<std::string, int> mode_to_idx;
    std::unordered_map<std::string, int> tense_to_idx;
    std::unordered_map<std::string, int> person_to_idx;

    /* Verb sets */
    std::unordered_set<std::string> known_verbs;
    std::unordered_set<std::string> etre_verbs;
    std::unordered_set<std::string> prono_verbs;
    std::unordered_set<std::string> h_aspire;
    std::unordered_set<std::string> invariable_pp;

    /* Exception table: "infinitive|mode|tense|person" → form */
    std::unordered_map<std::string, std::string> exceptions;

    /* Alias maps */
    std::unordered_map<std::string, std::string> mode_aliases;
    std::unordered_map<std::string, std::string> tense_aliases;
    std::unordered_map<std::string, std::string> person_aliases;

    /* Linguistic constants */
    std::set<StringPair> simple_tenses;
    std::unordered_map<std::string, CompoundInfo> compound_tense_map;
    std::unordered_map<std::string, std::string> pp_forme_etre;
};

/* ── UTF-8 helpers ──────────────────────────────────────────────────── */

static std::vector<char32_t> utf8_decode(const std::string &s) {
    std::vector<char32_t> result;
    size_t i = 0;
    while (i < s.size()) {
        char32_t cp;
        unsigned char c = static_cast<unsigned char>(s[i]);
        int len;
        if (c < 0x80)      { cp = c; len = 1; }
        else if (c < 0xC0) { ++i; continue; }
        else if (c < 0xE0) { cp = c & 0x1F; len = 2; }
        else if (c < 0xF0) { cp = c & 0x0F; len = 3; }
        else               { cp = c & 0x07; len = 4; }
        for (int j = 1; j < len && (i + j) < s.size(); ++j)
            cp = (cp << 6) | (static_cast<unsigned char>(s[i + j]) & 0x3F);
        result.push_back(cp);
        i += len;
    }
    return result;
}

static std::string utf8_encode_char(char32_t cp) {
    std::string out;
    if (cp < 0x80) {
        out += static_cast<char>(cp);
    } else if (cp < 0x800) {
        out += static_cast<char>(0xC0 | (cp >> 6));
        out += static_cast<char>(0x80 | (cp & 0x3F));
    } else if (cp < 0x10000) {
        out += static_cast<char>(0xE0 | (cp >> 12));
        out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
        out += static_cast<char>(0x80 | (cp & 0x3F));
    } else {
        out += static_cast<char>(0xF0 | (cp >> 18));
        out += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
        out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
        out += static_cast<char>(0x80 | (cp & 0x3F));
    }
    return out;
}

/* ── Static initializers ────────────────────────────────────────────── */

static void init_aliases(FRConjugationModel *m) {
    m->mode_aliases = {
        {"indicatif","indicatif"},{"ind","indicatif"},
        {"subjonctif","subjonctif"},{"sub","subjonctif"},
        {"conditionnel","conditionnel"},{"cond","conditionnel"},
        {"imperatif","imperatif"},{"impératif","imperatif"},{"imp","imperatif"},
        {"participe","participe"},{"part","participe"},
    };
    m->tense_aliases = {
        {"present","present"},{"présent","present"},
        {"imparfait","imparfait"},
        {"passe_simple","passe_simple"},{"passé_simple","passe_simple"},
        {"futur_simple","futur_simple"},{"futur","futur_simple"},
        {"passe_compose","passe_compose"},{"passé_composé","passe_compose"},
        {"plus_que_parfait","plus_que_parfait"},
        {"passe_anterieur","passe_anterieur"},{"passé_antérieur","passe_anterieur"},
        {"futur_anterieur","futur_anterieur"},{"futur_antérieur","futur_anterieur"},
        {"passe","passe"},{"passé","passe"},
        {"passe_sm","passe_sm"},{"passe_sf","passe_sf"},
        {"passe_pm","passe_pm"},{"passe_pf","passe_pf"},
    };
    m->person_aliases = {
        {"1s","1s"},{"2s","2s"},{"3s","3sm"},{"3sm","3sm"},{"3sf","3sf"},
        {"1p","1p"},{"2p","2p"},{"3p","3pm"},{"3pm","3pm"},{"3pf","3pf"},
        {"je","1s"},{"tu","2s"},{"il","3sm"},{"elle","3sf"},{"on","3sm"},
        {"nous","1p"},{"vous","2p"},{"ils","3pm"},{"elles","3pf"},
    };
}

static void init_tense_tables(FRConjugationModel *m) {
    m->simple_tenses = {
        {"indicatif","present"},{"indicatif","imparfait"},
        {"indicatif","passe_simple"},{"indicatif","futur_simple"},
        {"conditionnel","present"},
        {"subjonctif","present"},{"subjonctif","imparfait"},
        {"imperatif","present"},
    };
    m->compound_tense_map = {
        {"indicatif|passe_compose",    {"indicatif","present"}},
        {"indicatif|plus_que_parfait", {"indicatif","imparfait"}},
        {"indicatif|passe_anterieur",  {"indicatif","passe_simple"}},
        {"indicatif|futur_anterieur",  {"indicatif","futur_simple"}},
        {"conditionnel|passe",         {"conditionnel","present"}},
        {"subjonctif|passe",           {"subjonctif","present"}},
        {"subjonctif|plus_que_parfait",{"subjonctif","imparfait"}},
        {"imperatif|passe",            {"imperatif","present"}},
    };
    m->pp_forme_etre = {
        {"1s","passe_sm"},{"2s","passe_sm"},
        {"3sm","passe_sm"},{"3sf","passe_sf"},
        {"1p","passe_pm"},{"2p","passe_pm"},
        {"3pm","passe_pm"},{"3pf","passe_pf"},
    };
}

static std::string resolve_alias(
    const std::unordered_map<std::string, std::string> &map,
    const std::string &key
) {
    auto it = map.find(key);
    return (it != map.end()) ? it->second : key;
}

/* ── Neural prediction (greedy decode loop) ─────────────────────────── */

static std::string predict(
    FRConjugationModel *m,
    const std::string &infinitive,
    const std::string &mode,
    const std::string &tense,
    const std::string &person
) {
    /* 1. Exception table override */
    std::string exc_key = infinitive + "|" + mode + "|" + tense + "|" + person;
    auto exc_it = m->exceptions.find(exc_key);
    if (exc_it != m->exceptions.end())
        return exc_it->second;

    /* 2. Encode input characters */
    auto codepoints = utf8_decode(infinitive);
    std::vector<int64_t> ids;
    ids.reserve(codepoints.size() + 1);
    for (auto cp : codepoints) {
        auto it = m->char_to_idx.find(cp);
        if (it == m->char_to_idx.end()) return "";
        ids.push_back(it->second);
    }
    ids.push_back(EOS_IDX);

    auto mode_it   = m->mode_to_idx.find(mode);
    auto tense_it  = m->tense_to_idx.find(tense);
    auto person_it = m->person_to_idx.find(person);
    if (mode_it == m->mode_to_idx.end() ||
        tense_it == m->tense_to_idx.end() ||
        person_it == m->person_to_idx.end())
        return "";

    /* 3. Create tensors */
    auto src   = torch::tensor(ids, torch::kLong).unsqueeze(0);       // [1, seq_len]
    auto m_idx = torch::tensor({(int64_t)mode_it->second}, torch::kLong);
    auto t_idx = torch::tensor({(int64_t)tense_it->second}, torch::kLong);
    auto p_idx = torch::tensor({(int64_t)person_it->second}, torch::kLong);

    /* 4. Encoder */
    auto enc_out = m->encoder.forward({src}).toTuple();
    auto enc_outputs = enc_out->elements()[0].toTensor();  // [1, seq, H*2]
    auto enc_hidden  = enc_out->elements()[1].toTensor();  // [1, H*2]

    /* 5. Bridge → initial decoder hidden state */
    auto dec_hidden = m->bridge.forward({enc_hidden, m_idx, t_idx, p_idx}).toTensor();
    // dec_hidden: [1, 1, H]

    /* 6. Build source mask */
    auto src_mask = (src != PAD_IDX).to(torch::kFloat);  // [1, seq_len]

    /* 7. Greedy decode loop */
    auto dec_input = torch::tensor({(int64_t)SOS_IDX}, torch::kLong);
    std::string result;

    for (int step = 0; step < MAX_DECODE_LEN; ++step) {
        /* Attention: dec_hidden.squeeze(0) → [1, H] */
        auto context = m->attention.forward(
            {dec_hidden.squeeze(0), enc_outputs, src_mask}
        ).toTensor();

        /* Decoder step */
        auto dec_out = m->decoder_step.forward({dec_input, dec_hidden, context}).toTuple();
        auto logits     = dec_out->elements()[0].toTensor();  // [1, vocab]
        dec_hidden      = dec_out->elements()[1].toTensor();  // [1, 1, H]

        int64_t token_id = logits.argmax(-1).item<int64_t>();
        if (token_id == EOS_IDX) break;

        auto ch_it = m->idx_to_char.find(static_cast<int>(token_id));
        if (ch_it != m->idx_to_char.end())
            result += utf8_encode_char(ch_it->second);

        dec_input = torch::tensor({token_id}, torch::kLong);
    }

    return result;
}

/* ── Higher-level conjugation logic ─────────────────────────────────── */

static std::string get_participle_internal(
    FRConjugationModel *m,
    const std::string &infinitive,
    const std::string &forme
) {
    std::string actual = forme;
    if ((forme == "passe_sf" || forme == "passe_pm" || forme == "passe_pf") &&
        m->invariable_pp.count(infinitive))
        actual = "passe_sm";
    return predict(m, infinitive, "participe", actual, "-");
}

static std::string conjugate_compound(
    FRConjugationModel *m,
    const std::string &infinitive,
    const std::string &mode,
    const std::string &tense,
    const std::string &person
) {
    auto it = m->compound_tense_map.find(mode + "|" + tense);
    if (it == m->compound_tense_map.end()) return "";

    const auto &info = it->second;
    bool uses_etre = m->etre_verbs.count(infinitive) > 0;
    std::string aux_verb = uses_etre ? "être" : "avoir";

    std::string aux_form = predict(m, aux_verb, info.aux_mode, info.aux_tense, person);
    if (aux_form.empty()) return "";

    std::string pp_forme = "passe_sm";
    if (uses_etre) {
        auto pp_it = m->pp_forme_etre.find(person);
        if (pp_it != m->pp_forme_etre.end())
            pp_forme = pp_it->second;
    }
    std::string pp = predict(m, infinitive, "participe", pp_forme, "-");
    if (pp.empty()) return "";

    return aux_form + " " + pp;
}

static std::string conjugate_single(
    FRConjugationModel *m,
    const std::string &infinitive,
    const std::string &mode,
    const std::string &tense,
    const std::string &person
) {
    if (mode == "participe")
        return get_participle_internal(m, infinitive, tense);
    if (m->simple_tenses.count({mode, tense}))
        return predict(m, infinitive, mode, tense, person);
    if (m->compound_tense_map.count(mode + "|" + tense))
        return conjugate_compound(m, infinitive, mode, tense, person);
    return "";
}

/* ── Buffer helper ──────────────────────────────────────────────────── */

static int write_to_buf(const std::string &s, char *buf, size_t buf_size) {
    if (s.empty() || !buf || buf_size == 0) return -1;
    size_t n = std::min(s.size(), buf_size - 1);
    std::memcpy(buf, s.data(), n);
    buf[n] = '\0';
    return static_cast<int>(n);
}

/* ── Public C API ───────────────────────────────────────────────────── */

extern "C" {

FRConjugationModel *fr_conjugation_load(const char *model_dir) {
    try {
        std::string dir(model_dir);
        if (!dir.empty() && dir.back() != '/') dir += '/';

        auto *m = new FRConjugationModel();

        /* Load four traced modules */
        m->encoder      = torch::jit::load(dir + "conjugation_encoder.pt");
        m->bridge       = torch::jit::load(dir + "conjugation_bridge.pt");
        m->attention    = torch::jit::load(dir + "conjugation_attention.pt");
        m->decoder_step = torch::jit::load(dir + "conjugation_decoder.pt");
        m->encoder.eval();
        m->bridge.eval();
        m->attention.eval();
        m->decoder_step.eval();

        /* Load metadata JSON */
        std::ifstream f(dir + "conjugation_meta.json");
        if (!f.is_open()) { delete m; return nullptr; }
        json meta = json::parse(f);

        for (auto &[k, v] : meta["char_to_idx"].items()) {
            auto cps = utf8_decode(k);
            if (!cps.empty()) m->char_to_idx[cps[0]] = v.get<int>();
        }
        for (auto &[k, v] : meta["idx_to_char"].items()) {
            auto cps = utf8_decode(v.get<std::string>());
            if (!cps.empty()) m->idx_to_char[std::stoi(k)] = cps[0];
        }
        for (auto &[k, v] : meta["mode_to_idx"].items())
            m->mode_to_idx[k] = v.get<int>();
        for (auto &[k, v] : meta["tense_to_idx"].items())
            m->tense_to_idx[k] = v.get<int>();
        for (auto &[k, v] : meta["person_to_idx"].items())
            m->person_to_idx[k] = v.get<int>();

        for (auto &v : meta["known_verbs"])     m->known_verbs.insert(v.get<std::string>());
        for (auto &v : meta["etre_verbs"])      m->etre_verbs.insert(v.get<std::string>());
        for (auto &v : meta["prono_verbs"])     m->prono_verbs.insert(v.get<std::string>());
        for (auto &v : meta["h_aspire"])        m->h_aspire.insert(v.get<std::string>());
        for (auto &v : meta["invariable_pp_verbs"]) m->invariable_pp.insert(v.get<std::string>());

        for (auto &[k, v] : meta["exceptions"].items())
            m->exceptions[k] = v.get<std::string>();

        init_aliases(m);
        init_tense_tables(m);

        return m;
    } catch (const std::exception &e) {
        std::cerr << "fr_conjugation_load: " << e.what() << std::endl;
        return nullptr;
    }
}

void fr_conjugation_free(FRConjugationModel *model) { delete model; }

int fr_conjugation_verb_count(const FRConjugationModel *model) {
    return model ? static_cast<int>(model->known_verbs.size()) : 0;
}

bool fr_conjugation_has_verb(const FRConjugationModel *model, const char *infinitive) {
    return model && infinitive && model->known_verbs.count(infinitive);
}

bool fr_conjugation_is_h_aspire(const FRConjugationModel *model, const char *infinitive) {
    return model && infinitive && model->h_aspire.count(infinitive);
}

int fr_conjugation_auxiliary(
    const FRConjugationModel *model, const char *infinitive,
    char *out_buf, size_t buf_size
) {
    if (!model || !infinitive) return -1;
    std::string inf(infinitive), result;
    if (!model->etre_verbs.count(inf)) result += "avoir";
    if (model->etre_verbs.count(inf)) {
        if (!result.empty()) result += ",";
        result += "être";
    }
    if (model->prono_verbs.count(inf)) {
        if (!result.empty()) result += ",";
        result += "pronominal";
    }
    return write_to_buf(result, out_buf, buf_size);
}

int fr_conjugation_conjugate(
    const FRConjugationModel *model, const char *infinitive,
    const char *mode, const char *tense, const char *person,
    char *out_buf, size_t buf_size
) {
    if (!model || !infinitive || !mode || !tense || !person) return -1;
    auto *m = const_cast<FRConjugationModel *>(model);
    std::string r_mode   = resolve_alias(m->mode_aliases, mode);
    std::string r_tense  = resolve_alias(m->tense_aliases, tense);
    std::string r_person = resolve_alias(m->person_aliases, person);
    return write_to_buf(conjugate_single(m, infinitive, r_mode, r_tense, r_person), out_buf, buf_size);
}

int fr_conjugation_get_participle(
    const FRConjugationModel *model, const char *infinitive, const char *forme,
    char *out_buf, size_t buf_size
) {
    if (!model || !infinitive || !forme) return -1;
    auto *m = const_cast<FRConjugationModel *>(model);
    return write_to_buf(get_participle_internal(m, infinitive, forme), out_buf, buf_size);
}

} /* extern "C" */
