/**
 * conjugation.h — C API for the French verb conjugation model.
 *
 * This library wraps LibTorch traced components (encoder, bridge,
 * attention, decoder) together with metadata (vocabulary, exceptions,
 * linguistic rules) to provide a pure-C interface for conjugating
 * French verbs.
 *
 * Thread safety: each FRConjugationModel instance is independent.
 * Do not share one instance across threads without external locking.
 */

#ifndef CONJUGATION_H
#define CONJUGATION_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>
#include <stddef.h>

/* ── Opaque handle ─────────────────────────────────────────────────── */

typedef struct FRConjugationModel FRConjugationModel;

/* ── Lifecycle ─────────────────────────────────────────────────────── */

/**
 * Load the model from a directory containing the exported files.
 *
 * The directory must contain:
 *   conjugation_encoder.pt, conjugation_bridge.pt,
 *   conjugation_attention.pt, conjugation_decoder.pt,
 *   conjugation_meta.json
 *
 * @param model_dir  Path to the directory containing the exported files.
 * @return  A model handle, or NULL on failure.
 */
FRConjugationModel *fr_conjugation_load(const char *model_dir);

/**
 * Free all resources associated with the model.
 */
void fr_conjugation_free(FRConjugationModel *model);

/* ── Query ─────────────────────────────────────────────────────────── */

int fr_conjugation_verb_count(const FRConjugationModel *model);
bool fr_conjugation_has_verb(const FRConjugationModel *model, const char *infinitive);
bool fr_conjugation_is_h_aspire(const FRConjugationModel *model, const char *infinitive);

/**
 * Get the auxiliary for a verb.
 * Result is comma-separated if multiple (e.g. "avoir,pronominal").
 *
 * @return  Bytes written (excluding NUL), or -1 on error.
 */
int fr_conjugation_auxiliary(
    const FRConjugationModel *model,
    const char *infinitive,
    char *out_buf,
    size_t buf_size
);

/* ── Conjugation ───────────────────────────────────────────────────── */

/**
 * Conjugate a single form. All of mode, tense, person are required.
 *
 * @return  Bytes written (excluding NUL), or -1 on error / not found.
 */
int fr_conjugation_conjugate(
    const FRConjugationModel *model,
    const char *infinitive,
    const char *mode,
    const char *tense,
    const char *person,
    char *out_buf,
    size_t buf_size
);

/**
 * Get a participle form.
 * forme: "present", "passe_sm", "passe_sf", "passe_pm", "passe_pf".
 *
 * @return  Bytes written (excluding NUL), or -1 on error / not found.
 */
int fr_conjugation_get_participle(
    const FRConjugationModel *model,
    const char *infinitive,
    const char *forme,
    char *out_buf,
    size_t buf_size
);

#ifdef __cplusplus
}
#endif

#endif /* CONJUGATION_H */
