/**
 * conjugation.h — C API for the French verb conjugation model.
 *
 * Mirror of c_wrapper/conjugation.h — kept in sync manually.
 * This copy allows the Swift package to be self-contained.
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

FRConjugationModel *fr_conjugation_load(const char *model_dir);
void fr_conjugation_free(FRConjugationModel *model);

/* ── Query ─────────────────────────────────────────────────────────── */

int  fr_conjugation_verb_count(const FRConjugationModel *model);
bool fr_conjugation_has_verb(const FRConjugationModel *model, const char *infinitive);
bool fr_conjugation_is_h_aspire(const FRConjugationModel *model, const char *infinitive);

int fr_conjugation_auxiliary(
    const FRConjugationModel *model,
    const char *infinitive,
    char *out_buf,
    size_t buf_size
);

/* ── Conjugation ───────────────────────────────────────────────────── */

int fr_conjugation_conjugate(
    const FRConjugationModel *model,
    const char *infinitive,
    const char *mode,
    const char *tense,
    const char *person,
    char *out_buf,
    size_t buf_size
);

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
