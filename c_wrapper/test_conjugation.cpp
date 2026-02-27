/**
 * test_conjugation.cpp — Smoke test for the C conjugation library.
 *
 * Usage:
 *   ./test_conjugation <path/to/model_dir>
 *
 * The model_dir must contain the exported .pt files and conjugation_meta.json.
 */

#include "conjugation.h"
#include <cstdio>
#include <cstring>

static int passed = 0;
static int failed = 0;

static void check(
    const char *label,
    const FRConjugationModel *model,
    const char *infinitive,
    const char *mode,
    const char *tense,
    const char *person,
    const char *expected
) {
    char buf[256] = {0};
    int n = fr_conjugation_conjugate(model, infinitive, mode, tense, person, buf, sizeof(buf));
    if (n > 0 && std::strcmp(buf, expected) == 0) {
        printf("  [PASS] %s → %s\n", label, buf);
        ++passed;
    } else {
        printf("  [FAIL] %s → got \"%s\", expected \"%s\"\n", label, buf, expected);
        ++failed;
    }
}

static void check_participle(
    const char *label,
    const FRConjugationModel *model,
    const char *infinitive,
    const char *forme,
    const char *expected
) {
    char buf[256] = {0};
    int n = fr_conjugation_get_participle(model, infinitive, forme, buf, sizeof(buf));
    if (n > 0 && std::strcmp(buf, expected) == 0) {
        printf("  [PASS] %s → %s\n", label, buf);
        ++passed;
    } else {
        printf("  [FAIL] %s → got \"%s\", expected \"%s\"\n", label, buf, expected);
        ++failed;
    }
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <model_dir>\n", argv[0]);
        return 1;
    }

    printf("Loading model from: %s\n", argv[1]);
    FRConjugationModel *model = fr_conjugation_load(argv[1]);
    if (!model) {
        fprintf(stderr, "Failed to load model.\n");
        return 1;
    }

    printf("Verb count: %d\n\n", fr_conjugation_verb_count(model));

    /* Simple tenses */
    printf("— Simple tenses —\n");
    check("parler ind.present 1s",   model, "parler",  "indicatif", "present",      "1s",  "parle");
    check("finir ind.present 1s",    model, "finir",   "indicatif", "present",      "1s",  "finis");
    check("aller ind.present 1s",    model, "aller",   "indicatif", "present",      "1s",  "vais");
    check("être ind.present 1s",     model, "être",    "indicatif", "present",      "1s",  "suis");
    check("avoir ind.present 1s",    model, "avoir",   "indicatif", "present",      "1s",  "ai");
    check("faire ind.present 1s",    model, "faire",   "indicatif", "present",      "1s",  "fais");
    check("prendre ind.present 3sm", model, "prendre", "indicatif", "present",      "3sm", "prend");
    check("venir ind.present 1p",    model, "venir",   "indicatif", "present",      "1p",  "venons");

    /* Compound tenses */
    printf("\n— Compound tenses —\n");
    check("parler passe_compose 1s", model, "parler", "indicatif", "passe_compose", "1s",  "ai parlé");
    check("aller passe_compose 1s",  model, "aller",  "indicatif", "passe_compose", "1s",  "suis allé");
    check("aller passe_compose 3sf", model, "aller",  "indicatif", "passe_compose", "3sf", "est allée");

    /* Participles */
    printf("\n— Participles —\n");
    check_participle("parler present",  model, "parler", "present",  "parlant");
    check_participle("parler passe_sm", model, "parler", "passe_sm", "parlé");
    check_participle("finir passe_sm",  model, "finir",  "passe_sm", "fini");

    /* Helpers */
    printf("\n— Helpers —\n");
    if (fr_conjugation_has_verb(model, "parler")) {
        printf("  [PASS] has_verb(parler)\n"); ++passed;
    } else {
        printf("  [FAIL] has_verb(parler)\n"); ++failed;
    }
    if (!fr_conjugation_has_verb(model, "xyzfake")) {
        printf("  [PASS] !has_verb(xyzfake)\n"); ++passed;
    } else {
        printf("  [FAIL] !has_verb(xyzfake)\n"); ++failed;
    }

    char aux_buf[64];
    fr_conjugation_auxiliary(model, "parler", aux_buf, sizeof(aux_buf));
    if (std::strstr(aux_buf, "avoir") != nullptr) {
        printf("  [PASS] auxiliary(parler) → %s\n", aux_buf); ++passed;
    } else {
        printf("  [FAIL] auxiliary(parler) → got \"%s\"\n", aux_buf); ++failed;
    }

    fr_conjugation_auxiliary(model, "aller", aux_buf, sizeof(aux_buf));
    if (std::strstr(aux_buf, "être") != nullptr) {
        printf("  [PASS] auxiliary(aller) → %s\n", aux_buf); ++passed;
    } else {
        printf("  [FAIL] auxiliary(aller) → got \"%s\"\n", aux_buf); ++failed;
    }

    printf("\n========================================\n");
    printf("  %d/%d tests passed\n", passed, passed + failed);
    printf("========================================\n");

    fr_conjugation_free(model);
    return failed > 0 ? 1 : 0;
}
