#!/usr/bin/env python3
"""Test suite for the French conjugation model (v3)."""

from french_conjugation_model import ConjugationModel, get_model

m = ConjugationModel()
print(f"Model loaded: {m}")


def test(name, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    extra = f" -- {detail}" if detail and not condition else ""
    print(f"  [{status}] {name}{extra}")
    return condition


passes = 0
total = 0


def check(name, condition, detail=""):
    global passes, total
    total += 1
    if test(name, condition, detail):
        passes += 1


# -- 1. Simple tense conjugations -----------------------------------------

print("\n-- Simple tenses (active avoir) --")

f = m.conjugate("parler", voice="voix_active_avoir", mode="indicatif",
                tense="present", person="1sm")
check("parler ind.present 1sm", f == "parle", f"got '{f}'")

f = m.conjugate("parler", voice="voix_active_avoir", mode="indicatif",
                tense="present", person="2pm")
check("parler ind.present 2pm", f == "parlez", f"got '{f}'")

f = m.conjugate("parler", voice="voix_active_avoir", mode="indicatif",
                tense="imparfait", person="3sm")
check("parler ind.imparfait 3sm", f == "parlait", f"got '{f}'")

f = m.conjugate("finir", voice="voix_active_avoir", mode="indicatif",
                tense="present", person="3sm")
check("finir ind.present 3sm", f == "finit", f"got '{f}'")

f = m.conjugate("finir", voice="voix_active_avoir", mode="indicatif",
                tense="present", person="1pm")
check("finir ind.present 1pm", f == "finissons", f"got '{f}'")

f = m.conjugate("vendre", voice="voix_active_avoir", mode="indicatif",
                tense="present", person="1sm")
check("vendre ind.present 1sm", f == "vends", f"got '{f}'")

# -- 2. Irregular verbs (active) ------------------------------------------

print("\n-- Irregular verbs --")

f = m.conjugate("aller", voice="voix_active_etre", mode="indicatif",
                tense="present", person="1sm")
check("aller ind.present 1sm", f == "vais", f"got '{f}'")

f = m.conjugate("avoir", voice="voix_active_avoir", mode="indicatif",
                tense="present", person="1sm")
check("avoir ind.present 1sm", f == "ai", f"got '{f}'")

# Note: "etre" is the infinitive without accents in the DB
f = m.conjugate("\u00eatre", voice="voix_active_avoir", mode="indicatif",
                tense="present", person="1sm")
check("etre ind.present 1sm", f == "suis", f"got '{f}'")

f = m.conjugate("faire", voice="voix_active_avoir", mode="indicatif",
                tense="present", person="1sm")
check("faire ind.present 1sm", f == "fais", f"got '{f}'")

f = m.conjugate("venir", voice="voix_active_etre", mode="indicatif",
                tense="present", person="1sm")
check("venir ind.present 1sm", f == "viens", f"got '{f}'")

# -- 3. Compound tenses ---------------------------------------------------

print("\n-- Compound tenses --")

f = m.conjugate("parler", voice="voix_active_avoir", mode="indicatif",
                tense="passe_compose", person="1sm")
check("parler passe_compose 1sm", f == "ai parl\u00e9", f"got '{f}'")

f = m.conjugate("aller", voice="voix_active_etre", mode="indicatif",
                tense="passe_compose", person="1sm")
check("aller passe_compose 1sm (etre)", f == "suis all\u00e9", f"got '{f}'")

f = m.conjugate("aller", voice="voix_active_etre", mode="indicatif",
                tense="passe_compose", person="3sf")
check("aller passe_compose 3sf (agreement)", f == "est all\u00e9e",
      f"got '{f}'")

# -- 4. Passive voice ------------------------------------------------------

print("\n-- Passive voice --")

f = m.conjugate("aimer", voice="voix_passive", mode="indicatif",
                tense="present", person="1sm")
check("aimer passive ind.present 1sm", f == "suis aim\u00e9", f"got '{f}'")

f = m.conjugate("aimer", voice="voix_passive", mode="indicatif",
                tense="present", person="3sf")
check("aimer passive ind.present 3sf", f == "est aim\u00e9e", f"got '{f}'")

# -- 5. Pronominal voice --------------------------------------------------

print("\n-- Pronominal voice --")

f = m.conjugate("laver", voice="voix_prono", mode="indicatif",
                tense="present", person="1sm")
check("laver prono ind.present 1sm", f == "me lave", f"got '{f}'")

f = m.conjugate("laver", voice="voix_prono", mode="indicatif",
                tense="passe_compose", person="3sf")
check("laver prono passe_compose 3sf", f == "s'est lav\u00e9e", f"got '{f}'")

# -- 6. Participles -------------------------------------------------------

print("\n-- Participles --")

f = m.conjugate("parler", voice="voix_active_avoir", mode="participe",
                tense="present")
check("parler participe present", f == "parlant", f"got '{f}'")

f = m.conjugate("parler", voice="voix_active_avoir", mode="participe",
                tense="passe_sm")
check("parler participe passe_sm", f == "parl\u00e9", f"got '{f}'")

f = m.conjugate("finir", voice="voix_active_avoir", mode="participe",
                tense="passe_sm")
check("finir participe passe_sm", f == "fini", f"got '{f}'")

# -- 7. Layered parameter validation --------------------------------------

print("\n-- Layered parameters --")

try:
    m.conjugate("parler", mode="indicatif")
    check("mode without voice raises", False, "no exception raised")
except ValueError:
    check("mode without voice raises", True)

try:
    m.conjugate("parler", voice="voix_active_avoir", tense="present")
    check("tense without mode raises", False, "no exception raised")
except ValueError:
    check("tense without mode raises", True)

try:
    m.conjugate("parler", voice="voix_active_avoir", mode="indicatif",
                person="1sm")
    check("person without tense raises", False, "no exception raised")
except ValueError:
    check("person without tense raises", True)

# -- 8. Dict returns ------------------------------------------------------

print("\n-- Dict returns --")

result = m.conjugate("parler", voice="voix_active_avoir", mode="indicatif",
                     tense="present")
check("person dict return", isinstance(result, dict) and "1sm" in result,
      f"got {type(result)}: {result}")

result = m.conjugate("parler", voice="voix_active_avoir", mode="indicatif")
check("tense dict return",
      isinstance(result, dict) and "present" in result)

result = m.conjugate("parler", voice="voix_active_avoir")
check("mode dict return",
      isinstance(result, dict) and "indicatif" in result)

result = m.conjugate("parler")
check("full dict return",
      isinstance(result, dict) and "voix_active_avoir" in result)

# -- 9. Metadata -----------------------------------------------------------

print("\n-- Metadata --")

check("has_verb('parler')", m.has_verb("parler"))
check("not has_verb('xyzfake')", not m.has_verb("xyzfake"))
check("verb_count > 6000", m.verb_count > 6000, f"got {m.verb_count}")

vs = m.voices("parler")
check("voices('parler')", "voix_active_avoir" in vs, f"got {vs}")

vs = m.voices("aller")
check("voices('aller')", "voix_active_etre" in vs, f"got {vs}")

check("is_h_aspire('ha\u00efr')", m.is_h_aspire("ha\u00efr"))

# -- 10. Verb listing / prefix search -------------------------------------

print("\n-- Verb listing --")

all_verbs = m.verbs()
check("verbs() returns list", len(all_verbs) > 6000, f"got {len(all_verbs)}")
check("verbs() sorted", all_verbs == sorted(all_verbs))

a_verbs = m.verbs(prefix="a")
check("verbs(prefix='a')", len(a_verbs) > 0 and all(
    v.startswith("a") for v in a_verbs))

et_verbs = m.verbs(prefix="\u00eat")
check("verbs(prefix='et')", "\u00eatre" in et_verbs, f"got {et_verbs}")

# -- 11. Singleton ---------------------------------------------------------

print("\n-- Singleton --")

m2 = get_model()
check("get_model singleton works",
      m2.conjugate("parler", voice="voix_active_avoir", mode="indicatif",
                   tense="present", person="1sm") == "parle")

# -- 12. Aliases -----------------------------------------------------------

print("\n-- Aliases --")

f = m.conjugate("parler", voice="active_avoir", mode="ind",
                tense="present", person="je")
check("alias resolution", f == "parle", f"got '{f}'")

# -- Summary ---------------------------------------------------------------

print(f"\n{'='*40}")
print(f"  {passes}/{total} tests passed")
print(f"{'='*40}")
