#!/usr/bin/env python3
"""Test suite for the ML-based French conjugation model."""

from french_conjugation_model import ConjugationModel, get_model

m = ConjugationModel()
print(f"Model loaded: {m}")


def test(name, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    extra = f" — {detail}" if detail and not condition else ""
    print(f"  [{status}] {name}{extra}")
    return condition


passes = 0
total = 0


def check(name, condition, detail=""):
    global passes, total
    total += 1
    if test(name, condition, detail):
        passes += 1


# ── 1. Simple tense conjugations ─────────────────────────────────────────────

print("\n— Simple tenses —")

# Regular -er
check("parler ind.present 1s",
      m.conjugate("parler", mode="indicatif", tense="present", person="1s") == "parle")
check("parler ind.present 2p",
      m.conjugate("parler", mode="indicatif", tense="present", person="2p") == "parlez")
check("parler ind.imparfait 3sm",
      m.conjugate("parler", mode="indicatif", tense="imparfait", person="3sm") == "parlait")

# Regular -ir
f = m.conjugate("finir", mode="indicatif", tense="present", person="3sm")
check("finir ind.present 3sm", f == "finit", f"got '{f}'")
f = m.conjugate("finir", mode="indicatif", tense="present", person="1p")
check("finir ind.present 1p", f == "finissons", f"got '{f}'")

# Regular -re
f = m.conjugate("vendre", mode="indicatif", tense="present", person="1s")
check("vendre ind.present 1s", f == "vends", f"got '{f}'")

# Irregular
f = m.conjugate("aller", mode="indicatif", tense="present", person="1s")
check("aller ind.present 1s", f == "vais", f"got '{f}'")
f = m.conjugate("être", mode="indicatif", tense="present", person="1s")
check("être ind.present 1s", f == "suis", f"got '{f}'")
f = m.conjugate("avoir", mode="indicatif", tense="present", person="1s")
check("avoir ind.present 1s", f == "ai", f"got '{f}'")
f = m.conjugate("faire", mode="indicatif", tense="present", person="1s")
check("faire ind.present 1s", f == "fais", f"got '{f}'")
f = m.conjugate("prendre", mode="indicatif", tense="present", person="1s")
check("prendre ind.present 1s", f == "prends", f"got '{f}'")
f = m.conjugate("venir", mode="indicatif", tense="present", person="1s")
check("venir ind.present 1s", f == "viens", f"got '{f}'")

# ── 2. Pronoun / alias resolution ────────────────────────────────────────────

print("\n— Aliases —")

check("pronoun alias 'je'",
      m.conjugate("parler", mode="indicatif", tense="present", person="je") == "parle")
check("pronoun alias 'nous'",
      m.conjugate("parler", mode="indicatif", tense="present", person="nous") == "parlons")
check("mode alias 'ind'",
      m.conjugate("parler", mode="ind", tense="present", person="1s") == "parle")
check("tense alias 'présent'",
      m.conjugate("parler", mode="indicatif", tense="présent", person="1s") == "parle")

# ── 3. Compound tenses ──────────────────────────────────────────────────────

print("\n— Compound tenses —")

f = m.conjugate("parler", mode="indicatif", tense="passe_compose", person="1s")
check("parler passé composé 1s", f == "ai parlé", f"got '{f}'")

f = m.conjugate("parler", mode="indicatif", tense="plus_que_parfait", person="3sm")
check("parler PQP 3sm", f == "avait parlé", f"got '{f}'")

f = m.conjugate("aller", mode="indicatif", tense="passe_compose", person="1s")
check("aller passé composé 1s (être)", f == "suis allé", f"got '{f}'")

f = m.conjugate("aller", mode="indicatif", tense="passe_compose", person="3sf")
check("aller passé composé 3sf (agreement)", f == "est allée", f"got '{f}'")

# ── 4. Participles ──────────────────────────────────────────────────────────

print("\n— Participles —")

f = m.get_participle("parler", "present")
check("parler participe present", f == "parlant", f"got '{f}'")
f = m.get_participle("parler", "passe_sm")
check("parler participe passe_sm", f == "parlé", f"got '{f}'")
f = m.get_participle("finir", "passe_sm")
check("finir participe passe_sm", f == "fini", f"got '{f}'")

# ── 5. Dict returns ─────────────────────────────────────────────────────────

print("\n— Dict returns —")

result = m.conjugate("parler", mode="indicatif", tense="present")
check("dict return for partial spec", isinstance(result, dict) and "indicatif" in result)

result = m.conjugate("parler")
check("full conjugation dict",
      isinstance(result, dict) and "indicatif" in result and "subjonctif" in result)

# ── 6. Helpers ───────────────────────────────────────────────────────────────

print("\n— Helpers —")

check("has_verb('parler')", m.has_verb("parler"))
check("not has_verb('xyzfake')", not m.has_verb("xyzfake"))
check("auxiliary('parler')", "avoir" in m.auxiliary("parler"))
check("auxiliary('aller')", "être" in m.auxiliary("aller"))
check("verb_count > 6000", m.verb_count > 6000, f"got {m.verb_count}")

# ── 7. Singleton ────────────────────────────────────────────────────────────

print("\n— Singleton —")

m2 = get_model()
check("get_model singleton works",
      m2.conjugate("parler", mode="indicatif", tense="present", person="1s") == "parle")

# ── Summary ─────────────────────────────────────────────────────────────────

print(f"\n{'='*40}")
print(f"  {passes}/{total} tests passed")
print(f"{'='*40}")
