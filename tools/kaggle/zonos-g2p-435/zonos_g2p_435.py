#!/usr/bin/env python3
"""#435 follow-on: is the BUILT-IN G2P good enough to drop espeak-ng (GPL-3.0)?

#435 is fixed -- zonos now REFUSES instead of synthesising noise when it cannot
phonemise. That leaves the dependency itself. crispasr-core ships built-in G2P
for en/de/fr/es, and zonos can now reach it (CRISPASR_ZONOS_G2P=builtin|espeak|auto).
The open question is quality, and the honest answer needs numbers per language.

THE RISK THIS KERNEL EXISTS TO MEASURE
--------------------------------------
The built-ins emit IPA in the espeak dialect; zonos's phoneme map comes from its
own conditioning.py symbol list; and text_to_phoneme_ids() SILENTLY DROPS every
codepoint outside that list. That silent drop IS #435 -- Cyrillic became three
tokens at a success exit code. So "the builtin produced a wav" proves nothing. A
builtin path could drop 30% of its symbols and still sound like speech.

Therefore every language is scored on the PAYLOAD, not the container:
  * the phoneme-ID sequence from each path, compared position-by-position and by
    token-level Levenshtein similarity (a single inserted phoneme wrecks the
    positional score, so both are reported);
  * cp_dropped/cp_total per path -- how much of what the G2P emitted zonos could
    not map. A path that drops more is VISIBLY worse, not silently worse;
  * an ASR roundtrip (parakeet-tdt-0.6b-v3, 25 European languages) word-F1
    against the input text.

CONTROLS -- a readout that renders the same for good and bad input is worthless
------------------------------------------------------------------------------
  0. INSTRUMENT SELF-TEST, before anything heavy. Every metric is run on
     known-answer and DEGENERATE inputs (identical -> 1.0, disjoint -> 0.0, two
     EMPTY sequences -> 0.0 and NOT 1.0, which is the exact shape of the #435
     failure). If any of them is wrong the kernel exits before spending a
     session producing numbers from a broken ruler.
  1. WRONG-LANGUAGE END-TO-END CONTROL. English text phonemised as German,
     through the identical code path and scored by the identical metrics. If
     agreement and word-F1 do not FALL there, the comparison cannot fail and no
     "builtin agrees with espeak" result from this run means anything.
  2. DROP-COUNTER KNOWN-ANSWER CONTROL. After espeak is purged, "Test 123." at
     -l ru takes the raw-ASCII path; zonos's inventory has letters and
     punctuation but NO DIGITS, so cp_dropped must be >= 3 and the histogram
     must name U+0031..U+0033. A counter that reads 0 there is not measuring.
  3. LICENCE-GOAL ARM. espeak is physically removed and PROVEN gone (the same
     purge+verify as zonos-lang-435), then all four languages run again in
     builtin mode. That is the actual deliverable: does zonos work at all with
     no GPL dependency present? Russian in the same environment must still
     REFUSE -- the #435 guarantee must survive this change.

Nothing here flips a default. The default stays espeak-first; this run produces
the evidence that would justify moving it, per language, or not.
"""
import json, os, re, shutil, socket, subprocess, sys, time, unicodedata
from pathlib import Path

WORK = Path("/kaggle/working"); SCRATCH = Path("/tmp")
CLONE = SCRATCH / "CrispASR"
SCRIPT_VERSION = "2026-09-14-zonos-g2p-435-4"

# Short on purpose: zonos generates on CPU here and runtime scales with the
# audio length. ~10 words is enough for a word-F1 to mean something.
TEXTS = {
    "en": "The quick brown fox jumps over the lazy dog today.",
    "de": "Der schnelle braune Fuchs springt heute über den faulen Hund.",
    "fr": "Le rapide renard brun saute par dessus le chien paresseux.",
    # ACCENTED. v2 ran this line without its accents and the Spanish result was
    # not a measurement of the G2P: Spanish stress is carried by the written
    # accent, so "rapido" denies the phonemizer the only cue it has and the
    # built-in emitted `rapiðo` with no stress mark at all. A fixture that
    # withholds the input feature under test measures the fixture.
    "es": "El rápido zorro marrón salta sobre el perro perezoso hoy.",
    "ru": "Привет, это тест синтеза речи.",
}

# Spanish gets THREE sentences, the others one. Not arbitrary: en/de/fr produced
# byte-identical numbers across two independent runs, so one sentence is already
# reproducible for them. Spanish did not -- between run 2 and run 3 the espeak
# arm went 0.700 -> 0.111 and the built-in arm 0.421 -> 0.632 from an accent
# change alone, on a fixed seed. At N=1 the Spanish roundtrip is a coin, and a
# coin cannot decide a default. These three carry the features Spanish G2P
# actually has to get right: written accents, rr, and ñ.
ES_MULTI = [
    "El rápido zorro marrón salta sobre el perro perezoso hoy.",
    "Mañana por la tarde vamos a comprar pan y leche fresca.",
    "La niña pequeña corre por el parque con su perro blanco.",
]

def log(m):
    print(m, flush=True)
    try: (WORK/"progress.txt").open("a").write(f"{time.strftime('%H:%M:%S')} {m}\n")
    except Exception: pass

# ── 0. THE INSTRUMENT, AND ITS SELF-TEST ────────────────────────────────────
# Written and checked BEFORE any model is downloaded. Guards authored after the
# fact have never been observed to work.

def lev(a, b):
    if not a: return len(b)
    if not b: return len(a)
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]

def lev_sim(a, b):
    """1.0 identical, 0.0 disjoint. TWO EMPTY SEQUENCES ARE 0.0, NOT 1.0 --
    'both produced nothing' is the #435 failure, and it must never score as
    perfect agreement."""
    if not a or not b: return 0.0
    return max(0.0, 1.0 - lev(a, b) / max(len(a), len(b)))

def positional_match(a, b):
    """Fraction of positions that agree, over the LONGER sequence, so a length
    mismatch cannot be hidden. Empty -> 0.0 for the same reason as above."""
    if not a or not b: return 0.0
    n = min(len(a), len(b))
    return sum(1 for i in range(n) if a[i] == b[i]) / max(len(a), len(b))

_WORD = re.compile(r"[^\w]+", re.UNICODE)

def norm_words(s):
    s = unicodedata.normalize("NFKC", s or "").lower()
    return [w for w in _WORD.split(s) if w]

def word_f1(ref, hyp):
    """Token F1 with multiplicity. Empty on either side -> 0.0: an ASR that
    returned nothing must not tie with a perfect transcript."""
    r, h = norm_words(ref), norm_words(hyp)
    if not r or not h: return 0.0
    from collections import Counter
    inter = sum((Counter(r) & Counter(h)).values())
    if inter == 0: return 0.0
    p, rc = inter / len(h), inter / len(r)
    return 2 * p * rc / (p + rc)

def word_f1_micro(pairs):
    """Micro-average over several (ref, hyp) pairs: pool the counts, then one
    P/R/F1. NOT the mean of per-sentence F1s -- that lets one short sentence
    outvote a long one, and it hides a single catastrophic arm inside an
    average. Empty on either side still contributes 0 matches, so a collapsed
    sentence drags the pooled score down instead of being skipped."""
    from collections import Counter
    inter = nref = nhyp = 0
    for ref, hyp in pairs:
        r, h = norm_words(ref), norm_words(hyp)
        inter += sum((Counter(r) & Counter(h)).values())
        nref += len(r); nhyp += len(h)
    if not nref or not nhyp or inter == 0:
        return 0.0
    p, rc = inter / nhyp, inter / nref
    return 2 * p * rc / (p + rc)

def self_test():
    """Prove each metric distinguishes the two states it must distinguish."""
    checks = [
        ("lev_sim identical",        lev_sim([1,2,3],[1,2,3]), 1.0),
        ("lev_sim disjoint",         lev_sim([1,2,3],[7,8,9]), 0.0),
        ("lev_sim one substitution", lev_sim([1,2,3],[1,9,3]), 2/3),
        ("lev_sim BOTH EMPTY",       lev_sim([],[]),           0.0),
        ("lev_sim one empty",        lev_sim([1,2],[]),        0.0),
        ("pos identical",            positional_match([1,2,3],[1,2,3]), 1.0),
        ("pos disjoint",             positional_match([1,2,3],[7,8,9]), 0.0),
        ("pos length mismatch",      positional_match([1,2,3],[1,2,3,4]), 0.75),
        ("pos BOTH EMPTY",           positional_match([],[]),  0.0),
        ("f1 identical",             word_f1("the quick brown fox","The quick brown fox!"), 1.0),
        ("f1 disjoint",              word_f1("the quick brown fox","zzz yyy xxx"), 0.0),
        ("f1 half",                  word_f1("a b c d","a b x y"), 0.5),
        ("f1 BOTH EMPTY",            word_f1("",""),           0.0),
        ("f1 empty hyp",             word_f1("a b c",""),      0.0),
        ("micro identical",          word_f1_micro([("a b","a b"),("c d","c d")]), 1.0),
        ("micro disjoint",           word_f1_micro([("a b","x y"),("c d","z w")]), 0.0),
        # One perfect sentence must NOT hide one collapsed sentence: pooled
        # 2 matches over 4 ref and 2 hyp tokens -> P=1.0, R=0.5, F1=2/3.
        ("micro one arm collapsed",  word_f1_micro([("a b","a b"),("c d","")]),    2/3),
        ("micro all empty",          word_f1_micro([("",""),("","")]),             0.0),
    ]
    bad = [(n, got, want) for n, got, want in checks if abs(got - want) > 1e-9]
    for n, got, want in checks:
        log(f"[selftest] {n}: {got:.4f} (want {want:.4f}) {'OK' if abs(got-want)<=1e-9 else 'FAIL'}")
    return bad

_bad = self_test()
if _bad:
    (WORK/"results.json").write_text(json.dumps(
        {"script_version": SCRIPT_VERSION, "conclusive": False,
         "reason": "metric self-test failed; refusing to report numbers from a broken instrument",
         "failures": [{"check": n, "got": g, "want": w} for n, g, w in _bad]}, indent=2))
    log("[selftest] FAILED — aborting before any measurement")
    raise SystemExit(1)
log("[selftest] all metrics distinguish good from bad, including the empty-vs-empty case")

# ── 1. build ────────────────────────────────────────────────────────────────
if not CLONE.exists():
    subprocess.check_call(["git","clone","--depth","1","--recurse-submodules",
                           "--shallow-submodules","-b","feat/zonos-builtin-g2p",
                           "https://github.com/CrispStrobe/CrispASR.git",str(CLONE)])
sys.path.insert(0, str(CLONE/"tools"/"kaggle"))
import kaggle_harness as kh  # noqa: E402
kh.init_progress()
sha = subprocess.run(["git","-C",str(CLONE),"rev-parse","--short","HEAD"],
                     capture_output=True,text=True).stdout.strip()
log(f"[g2p] script_version={SCRIPT_VERSION} clone={sha}")
HF_TOKEN = kh.resolve_hf_token(); os.environ.setdefault("HF_TOKEN", HF_TOKEN or "")

res = {"script_version": SCRIPT_VERSION, "clone": sha, "langs": {}, "controls": {}}
def save():
    (WORK/"results.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))

kh.install_build_toolchain()
BUILD = SCRATCH/"build"
# BUILD_SHARED_LIBS=OFF on purpose: phonemizer.cpp moved from the kokoro target
# into crispasr-core, and the bug that arrangement originally fixed (#316 AUR)
# only shows up in a STATIC link, where library order decides whether
# crispasr_cache.o survives long enough for phonemizer.o to reference it.
r = subprocess.run(["cmake","-S",str(CLONE),"-B",str(BUILD),"-G","Ninja",
                    "-DCMAKE_BUILD_TYPE=Release","-DBUILD_SHARED_LIBS=OFF"]+kh.cache_and_link_flags(),
                   capture_output=True,text=True)
if r.returncode != 0:
    log("configure FAILED"); log((r.stdout or "")[-3000:]); log((r.stderr or "")[-3000:]); raise SystemExit(1)
with kh.build_heartbeat("build.crispasr"):
    r = subprocess.run(f"cmake --build {BUILD} --target crispasr -j{kh.safe_build_jobs(gpu=False)}",
                       shell=True, capture_output=True, text=True)
if r.returncode != 0:
    log(f"build FAILED rc={r.returncode}")
    log((r.stdout or "<empty>")[-6000:]); log((r.stderr or "<empty>")[-6000:]); raise SystemExit(1)
CRISPASR = BUILD/"bin"/"crispasr"
if not CRISPASR.is_file():
    log("build claimed success but produced no binary"); raise SystemExit(1)
# Proof-of-work, not an exit code: the linker's own final line.
link_lines = [l for l in (r.stdout or "").splitlines() if "Linking" in l and "crispasr" in l]
log(f"[g2p] static link OK: {link_lines[-1] if link_lines else '(ccache/no-op build)'}")
res["static_link_ok"] = True
save()

# crispasr-diff is the target the #316 AUR report actually failed to link.
# Building it is the specific regression check for the phonemizer move.
with kh.build_heartbeat("build.crispasr-diff"):
    rd = subprocess.run(f"cmake --build {BUILD} --target crispasr-diff -j{kh.safe_build_jobs(gpu=False)}",
                        shell=True, capture_output=True, text=True)
res["crispasr_diff_static_link_ok"] = (rd.returncode == 0 and (BUILD/"bin"/"crispasr-diff").is_file())
log(f"[g2p] crispasr-diff static link: {res['crispasr_diff_static_link_ok']}")
if not res["crispasr_diff_static_link_ok"]:
    log((rd.stdout or "")[-4000:]); log((rd.stderr or "")[-4000:])
save()

subprocess.run("apt-get install -y espeak-ng >/dev/null 2>&1 || true", shell=True)
have_espeak = shutil.which("espeak-ng") is not None
log(f"[g2p] espeak-ng present: {have_espeak}")
res["espeak_installed"] = have_espeak
if not have_espeak:
    res["conclusive"] = False
    res["reason"] = "espeak-ng unavailable; there is no control arm to compare the builtin against"
    save(); raise SystemExit(0)

from huggingface_hub import hf_hub_download
M  = hf_hub_download("cstr/zonos-v0.1-transformer-GGUF","zonos-v0.1-transformer-q8_0.gguf",local_dir=str(SCRATCH/"m"))
C  = hf_hub_download("cstr/dac-44khz-GGUF","dac-44khz-f16.gguf",local_dir=str(SCRATCH/"m"))
AS = hf_hub_download("cstr/parakeet-tdt-0.6b-v3-GGUF","parakeet-tdt-0.6b-v3-q4_k.gguf",local_dir=str(SCRATCH/"m"))
log(f"[g2p] model={M}\n[g2p] codec={C}\n[g2p] asr={AS}")

# ── 2. one synth, fully instrumented ────────────────────────────────────────
G2P_RE = re.compile(r"^zonos_g2p: (.*)$")

def parse_g2p(stderr):
    """Pull the instrumented readout out of the backend's own stderr.

    Keeps the WHOLE stream, not a tail: the readout is printed before synthesis
    and the generation logs are long enough to push it out of any window (the
    mistake zonos-lang-435 v1 made, which made every arm read lang='')."""
    out = {"path": "", "fields": {}, "ipa": "", "ids": [], "dropped": {}}
    for line in stderr.splitlines():
        m = G2P_RE.match(line.strip())
        if not m: continue
        body = m.group(1)
        if body.startswith("ipa="):
            out["ipa"] = body[4:]
        elif body.startswith("ids="):
            out["ids"] = [int(x) for x in body[4:].split(",") if x.strip()]
        elif body.startswith("dropped="):
            d = {}
            for part in body[8:].split(","):
                if ":" in part:
                    k, v = part.split(":", 1); d[k] = int(v)
            out["dropped"] = d
        else:
            for kv in body.split():
                if "=" in kv:
                    k, v = kv.split("=", 1)
                    out["fields"][k] = v
            out["path"] = out["fields"].get("path", out["path"])
    return out

def synth(text, lang, mode, out_path, tag):
    env = dict(os.environ)
    env["CRISPASR_ZONOS_G2P"] = mode
    env["CRISPASR_ZONOS_G2P_DEBUG"] = "1"
    if out_path.exists(): out_path.unlink()
    cmd = [str(CRISPASR),"--backend","zonos","-m",M,"--codec-model",C,
           "-l",lang,"--seed","42","--tts",text,"--tts-output",str(out_path)]
    t0 = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=3600)
    err = p.stderr or ""
    g = parse_g2p(err)
    rec = {"tag": tag, "mode": mode, "lang": lang, "rc": p.returncode,
           "wav_bytes": out_path.stat().st_size if out_path.exists() else 0,
           "secs": round(time.time() - t0, 1),
           "g2p_path": g["path"], "ipa": g["ipa"], "ids": g["ids"],
           "dropped_hist": g["dropped"],
           "cp_total": int(g["fields"].get("cp_total", -1)),
           "cp_mapped": int(g["fields"].get("cp_mapped", -1)),
           "cp_dropped": int(g["fields"].get("cp_dropped", -1)),
           "resolved_lang": g["fields"].get("lang", ""),
           "refused": ("Refusing to synthesise noise" in err or "no phoneme tokens" in err),
           # Which dictionaries actually loaded. Without this, an EN arm that
           # silently fell back to letter-to-sound rules would be reported as
           # "the builtin", and the number would be about a different thing.
           "g2p_dicts": [l.strip() for l in err.splitlines() if l.strip().startswith("g2p: ")],
           "stderr_tail": err[-800:]}
    rec["drop_rate"] = (rec["cp_dropped"] / rec["cp_total"]) if rec["cp_total"] > 0 else None
    log(f"[g2p] {tag}: rc={rec['rc']} path={rec['g2p_path']} ntok={len(rec['ids'])} "
        f"drop={rec['cp_dropped']}/{rec['cp_total']} wav={rec['wav_bytes']} {rec['secs']}s")
    # WHICH DICTIONARY, in the log. In v2 this landed only in results.json and
    # the English arm ran with NO dictionary at all -- letter-to-sound rules
    # phonemising "quick brown" as "cook bone" -- which was invisible until the
    # json was pulled afterwards. A builtin arm running on the LTS fallback is
    # not measuring the built-in G2P, so the log has to say so while the run is
    # still readable.
    if rec["g2p_path"] == "builtin":
        log(f"[g2p]    dicts: {rec['g2p_dicts'] or 'NONE — letter-to-sound rules only'}")
        log(f"[g2p]    ipa: {rec['ipa'][:160]}")
    return rec

def asr(wav, lang):
    if not Path(wav).is_file() or Path(wav).stat().st_size < 1000:
        return ""
    p = subprocess.run([str(CRISPASR),"--backend","parakeet","-m",AS,"-f",str(wav),
                        "-l",lang,"-nt"], capture_output=True, text=True, timeout=1800)
    diag = ("crispasr","parakeet:","ggml","main:","system_info:","whisper")
    cand = [l.strip() for l in (p.stdout or "").splitlines()
            if l.strip() and not l.strip().lower().startswith(diag)]
    return cand[-1] if cand else ""

# ── 3. per-language arms, espeak present ────────────────────────────────────
for lang in ("en","de","fr","es"):
    text = TEXTS[lang]
    e = synth(text, lang, "espeak",  SCRATCH/f"{lang}-espeak.wav",  f"{lang}/espeak")
    b = synth(text, lang, "builtin", SCRATCH/f"{lang}-builtin.wav", f"{lang}/builtin")
    e["asr"] = asr(SCRATCH/f"{lang}-espeak.wav", lang)
    b["asr"] = asr(SCRATCH/f"{lang}-builtin.wav", lang)
    e["word_f1"] = word_f1(text, e["asr"])
    b["word_f1"] = word_f1(text, b["asr"])
    # An arm only counts if the path it was supposed to exercise is the path
    # that ran. A builtin arm that silently fell through to espeak would
    # otherwise report perfect agreement with espeak -- with itself.
    ok_paths = e["g2p_path"].startswith("espeak") and b["g2p_path"] == "builtin"
    res["langs"][lang] = {
        "text": text, "espeak": e, "builtin": b,
        "paths_as_intended": ok_paths,
        "ntok_espeak": len(e["ids"]), "ntok_builtin": len(b["ids"]),
        "positional_match": positional_match(e["ids"], b["ids"]),
        "lev_sim": lev_sim(e["ids"], b["ids"]),
        "drop_rate_espeak": e["drop_rate"], "drop_rate_builtin": b["drop_rate"],
        "word_f1_espeak": e["word_f1"], "word_f1_builtin": b["word_f1"],
    }
    L = res["langs"][lang]
    log(f"[g2p] == {lang}: pos={L['positional_match']:.3f} lev={L['lev_sim']:.3f} "
        f"drop e={L['drop_rate_espeak']} b={L['drop_rate_builtin']} "
        f"f1 e={L['word_f1_espeak']:.3f} b={L['word_f1_builtin']:.3f} intended={ok_paths}")
    log(f"[g2p]    espeak  asr: {e['asr'][:110]!r}")
    log(f"[g2p]    builtin asr: {b['asr'][:110]!r}")
    save()

# ── 3b. Spanish, three sentences, because one was not enough ────────────────
# The single-sentence Spanish result was not reproducible (see ES_MULTI). Pool
# the tokens across three sentences and score once, so no single arm decides.
es_pairs = {"espeak": [], "builtin": []}
es_arms = []
for i, text in enumerate(ES_MULTI):
    for mode in ("espeak", "builtin"):
        a = synth(text, "es", mode, SCRATCH/f"es{i}-{mode}.wav", f"es[{i}]/{mode}")
        a["asr"] = asr(SCRATCH/f"es{i}-{mode}.wav", "es")
        a["ref"] = text
        a["word_f1_sentence"] = word_f1(text, a["asr"])
        es_pairs[mode].append((text, a["asr"]))
        es_arms.append(a)
        log(f"[g2p]    es[{i}]/{mode} f1={a['word_f1_sentence']:.3f} asr={a['asr'][:90]!r}")
        save()
es_ids_ok = all(x["g2p_path"].startswith("espeak") for x in es_arms if x["mode"] == "espeak") and \
            all(x["g2p_path"] == "builtin" for x in es_arms if x["mode"] == "builtin")
res["es_multi"] = {
    "sentences": ES_MULTI,
    "paths_as_intended": es_ids_ok,
    "word_f1_micro_espeak": word_f1_micro(es_pairs["espeak"]),
    "word_f1_micro_builtin": word_f1_micro(es_pairs["builtin"]),
    # es_arms is flat and interleaved (sentence0/espeak, sentence0/builtin, ...),
    # so the SENTENCE index is k//2 -- labelling it with the flat index would
    # report sentence 2 as sentence 5.
    "per_sentence": [{"sentence": k // 2, "mode": a["mode"], "f1": a["word_f1_sentence"],
                      "ref": a["ref"], "asr": a["asr"], "ipa": a["ipa"]}
                     for k, a in enumerate(es_arms)],
    "lev_sim_per_sentence": [
        lev_sim([x for x in es_arms[2*i]["ids"]], [x for x in es_arms[2*i+1]["ids"]])
        for i in range(len(ES_MULTI))],
}
EM = res["es_multi"]
log(f"[g2p] == es MULTI({len(ES_MULTI)} sentences): micro-f1 espeak={EM['word_f1_micro_espeak']:.3f} "
    f"builtin={EM['word_f1_micro_builtin']:.3f} lev_per_sentence="
    + ",".join(f"{v:.3f}" for v in EM["lev_sim_per_sentence"]))
save()

# Russian: no built-in covers it. Recorded so the report can say what espeak is
# still load-bearing for, rather than implying the licence goal is complete.
ru = synth(TEXTS["ru"], "ru", "builtin", SCRATCH/"ru-builtin-espeakpresent.wav", "ru/builtin-with-espeak")
res["langs"]["ru"] = {"text": TEXTS["ru"], "builtin_mode": ru,
                      "note": "no built-in G2P for ru; builtin mode must fall through to espeak"}
res["controls"]["ru_falls_through_to_espeak"] = ru["g2p_path"].startswith("espeak")
save()

# ── 4. CONTROL 1: wrong-language, end to end ────────────────────────────────
# English text phonemised as German. Same binary, same metrics. If this does not
# score WORSE than en/en, the comparison above cannot fail and means nothing.
wl = synth(TEXTS["en"], "de", "espeak", SCRATCH/"control-en-as-de.wav", "control/en-text-as-de")
wl["asr"] = asr(SCRATCH/"control-en-as-de.wav", "en")
wl["word_f1"] = word_f1(TEXTS["en"], wl["asr"])
en_ids = res["langs"]["en"]["espeak"]["ids"]
res["controls"]["wrong_language"] = {
    "arm": wl,
    "positional_match_vs_en_espeak": positional_match(en_ids, wl["ids"]),
    "lev_sim_vs_en_espeak": lev_sim(en_ids, wl["ids"]),
    "word_f1": wl["word_f1"],
    "word_f1_en_espeak": res["langs"]["en"]["word_f1_espeak"],
}
c = res["controls"]["wrong_language"]
log(f"[g2p] CONTROL wrong-language: pos={c['positional_match_vs_en_espeak']:.3f} "
    f"lev={c['lev_sim_vs_en_espeak']:.3f} f1={c['word_f1']:.3f} "
    f"(en/en f1 was {c['word_f1_en_espeak']:.3f})")
save()

# ── 5. purge espeak, then the licence-goal arms ─────────────────────────────
def remove_espeak():
    subprocess.run("apt-get remove -y --purge espeak-ng espeak-ng-data libespeak-ng1 "
                   ">/dev/null 2>&1 || true", shell=True)
    subprocess.run("for f in $(find /usr/lib /usr/local/lib /lib -name 'libespeak*' 2>/dev/null); "
                   "do mv \"$f\" \"$f.hidden\" 2>/dev/null || true; done", shell=True)
    leftover = subprocess.run("find /usr/lib /usr/local/lib /lib -name 'libespeak*' "
                              "! -name '*.hidden' 2>/dev/null", shell=True,
                              capture_output=True, text=True).stdout.strip()
    on_path = shutil.which("espeak-ng") or shutil.which("espeak")
    if on_path or leftover:
        return False, f"espeak still reachable (binary={on_path!r}, libs={leftover!r})"
    return True, ""

gone, why = remove_espeak()
res["espeak_removed"] = gone; res["espeak_removed_note"] = why
log(f"[g2p] espeak removed: {gone} {why}")
save()

if gone:
    # THE DELIVERABLE: zonos with no GPL dependency present at all.
    for lang in ("en","de","fr","es"):
        text = TEXTS[lang]
        a = synth(text, lang, "builtin", SCRATCH/f"{lang}-noespeak.wav", f"{lang}/builtin-no-espeak")
        a["asr"] = asr(SCRATCH/f"{lang}-noespeak.wav", lang)
        a["word_f1"] = word_f1(text, a["asr"])
        base = res["langs"][lang]["builtin"]["ids"]
        res["langs"][lang]["no_espeak"] = {
            "arm": a, "word_f1": a["word_f1"],
            # Must be identical to the with-espeak builtin arm: same G2P, same
            # input. A difference means espeak was leaking into the builtin path.
            "ids_identical_to_builtin_arm": a["ids"] == base,
        }
        log(f"[g2p] == {lang} NO-ESPEAK: path={a['g2p_path']} wav={a['wav_bytes']} "
            f"f1={a['word_f1']:.3f} ids_same={a['ids']==base}")
        save()

    # THE ONE BEHAVIOURAL DELTA of the `auto` default: on a box with no espeak,
    # English used to take the raw-ASCII path (letters tokenised as if they were
    # IPA -- zonos's inventory does contain A-Za-z, so it produces sound). `auto`
    # now puts the built-in in front of it. Measure that displaced path directly,
    # so the change is justified by a number and not by "IPA must beat letters".
    asc = synth(TEXTS["en"], "en", "espeak", SCRATCH/"en-ascii.wav", "en/ascii-no-espeak")
    asc["asr"] = asr(SCRATCH/"en-ascii.wav", "en")
    asc["word_f1"] = word_f1(TEXTS["en"], asc["asr"])
    res["langs"]["en"]["ascii_no_espeak"] = {
        "arm": asc, "word_f1": asc["word_f1"],
        "path_was_ascii": asc["g2p_path"] == "ascii",
    }
    log(f"[g2p] == en ASCII-FALLBACK (what `auto` now displaces): path={asc['g2p_path']} "
        f"f1={asc['word_f1']:.3f} asr={asc['asr'][:90]!r}")
    save()

    # The #435 guarantee must survive: no phonemizer for ru => refuse, 0 bytes.
    ru2 = synth(TEXTS["ru"], "ru", "builtin", SCRATCH/"ru-noespeak.wav", "ru/no-espeak")
    res["controls"]["ru_refuses_without_espeak"] = bool(ru2["refused"]) and ru2["wav_bytes"] == 0
    res["langs"]["ru"]["no_espeak"] = ru2
    log(f"[g2p] ru without espeak: refused={ru2['refused']} wav={ru2['wav_bytes']} rc={ru2['rc']}")

    # ── CONTROL 2: the drop counter's known answer ──────────────────────────
    # "Test 123." at -l ru: no espeak, no builtin for ru, text is pure ASCII ->
    # the raw-ASCII path. zonos's inventory has letters and punctuation but NO
    # DIGITS, so exactly the three digits must be counted as dropped.
    dc = synth("Test 123.", "ru", "espeak", SCRATCH/"control-drop.wav", "control/drop-counter")
    digits_seen = {k for k in dc["dropped_hist"] if k in ("U+0031","U+0032","U+0033")}
    res["controls"]["drop_counter"] = {
        "arm": dc, "path": dc["g2p_path"], "cp_dropped": dc["cp_dropped"],
        "digit_codepoints_named": sorted(digits_seen),
        "fires": dc["g2p_path"] == "ascii" and dc["cp_dropped"] >= 3 and len(digits_seen) == 3,
    }
    log(f"[g2p] CONTROL drop-counter: path={dc['g2p_path']} dropped={dc['cp_dropped']} "
        f"hist={dc['dropped_hist']} fires={res['controls']['drop_counter']['fires']}")
    save()
else:
    log("[g2p] espeak could not be removed — the licence-goal arms did NOT run. "
        "Reported as inconclusive, NOT as failures.")

# ── 6. verdict ──────────────────────────────────────────────────────────────
# Per language, and deliberately not a single boolean: the answer is expected to
# differ by language, and "builtin is fine for es, worse for en" is the useful
# result, not a pass/fail.
ctl = res["controls"]
wlc = ctl.get("wrong_language", {})
controls_fire = {
    # The wrong-language arm must score materially WORSE on both the ID
    # agreement and the roundtrip than the matched-language arm did.
    "wrong_language_lowers_agreement": wlc.get("lev_sim_vs_en_espeak", 1.0) < 0.75,
    "wrong_language_lowers_word_f1":
        wlc.get("word_f1", 1.0) < max(0.0, wlc.get("word_f1_en_espeak", 0.0) - 0.15),
    "drop_counter_fires": bool(ctl.get("drop_counter", {}).get("fires")),
    "ru_still_refuses_without_espeak": bool(ctl.get("ru_refuses_without_espeak")),
}
res["controls_fire"] = controls_fire
# If the controls did not fire, the per-language numbers are not evidence.
res["conclusive"] = bool(gone) and all(controls_fire.values())

per_lang = {}
for lang in ("en","de","fr","es"):
    L = res["langs"].get(lang, {})
    if not L: continue
    ne = L.get("no_espeak", {})
    per_lang[lang] = {
        "paths_as_intended": L.get("paths_as_intended"),
        "lev_sim": round(L.get("lev_sim", 0.0), 4),
        "positional_match": round(L.get("positional_match", 0.0), 4),
        "drop_rate_espeak": L.get("drop_rate_espeak"),
        "drop_rate_builtin": L.get("drop_rate_builtin"),
        "word_f1_espeak": round(L.get("word_f1_espeak", 0.0), 4),
        "word_f1_builtin": round(L.get("word_f1_builtin", 0.0), 4),
        "word_f1_builtin_no_espeak": round(ne.get("word_f1", 0.0), 4) if ne else None,
        "word_f1_ascii_no_espeak":
            round(L["ascii_no_espeak"]["word_f1"], 4) if L.get("ascii_no_espeak") else None,
        # The proposal, stated as a claim that the numbers above support or not.
        # Deliberately conservative: the builtin must not lose more than 0.10 of
        # word-F1 and must not drop a larger share of its own symbols.
        "builtin_dicts_loaded": L.get("builtin", {}).get("g2p_dicts", []),
        "builtin_ran_on_lts_only_no_dict": not L.get("builtin", {}).get("g2p_dicts"),
        # A comparison needs a BASELINE that itself works. Run 3 scored Spanish
        # "good enough to default" purely because the espeak control arm had
        # collapsed to 0.111 -- the built-in's 0.632 cleared `espeak - 0.10`
        # while being the WORST built-in score of the four languages. A
        # predicate that passes for that reason is not measuring what it is
        # named for, so a control arm below this floor makes the language
        # INCONCLUSIVE rather than a pass.
        "espeak_baseline_usable": L.get("word_f1_espeak", 0.0) >= 0.60,
        "builtin_good_enough_to_default":
            bool(L.get("paths_as_intended")
                 and L.get("builtin", {}).get("g2p_dicts")
                 and L.get("word_f1_espeak", 0.0) >= 0.60
                 and L.get("word_f1_builtin", 0.0) >= L.get("word_f1_espeak", 0.0) - 0.10
                 and (L.get("drop_rate_builtin") or 0.0) <= (L.get("drop_rate_espeak") or 0.0) + 0.02),
    }
# Spanish is judged on the POOLED three-sentence arm; the single-sentence entry
# stays visible so the instability that motivated it is still on the record.
EM = res.get("es_multi")
if EM and "es" in per_lang:
    per_lang["es"]["single_sentence_word_f1_espeak"] = per_lang["es"]["word_f1_espeak"]
    per_lang["es"]["single_sentence_word_f1_builtin"] = per_lang["es"]["word_f1_builtin"]
    per_lang["es"]["word_f1_espeak"] = round(EM["word_f1_micro_espeak"], 4)
    per_lang["es"]["word_f1_builtin"] = round(EM["word_f1_micro_builtin"], 4)
    per_lang["es"]["n_sentences"] = len(EM["sentences"])
    per_lang["es"]["espeak_baseline_usable"] = EM["word_f1_micro_espeak"] >= 0.60
    per_lang["es"]["builtin_good_enough_to_default"] = bool(
        EM["paths_as_intended"]
        and res["langs"]["es"]["builtin"].get("g2p_dicts")
        and EM["word_f1_micro_espeak"] >= 0.60
        and EM["word_f1_micro_builtin"] >= EM["word_f1_micro_espeak"] - 0.10)

res["per_language_verdict"] = per_lang
en = res["langs"].get("en", {})
if en.get("ascii_no_espeak") and en.get("no_espeak"):
    res["auto_fallback_justified_for_en"] = {
        "builtin_word_f1": en["no_espeak"]["word_f1"],
        "ascii_word_f1": en["ascii_no_espeak"]["word_f1"],
        "ascii_path_confirmed": en["ascii_no_espeak"]["path_was_ascii"],
        # The `auto` default only ever displaces the ASCII path, so this is the
        # only comparison that can justify or condemn it.
        "builtin_beats_ascii":
            en["no_espeak"]["word_f1"] > en["ascii_no_espeak"]["word_f1"],
    }
    log("[g2p] AUTO-FALLBACK " + json.dumps(res["auto_fallback_justified_for_en"]))
save()
log("[g2p] CONTROLS " + json.dumps(controls_fire))
log("[g2p] PER-LANGUAGE " + json.dumps(per_lang, ensure_ascii=False))
if not res["conclusive"]:
    log("[g2p] INCONCLUSIVE: a control did not fire, or espeak could not be removed. "
        "The per-language numbers above are NOT evidence in that state.")
