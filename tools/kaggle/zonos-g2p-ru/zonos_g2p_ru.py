#!/usr/bin/env python3
"""Russian G2P without espeak-ng: is the new built-in good enough to default to?

Direct continuation of chr1s4/crispasr-zonos-g2p-435, which answered the same
question for en/de/fr/es and left Russian as the one language zonos still
needed espeak-ng (GPL-3.0) for. crispasr-core now ships a Russian G2P:
a 812,953-entry IPA dictionary with lexical stress already resolved
(bene-ges/ru_g2p_ipa_bert_large, CC-BY-4.0) in front of letter-to-sound rules.

This is a NEW kernel slug rather than a re-push of zonos-g2p-435 on purpose:
re-pushing a kernel makes the previous run's log unreachable, and that run is
the evidence behind the en/de/fr defaults currently shipping. The METHOD is
inherited unchanged — same metrics, same self-test, same control structure —
because the method is the part worth reusing.

THE RISK THIS KERNEL EXISTS TO MEASURE
--------------------------------------
zonos's phoneme map comes from its own conditioning.py symbol list, and
text_to_phoneme_ids() SILENTLY DROPS every codepoint outside it. That silent
drop IS #435 -- Cyrillic became three tokens at a success exit code. The
Russian dictionary's stress marker upstream is a BACKTICK (U+0060), which is in
no TTS inventory anywhere and accounts for 7.57% of all symbols in the file; it
was rewritten to U+02C8 when the dictionary was published. If that rewrite had
been missed, the built-in path would have produced audio, returned 0, and
quietly lost every stress mark. So "the builtin produced a wav" proves nothing
and this kernel scores the PAYLOAD:

  * the phoneme-ID sequence from each path, compared position-by-position and
    by token-level Levenshtein similarity;
  * cp_dropped/cp_total per path -- how much of what the G2P emitted zonos could
    not map. A path that drops more is VISIBLY worse, not silently worse;
  * an ASR roundtrip (parakeet-tdt-0.6b-v3) word-F1 against the input text,
    micro-averaged over THREE Russian sentences, because a single sentence was
    already shown not to be reproducible when Spanish was scored this way.

CONTROLS -- a readout that renders the same for good and bad input is worthless
------------------------------------------------------------------------------
  0. INSTRUMENT SELF-TEST, before anything heavy. Every metric is run on
     known-answer and DEGENERATE inputs (identical -> 1.0, disjoint -> 0.0, two
     EMPTY sequences -> 0.0 and NOT 1.0, which is the exact shape of the #435
     failure). If any of them is wrong the kernel exits before spending a
     session producing numbers from a broken ruler.
  1. WRONG-LANGUAGE END-TO-END CONTROL. The Russian text phonemised as German,
     through the identical code path and scored by the identical metrics. If
     agreement and word-F1 do not FALL there, the comparison cannot fail.
  2. DROP-COUNTER, TWO KNOWN ANSWERS.
     (a) generic: "Test 123." at -l ja after espeak is purged takes the
         raw-ASCII path; zonos's inventory has letters but NO DIGITS, so
         cp_dropped must be >= 3 and the histogram must name U+0031..U+0033.
     (b) SPECIFIC TO THIS CHANGE: a one-line Russian dictionary is injected via
         CRISPASR_RU_DICT_PATH whose IPA still contains the upstream BACKTICK.
         The builtin ru path must then report cp_dropped >= 1 naming U+0060.
         Without (b), "the builtin ru path dropped 0 codepoints" is a claim the
         instrument has never been shown capable of contradicting ON THAT PATH.
  3. LICENCE-GOAL ARM. espeak is physically removed and PROVEN gone, then
     Russian runs again in builtin mode and must produce the IDENTICAL ID
     sequence. A language with NO built-in (ja) must still REFUSE in that
     state -- the #435 guarantee has to survive Russian gaining a G2P.
  4. BASELINE SANITY. If the espeak arm's own word-F1 is below 0.60 the
     comparison says more about zonos's Russian than about the G2P, and the
     verdict is INCONCLUSIVE rather than a pass. Spanish was once passed for
     exactly that wrong reason.

A NOTE ON SENSITIVITY, so the two instruments are not weighted equally: in the
en/de/fr/es run the wrong-language control moved word-F1 by only 0.20 while
moving ID agreement to 0.494. The roundtrip is the BLUNTER instrument. Gaps
under ~0.2 of word-F1 are not evidence of anything.

WHAT v1 FOUND, AND WHY v2 HAS A THIRD ARM
-----------------------------------------
v1 (saved as zonos-g2p-ru-v1.log; re-pushing destroys the previous log, so it
was pulled first) came back with every control firing except one, and one clear
finding: the built-in path drops ZERO codepoints, proven by a control that
fires on that exact path. But the roundtrip favoured espeak, 0.585 to 0.293,
and the espeak baseline itself was under the 0.60 gate, so it was INCONCLUSIVE.

Running espeak's ru voice over 2,200 dictionary words locally said why. The two
G2Ps describe the same sounds in DIFFERENT SYMBOLS — raw agreement 57.7%, not
one word identical:

  ours   məɫɐkˈo i xlʲep lʲɪʐˈat na stɐlʲˈe v bɐlʲʂˈoj kˈomnətʲe
  espeak mʌɭʌkˈo ɪ xɭʲˈep ɭʲiʒˈɑt nə stʌɭʲˈe v bʌɭʃˈoj kˈomnʌtʲi

Every symbol on BOTH sides is inside zonos's inventory, so the drop counter is
blind to this by construction — it is the #316 problem (Kokoro trained on
misaki's spelling) one language further on. g2p_ru::to_espeak_dialect now
rewrites the former into the latter, taking agreement to 88.0%, and
CRISPASR_ZONOS_RU_DIALECT=espeak selects it. Whether that helps the AUDIO is
what this run is for; the third arm is the test.

TWO FIXES TO THE INSTRUMENT ITSELF, from v1:
  * the wrong-language F1 leg compared against ru sentence 0, whose espeak
    baseline was itself 0.000 — so it could not fire no matter what the control
    produced. It now compares against the espeak MICRO F1 over all three
    sentences. (Its agreement leg fired decisively either way: lev 0.063.)
  * per-sentence espeak IPA is now logged, not just the builtin's, so the two
    spellings can be read off the log instead of inferred.

Nothing here flips a default. This run produces the evidence that would justify
moving Russian to builtin-first, or not, and the same for the dialect switch.
"""
import json, os, re, shutil, socket, subprocess, sys, time, unicodedata
from pathlib import Path

WORK = Path("/kaggle/working"); SCRATCH = Path("/tmp")
CLONE = SCRATCH / "CrispASR"
SCRIPT_VERSION = "2026-09-15-zonos-g2p-ru-2"

# Short on purpose: zonos generates on CPU here and runtime scales with the
# audio length. ~10 words is enough for a word-F1 to mean something.
TEXTS = {
    # The en arm is an ANCHOR, not the subject. It is already settled
    # (agreement 0.95, F1 1.00 both paths), so it says whether THIS run's
    # harness is behaving before any Russian number is read.
    "en": "The quick brown fox jumps over the lazy dog today.",
    "ru": "Привет, это тест синтеза речи.",
}

# Russian gets THREE sentences for the same reason Spanish did: at N=1 the
# roundtrip is a coin, and a coin cannot decide a default. These carry the
# features Russian G2P has to get right and that the dictionary alone does not
# settle:
#   [0] ordinary prose containing `тест`, which is genuinely ABSENT from the
#       vocabulary (it is on the heteronym list), so this sentence exercises the
#       letter-to-sound path inside an otherwise dictionary-covered sentence;
#   [1] ъ, ё and щ, plus consonant clusters — ё is the one stress signal the
#       input can carry for free, and the dictionary's own keys fold it away;
#   [2] unstressed-vowel reduction over several syllables: `молоко` is
#       [məɫɐkˈo], the same letter as three different vowels, chosen entirely
#       by distance from the stress.
RU_MULTI = [
    "Привет, это тест синтеза речи.",
    "Съешь ещё этих мягких французских булок.",
    "Молоко и хлеб лежат на столе в большой комнате.",
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
                           "--shallow-submodules","-b","feat/g2p-russian",
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

def synth(text, lang, mode, out_path, tag, extra_env=None):
    env = dict(os.environ)
    env["CRISPASR_ZONOS_G2P"] = mode
    env["CRISPASR_ZONOS_G2P_DEBUG"] = "1"
    # Used by the backtick control, which has to reach the Russian DICTIONARY
    # loader specifically -- no other knob can put a known-bad symbol on the
    # builtin ru path.
    if extra_env:
        env.update(extra_env)
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
           # Which input words the upstream project flagged as genuinely
           # ambiguous. These are exactly the words with NO dictionary reading
           # (the heteronym list and the vocabulary are disjoint), so they are
           # the words the letter-to-sound rules had to invent a reading for.
           # Recorded so the limitation can be reported with names in it rather
           # than as a hand-wave.
           "heteronyms_hit": [l.strip() for l in err.splitlines() if l.strip().startswith("g2p_ru: ")],
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


# ── 3. the anchor arm (en), so the harness is known good before ru is read ──
for lang in ("en",):
    text = TEXTS[lang]
    e = synth(text, lang, "espeak",  SCRATCH/f"{lang}-espeak.wav",  f"{lang}/espeak")
    b = synth(text, lang, "builtin", SCRATCH/f"{lang}-builtin.wav", f"{lang}/builtin")
    e["asr"] = asr(SCRATCH/f"{lang}-espeak.wav", lang)
    b["asr"] = asr(SCRATCH/f"{lang}-builtin.wav", lang)
    e["word_f1"] = word_f1(text, e["asr"])
    b["word_f1"] = word_f1(text, b["asr"])
    ok_paths = e["g2p_path"].startswith("espeak") and b["g2p_path"] == "builtin"
    res["langs"][lang] = {
        "text": text, "espeak": e, "builtin": b,
        "paths_as_intended": ok_paths,
        "positional_match": positional_match(e["ids"], b["ids"]),
        "lev_sim": lev_sim(e["ids"], b["ids"]),
        "drop_rate_espeak": e["drop_rate"], "drop_rate_builtin": b["drop_rate"],
        "word_f1_espeak": e["word_f1"], "word_f1_builtin": b["word_f1"],
    }
    L = res["langs"][lang]
    log(f"[g2p] == ANCHOR {lang}: pos={L['positional_match']:.3f} lev={L['lev_sim']:.3f} "
        f"f1 e={L['word_f1_espeak']:.3f} b={L['word_f1_builtin']:.3f} intended={ok_paths}")
    save()

# ── 4. Russian, three sentences, both paths ─────────────────────────────────
# Pool the ASR tokens across the three sentences and score ONCE (micro-average),
# so no single sentence decides. Per-sentence ID agreement is kept separately:
# it is the sharper instrument and it does not need pooling.
# Ask the Russian G2P to name the words it KNOWS it cannot resolve. It costs one
# extra download and turns "heteronyms are a limitation" into a list of the
# specific words in these specific sentences that took the rule path.
RU_ENV = {"CRISPASR_G2P_RU_HETERONYM_WARN": "1"}
# THREE arms, not two. The third is the built-in G2P with its output rewritten
# into espeak's Russian spelling — same sounds, the symbols the model was
# trained on. Naming them explicitly (rather than by the CRISPASR_ZONOS_G2P mode
# alone) because two of them run in the SAME mode and differ only by the dialect
# switch, and an arm that cannot be told apart in the log is not an arm.
ARMS = [
    ("espeak",  "espeak",  {}),
    ("builtin", "builtin", {}),
    ("builtin-espeak-dialect", "builtin", {"CRISPASR_ZONOS_RU_DIALECT": "espeak"}),
]
ru_pairs = {name: [] for name, _, _ in ARMS}
ru_arms = []
for i, text in enumerate(RU_MULTI):
    for name, mode, env in ARMS:
        a = synth(text, "ru", mode, SCRATCH/f"ru{i}-{name}.wav", f"ru[{i}]/{name}",
                  extra_env={**RU_ENV, **env})
        a["arm"] = name
        a["asr"] = asr(SCRATCH/f"ru{i}-{name}.wav", "ru")
        a["ref"] = text
        a["word_f1_sentence"] = word_f1(text, a["asr"])
        ru_pairs[name].append((text, a["asr"]))
        ru_arms.append(a)
        log(f"[g2p]    ru[{i}]/{name} path={a['g2p_path']} f1={a['word_f1_sentence']:.3f} "
            f"drop={a['cp_dropped']}/{a['cp_total']} asr={a['asr'][:90]!r}")
        # The IPA of EVERY arm, so the spellings can be compared in the log.
        log(f"[g2p]       ipa: {a['ipa'][:200]}")
        for h in a["heteronyms_hit"]:
            log(f"[g2p]       {h}")
        save()

# Which dictionary actually loaded. Without this an arm that fell back to the
# letter-to-sound rules would be reported as "the builtin", and the number would
# be about a different thing entirely. This is the trap the en arm fell into in
# an earlier run of the sibling kernel.
ru_builtin_arms = [a for a in ru_arms if a["arm"] == "builtin"]
ru_dicts = ru_builtin_arms[0]["g2p_dicts"] if ru_builtin_arms else []
ru_dict_loaded = any("Russian IPA dict" in d for d in ru_dicts)
log(f"[g2p] ru builtin dicts: {ru_dicts or 'NONE — letter-to-sound rules only'}")

NA = len(ARMS)
def arm_at(sent, name):
    """The arm for one sentence. ru_arms is flat and interleaved, so indexing it
    by the flat position would report sentence 2 as sentence 5 — the exact
    off-by-N the sibling kernel had to fix once already."""
    k = [n for n, _, _ in ARMS].index(name)
    return ru_arms[sent * NA + k]

res["ru"] = {
    "sentences": RU_MULTI,
    "paths_as_intended":
        all(a["g2p_path"].startswith("espeak") for a in ru_arms if a["arm"] == "espeak") and
        all(a["g2p_path"] == "builtin" for a in ru_arms if a["arm"] != "espeak"),
    "dict_loaded": ru_dict_loaded,
    "dict_lines": ru_dicts,
    "word_f1_micro": {name: word_f1_micro(ru_pairs[name]) for name, _, _ in ARMS},
    "word_f1_micro_espeak": word_f1_micro(ru_pairs["espeak"]),
    "word_f1_micro_builtin": word_f1_micro(ru_pairs["builtin"]),
    "word_f1_micro_builtin_espeak_dialect": word_f1_micro(ru_pairs["builtin-espeak-dialect"]),
    "per_sentence": [{"sentence": k // NA, "arm": a["arm"], "f1": a["word_f1_sentence"],
                      "path": a["g2p_path"], "ref": a["ref"], "asr": a["asr"], "ipa": a["ipa"],
                      "cp_total": a["cp_total"], "cp_dropped": a["cp_dropped"],
                      "heteronyms_hit": a["heteronyms_hit"],
                      "dropped_hist": a["dropped_hist"], "ntok": len(a["ids"])}
                     for k, a in enumerate(ru_arms)],
    # Agreement with the espeak arm, per sentence, for BOTH built-in spellings.
    # The dialect conversion is a claim that this number moves; here it either
    # does or it does not.
    "lev_sim_per_sentence": [lev_sim(arm_at(i, "espeak")["ids"], arm_at(i, "builtin")["ids"])
                             for i in range(len(RU_MULTI))],
    "positional_match_per_sentence": [positional_match(arm_at(i, "espeak")["ids"],
                                                       arm_at(i, "builtin")["ids"])
                                      for i in range(len(RU_MULTI))],
    "lev_sim_dialect_per_sentence": [
        lev_sim(arm_at(i, "espeak")["ids"], arm_at(i, "builtin-espeak-dialect")["ids"])
        for i in range(len(RU_MULTI))],
    "positional_match_dialect_per_sentence": [
        positional_match(arm_at(i, "espeak")["ids"], arm_at(i, "builtin-espeak-dialect")["ids"])
        for i in range(len(RU_MULTI))],
    # THE headline drop numbers. The builtin ru dictionary was published with its
    # upstream backtick stress marker rewritten to U+02C8 precisely so this
    # number is 0; if the rewrite had been missed it would be ~7.6% and the
    # audio would still have sounded like speech.
    "cp_dropped": {name: sum(a["cp_dropped"] for a in ru_arms if a["arm"] == name)
                   for name, _, _ in ARMS},
    "cp_total": {name: sum(a["cp_total"] for a in ru_arms if a["arm"] == name)
                 for name, _, _ in ARMS},
    "cp_dropped_espeak": sum(a["cp_dropped"] for a in ru_arms if a["arm"] == "espeak"),
    "cp_total_espeak": sum(a["cp_total"] for a in ru_arms if a["arm"] == "espeak"),
    "cp_dropped_builtin": sum(a["cp_dropped"] for a in ru_arms if a["arm"] == "builtin"),
    "cp_total_builtin": sum(a["cp_total"] for a in ru_arms if a["arm"] == "builtin"),
}
R = res["ru"]
R["drop_rate_espeak"] = R["cp_dropped_espeak"] / R["cp_total_espeak"] if R["cp_total_espeak"] > 0 else None
R["drop_rate_builtin"] = R["cp_dropped_builtin"] / R["cp_total_builtin"] if R["cp_total_builtin"] > 0 else None
R["drop_rate"] = {name: (R["cp_dropped"][name] / R["cp_total"][name] if R["cp_total"][name] else None)
                  for name, _, _ in ARMS}
log(f"[g2p] == ru({len(RU_MULTI)} sentences) micro-f1: " +
    "  ".join(f"{n}={R['word_f1_micro'][n]:.3f}" for n, _, _ in ARMS))
log("[g2p]    lev vs espeak  native : " + ",".join(f"{v:.3f}" for v in R["lev_sim_per_sentence"]))
log("[g2p]    lev vs espeak  dialect: " + ",".join(f"{v:.3f}" for v in R["lev_sim_dialect_per_sentence"]))
log("[g2p]    drop rate: " + "  ".join(f"{n}={R['drop_rate'][n]}" for n, _, _ in ARMS) +
    f"  dict_loaded={ru_dict_loaded}")
save()

# ── 5. CONTROL 1: wrong-language, end to end ────────────────────────────────
# The Russian text phonemised as GERMAN. Same binary, same metrics. If this does
# not score WORSE than ru/ru, the comparison above cannot fail and means nothing.
wl = synth(RU_MULTI[0], "de", "espeak", SCRATCH/"control-ru-as-de.wav", "control/ru-text-as-de")
wl["asr"] = asr(SCRATCH/"control-ru-as-de.wav", "ru")
wl["word_f1"] = word_f1(RU_MULTI[0], wl["asr"])
ru0_espeak_ids = arm_at(0, "espeak")["ids"]
res["controls"]["wrong_language"] = {
    "arm": wl,
    "positional_match_vs_ru_espeak": positional_match(ru0_espeak_ids, wl["ids"]),
    "lev_sim_vs_ru_espeak": lev_sim(ru0_espeak_ids, wl["ids"]),
    "word_f1": wl["word_f1"],
    # v1 compared this against SENTENCE 0, whose espeak baseline was itself
    # 0.000 — so the leg could not fire whatever the control produced. A control
    # that cannot fail is not a control. The reference is now the pooled espeak
    # score over all three sentences.
    "word_f1_ru_espeak_micro": R["word_f1_micro_espeak"],
    "word_f1_ru_espeak_sentence0": arm_at(0, "espeak")["word_f1_sentence"],
}
c = res["controls"]["wrong_language"]
log(f"[g2p] CONTROL wrong-language: pos={c['positional_match_vs_ru_espeak']:.3f} "
    f"lev={c['lev_sim_vs_ru_espeak']:.3f} f1={c['word_f1']:.3f} "
    f"(ru/ru espeak micro-f1 was {c['word_f1_ru_espeak_micro']:.3f})")
save()

# ── 6. CONTROL 2b: the drop counter, ON THE BUILTIN RU PATH ─────────────────
# The generic digit control (2a, below) exercises the ASCII path. It says
# nothing about whether the counter can fire on the path this kernel is
# actually judging. So: inject a one-line Russian dictionary whose IPA still
# carries the UPSTREAM BACKTICK (U+0060) -- the exact symbol the published
# dictionary rewrote to U+02C8 -- and require the counter to name it.
#
# Without this arm, "the builtin ru path dropped 0 codepoints" is a negative
# that the instrument has never been shown capable of contradicting on that
# path, and a negative that cannot fire is not a measurement.
BAD_DICT = SCRATCH/"ru-backtick.tsv"
BAD_DICT.write_text("word\tipa\n" "привет\t/prʲɨvʲ`et/\n", encoding="utf-8")
bt = synth("Привет", "ru", "builtin", SCRATCH/"control-backtick.wav",
           "control/backtick-in-ru-dict", extra_env={"CRISPASR_RU_DICT_PATH": str(BAD_DICT)})
res["controls"]["backtick_drop_on_builtin_ru"] = {
    "arm": bt, "path": bt["g2p_path"], "ipa": bt["ipa"],
    "cp_dropped": bt["cp_dropped"], "dropped_hist": bt["dropped_hist"],
    "fires": bt["g2p_path"] == "builtin" and bt["cp_dropped"] >= 1 and "U+0060" in bt["dropped_hist"],
}
bc = res["controls"]["backtick_drop_on_builtin_ru"]
log(f"[g2p] CONTROL backtick-on-builtin-ru: path={bc['path']} ipa={bc['ipa']!r} "
    f"dropped={bc['cp_dropped']} hist={bc['dropped_hist']} fires={bc['fires']}")
save()

# ── 7. purge espeak, then the licence-goal arms ─────────────────────────────
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
    # THE DELIVERABLE: Russian through zonos with no GPL dependency present.
    # Both built-in spellings, so the licence-goal arm covers whichever one the
    # verdict ends up recommending.
    ne = {}
    for name, _, env in ARMS[1:]:
        pairs = []
        arms = []
        for i, text in enumerate(RU_MULTI):
            a = synth(text, "ru", "builtin", SCRATCH/f"ru{i}-noespeak-{name}.wav",
                      f"ru[{i}]/{name}-no-espeak", extra_env={**RU_ENV, **env})
            a["asr"] = asr(SCRATCH/f"ru{i}-noespeak-{name}.wav", "ru")
            a["word_f1_sentence"] = word_f1(text, a["asr"])
            pairs.append((text, a["asr"]))
            arms.append(a)
            log(f"[g2p] == ru[{i}] NO-ESPEAK {name}: path={a['g2p_path']} wav={a['wav_bytes']} "
                f"f1={a['word_f1_sentence']:.3f} ids_same={a['ids']==arm_at(i, name)['ids']}")
            save()
        ne[name] = {
            "word_f1_micro": word_f1_micro(pairs),
            # Must be identical to the with-espeak arm of the same name: same
            # G2P, same input. A difference means espeak was leaking into the
            # built-in path.
            "ids_identical_to_with_espeak_arm":
                all(arms[i]["ids"] == arm_at(i, name)["ids"] for i in range(len(RU_MULTI))),
            "per_sentence": [{"sentence": i, "f1": arms[i]["word_f1_sentence"],
                              "path": arms[i]["g2p_path"], "asr": arms[i]["asr"],
                              "wav_bytes": arms[i]["wav_bytes"]} for i in range(len(RU_MULTI))],
        }
        log(f"[g2p] ru NO-ESPEAK {name}: micro-f1={ne[name]['word_f1_micro']:.3f} "
            f"ids_identical={ne[name]['ids_identical_to_with_espeak_arm']}")
        save()
    res["ru"]["no_espeak"] = ne

    # The #435 guarantee must SURVIVE Russian gaining a G2P: a language with no
    # built-in and non-ASCII text must still refuse rather than synthesise noise.
    ja = synth("これはテストです。", "ja", "builtin", SCRATCH/"ja-noespeak.wav", "ja/no-espeak")
    res["controls"]["no_builtin_language_still_refuses"] = bool(ja["refused"]) and ja["wav_bytes"] == 0
    res["controls"]["ja_arm"] = ja
    log(f"[g2p] ja without espeak: refused={ja['refused']} wav={ja['wav_bytes']} rc={ja['rc']}")
    save()

    # ── CONTROL 2a: the drop counter's generic known answer ─────────────────
    # "Test 123." at -l ja: no espeak, no builtin for ja, text is pure ASCII ->
    # the raw-ASCII path. zonos's inventory has letters and punctuation but NO
    # DIGITS, so exactly the three digits must be counted as dropped.
    dc = synth("Test 123.", "ja", "espeak", SCRATCH/"control-drop.wav", "control/drop-counter")
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

# ── 8. verdict ──────────────────────────────────────────────────────────────
ctl = res["controls"]
wlc = ctl.get("wrong_language", {})
ru_f1_e = R["word_f1_micro_espeak"]
ru_f1_b = R["word_f1_micro_builtin"]
controls_fire = {
    # The wrong-language arm must score materially WORSE on the ID agreement
    # than the matched-language arm did. Agreement is the sharp instrument; the
    # roundtrip is checked too but with a looser threshold, because in the
    # sibling run the same control moved F1 by only 0.20.
    "wrong_language_lowers_agreement": wlc.get("lev_sim_vs_ru_espeak", 1.0) < 0.75,
    "wrong_language_lowers_word_f1":
        wlc.get("word_f1", 1.0) < max(0.0, wlc.get("word_f1_ru_espeak_micro", 0.0) - 0.15),
    "drop_counter_fires": bool(ctl.get("drop_counter", {}).get("fires")),
    "backtick_drop_fires_on_builtin_ru": bool(ctl.get("backtick_drop_on_builtin_ru", {}).get("fires")),
    "no_builtin_language_still_refuses": bool(ctl.get("no_builtin_language_still_refuses")),
    "ru_builtin_dictionary_actually_loaded": bool(R["dict_loaded"]),
    "ru_paths_as_intended": bool(R["paths_as_intended"]),
}
res["controls_fire"] = controls_fire
res["conclusive"] = bool(gone) and all(controls_fire.values())

# The proposal, stated as a claim the numbers support or do not. Deliberately
# conservative, and with the baseline gate that Spanish taught us to add: a
# comparison needs a control arm that itself WORKS. An espeak arm below 0.60
# makes the language INCONCLUSIVE, not a pass -- a built-in can otherwise clear
# "espeak - 0.10" by being merely less bad than a collapsed baseline.
res["ru_verdict"] = {
    "paths_as_intended": R["paths_as_intended"],
    "dict_loaded": R["dict_loaded"],
    "n_sentences": len(RU_MULTI),
    "lev_sim_per_sentence": R["lev_sim_per_sentence"],
    "positional_match_per_sentence": R["positional_match_per_sentence"],
    "drop_rate_espeak": R["drop_rate_espeak"],
    "drop_rate_builtin": R["drop_rate_builtin"],
    "word_f1_micro_espeak": ru_f1_e,
    "word_f1_micro_builtin": ru_f1_b,
    "word_f1_micro_builtin_no_espeak":
        res["ru"].get("no_espeak", {}).get("builtin", {}).get("word_f1_micro"),
    "word_f1_micro_builtin_espeak_dialect": R["word_f1_micro_builtin_espeak_dialect"],
    "lev_sim_dialect_per_sentence": R["lev_sim_dialect_per_sentence"],
    "positional_match_dialect_per_sentence": R["positional_match_dialect_per_sentence"],
    "drop_rate_builtin_espeak_dialect": R["drop_rate"]["builtin-espeak-dialect"],
    # The dialect switch is a claim about SPELLING, so it is judged on the
    # agreement metric first (where it either moves or does not) and on the
    # roundtrip second (which is the blunter instrument).
    "dialect_raises_agreement":
        (sum(R["lev_sim_dialect_per_sentence"]) > sum(R["lev_sim_per_sentence"])),
    "dialect_raises_word_f1":
        R["word_f1_micro_builtin_espeak_dialect"] > R["word_f1_micro_builtin"],
    "espeak_baseline_usable": ru_f1_e >= 0.60,
    # Judged on the BEST of the two built-in spellings: the question the flip
    # answers is "can zonos drop espeak for Russian", and either spelling
    # answers it if it holds up.
    "builtin_good_enough_to_default": bool(
        R["paths_as_intended"] and R["dict_loaded"]
        and ru_f1_e >= 0.60
        and max(ru_f1_b, R["word_f1_micro_builtin_espeak_dialect"]) >= ru_f1_e - 0.10
        and (R["drop_rate_builtin"] or 0.0) <= (R["drop_rate_espeak"] or 0.0) + 0.02),
    # Stated separately because it is TRUE even when the roundtrip is
    # inconclusive: zonos may simply not speak Russian well enough for an ASR
    # comparison to mean anything, and that is a fact about zonos, not about
    # the G2P. The drop rate is not affected by it.
    "builtin_loses_nothing_to_the_inventory": (R["drop_rate_builtin"] == 0.0),
}
res["anchor_en_ok"] = bool(res["langs"].get("en", {}).get("paths_as_intended"))
save()
log("[g2p] CONTROLS " + json.dumps(controls_fire))
log("[g2p] RU VERDICT " + json.dumps(res["ru_verdict"], ensure_ascii=False))
if not res["conclusive"]:
    log("[g2p] INCONCLUSIVE: a control did not fire, or espeak could not be removed. "
        "The numbers above are NOT evidence in that state.")
if res["ru_verdict"]["espeak_baseline_usable"] is False:
    log("[g2p] NOTE: the espeak ru baseline is itself below 0.60, so the ROUNDTRIP comparison "
        "says more about zonos's Russian than about the G2P. The ID-agreement and drop-rate "
        "numbers are unaffected and remain valid.")
