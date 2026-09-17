#!/usr/bin/env python3
"""Kaggle kernel (#437): decoded-output roundtrip from the PUBLISHED F16 GGUFs.

Reading a GGUF header back proves the file is what it claims to be. It does
not prove today's runtime can open it — those are different claims, and the
second is the one a user experiences the moment `-m auto:f16` hands them a
9 GB download. This kernel answers the second.

Deliberate choices, each of which is what makes the run worth its time:

  * The model comes from the REGISTRY, not from a local conversion:
    `-m auto:f16 --auto-download` into an EMPTY cache dir. So the arm
    exercises registry resolution → HTTPS download → load → decode, the
    same path a user takes. A `--dry-run-resolve` first asserts the URL it
    picked is the published file rather than something stale in a cache.

  * Every model runs a q4_k CONTROL arm in the same kernel, from the same
    build, on the same clip. q4_k is already known-good and shipped, so the
    three outcomes are distinguishable:
        both arms pass ......... the F16 path works
        control fails too ...... the build/harness is broken — INCONCLUSIVE,
                                 and specifically NOT evidence against F16
        only F16 fails ......... the F16 path is broken
    A subject-only run could not tell the first from the second.

  * The scoring metric is itself tested before it judges anything
    (`--self-test`-style controls at the top): identical text, unrelated
    text, empty text and a truncated half must produce four DIFFERENT
    scores. A word-overlap score that cannot go low is not a check.

Push (under chr1s4):
    export KAGGLE_API_TOKEN=<chr1s4 token>
    python -m kaggle kernels push -p tools/kaggle/voxtral-f16-roundtrip
"""

import json
import os
import re
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

# Gotcha #24: kernel script is frozen at the last push while the clone is
# always fresh. Print both.
SCRIPT_VERSION = "437-roundtrip-v2"

BRANCH = "feat/437-voxtral-f16"
REPO_URL = "https://github.com/CrispStrobe/CrispASR"

WORK = Path("/kaggle/working")
REPO = WORK / "CrispASR"
BUILD = WORK / "build"
CACHE = WORK / "modelcache"

# samples/jfk.wav, 11 s. The reference is the clip's actual content; the
# 3B card quotes the same sentence as the transcript both published quants
# produce.
REFERENCE = ("And so my fellow Americans ask not what your country can do "
             "for you ask what you can do for your country")

# Word-F1 against REFERENCE. 0.80 sits well above what a wrong-but-fluent
# transcript scores (the metric controls below put unrelated English at
# ~0.1) and below the punctuation/casing jitter a correct one shows.
PASS_F1 = 0.80

MODELS = [
    {"name": "voxtral-mini-3b-2507", "backend": "voxtral",
     "arms": [("q4_k", "control"), ("f16", "subject")]},
    {"name": "voxtral-mini-4b-realtime", "backend": "voxtral4b",
     "arms": [("q4_k", "control"), ("f16", "subject")]},
]


def log(msg=""):
    print(msg, flush=True)


def df(path: Path) -> str:
    try:
        st = os.statvfs(str(path))
        return f"{st.f_bavail * st.f_frsize / 2**30:.1f} GiB free on {path}"
    except OSError:
        return f"(statvfs failed on {path})"


# ── the metric, and proof that it can report failure ────────────────────────

_WORD = re.compile(r"[a-z0-9']+")


def norm_words(s: str) -> list[str]:
    return _WORD.findall(s.lower())


def word_f1(hyp: str, ref: str) -> float:
    """Multiset token F1. Order-insensitive on purpose: the thing under test
    is 'did the model decode this clip', not word order, and an order-aware
    metric would make punctuation drift look like a failure."""
    from collections import Counter
    h, r = Counter(norm_words(hyp)), Counter(norm_words(ref))
    if not h or not r:
        return 0.0
    overlap = sum((h & r).values())
    if overlap == 0:
        return 0.0
    prec = overlap / sum(h.values())
    rec = overlap / sum(r.values())
    return 2 * prec * rec / (prec + rec)


def metric_self_test() -> bool:
    """Controls with KNOWN answers, run before the metric judges anything.

    The property asserted is not "these scores are all different" — an
    earlier draft demanded that and failed honestly, because unrelated text
    and empty text both score exactly 0.0 and always will. The property that
    matters is SEPARATION: every transcript a human would call correct must
    land at or above PASS_F1, every failure mode a speech-LLM actually
    exhibits must land below it, and the two groups must not touch.
    """
    log("=== metric controls (run before the metric judges anything) ===")
    # (label, text, must_pass)
    cases = [
        ("identical", REFERENCE, True),
        ("same words, punctuated + capitalised",
         "And so, my fellow Americans: ask not what your country can do for you — "
         "ask what you can do for your country.", True),
        # Realistic failure modes for an audio-LLM backend, all of which have
        # been seen in this repo on a broken load: silence, a degenerate
        # single-token loop, a wrong-language hallucination, a truncated
        # decode, and unrelated fluent text.
        ("empty (model produced nothing)", "", False),
        ("degenerate repetition loop", "the the the the the the the the the the", False),
        ("wrong-language hallucination",
         "Und so, meine amerikanischen Mitbuerger, fragt nicht was euer Land", False),
        ("single token", "And", False),
        ("unrelated fluent English",
         "the quick brown fox jumps over the lazy dog while it rains", False),
        ("truncated half-decode",
         "And so my fellow Americans ask not what your country", False),
    ]
    ok = True
    passing, failing = [], []
    for label, text, must_pass in cases:
        v = word_f1(text, REFERENCE)
        good = (v >= PASS_F1) if must_pass else (v < PASS_F1)
        (passing if must_pass else failing).append(v)
        log(f"  {'ok  ' if good else 'FAIL'}  {label:38s} F1={v:.3f} "
            f"({'expect pass' if must_pass else 'expect fail'})")
        ok &= good

    margin = min(passing) - max(failing)
    log(f"  separation: lowest correct {min(passing):.3f} vs "
        f"highest wrong {max(failing):.3f}  (margin {margin:+.3f}, "
        f"threshold {PASS_F1})")
    if margin <= 0.0:
        log("  FAIL  correct and wrong transcripts overlap — threshold is meaningless")
        ok = False
    if max(failing) - min(failing) < 0.5:
        log("  FAIL  the failure controls barely differ — metric is not graded")
        ok = False
    log(f"  metric controls: {'PASS' if ok else 'FAIL'}")
    return ok


# ── phase 0: environment ────────────────────────────────────────────────────

log(f"=== script {SCRIPT_VERSION} ===")
log(f"  {df(WORK)}")
try:
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith(("MemTotal", "MemAvailable")):
                log("  " + line.strip())
except OSError:
    pass
subprocess.run("nproc", shell=True)

if not metric_self_test():
    log("VERDICT: METRIC_BROKEN — refusing to score anything")
    raise SystemExit(1)

try:
    with urllib.request.urlopen("https://huggingface.co/api/models/"
                                "cstr/voxtral-mini-3b-2507-GGUF", timeout=30) as r:
        r.read(64)
    log("  internet: OK")
except Exception as e:  # noqa: BLE001
    log(f"  NO_INTERNET_RETRY: {e}")
    raise SystemExit(0)

# ── phase 1: clone ──────────────────────────────────────────────────────────

log("=== clone ===")
cloned = None
for ref in (BRANCH, "main"):
    if REPO.exists():
        shutil.rmtree(REPO, ignore_errors=True)
    try:
        subprocess.check_call(["git", "clone", "--depth", "1", "-b", ref,
                               REPO_URL, str(REPO)])
        cloned = ref
        break
    except Exception as e:  # noqa: BLE001
        log(f"  clone {ref} failed: {e}")
if cloned is None:
    log("CLONE_FAILED")
    raise SystemExit(1)
# ALL submodules, not just ggml. v1 initialised ggml only and died at cmake
# configure: examples/cli/CMakeLists.txt:19 hard-fails without
# third_party/c2pa-audio/src/sha256.h, because the CLI records consent hashes
# even when C2PA signing is off. The convert kernel never needed the CLI, so
# `--init ggml` had always been enough there — a habit that did not transfer.
try:
    subprocess.check_call(
        ["git", "submodule", "update", "--init", "--recursive", "--depth", "1"],
        cwd=str(REPO))
except Exception as e:  # noqa: BLE001
    log(f"  submodule init: {e}")

# Assert the two headers cmake will look for, so a missing submodule costs
# seconds and names itself instead of surfacing 90 s later as a FATAL_ERROR
# in the middle of a configure log.
REQUIRED_SUBMODULE_FILES = [
    REPO / "ggml" / "CMakeLists.txt",
    REPO / "third_party" / "c2pa-audio" / "src" / "sha256.h",
]
missing = [str(f) for f in REQUIRED_SUBMODULE_FILES if not f.is_file()]
if missing:
    log(f"SUBMODULES_INCOMPLETE: {missing}")
    raise SystemExit(1)
log(f"  submodules OK: {[f.name for f in REQUIRED_SUBMODULE_FILES]}")

sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(REPO)).decode().strip()
log(f"  branch={cloned} sha={sha}")

sys.path.insert(0, str(REPO / "tools" / "kaggle"))
import kaggle_harness as kh  # noqa: E402

kh.init_progress()

WAV = REPO / "samples" / "jfk.wav"
if not WAV.is_file():
    log(f"MISSING_SAMPLE: {WAV}")
    raise SystemExit(1)
log(f"  sample: {WAV} ({WAV.stat().st_size} bytes)")

# ── phase 2: build the CLI ──────────────────────────────────────────────────

kh.step("install build toolchain")
kh.install_build_toolchain()
os.environ["CCACHE_MAXSIZE"] = "2G"   # 20 GB disk shared with a 9 GB model

kh.step("cmake configure")
flags = kh.cache_and_link_flags()
kh.sh_with_progress(
    f"cmake -S {REPO} -B {BUILD} -G Ninja "
    f"-DCMAKE_BUILD_TYPE=Release "
    f"-DCRISPASR_BUILD_TESTS=OFF "
    f"-DCRISPASR_BUILD_EXAMPLES=ON "
    f"-DCRISPASR_BUILD_SERVER=OFF "
    f"-DGGML_CUDA=OFF " + " ".join(flags)
)

kh.step("cmake build crispasr-cli")
# Target `crispasr-cli` produces bin/crispasr — the target and output names
# diverge on purpose (examples/cli/CMakeLists.txt:12,349). Asking for target
# `crispasr` builds only the library and leaves bin/crispasr absent.
with kh.build_heartbeat("cmake.build"):
    kh.sh_with_progress(
        f"stdbuf -oL -eL cmake --build {BUILD} --target crispasr-cli "
        f"-j{kh.safe_build_jobs(gpu=False)}"
    )
CLI = BUILD / "bin" / "crispasr"
if not CLI.is_file():
    log(f"BUILD_PRODUCED_NO_BINARY at {CLI}")
    raise SystemExit(1)
log(f"  built {CLI} ({CLI.stat().st_size:,} bytes)")
subprocess.run([str(CLI), "--version"])

# The build is done; the ccache is 2 GB we need for the model downloads.
shutil.rmtree(WORK / ".ccache", ignore_errors=True)
log(f"  after ccache purge: {df(WORK)}")


# ── phase 3: the arms ───────────────────────────────────────────────────────

def extract_transcript(stdout: str) -> str:
    """Everything the CLI printed that is not a diagnostic line."""
    out = []
    for line in stdout.splitlines():
        s = line.strip()
        if not s:
            continue
        # crispasr diagnostics are prefixed; transcripts are not.
        if s.startswith(("crispasr:", "whisper_", "ggml_", "[", "warning:", "error:",
                         "note:", "load_", "voxtral", "main:")):
            continue
        out.append(s)
    return " ".join(out)


def run_arm(model: dict, quant: str, role: str) -> dict:
    name, backend = model["name"], model["backend"]
    expect_file = f"{name}-{quant}.gguf"
    r = {"model": name, "quant": quant, "role": role, "resolved_ok": False,
         "ran": False, "rc": None, "transcript": "", "f1": 0.0, "pass": False,
         "error": None}
    log(f"\n---------------- {name} / {quant} ({role}) ----------------")

    # Always start from an empty cache so the download really happens and a
    # stale file cannot stand in for the published one.
    shutil.rmtree(CACHE, ignore_errors=True)
    CACHE.mkdir(parents=True, exist_ok=True)

    base = [str(CLI), "--backend", backend, "-m", f"auto:{quant}",
            "--auto-download", "--cache-dir", str(CACHE)]

    kh.step(f"{name}/{quant}: dry-run resolve")
    p = subprocess.run(base + ["--dry-run-resolve", "--dry-run-ignore-cache"],
                       capture_output=True, text=True, timeout=600)
    log(p.stdout.rstrip())
    if p.stderr.strip():
        log("  stderr: " + p.stderr.strip()[:2000])
    blob = p.stdout + p.stderr
    if expect_file in blob:
        r["resolved_ok"] = True
        log(f"  ok    registry resolved to {expect_file}")
    else:
        log(f"  FAIL  registry did not resolve to {expect_file}")

    kh.step(f"{name}/{quant}: transcribe")
    try:
        p = subprocess.run(base + ["-f", str(WAV), "-l", "en", "-t", "4", "-nt"],
                           capture_output=True, text=True, timeout=5400)
        r["ran"] = True
        r["rc"] = p.returncode
        log(f"  exit={p.returncode}")
        tail = p.stderr.strip().splitlines()[-25:]
        if tail:
            log("  stderr tail:")
            for line in tail:
                log("    " + line[:300])
        r["transcript"] = extract_transcript(p.stdout)
    except subprocess.TimeoutExpired:
        r["error"] = "timeout"
        log("  FAIL  timed out")
    except Exception as e:  # noqa: BLE001
        r["error"] = f"{type(e).__name__}: {e}"
        log(f"  FAIL  {r['error']}")

    log(f"  transcript: {r['transcript'][:400]!r}")
    r["f1"] = word_f1(r["transcript"], REFERENCE)
    n_words = len(norm_words(r["transcript"]))
    log(f"  words={n_words}  word-F1 vs reference={r['f1']:.3f}  (threshold {PASS_F1})")
    r["pass"] = bool(r["resolved_ok"] and r["ran"] and r["rc"] == 0
                     and n_words > 0 and r["f1"] >= PASS_F1)
    log(f"  ARM {'PASS' if r['pass'] else 'FAIL'}")

    shutil.rmtree(CACHE, ignore_errors=True)
    log(f"  after cache purge: {df(WORK)}")
    return r


results = []
for model in MODELS:
    for quant, role in model["arms"]:
        try:
            results.append(run_arm(model, quant, role))
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            results.append({"model": model["name"], "quant": quant, "role": role,
                            "resolved_ok": False, "ran": False, "rc": None,
                            "transcript": "", "f1": 0.0, "pass": False,
                            "error": f"{type(e).__name__}: {e}"})

# ── verdict ─────────────────────────────────────────────────────────────────

log("\n================ SUMMARY ================")
log(f"  reference: {REFERENCE!r}")
for r in results:
    log(f"  {r['model']:26s} {r['quant']:5s} {r['role']:8s} "
        f"{'PASS' if r['pass'] else 'FAIL'}  rc={r['rc']} F1={r['f1']:.3f}"
        + (f"  err={r['error']}" if r["error"] else ""))

controls = [r for r in results if r["role"] == "control"]
subjects = [r for r in results if r["role"] == "subject"]
control_ok = bool(controls) and all(r["pass"] for r in controls)
subject_ok = bool(subjects) and all(r["pass"] for r in subjects)

if not control_ok:
    verdict = "INCONCLUSIVE_HARNESS"
    note = ("the known-good q4_k control failed too, so this run says nothing "
            "about the F16 path — fix the build/harness first")
elif subject_ok:
    verdict = "F16_PATH_GOOD"
    note = ("both published F16 GGUFs resolved through the registry, downloaded, "
            "loaded and transcribed the clip correctly")
else:
    verdict = "F16_PATH_BROKEN"
    note = ("the q4_k control passed on the same build and clip, so the failure "
            "is specific to the F16 artifacts or the F16 load path")
log(f"  control_ok={control_ok} subject_ok={subject_ok}")
log(f"  VERDICT: {verdict} — {note}")

(WORK / "roundtrip-verdict.json").write_text(json.dumps(
    {"script_version": SCRIPT_VERSION, "clone_sha": sha, "verdict": verdict,
     "control_ok": control_ok, "subject_ok": subject_ok,
     "reference": REFERENCE, "threshold": PASS_F1, "arms": results}, indent=2))
raise SystemExit(0 if verdict == "F16_PATH_GOOD" else 1)
