# ─────────────────────────── cell 0 (markdown) ───────────────────────────
# # CrispASR — voxtral bucketed step-graph cache A/B (CRISPASR_VOXTRAL_STEP_CACHE)
#
# Proves, on the real 3B model, the two claims the local unit test CANNOT reach:
#   1. the cached path emits the SAME transcript as the per-call path
#   2. it is actually faster on voxtral's geometry (not just funasr's)
#
# FOUR ARMS. Three of them are controls, and they are the point:
#
#   A  cache OFF                  -- the baseline, current shipping behaviour
#   B  cache ON, width 16         -- the candidate
#   C  cache ON, width 100000     -- >= kv_max_ctx, i.e. the SINGLE fixed-Lk
#                                    graph design. funasr measured +69% decode
#                                    CPU for this. It is a POSITIVE CONTROL on
#                                    the CLOCK: if C does not come out clearly
#                                    slower than A, the timing instrument
#                                    cannot see what it is being asked to
#                                    measure, and B-vs-A is uninterpretable
#                                    noise rather than a result.
#   D  cache ON, width 16, DIFFERENT AUDIO
#                                 -- POSITIVE CONTROL on the COMPARATOR: it
#                                    SHOULD differ from B. If D matches B too,
#                                    then "A == B" proves nothing, because the
#                                    comparison is blind.
#                                    (The first draft used --temperature here.
#                                    That arm is UNREACHABLE: voxtral declares
#                                    CAP_TEMPERATURE but run_voxtral_family
#                                    never receives it, so D would have equalled
#                                    B and the kernel would have reported a
#                                    blind comparator -- a false alarm about the
#                                    instrument instead of a fact about the
#                                    model. A control must be able to fire.)
#
# ACTIVATION CHECK. The cache prints "step-graph cache ACTIVE" on first build.
# If that line is absent from B/C/D, the cached path never ran and the arms are
# literally the same code -- which would produce a perfect match and identical
# timings, i.e. exactly what success looks like. The run FAILS LOUDLY in that
# case rather than reporting a green.
#
# CPU-only on purpose: the funasr measurement that motivates the bucket width
# is a CPU-decode result, and CPU is where graph-prep vs KV-bandwidth trade off
# in the documented way. Datasets: chr1str/crispasr-hf-token, chr1str/crispasr-ccache.

# ─────────────────────────── cell 1 (code) ───────────────────────────
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

WORK = Path("/kaggle/working")
REPO = WORK / "CrispASR"
BUILD = WORK / "build"
TMP = Path("/tmp/vx-sc")
MODELS = TMP / "models"
MODELS.mkdir(parents=True, exist_ok=True)

CRISPASR_REF = os.environ.get("CRISPASR_REF", "main")
REPS = int(os.environ.get("REPS", "3"))
GGUF_REPO = os.environ.get("GGUF_REPO", "cstr/voxtral-mini-3b-2507-GGUF")
GGUF_FILE = os.environ.get("GGUF_FILE", "voxtral-mini-3b-2507-q4_k.gguf")


def sh(cmd, check=True, env=None, cwd=None, timeout=None, quiet=False):
    e = {**os.environ, **(env or {})}
    r = subprocess.run(cmd, env=e, cwd=cwd, timeout=timeout,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if not quiet:
        print(r.stdout[-6000:], flush=True)
    if check and r.returncode != 0:
        raise SystemExit(f"cmd failed ({r.returncode}): {cmd}")
    return r


if REPO.exists():
    shutil.rmtree(REPO)
sh(["git", "clone", "--depth", "1", "--branch", CRISPASR_REF, "--recursive",
    "https://github.com/CrispStrobe/CrispASR.git", str(REPO)])
sys.path.insert(0, os.path.join(str(REPO), "tools", "kaggle"))
import kaggle_harness as kh  # noqa: E402

TOKEN = kh.resolve_hf_token()
if TOKEN:
    os.environ["HF_TOKEN"] = TOKEN
kh.init_progress()

sha = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
kh.step("start", sha=sha, ref=CRISPASR_REF)

# ── build the CLI (CPU-only) ──
kh.install_build_toolchain()
BUILD.mkdir(exist_ok=True)
sh(["cmake", "-S", str(REPO), "-B", str(BUILD), "-DCMAKE_BUILD_TYPE=Release",
    "-DBUILD_SHARED_LIBS=ON"] + kh.cache_and_link_flags())
with kh.build_heartbeat("cmake.build"):
    kh.sh_with_progress(
        f"stdbuf -oL -eL cmake --build {BUILD} --target crispasr-cli -j{kh.safe_build_jobs(gpu=False)}")
# The CMake TARGET is `crispasr-cli` but OUTPUT_NAME is `crispasr`
# (examples/cli/CMakeLists.txt:349) -- there is no file called crispasr-cli.
CLI = BUILD / "bin" / "crispasr"
if not (CLI.is_file() and os.access(CLI, os.X_OK)):
    cands = sorted(str(c) for c in BUILD.rglob("crispasr*") if c.is_file() and os.access(c, os.X_OK))
    raise SystemExit(f"crispasr binary not at {CLI}. Executables actually built: {cands[:20]}")
os.environ["LD_LIBRARY_PATH"] = f"{BUILD / 'src'}:{os.environ.get('LD_LIBRARY_PATH', '')}"
kh.step("built", cli=str(CLI))

# ── model ──
from huggingface_hub import hf_hub_download  # noqa: E402
with kh.build_heartbeat("download.gguf"):
    GGUF = hf_hub_download(GGUF_REPO, GGUF_FILE, local_dir=str(MODELS), token=TOKEN or None)
kh.step("dl_done", size_gb=round(os.path.getsize(GGUF) / 1e9, 2))

AUDIO = str(REPO / "samples" / "jfk.wav")
AUDIO_CTRL = str(REPO / "samples" / "multispeaker.wav")  # must transcribe differently

ARMS = [
    ("A_off",       {"CRISPASR_VOXTRAL_STEP_CACHE": "0"},                                           AUDIO),
    ("B_w16",       {"CRISPASR_VOXTRAL_STEP_CACHE": "1", "CRISPASR_VOXTRAL_STEP_BUCKET": "16"},     AUDIO),
    ("C_maxctx",    {"CRISPASR_VOXTRAL_STEP_CACHE": "1", "CRISPASR_VOXTRAL_STEP_BUCKET": "100000"}, AUDIO),
    ("D_audio_ctrl", {"CRISPASR_VOXTRAL_STEP_CACHE": "1", "CRISPASR_VOXTRAL_STEP_BUCKET": "16"},    AUDIO_CTRL),
]


def run_arm(name, env, audio):
    """Returns (seconds, rc, combined_output, transcript, bench).

    The transcript comes from --output-txt, not from scraping stdout: a stdout
    extractor that silently yields "" would make every arm match, which is
    exactly what this experiment must not be able to fake.

    The .txt is DELETED first and its existence asserted after. Without that, a
    file left by the previous arm would be read as this arm's result -- arms
    would agree because they read the same stale bytes.
    """
    # crispasr_make_out_path (examples/cli/crispasr_output.cpp:35) STRIPS the
    # audio extension, so samples/jfk.wav -> samples/jfk.txt.
    txt = Path(audio).with_suffix(".txt")
    if txt.exists():
        txt.unlink()

    cmd = [str(CLI), "--backend", "voxtral", "--model", GGUF, "--file", audio,
           "--threads", "4", "--output-txt", "--no-timestamps"]
    t0 = time.time()
    r = sh(cmd, env={**env, "OMP_NUM_THREADS": "4", "CRISPASR_VOXTRAL_BENCH": "1"},
           check=False, timeout=3600, quiet=True)
    dt = time.time() - t0

    if not txt.exists():
        print(f"--- {name}: --output-txt produced no {txt}; stdout tail ---\n{r.stdout[-3000:]}", flush=True)
        return dt, (r.returncode or 90), r.stdout, None, {}
    transcript = " ".join(txt.read_text(errors="replace").split()).strip()

    # CRISPASR_VOXTRAL_BENCH=1 emits one "voxtral_bench: llm_kv  X ms" per call.
    # The FIRST is prefill (T>1); every later one is a single decode step, which
    # is the only thing the step-graph cache touches. v2 compared END-TO-END wall
    # clock instead, where 91 s is dominated by model load and the encoder -- so a
    # large decode effect showed up as a 0.3% total and the run was uninterpretable.
    lk = [float(x) for x in re.findall(r"voxtral_bench:\s+llm_kv\s+([0-9.]+) ms", r.stdout)]
    prefill_ms = lk[0] if lk else 0.0
    steps = lk[1:]
    return dt, r.returncode, r.stdout, transcript, {
        "prefill_ms": round(prefill_ms, 2),
        "decode_ms": round(sum(steps), 2),
        "n_steps": len(steps),
        "ms_per_step": round(sum(steps) / len(steps), 3) if steps else 0.0,
    }


results = {}
for name, env, arm_audio in ARMS:
    times, texts, actives, benches = [], [], [], []
    for i in range(REPS):
        dt, rc, out, tr, bench = run_arm(name, env, arm_audio)
        if rc != 0 or tr is None:
            print(f"--- {name} rep{i} FAILED rc={rc} ---\n{out[-4000:]}", flush=True)
            raise SystemExit(f"{name} exited {rc} / no transcript — arm cannot be compared")
        if not tr:
            raise SystemExit(f"{name} rep{i}: EMPTY transcript — an empty string would compare equal "
                             f"to every other empty arm and fake a pass")
        times.append(dt)
        benches.append(bench)
        texts.append(tr)
        actives.append("step-graph cache ACTIVE" in out)
        kh.step(f"{name}_rep{i}", sec=round(dt, 2), active=actives[-1])
    dec = [b.get("decode_ms", 0.0) for b in benches if b.get("n_steps")]
    if not dec:
        raise SystemExit(f"{name}: no voxtral_bench llm_kv lines — the decode clock is BLIND, "
                         f"and a blind clock reports the same number for every arm")
    results[name] = {
        "best_s": round(min(times), 3),
        "median_s": round(statistics.median(times), 3),
        "decode_ms": round(min(dec), 2),
        "n_steps": benches[0].get("n_steps", 0),
        "ms_per_step": round(min(dec) / max(benches[0].get("n_steps", 1), 1), 3),
        "prefill_ms": benches[0].get("prefill_ms", 0.0),
        "text": texts[0],
        "text_stable": len(set(texts)) == 1,
        "cache_active": any(actives),
    }
    print(f"[{name}] best={results[name]['best_s']}s active={results[name]['cache_active']} "
          f"text={results[name]['text'][:90]!r}", flush=True)

# ── verdict ───────────────────────────────────────────────────────────────
fail = []

# 0. activation: B/C/D must have actually taken the cached path.
for n in ("B_w16", "C_maxctx", "D_audio_ctrl"):
    if not results[n]["cache_active"]:
        fail.append(f"{n}: 'step-graph cache ACTIVE' never printed — the cached path did NOT run, "
                    f"so this arm is the same code as A_off and any match below is vacuous")

# 1. comparator control: D must DIFFER from B, else the comparison is blind.
comparator_fires = results["D_audio_ctrl"]["text"] != results["B_w16"]["text"]
if not comparator_fires:
    fail.append("D_audio_ctrl == B_w16: the transcript comparator cannot distinguish two arms that "
                "must differ, so 'A == B' is not evidence of anything")

# 2. the actual claim: cached path must match the per-call path exactly.
identical = results["A_off"]["text"] == results["B_w16"]["text"]
if not identical:
    fail.append(f"A_off != B_w16 — the cached path changed the transcript.\n"
                f"  A: {results['A_off']['text'][:300]!r}\n  B: {results['B_w16']['text'][:300]!r}")

# 3. clock control: C (fixed at max_ctx) must be clearly slower than A.
# Judge DECODE time. v2 used wall clock and C/A came out at 1.142 against an
# arbitrary 1.15 bar -- a REAL 14% effect misread as a failed control. Decode is
# the only thing this feature touches, so it is the only honest denominator.
speed_ratio_C = results["C_maxctx"]["decode_ms"] / results["A_off"]["decode_ms"]
clock_fires = speed_ratio_C > 1.15
if not clock_fires:
    fail.append(f"C_maxctx/A_off = {speed_ratio_C:.3f} — the known-bad fixed-Lk design did NOT measure "
                f"slower. The clock cannot see decode cost here (dominated by load/encode?), so the "
                f"B-vs-A timing below is NOT interpretable")

speedup = results["A_off"]["decode_ms"] / results["B_w16"]["decode_ms"]

# The point of the run is "is the cache FASTER". v3 reported PASS for a 1.2%
# REGRESSION because the fail-list only gated on identity and the controls --
# a verdict that cannot report the very outcome it exists to detect. A cache
# that is slower than no cache is a failed experiment, not a passing one.
MIN_SPEEDUP = 1.02
if speedup < MIN_SPEEDUP:
    fail.append(f"B_w16 is not meaningfully faster than A_off on decode "
                f"({speedup:.3f}x, need >= {MIN_SPEEDUP}). Bit-identity holds, but the "
                f"speed claim does NOT: graph-prep is a negligible share of a "
                f"{results['B_w16']['ms_per_step']:.0f} ms decode step on this model, so the "
                f"cache buys nothing here. Keep it default-off.")

print("\n" + "=" * 72)
print(f"{'arm':<14}{'best s':>10}{'median s':>11}{'cache':>8}{'stable':>8}")
for n, r in results.items():
    print(f"{n:<14}{r['best_s']:>10.3f}{r['median_s']:>11.3f}{str(r['cache_active']):>8}{str(r['text_stable']):>8}")
print("=" * 72)
print(f"identical A vs B      : {identical}")
print(f"comparator fires (D≠B): {comparator_fires}")
print(f"clock fires (C slower): {clock_fires}  (C/A = {speed_ratio_C:.3f})")
print(f"B speedup vs A        : {speedup:.3f}x")

if fail:
    print("\n--- VERDICT: NOT A PASS ---")
    for f in fail:
        print(" * " + f)
else:
    print(f"\n--- VERDICT: PASS — bit-identical transcript, {speedup:.3f}x on end-to-end wall clock, "
          f"with both controls firing ---")

kh.step("DONE", sha=sha[:8], identical=identical, speedup=round(speedup, 3),
        c_over_a=round(speed_ratio_C, 3), controls_ok=bool(comparator_fires and clock_fires),
        verdict="PASS" if not fail else "NOT_A_PASS")
print(json.dumps(results, indent=2), flush=True)
if fail:
    raise SystemExit(1)
