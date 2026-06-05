"""
CrispASR — mimo-asr CPU-path validation (PLAN #115)

Question: on current main, mimo-asr produces zero output on JFK (11 s) on
M1 Metal and segfaults on a 5 min clip. The smoking-gun commit by
inspection is `89111260` ("perf #72: load weights to GPU when use_gpu=true")
which flipped `core_gguf::load_weights(..., ctx->backend_cpu, ...)` to
`..., ctx->backend, ...`. That commit's own message foresees the regression:
"If a platform regresses, add a CRISPASR_FORCE_CPU_WEIGHTS=1 escape hatch
— none seen yet." Now we have one.

We can't safely repro on the local M1 box (4.5 GB CPU mimo Q4_K + already
loaded benchmark sweep risks OOM). This kernel runs on a Kaggle CPU
notebook (separate quota from GPU, no queue starvation) and answers:

    Does HEAD mimo-asr produce a JFK transcript on a pure-CPU build?

If yes → bug is GPU-specific (PLAN #72 broke the GPU residency path only).
If no  → CPU path is also broken; more regressions stacked since HISTORY §56.

Reference (HISTORY §56): the JFK transcript should be verbatim
"And so, my fellow Americans, ask not what your country can do for you.
Ask what you can do for your country."

All build-speed + reporting plumbing now comes from the shared harness
tools/kaggle/kaggle_harness.py (imported right after the clone), which is
the union of the best parts of the per-kernel copies:
  - kh.step() / progress.jsonl + HF mirror at cstr/crispasr-kaggle-progress
  - kh.sh_with_progress() Popen-based build streamer + ninja [X/N] parsing
  - kh.build_heartbeat() ticker (now also reports VmRSS + free disk) so
    cmake/ninja hangs — and the climbing-RSS OOM signature — show in
    progress.jsonl
  - kh.install_build_toolchain() + kh.cache_and_link_flags(): ninja +
    ccache (primed at /kaggle/working/.ccache, persisted across runs) +
    mold linker, so re-runs reuse cached objects and link faster
  - kh.resolve_hf_token(): 3-tier auth env → Kaggle Secret (retry) →
    chr1str/crispasr-hf-token dataset (kernel-metadata.json:dataset_sources)
"""

import os
import re
import subprocess
import time
from pathlib import Path

WORK = Path("/kaggle/working")
REPO = WORK / "CrispASR"
BUILD = WORK / "build"
MODELS = WORK / "models"
SAMPLE = WORK / "jfk.wav"

_T0 = time.time()

# ── Clone the repo FIRST so the shared harness is importable ──────────────
# The shared harness (tools/kaggle/kaggle_harness.py) provides line-buffered
# I/O + progress.jsonl + HF mirror, the Popen build streamer, the build
# heartbeat (now with RSS/free-disk reporting), 3-tier HF auth, and the
# ccache/mold build toolchain. It lives inside the repo, so it can only be
# imported after the clone. Until then we have a single minimal inline
# helper for the pre-clone span.
CRISPASR_REF = os.environ.get("CRISPASR_REF", "main")


def _sh_preclone(cmd: str) -> None:
    print(f"$ {cmd}", flush=True)
    subprocess.run(cmd, shell=True, check=True)


print("[pre-clone] cloning CrispASR for shared harness", flush=True)
if not REPO.exists():
    _sh_preclone(
        f"git clone --depth 1 --branch {CRISPASR_REF} "
        f"https://github.com/CrispStrobe/CrispASR.git {REPO}"
    )
else:
    _sh_preclone(f"git -C {REPO} pull --ff-only")

import sys
sys.path.insert(0, os.path.join(str(REPO), "tools", "kaggle"))
import kaggle_harness as kh
kh.init_progress()

# 3-tier HF auth (env → Kaggle Secret w/ retry → mounted dataset).
if kh.resolve_hf_token():
    print("HF auth: token loaded (progress mirror enabled)", flush=True)
else:
    print("HF auth: anonymous (progress mirror disabled; local JSONL only)", flush=True)

kh.step("script.start")

# Branch-parametrized so re-runs against fixes are one env var away.
# (CRISPASR_REF was read before the clone above so the harness could be
# imported; the repo is already checked out at this ref.)
EXPECTED_JFK = "ask not what your country can do for you"

# ── Step 1: record cloned SHA (clone happened pre-import, above) ──────────

kh.step("clone.done.ref", ref=CRISPASR_REF)
sha = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
kh.step("clone.done", sha=sha)

# ── Step 2: CPU-only build ───────────────────────────────────────────────
# We are explicitly on a CPU kernel (enable_gpu=false in kernel-metadata)
# because: (a) GPU quota was queue-blocking us for 14+ h on the prior
# attempt; (b) the question we're answering — "does the CPU path produce
# any text at all on HEAD?" — doesn't need GPU. The full GPU vs CPU
# regression bisect can run when GPU quota frees, with this same script
# pointed at enable_gpu=true.

kh.step("build.begin")
BUILD.mkdir(exist_ok=True)
# ninja + ccache + mold (ccache primed at /kaggle/working/.ccache so a
# re-run reuses cached objects; mold links faster than ld).
kh.install_build_toolchain()
cmake_cmd = (
    f"cmake {REPO} -B{BUILD} -GNinja "
    "-DCMAKE_BUILD_TYPE=Release "
    "-DBUILD_SHARED_LIBS=ON "
    "-DCRISPASR_BUILD_TESTS=OFF "
    + " ".join(kh.cache_and_link_flags())
)
njobs = kh.safe_build_jobs(gpu=False)
with kh.build_heartbeat("cmake-configure"):
    kh.sh_with_progress(cmake_cmd)
kh.step("build.configured")
with kh.build_heartbeat("cmake-build"):
    kh.sh_with_progress(
        f"stdbuf -oL -eL cmake --build {BUILD} --target crispasr-cli -- -j{njobs}"
    )
CRISPASR = BUILD / "bin" / "crispasr"
assert CRISPASR.is_file(), f"crispasr binary missing at {CRISPASR}"
kh.step("build.done", binary=str(CRISPASR))

# ── Step 3: download mimo-asr + tokenizer ─────────────────────────────────

kh.step("download.begin")
MODELS.mkdir(exist_ok=True)
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
kh.sh_with_progress("pip install -q huggingface_hub hf_transfer")
from huggingface_hub import hf_hub_download

for repo_id, fname in [
    ("cstr/mimo-asr-GGUF", "mimo-asr-q4_k.gguf"),
    ("cstr/mimo-tokenizer-GGUF", "mimo-tokenizer-q4_k.gguf"),
]:
    kh.step(f"download.{fname}.begin", repo=repo_id)
    p = hf_hub_download(repo_id=repo_id, filename=fname, local_dir=str(MODELS), local_dir_use_symlinks=False)
    kh.step(f"download.{fname}.done", path=p, size_mib=Path(p).stat().st_size // (1 << 20))

mimo_path = MODELS / "mimo-asr-q4_k.gguf"
tok_path = MODELS / "mimo-tokenizer-q4_k.gguf"

# JFK sample from repo
subprocess.run(["cp", f"{REPO}/samples/jfk.wav", str(SAMPLE)], check=False)
assert SAMPLE.is_file(), "jfk.wav missing"
kh.step("download.done")

# ── Step 4: run mimo-asr on JFK ──────────────────────────────────────────

def run_mimo(label: str, extra_args: list[str]) -> tuple[str, bool]:
    out_stem = WORK / f"mimo-jfk-{label}"
    for ext in [".txt", ".srt"]:
        f = out_stem.with_suffix(ext)
        if f.exists():
            f.unlink()
    cmd = [
        str(CRISPASR),
        "-m", str(mimo_path),
        "--backend", "mimo-asr",
        "-f", str(SAMPLE),
        "-of", str(out_stem),
        "-otxt",
        "-np",
    ] + extra_args
    kh.step(f"run.{label}.begin", cmd=" ".join(cmd))
    t0 = time.time()
    r = subprocess.run(cmd, env={**os.environ, "MIMO_ASR_BENCH": "1"},
                       capture_output=True, text=True, timeout=600)
    dt = time.time() - t0
    log = (r.stdout or "") + (r.stderr or "")
    bench = re.search(r"mimo_asr_bench:.*", log)
    txt_path = out_stem.with_suffix(".txt")
    has_file = txt_path.exists() and txt_path.stat().st_size > 0
    text = txt_path.read_text().strip() if has_file else ""
    ok = EXPECTED_JFK in text.lower()
    kh.step(
        f"run.{label}.done",
        exit=r.returncode,
        wall_s=round(dt, 1),
        bench=bench.group(0) if bench else None,
        has_output=has_file,
        chars=len(text),
        ok=ok,
    )
    if not has_file or not text:
        # Surface tail of stderr to logs for diagnosis
        print("--- last 30 lines of stderr ---", flush=True)
        for line in log.splitlines()[-30:]:
            print(line, flush=True)
    return text, ok


kh.step("run.section.begin")
text, ok = run_mimo("cpu", [])

# ── Summary ──────────────────────────────────────────────────────────────

kh.step("summary", text=text[:300], ok=ok, sha=sha)
print("\n" + "=" * 70)
print(f"SUMMARY — mimo-asr JFK on HEAD ({sha[:8]}) — CPU-only kernel")
print("=" * 70)
status = "PASS (matches reference)" if ok else ("EMPTY (no output)" if not text else "WRONG (text differs from reference)")
print(f"  result: {status}")
print(f"  text:   {text[:200]}")
print()
print("Interpretation:")
if ok:
    print("  CPU path works on HEAD. The mimo-asr regression observed on M1 Metal")
    print("  is GPU-specific (PLAN #72 commit 89111260 'load weights to GPU').")
    print("  Fix: either revert the one-line backend_cpu→backend swap, or fix the")
    print("  GPU tensor routing in mimo_asr_build_prefill_graph.")
elif text:
    print("  Output produced but doesn't match the reference. Could be a separate")
    print("  text bug (detokenizer / stop-token handling) or different model.")
else:
    print("  No output even on CPU. There are more regressions stacked on top of")
    print("  #72. Bisect further: HISTORY §56 (dae361f2) was the last known good.")

kh._push_progress_to_hf(force=True)
kh.step("script.end")
