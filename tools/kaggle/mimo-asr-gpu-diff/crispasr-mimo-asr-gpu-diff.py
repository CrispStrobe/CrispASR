"""
CrispASR — mimo-asr GPU prefill diagnostic (PLAN #115 option C)

The CPU path works (sibling kernel crispasr-mimo-asr-cpu-validate, JFK PASS).
The GPU path is broken: with weights GPU-resident + GPU compute (commit
89111260) mimo-asr emits no tokens (exit 0, empty). Option A force-pins to
CPU (correct, slow). Option C is the proper GPU prefill fix, but it can't be
debugged on the local M1 (4.2 GB model, box memory-saturated) — so we run it
on a Kaggle GPU box, mirroring the funasr-cuda-debug kernel.

Plan (same shape as crispasr-funasr-cuda-debug):
  1. CUDA build of crispasr-cli.
  2. mimo-asr on jfk.wav, two runs, MIMO_ASR_DUMP_STAGES=1 on both:
       cpu  — default (force-CPU, option A). Ground truth.
       gpu  — CRISPASR_MIMO_FORCE_GPU=1 (weights+compute on GPU). The bug.
  3. Parse the `mimo_dump: <stage> ... nan= inf=` lines and compare the two
     runs stage-by-stage. The first stage where GPU NaNs / diverges from CPU
     localises the broken op in mimo_asr_build_prefill_graph.

Reference (HISTORY §56) JFK transcript: "...ask not what your country can do
for you. Ask what you can do for your country."

Build/report plumbing from the shared harness tools/kaggle/kaggle_harness.py.
enable_gpu=true in kernel-metadata.json.
"""

import os
import re
import subprocess
import sys
import time
from pathlib import Path

WORK = Path("/kaggle/working")
REPO = WORK / "CrispASR"
BUILD = WORK / "build"
RESULTS = WORK / "results"
RESULTS.mkdir(parents=True, exist_ok=True)
SAMPLE = REPO / "samples" / "jfk.wav"

CRISPASR_REF = os.environ.get("CRISPASR_REF", "main")
CRISPASR_REPO = os.environ.get("CRISPASR_REPO", "https://github.com/CrispStrobe/CrispASR.git")
EXPECTED_JFK = "ask not what your country can do for you"


def run(cmd, check=True, env=None, timeout=None):
    print(f"\n$ {' '.join(str(c) for c in cmd)}", flush=True)
    e = os.environ.copy()
    if env:
        e.update(env)
    r = subprocess.run(cmd, env=e, timeout=timeout)
    if check and r.returncode != 0:
        raise SystemExit(f"command failed (rc={r.returncode}): {cmd}")
    return r


# ── Clone + CUDA build (mirrors funasr-cuda-debug) ───────────────────────
print(f"[start] ref={CRISPASR_REF}", flush=True)
if REPO.exists():
    import shutil
    shutil.rmtree(REPO)
run(["git", "clone", "--depth", "1", "--branch", CRISPASR_REF, "--recursive", CRISPASR_REPO, str(REPO)])

sys.path.insert(0, os.path.join(str(REPO), "tools", "kaggle"))
import kaggle_harness as kh
kh.init_progress()
kh.resolve_hf_token()

sha = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
kh.step("cloned", sha=sha, ref=CRISPASR_REF)

run(["nvidia-smi", "-L"])
gpu_name = subprocess.check_output(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True).strip()
kh.step("gpu", gpu_name=gpu_name)

kh.install_build_toolchain()
arch = kh.detect_cuda_arch()
kh.step("cuda_arch", arch=arch)

BUILD.mkdir(parents=True, exist_ok=True)
cmake_args = (
    ["cmake", "-S", str(REPO), "-B", str(BUILD), "-DCMAKE_BUILD_TYPE=Release", "-DBUILD_SHARED_LIBS=ON",
     "-DCRISPASR_BUILD_TESTS=OFF"]
    + kh.cuda_build_flags(arch)
    + kh.cache_and_link_flags()
)
run(cmake_args)
kh.step("cmake_done")
with kh.build_heartbeat("cmake.build"):
    kh.sh_with_progress(f"stdbuf -oL -eL cmake --build {BUILD} --target crispasr-cli -j{kh.safe_build_jobs(gpu=True)}")

CLI = BUILD / "bin" / "crispasr"
if not CLI.exists():
    cands = [c for c in BUILD.rglob("crispasr") if c.is_file() and os.access(c, os.X_OK)]
    assert cands, "crispasr binary not found after build"
    CLI = cands[0]
os.environ["LD_LIBRARY_PATH"] = f"{BUILD / 'src'}:{os.environ.get('LD_LIBRARY_PATH', '')}"
kh.step("build_done", cli=str(CLI))

# ── Run mimo-asr on JFK: cpu baseline + gpu repro, both dumping stages ────
STAGE_RE = re.compile(r"mimo_dump:\s+(\S+)\s+(.*)")
STAT_RE = re.compile(r"nan=(\d+)\s+inf=(\d+)")


def run_mimo(label: str, extra_env: dict, timeout: int = 900) -> dict:
    kh.step(f"{label}.start")
    out_stem = WORK / f"mimo-jfk-{label}"
    for ext in (".txt", ".srt"):
        f = out_stem.with_suffix(ext)
        if f.exists():
            f.unlink()
    cmd = [str(CLI), "--backend", "mimo-asr", "-m", "auto", "--auto-download",
           "-f", str(SAMPLE), "-of", str(out_stem), "-otxt"]
    env = {"MIMO_ASR_DUMP_STAGES": "1", "MIMO_ASR_BENCH": "1", **extra_env}
    t0 = time.time()
    try:
        r = subprocess.run(cmd, env={**os.environ, **env}, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                           text=True, timeout=timeout)
        rc, stderr = r.returncode, r.stderr
    except subprocess.TimeoutExpired as ex:
        rc = -1
        stderr = (ex.stderr or b"").decode(errors="replace") if isinstance(ex.stderr, bytes) else (ex.stderr or "")
    elapsed = round(time.time() - t0, 1)
    (RESULTS / f"{label}_stderr.txt").write_text(stderr)

    txt_path = out_stem.with_suffix(".txt")
    text = txt_path.read_text().strip() if (txt_path.exists() and txt_path.stat().st_size > 0) else ""
    ok = EXPECTED_JFK in text.lower()

    stages = {}
    for line in stderr.splitlines():
        m = STAGE_RE.match(line.strip())
        if m:
            stages[m.group(1)] = m.group(2)

    print(f"\n{'='*64}\nRun: {label}  rc={rc}  elapsed={elapsed}s  ok={ok}", flush=True)
    print(f"  transcript: {text[:160]!r}", flush=True)
    for sn, sv in stages.items():
        print(f"  {sn:28s} {sv[:110]}", flush=True)
    if not text:
        print("  --- stderr tail (no transcript) ---", flush=True)
        for ln in stderr.splitlines()[-25:]:
            print("   " + ln, flush=True)
    kh.step(f"{label}.done", rc=rc, elapsed=elapsed, ok=ok, n_stages=len(stages), chars=len(text))
    return {"label": label, "rc": rc, "text": text, "ok": ok, "stages": stages}


cpu = run_mimo("cpu", {})
gpu = run_mimo("gpu", {"CRISPASR_MIMO_FORCE_GPU": "1"})

# ── Compare CPU vs GPU per stage — find the first divergence ──────────────
print("\n" + "=" * 64, flush=True)
print("STAGE COMPARISON (cpu = ground truth)", flush=True)
print("=" * 64, flush=True)
first_bad = None
for sn in cpu["stages"]:
    cv, gv = cpu["stages"].get(sn, ""), gpu["stages"].get(sn, "")
    g_nan = STAT_RE.search(gv)
    bad = (gv == "" or "[not found]" in gv or (g_nan and (int(g_nan.group(1)) or int(g_nan.group(2)))))
    flag = " <-- DIVERGES" if bad else ""
    if bad and first_bad is None:
        first_bad = sn
    print(f"[{sn}]{flag}\n  cpu: {cv[:120]}\n  gpu: {gv[:120]}", flush=True)

print("\n" + "=" * 64, flush=True)
print(f"SUMMARY — mimo-asr GPU prefill — {sha[:8]} on {gpu_name}", flush=True)
print(f"  cpu: {'PASS' if cpu['ok'] else ('EMPTY' if not cpu['text'] else 'WRONG')}", flush=True)
print(f"  gpu: {'PASS' if gpu['ok'] else ('EMPTY' if not gpu['text'] else 'WRONG')}", flush=True)
print(f"  first GPU-diverging stage: {first_bad or '(none — stages match; bug is post-prefill / decode)'}", flush=True)

kh.step("summary", cpu_ok=cpu["ok"], gpu_ok=gpu["ok"], first_bad=first_bad,
        gpu_status=("PASS" if gpu["ok"] else ("EMPTY" if not gpu["text"] else "WRONG")), sha=sha)
kh._push_progress_to_hf(force=True)
kh.step("script.end")
