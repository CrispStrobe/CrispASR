"""
CrispASR -- funasr CUDA debug: flash-attn ablation + per-stage tensor dump.

Purpose: localize the funasr CUDA !-loop bug (issue #125/#126). On CUDA,
funasr's greedy decode degenerates into token-0 repeats. This kernel:

  1. Builds CrispASR with CUDA.
  2. Downloads funasr-nano-2512-q8_0.gguf (the default auto-download quant).
  3. Run A: CUDA + FA on (default) + FUNASR_DUMP_STAGES=1.  Expect: !-loop.
  4. Run B: CUDA + FA off (FUNASR_NO_FA=1) + dump.         Expect: TBD.
  5. Run C: CPU forced (GGML_CUDA_DISABLE=1) + dump.        Expect: correct.
  6. Compare per-stage tensor stats across all three runs.

The per-stage dump (FUNASR_DUMP_STAGES=1) prints min/max/mean/L2/first8 for
13 key stages (every 10th encoder layer + boundaries + adaptor + spliced
embeddings + prefill logits). CPU vs CUDA divergence in these stats locates
the first broken stage; FA-on vs FA-off isolates whether flash_attn_ext is
the culprit.
"""

import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ["PYTHONUNBUFFERED"] = "1"
try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except (AttributeError, ValueError):
    pass

WORK = Path("/kaggle/working")
REPO = WORK / "CrispASR"
BUILD = WORK / "build"
RESULTS = WORK / "results"
RESULTS.mkdir(parents=True, exist_ok=True)
SAMPLE = REPO / "samples" / "jfk.wav"

CRISPASR_REF = os.environ.get("CRISPASR_REF", "main")
CRISPASR_REPO = os.environ.get(
    "CRISPASR_REPO", "https://github.com/CrispStrobe/CrispASR.git"
)

PROGRESS = RESULTS / "progress.jsonl"
_T0 = time.time()


def step(name, **kv):
    payload = {"t": round(time.time() - _T0, 2), "step": name, **kv}
    line = json.dumps(payload)
    print(f"[step] {line}", flush=True)
    with open(PROGRESS, "a") as f:
        f.write(line + "\n")


def run(cmd, check=True, capture=False, env=None, cwd=None, timeout=None):
    print(f"\n$ {' '.join(str(c) for c in cmd)}", flush=True)
    e = os.environ.copy()
    if env:
        e.update(env)
    if capture:
        r = subprocess.run(
            cmd,
            env=e,
            cwd=cwd,
            timeout=timeout,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        print(r.stdout[-4096:] if len(r.stdout) > 4096 else r.stdout, flush=True)
        if check and r.returncode != 0:
            raise SystemExit(f"command failed (rc={r.returncode}): {cmd}")
        return r
    r = subprocess.run(cmd, env=e, cwd=cwd, timeout=timeout)
    if check and r.returncode != 0:
        raise SystemExit(f"command failed (rc={r.returncode}): {cmd}")
    return r


# ──────────────────────────────────────────────────────────────────────────
# Clone + build
# ──────────────────────────────────────────────────────────────────────────
step("start", ref=CRISPASR_REF)

if REPO.exists():
    import shutil

    shutil.rmtree(REPO)
run(["git", "clone", "--depth", "1", "--branch", CRISPASR_REF, "--recursive",
     CRISPASR_REPO, str(REPO)])

sys.path.insert(0, os.path.join(str(REPO), "tools", "kaggle"))
import kaggle_harness as kh

kh.init_progress()

sha = subprocess.check_output(
    ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True
).strip()
step("cloned", sha=sha)

run(["nvidia-smi", "-L"])
gpu_name = subprocess.check_output(
    ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True
).strip()
step("gpu", gpu_name=gpu_name)

kh.install_build_toolchain()
arch = kh.detect_cuda_arch()
step("cuda_arch", arch=arch)

BUILD.mkdir(parents=True, exist_ok=True)
cmake_args = (
    [
        "cmake", "-S", str(REPO), "-B", str(BUILD),
        "-DCMAKE_BUILD_TYPE=Release",
        "-DBUILD_SHARED_LIBS=ON",
    ]
    + kh.cuda_build_flags(arch)
    + kh.cache_and_link_flags()
)
run(cmake_args)
step("cmake_done")

with kh.build_heartbeat("cmake.build"):
    kh.sh_with_progress(
        f"stdbuf -oL -eL cmake --build {BUILD} --target crispasr-cli "
        f"-j{kh.safe_build_jobs(gpu=True)}"
    )
step("build_done")

CLI = BUILD / "bin" / "crispasr"
if not CLI.exists():
    candidates = list(BUILD.rglob("crispasr"))
    candidates = [c for c in candidates if c.is_file() and os.access(c, os.X_OK)]
    if not candidates:
        raise SystemExit("crispasr binary not found after build")
    CLI = candidates[0]
print(f"crispasr binary: {CLI}", flush=True)
step("cli_found", path=str(CLI))

LIB_DIR = BUILD / "src"
os.environ["LD_LIBRARY_PATH"] = (
    f"{LIB_DIR}:{os.environ.get('LD_LIBRARY_PATH', '')}"
)


# ──────────────────────────────────────────────────────────────────────────
# Run helper: transcribe jfk.wav with given env overrides, capture output
# ──────────────────────────────────────────────────────────────────────────
def run_funasr(label: str, extra_env: dict, timeout: int = 300) -> dict:
    """Run funasr on jfk.wav. Returns {label, transcript, rc, stages, elapsed}."""
    step(f"{label}_start")
    env = {
        "FUNASR_DUMP_STAGES": "1",
        "CRISPASR_VERBOSE": "1",
        **extra_env,
    }
    cmd = [
        str(CLI),
        "--backend", "funasr",
        "-m", "auto",
        "--auto-download",
        "-f", str(SAMPLE),
        "--no-prints",
    ]
    t0 = time.time()
    try:
        r = subprocess.run(
            cmd,
            timeout=timeout,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env={**os.environ, **env},
        )
        rc = r.returncode
        stdout = r.stdout
        stderr = r.stderr
    except subprocess.TimeoutExpired as ex:
        rc = -1
        stdout = (ex.stdout or b"").decode(errors="replace")
        stderr = (ex.stderr or b"").decode(errors="replace")
    elapsed = round(time.time() - t0, 2)

    # Save raw logs
    (RESULTS / f"{label}_stdout.txt").write_text(stdout)
    (RESULTS / f"{label}_stderr.txt").write_text(stderr)

    # Parse transcript (stdout, stripped)
    transcript = stdout.strip()

    # Parse stage dumps from stderr
    stages = {}
    for line in stderr.splitlines():
        if line.startswith("funasr_dump:"):
            # e.g. "funasr_dump: encoder_layer_0              n=93696 ..."
            parts = line.split(None, 2)
            if len(parts) >= 3:
                stage_name = parts[1]
                stages[stage_name] = parts[2]

    # Check for degeneration warning
    degenerated = "greedy decode degenerated" in stderr

    result = {
        "label": label,
        "transcript": transcript[:500],
        "rc": rc,
        "elapsed_s": elapsed,
        "degenerated": degenerated,
        "n_stages": len(stages),
        "stages": stages,
    }

    # Print summary
    print(f"\n{'='*60}", flush=True)
    print(f"Run: {label}", flush=True)
    print(f"  RC: {rc}  elapsed: {elapsed}s  degenerated: {degenerated}", flush=True)
    print(f"  Transcript: {transcript[:200]!r}", flush=True)
    print(f"  Stages dumped: {len(stages)}", flush=True)
    for sname, sval in sorted(stages.items()):
        print(f"  {sname}: {sval[:120]}", flush=True)
    print(f"{'='*60}\n", flush=True)

    step(f"{label}_done", rc=rc, elapsed=elapsed, degenerated=degenerated,
         transcript=transcript[:200])
    return result


# ──────────────────────────────────────────────────────────────────────────
# Run the three experiments
# ──────────────────────────────────────────────────────────────────────────
results = []

# Run A: CUDA default (FA on) — should reproduce the bug
r_cuda_fa = run_funasr("cuda_fa_on", {})
results.append(r_cuda_fa)

# Run B: CUDA with FA off — test if flash-attn is the culprit
r_cuda_nofa = run_funasr("cuda_fa_off", {"FUNASR_NO_FA": "1"})
results.append(r_cuda_nofa)

# Run C: CPU forced — ground truth
# Force CPU by setting GGML_CUDA_NO_PEER_COPY and using a trick:
# Build has CUDA but we can force CPU by setting the backend env.
# Actually the cleanest way is CUDA_VISIBLE_DEVICES="" to hide GPUs.
r_cpu = run_funasr("cpu_baseline", {"CUDA_VISIBLE_DEVICES": ""})
results.append(r_cpu)


# ──────────────────────────────────────────────────────────────────────────
# Compare stages across runs
# ──────────────────────────────────────────────────────────────────────────
def parse_stage_stats(raw: str) -> dict:
    """Parse 'n=93696 min=... max=... mean=... L2=... nan=0 inf=0 first8=[...]'."""
    d = {}
    for m in re.finditer(r"(\w+)=([\d.e+\-]+|\[[\d.,e+\-]+\])", raw):
        k, v = m.group(1), m.group(2)
        if v.startswith("["):
            d[k] = [float(x) for x in v.strip("[]").split(",") if x]
        else:
            try:
                d[k] = float(v)
            except ValueError:
                d[k] = v
    return d


print("\n" + "=" * 80, flush=True)
print("STAGE-BY-STAGE COMPARISON", flush=True)
print("=" * 80, flush=True)

comparison = {}
stage_names_ordered = [
    "encoder_layer_0", "encoder_layer_9", "encoder_layer_19",
    "encoder_layer_29", "encoder_layer_39", "encoder_layer_49",
    "encoder_layer_59", "encoder_layer_69",
    "encoder_main_out", "encoder_output",
    "audio_adaptor_layer_0", "audio_adaptor_layer_1",
    "audio_adaptor_output", "spliced_audio_embeds", "prefill_logits",
]

for sname in stage_names_ordered:
    cpu_raw = r_cpu["stages"].get(sname, "")
    cuda_fa_raw = r_cuda_fa["stages"].get(sname, "")
    cuda_nofa_raw = r_cuda_nofa["stages"].get(sname, "")

    cpu_s = parse_stage_stats(cpu_raw)
    cuda_fa_s = parse_stage_stats(cuda_fa_raw)
    cuda_nofa_s = parse_stage_stats(cuda_nofa_raw)

    comp = {"cpu": cpu_s, "cuda_fa": cuda_fa_s, "cuda_nofa": cuda_nofa_s}
    comparison[sname] = comp

    # Print comparison
    print(f"\n--- {sname} ---", flush=True)
    for metric in ["min", "max", "mean", "L2", "nan", "inf"]:
        cv = cpu_s.get(metric, "?")
        fv = cuda_fa_s.get(metric, "?")
        nv = cuda_nofa_s.get(metric, "?")
        flag = ""
        if isinstance(cv, (int, float)) and isinstance(fv, (int, float)):
            if cv != 0 and abs(fv - cv) / max(abs(cv), 1e-10) > 0.01:
                flag = " *** DIVERGED"
            if isinstance(fv, float) and (fv != fv):  # NaN
                flag = " *** NaN"
        print(f"  {metric:6s}  cpu={cv!s:>14s}  cuda_fa={fv!s:>14s}  cuda_nofa={nv!s:>14s}{flag}", flush=True)

    # Compare first8 values
    cpu_f8 = cpu_s.get("first8", [])
    cuda_fa_f8 = cuda_fa_s.get("first8", [])
    cuda_nofa_f8 = cuda_nofa_s.get("first8", [])
    if cpu_f8 and cuda_fa_f8:
        max_diff = max(abs(a - b) for a, b in zip(cpu_f8, cuda_fa_f8)) if len(cpu_f8) == len(cuda_fa_f8) else float("inf")
        print(f"  first8 max|cpu-cuda_fa| = {max_diff:.6f}", flush=True)
    if cpu_f8 and cuda_nofa_f8:
        max_diff = max(abs(a - b) for a, b in zip(cpu_f8, cuda_nofa_f8)) if len(cpu_f8) == len(cuda_nofa_f8) else float("inf")
        print(f"  first8 max|cpu-cuda_nofa| = {max_diff:.6f}", flush=True)

# Also compare argmax for prefill_logits if present
for sname in ["prefill_logits"]:
    for label, r in [("cuda_fa", r_cuda_fa), ("cuda_nofa", r_cuda_nofa), ("cpu", r_cpu)]:
        raw = r["stages"].get(sname, "")
        m = re.search(r"argmax=(\d+)", raw)
        if m:
            print(f"  {label} argmax = {m.group(1)}", flush=True)

# ──────────────────────────────────────────────────────────────────────────
# Summary
# ──────────────────────────────────────────────────────────────────────────
summary = {
    "ts": datetime.now(timezone.utc).isoformat(),
    "ref": CRISPASR_REF,
    "sha": sha,
    "gpu": gpu_name,
    "cuda_arch": arch,
    "runs": {r["label"]: {
        "rc": r["rc"],
        "degenerated": r["degenerated"],
        "transcript": r["transcript"][:300],
        "elapsed_s": r["elapsed_s"],
    } for r in results},
    "comparison": comparison,
}
(RESULTS / "summary.json").write_text(json.dumps(summary, indent=2))

print("\n" + "=" * 60, flush=True)
print("VERDICT:", flush=True)
for r in results:
    status = "DEGENERATED" if r["degenerated"] else ("PASS" if "ask not" in r["transcript"].lower() else "UNKNOWN")
    print(f"  {r['label']:20s} -> {status}: {r['transcript'][:80]!r}", flush=True)
print("=" * 60, flush=True)

# Determine if FA is the culprit
if r_cuda_fa["degenerated"] and not r_cuda_nofa["degenerated"]:
    print("\nCONCLUSION: Flash-attention (ggml_flash_attn_ext) is the CUDA culprit.", flush=True)
    print("Fix: disable FA on CUDA for funasr encoder, or fix the FA CUDA kernel.", flush=True)
elif r_cuda_fa["degenerated"] and r_cuda_nofa["degenerated"]:
    print("\nCONCLUSION: Bug persists with FA off. Flash-attention is NOT the sole cause.", flush=True)
    print("Check stage comparison above for first diverging stage.", flush=True)
elif not r_cuda_fa["degenerated"]:
    print("\nSURPRISE: CUDA FA-on did NOT degenerate. Bug may be fixed or intermittent.", flush=True)
else:
    print("\nUNCLEAR: check logs.", flush=True)

step("done", all_pass=all(not r["degenerated"] for r in results))
