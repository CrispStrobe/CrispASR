# ─────────────────────────── cell 0 (markdown) ───────────────────────────
# # CrispASR — fusion A/B (pre- vs post-norm_affine+siglu) on Kaggle T4
#
# Reproduces the issue #81 reporter's setup (parakeet TDT 0.6B v3, ONNX vs
# CrispASR on CUDA) on a Kaggle T4 / P100, and adds the A/B that the
# reporter couldn't run: **CrispASR built before vs after `c2423313`** —
# the norm_affine + siglu fusion (2026-05-13) that the committer reports gives ~1.2% on CPU; on CUDA they expect "structural" gains from fewer backend splits + more graph capture
# but didnt measure (left for this notebook).
#
# What it does:
# 1. Clone CrispASR.
# 2. Build at `d758fe69~1` (PRE)  → save `libcrispasr-pre.so`.
# 3. `git checkout c2423313` (POST) → incremental rebuild → save `libcrispasr-post.so`.
# 4. Pre-download `parakeet-tdt-0.6b-v3` GGUF (q8_0) and ONNX (int8 + fp32).
# 5. Run `tools/benchmark_asr_engines.py` three times, all with `--runs 10
#    --warmups 1 --prewarm` on the 60 s tiled-JFK clip + 11 s real JFK:
#      a) `--engine crispasr --crispasr-lib libcrispasr-pre.so`
#      b) `--engine crispasr --crispasr-lib libcrispasr-post.so`
#      c) `--engine onnx --providers CUDAExecutionProvider`
# 6. Aggregate the three JSON sidecars into one comparison table:
#      - mean / p50 / p95 per cell
#      - **per-run series** (10 numbers) so we can see if the issue #81
#        reporter's "rising runs" pattern (`[2.45, 2.83, 3.39]`) reproduces
#        on a clean Kaggle GPU.
#      - delta PRE → POST (% of flash-attn fusion gain on dGPU)
#      - delta POST → onnx-CUDA (the remaining gap to close)
#
# Requirements:
# - Kaggle accelerator: GPU T4 ×1 or P100 (CPU works but defeats the point).
# - Internet ON (model downloads).
# - Disk: ~8 GB (CrispASR build ~1 GB + GGUF ~700 MB + ONNX ~3 GB + deps).
# - Optional secret: `GH_GIST_TOKEN` to push results to a gist.

# ─────────────────────────── cell 1 (code) ───────────────────────────
# ── Configuration ──────────────────────────────────────────────────────────
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# ── Unbuffered I/O + step checkpointing ──────────────────────────────
# Kaggle persists logs only at cell/process end, so a hang past stdout
# buffer fill is invisible until termination. Line-buffer stdio +
# write a per-step JSONL to /kaggle/working/progress.jsonl so even
# when the run hangs we can fetch progress.jsonl (tiny file) and see
# the last completed step. Mirrors the helper in
# tools/kaggle/crispasr-regression.py.
os.environ["PYTHONUNBUFFERED"] = "1"
try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except (AttributeError, ValueError):
    pass

_PROGRESS_PATH = Path("/kaggle/working/progress.jsonl")
_T0 = time.time()

# Live HF progress push — see the helper in
# tools/kaggle/crispasr-regression.py for the full rationale.
# Rolls the local progress.jsonl up to cstr/crispasr-kaggle-progress
# (rate-limited to every 30s) so mid-run state is fetchable via
# `huggingface.co/datasets/cstr/crispasr-kaggle-progress` even
# while the Kaggle kernel is mid-execution / hung / crashed.
_HF_PROGRESS_REPO = "cstr/crispasr-kaggle-progress"
_HF_PROGRESS_PATH = (
    f"runs/{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    f"-{os.environ.get('KAGGLE_KERNEL_RUN_TYPE', 'local').lower()}"
    f"-{os.environ.get('KAGGLE_KERNEL_REF', 'unknown').split('/')[-1] or 'unknown'}"
    f".jsonl"
)
_HF_PUSH_INTERVAL_S = 30.0
_HF_LAST_PUSH = 0.0


def _push_progress_to_hf(force: bool = False) -> None:
    global _HF_LAST_PUSH
    now = time.time()
    if not force and (now - _HF_LAST_PUSH) < _HF_PUSH_INTERVAL_S:
        return
    if not os.environ.get("HF_TOKEN"):
        return
    if not _PROGRESS_PATH.exists():
        return
    try:
        from huggingface_hub import HfApi
        HfApi(token=os.environ["HF_TOKEN"]).upload_file(
            path_or_fileobj=str(_PROGRESS_PATH),
            path_in_repo=_HF_PROGRESS_PATH,
            repo_id=_HF_PROGRESS_REPO,
            repo_type="dataset",
            commit_message=f"progress @ {now - _T0:.0f}s",
        )
        _HF_LAST_PUSH = now
    except Exception:
        pass


def step(name: str, **extra) -> None:
    rec = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "elapsed_s": round(time.time() - _T0, 2),
        "step": name,
        **extra,
    }
    try:
        _PROGRESS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with _PROGRESS_PATH.open("a") as f:
            f.write(json.dumps(rec) + "\n")
    except Exception:
        pass
    print(f"[step {rec['elapsed_s']:>7.1f}s] {name}" +
          (f"  {extra}" if extra else ""), flush=True)
    _push_progress_to_hf()


step("script.start")

WORK = Path("/kaggle/working")
REPO = WORK / "CrispASR"
BUILD = WORK / "build"
LIB_PRE = WORK / "libcrispasr-pre.so"
LIB_POST = WORK / "libcrispasr-post.so"
GGUF_DIR = WORK / "models-gguf"
ONNX_DIR = WORK / "models-onnx"
RESULTS_DIR = WORK / "results"

# A/B SHAs. PRE_SHA is just before the norm_affine+siglu fusion landed.
PRE_SHA = "d758fe69~1"   # "docs(perf): issue #81 round 2 — Tiger Lake CPU diagnosis"
POST_SHA = "d758fe69"    # "perf(ggml): fused norm_affine + siglu ops for FastConformer (issue #81)"
# To compare against current HEAD instead of the exact fusion commit, set
# POST_SHA = "main" (will fetch + check out) — picks up post-fusion fixes too.

# Bench knobs. RUNS=10 is double the issue #81 reporter's RUNS=3 — needed
# to separate steady-state from the rising-runs pattern they saw.
RUNS = 10
WARMUPS = 1
WINDOW_S = 4.0
GGUF_QUANTS = "q4_k,q5_0,q8_0,f16"   # sweep all parakeet-tdt-0.6b-v3 quants
ONNX_QUANTS = "int8,fp32"      # int8 is the reporter's; fp32 for context
AUDIOS = "both"                # short (jfk 11s) + long (60s tiled)
MODES = "chunked"              # streaming-relevant shape; matches reporter

for d in (RESULTS_DIR, GGUF_DIR, ONNX_DIR):
    d.mkdir(parents=True, exist_ok=True)

print(f"CrispASR issue #81 CUDA A/B — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
print(f"PRE  = {PRE_SHA}")
print(f"POST = {POST_SHA}")
print(f"runs={RUNS} warmups={WARMUPS} window={WINDOW_S}s quants={GGUF_QUANTS}")

# ─────────────────────────── cell 1b (code) ───────────────────────────
# ── Wire Kaggle secrets up front (HF_TOKEN before any hf_hub_download) ────
# Without HF_TOKEN, huggingface_hub falls back to anonymous requests:
# strict rate limits, slower downloads, and the warning
# "You are sending unauthenticated requests to the HF Hub". Pull from
# Kaggle's UserSecretsClient if available, then expose via every env var
# huggingface_hub recognises so all of `hf_hub_download`, `snapshot_download`,
# and the cli see it.
HF_TOKEN = ""
GH_GIST_TOKEN = ""
try:
    from kaggle_secrets import UserSecretsClient
    _secrets = UserSecretsClient()
    try:
        HF_TOKEN = _secrets.get_secret("HF_TOKEN")
    except Exception:
        pass
    try:
        GH_GIST_TOKEN = _secrets.get_secret("GH_GIST_TOKEN")
    except Exception:
        pass
except Exception:
    HF_TOKEN = os.environ.get("HF_TOKEN", "")
    GH_GIST_TOKEN = os.environ.get("GH_GIST_TOKEN", "")

if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN
    os.environ["HUGGING_FACE_HUB_TOKEN"] = HF_TOKEN
    os.environ["HF_HUB_TOKEN"] = HF_TOKEN
    print("✓ HF_TOKEN wired from Kaggle secrets")
else:
    print("⚠ no HF_TOKEN — model downloads will be unauthenticated and "
          "subject to anonymous rate limits")

# ─────────────────────────── cell 2 (code) ───────────────────────────
step("cell_2_begin")
# ── Install deps ──────────────────────────────────────────────────────────
# onnxruntime-gpu wheel selection: Kaggle T4/P100 images currently ship
# CUDA 12.x; the default `onnxruntime-gpu` wheel matches. If you see
# `LoadLibrary failed for libonnxruntime_providers_cuda.so`, pin to the
# wheel that matches /usr/local/cuda/version.json (e.g.
# `onnxruntime-gpu==1.20.0` for CUDA 12.x, `==1.18.x` for CUDA 11.8).
DEPS = [
    "huggingface_hub",
    "hf_transfer",
    "jiwer",          # for WER on the short clip
    "numpy",
    "soundfile",
    "onnx-asr",       # istupakov/onnx-asr — the package name in #81
    "onnxruntime-gpu",
]
print("Installing:", " ".join(DEPS))
subprocess.run([sys.executable, "-m", "pip", "install", "-q", *DEPS], check=True)
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
print("✓ deps installed")

# ─────────────────────────── cell 3 (code) ───────────────────────────
step("cell_3_begin")
# ── Clone repo (full history — we need to check out two SHAs) ──────────────
if REPO.is_dir():
    subprocess.run("git fetch --all --tags", cwd=REPO, shell=True, check=True)
    print("✓ repo fetched")
else:
    # Note: NOT --depth 1 — we need the history to reach PRE_SHA.
    subprocess.run(
        f"git clone https://github.com/CrispStrobe/CrispASR.git {REPO}",
        shell=True, check=True,
    )
    print("✓ repo cloned")

# Sanity: both SHAs reachable?
for sha in (PRE_SHA, POST_SHA):
    r = subprocess.run(f"git -C {REPO} rev-parse {sha}",
                       shell=True, capture_output=True, text=True)
    assert r.returncode == 0, f"can't resolve {sha}: {r.stderr}"
    print(f"  {sha:14s} → {r.stdout.strip()[:12]}")

# ─────────────────────────── cell 4 (code) ───────────────────────────
step("cell_4_begin")
# ── Detect GPU + install fast build toolchain (ninja, ccache, mold) ────────
HAS_GPU = Path("/usr/local/cuda/bin/nvcc").is_file()
if not HAS_GPU:
    print("⚠ WARNING: nvcc not found — this run will use CPU and the "
          "issue #81 framing (CUDA EP vs CUDA backend) won't apply.")

# Install everything in one apt call so the layer cost is paid once.
# - ninja-build: parallel build graph (default Make is sequential per dir)
# - ccache:      caches compile results across rebuilds — POST build after
#                PRE only recompiles the few files c2423313 actually
#                touches; ccache makes that hit-rate near 100 %
# - mold:        ~5–10× faster linker than GNU ld for big projects
# `apt-get install` is silenced unless it actually fails; `pip install ninja`
# is the fallback for Kaggle images that don't allow apt.
print("Installing build toolchain (ninja, ccache, mold)…")
r = subprocess.run(
    "apt-get install -y ninja-build ccache mold 2>&1",
    shell=True, capture_output=True, text=True,
)
if r.returncode != 0:
    # Fallback: at least get ninja from pip; ccache/mold are unavailable.
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ninja"], check=True)
    print("  apt failed; ninja via pip, ccache+mold unavailable")
else:
    print("  apt: ninja+ccache+mold OK")

HAS_NINJA = shutil.which("ninja") is not None
HAS_CCACHE = shutil.which("ccache") is not None
HAS_MOLD = shutil.which("mold") is not None

# Persist ccache across notebook re-runs in /kaggle/working — survives
# session restart if you save the working dir; even within one session,
# the POST build benefits from PRE's compile artifacts.
CCACHE_DIR = WORK / ".ccache"
CCACHE_DIR.mkdir(exist_ok=True)
os.environ["CCACHE_DIR"] = str(CCACHE_DIR)
os.environ["CCACHE_MAXSIZE"] = "5G"  # plenty for two CUDA builds
if HAS_CCACHE:
    subprocess.run("ccache -M 5G && ccache -z", shell=True, capture_output=True)

# Detect the actual compute capability of the allocated GPU. The default
# ggml-cuda CMakeLists builds for 5–7 architectures (50/61/70/75/80/86,
# +89 on CUDA ≥ 11.8, +120a/121a on CUDA ≥ 12.8) — ~5× more nvcc work
# than we need on Kaggle. T4 = sm_75, P100 = sm_60, A100 = sm_80, L4 =
# sm_89. Querying nvidia-smi instead of hard-coding picks the right one
# regardless of which Kaggle accelerator type is allocated.
CUDA_ARCH = None
if HAS_GPU:
    r = subprocess.run(
        "nvidia-smi --query-gpu=compute_cap --format=csv,noheader",
        shell=True, capture_output=True, text=True,
    )
    if r.returncode == 0 and r.stdout.strip():
        # "7.5" → "75". Pick the lowest-compute device if multiple.
        cc = sorted(line.strip() for line in r.stdout.strip().splitlines())[0]
        CUDA_ARCH = cc.replace(".", "")
        print(f"  GPU compute cap: {cc} → CMAKE_CUDA_ARCHITECTURES={CUDA_ARCH}")
    else:
        print("  nvidia-smi compute_cap query failed; nvcc will build "
              "for the default fat-binary arch list (slow)")

CMAKE_FLAGS = [
    "-DCMAKE_BUILD_TYPE=Release",
    # Skip everything we don't need for libcrispasr.so. The CI windows
    # release job uses the same trio (release.yml:281-283).
    "-DCRISPASR_BUILD_TESTS=OFF",
    "-DCRISPASR_BUILD_EXAMPLES=OFF",
    "-DCRISPASR_BUILD_SERVER=OFF",
    "-DCMAKE_C_FLAGS=-fopenmp",
    "-DCMAKE_CXX_FLAGS=-fopenmp",
]
if HAS_CCACHE:
    CMAKE_FLAGS += [
        "-DCMAKE_C_COMPILER_LAUNCHER=ccache",
        "-DCMAKE_CXX_COMPILER_LAUNCHER=ccache",
        "-DCMAKE_CUDA_COMPILER_LAUNCHER=ccache",
    ]
if HAS_MOLD:
    # CMake 3.29+ added LINKER_TYPE but Kaggle's image (CMake 3.31 +
    # GCC 11.4 + nvcc) errors with "LINKER_TYPE 'MOLD' is unknown or
    # not supported by this toolchain." on the CUDA try_compile. Use
    # the linker flag directly instead — works on every cmake.
    for kind in ("EXE", "SHARED", "MODULE"):
        CMAKE_FLAGS.append(f"-DCMAKE_{kind}_LINKER_FLAGS=-fuse-ld=mold")
if HAS_GPU:
    cuda_stubs = "/usr/local/cuda/lib64/stubs"
    if Path(cuda_stubs).is_dir():
        os.environ["LIBRARY_PATH"] = f"{cuda_stubs}:{os.environ.get('LIBRARY_PATH', '')}"
    CMAKE_FLAGS += [
        "-DGGML_CUDA=ON",
        "-DGGML_CUDA_NO_VMM=ON",
        "-DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc",
    ]
    if CUDA_ARCH:
        CMAKE_FLAGS.append(f"-DCMAKE_CUDA_ARCHITECTURES={CUDA_ARCH}")
else:
    CMAKE_FLAGS.append("-DGGML_CUDA=OFF")

GENERATOR = ["-G", "Ninja"] if HAS_NINJA else []
print(f"  ninja={HAS_NINJA}  ccache={HAS_CCACHE}  mold={HAS_MOLD}  "
      f"cuda_arch={CUDA_ARCH or 'default(slow)'}")


def build_at(sha: str, out_lib: Path) -> None:
    """Check out `sha`, (re)build libcrispasr, copy the .so to `out_lib`.

    Incremental: keeps BUILD/ across SHAs so only files that actually
    changed between PRE_SHA and POST_SHA are recompiled. The flash-attn
    fusion commit touches a handful of conformer/parakeet files, so the
    POST build after PRE typically completes in <30 s with ccache hot.

    Build target is just the `crispasr` shared lib — examples/server/tests
    are off via CMAKE_FLAGS so cmake never even configures them.
    """
    print(f"\n=== build {sha} → {out_lib.name} ===")
    subprocess.run(f"git -C {REPO} checkout --quiet {sha}", shell=True, check=True)
    head = subprocess.run(f"git -C {REPO} log --oneline -1",
                          shell=True, capture_output=True, text=True).stdout.strip()
    print(f"  HEAD: {head}")
    BUILD.mkdir(exist_ok=True)
    if not (BUILD / "CMakeCache.txt").is_file():
        t_cfg = time.time()
        subprocess.run(
            ["cmake", "-S", str(REPO), "-B", str(BUILD)] + GENERATOR + CMAKE_FLAGS,
            check=True,
        )
        print(f"  cmake configure: {time.time() - t_cfg:.0f}s")
    t0 = time.time()
    # Ninja parallelises automatically; explicit -j$(nproc) is a no-op for
    # Ninja but matters for the Make fallback. CMAKE_BUILD_PARALLEL_LEVEL
    # gives the same effect cross-generator.
    env = {**os.environ, "CMAKE_BUILD_PARALLEL_LEVEL": str(os.cpu_count() or 4)}
    subprocess.run(
        f"cmake --build {BUILD} --target crispasr -j$(nproc)",
        shell=True, check=True, env=env,
    )
    el = time.time() - t0
    # Locate the .so. Layout differs between cmake/ninja generators slightly.
    candidates = list(BUILD.rglob("libcrispasr.so"))
    assert candidates, f"libcrispasr.so not found under {BUILD}"
    src = candidates[0]
    shutil.copy2(src, out_lib)
    print(f"  built in {el:.0f}s, copied {src.name} → {out_lib} "
          f"({out_lib.stat().st_size // (1024*1024)} MB)")
    if HAS_CCACHE:
        # Show the cache hit-rate so it's obvious why POST is so much faster
        # than PRE (or why it's not, if something invalidates the cache).
        r = subprocess.run("ccache -s | grep -E 'cache hit|cache miss|files in cache'",
                           shell=True, capture_output=True, text=True)
        if r.stdout:
            print("  ccache stats:")
            for line in r.stdout.strip().splitlines():
                print(f"    {line.strip()}")


# ─────────────────────────── cell 5 (code) ───────────────────────────
step("cell_5_begin")
# ── Build PRE then POST (incremental) ──────────────────────────────────────
build_at(PRE_SHA, LIB_PRE)
build_at(POST_SHA, LIB_POST)

# Restore the worktree to a SHA that has tools/benchmark_asr_engines.py.
# That script was added in 18801e29 (2026-05-10), AFTER c2423313 (the
# flash-attn fusion, 2026-05-09). After building POST_SHA, the worktree
# is on c2423313 and the bench script doesn't exist there. The two .so
# we need are already snapshotted to LIB_PRE / LIB_POST; the worktree
# state from here on only matters for the matrix runner. `main` always
# has the latest matrix runner.
print("\nrestoring worktree to main for the bench runner…")
subprocess.run(f"git -C {REPO} checkout --quiet main", shell=True, check=True)
head = subprocess.run(f"git -C {REPO} log --oneline -1",
                      shell=True, capture_output=True, text=True).stdout.strip()
print(f"  worktree HEAD: {head}")

print(f"\nlib sizes:")
for p in (LIB_PRE, LIB_POST):
    print(f"  {p.name:24s} {p.stat().st_size // (1024*1024):4d} MB")

# ─────────────────────────── cell 6 (code) ───────────────────────────
step("cell_6_begin")
# ── Pre-download GGUF + ONNX models ────────────────────────────────────────
from huggingface_hub import hf_hub_download, snapshot_download  # noqa: E402

GGUF_FILES = {
    "q4_k": "parakeet-tdt-0.6b-v3-q4_k.gguf",
    "q5_0": "parakeet-tdt-0.6b-v3-q5_0.gguf",
    "q8_0": "parakeet-tdt-0.6b-v3-q8_0.gguf",
    "f16":  "parakeet-tdt-0.6b-v3.gguf",
}
for q in GGUF_QUANTS.split(","):
    fname = GGUF_FILES[q.strip()]
    dst = GGUF_DIR / fname
    if not dst.is_file():
        print(f"  ↓ {fname}")
        hf_hub_download("cstr/parakeet-tdt-0.6b-v3-GGUF", fname,
                        local_dir=str(GGUF_DIR))
    else:
        print(f"  ✓ cached {fname}")

# ONNX: snapshot pulls encoder + decoder + joint + vocab + config.
print("  ↓ ONNX snapshot (istupakov/parakeet-tdt-0.6b-v3-onnx)")
snapshot_download("istupakov/parakeet-tdt-0.6b-v3-onnx",
                  local_dir=str(ONNX_DIR))
print("✓ models ready")

# ─────────────────────────── cell 7 (code) ───────────────────────────
step("cell_7_begin")
# ── Sanity: confirm onnxruntime sees CUDA ──────────────────────────────────
import onnxruntime as ort  # noqa: E402
avail = ort.get_available_providers()
print("onnxruntime version :", ort.__version__)
print("onnxruntime EPs     :", avail)
if "CUDAExecutionProvider" not in avail:
    print("⚠ CUDAExecutionProvider not available — onnx-asr will fall back.")
    print("  Likely cause: CUDA major version mismatch between Kaggle image")
    print("  and the onnxruntime-gpu wheel. Check /usr/local/cuda/version.json")
    print("  and pin onnxruntime-gpu to the matching wheel.")

# ─────────────────────────── cell 8 (code) ───────────────────────────
step("cell_8_begin")
# ── Run the three benchmarks ───────────────────────────────────────────────
BENCH = REPO / "tools" / "benchmark_asr_engines.py"
assert BENCH.is_file(), f"{BENCH} not present at HEAD; check POST_SHA"


def run_bench(label: str, lib: Path | None, engine: str,
              extra: list[str]) -> Path:
    """Invoke the existing matrix bench with one engine selection.

    We force `--engine {crispasr,onnx}` rather than `both` per call so each
    JSON sidecar is unambiguous about which library it represents (PRE,
    POST, or onnx-CUDA), and so a CUDA-OOM in one cell doesn't poison
    the other two.
    """
    json_path = RESULTS_DIR / f"{label}.json"
    cmd = [
        sys.executable, str(BENCH),
        "--engine", engine,
        "--mode", MODES,
        "--audio", AUDIOS,
        "--gguf-quants", GGUF_QUANTS,
        "--onnx-quants", ONNX_QUANTS,
        "--window-s", str(WINDOW_S),
        "--warmups", str(WARMUPS),
        "--runs", str(RUNS),
        "--gguf-dir", str(GGUF_DIR),
        "--onnx-dir", str(ONNX_DIR),
        "--gpu-backend", "cuda" if HAS_GPU else "cpu",
        "--prewarm",
        "--json", str(json_path),
        *extra,
    ]
    if lib is not None:
        cmd += ["--crispasr-lib", str(lib)]
    print(f"\n=== {label} ===")
    print(" ".join(cmd))
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True)
    el = time.time() - t0
    # Stream the bench's own table to the notebook output.
    print(r.stdout[-4000:])
    if r.returncode != 0:
        print(f"⚠ {label} exited with {r.returncode}; stderr tail:")
        print(r.stderr[-2000:])
    print(f"  wallclock: {el:.0f}s  →  {json_path}")
    return json_path


PROVIDERS_CUDA = ["CUDAExecutionProvider", "CPUExecutionProvider"]

j_pre  = run_bench("crispasr-pre",  LIB_PRE,  "crispasr", [])
j_post = run_bench("crispasr-post", LIB_POST, "crispasr", [])
j_onnx = run_bench("onnx-cuda",     None,     "onnx",
                   ["--providers", ",".join(PROVIDERS_CUDA)])

# ─────────────────────────── cell 9 (code) ───────────────────────────
step("cell_9_begin")
# ── Aggregate ──────────────────────────────────────────────────────────────
def load(json_path: Path) -> dict:
    with json_path.open() as f:
        return json.load(f)


def fmt_ms(s: float) -> str:
    return f"{s * 1000:.0f}ms" if s else "—"


def cell_key(r: dict) -> tuple:
    """Group cells across the three runs by (audio, mode, quant-family)."""
    return (r["audio"], r["mode"])


pre_results = load(j_pre)["results"]
post_results = load(j_post)["results"]
onnx_results = load(j_onnx)["results"]


def summarize(results: list[dict], engine_label: str) -> dict:
    """Index by (audio, mode, quant) → row-summary dict."""
    out = {}
    for r in results:
        if "error" in r.get("extra", {}):
            continue
        key = (r["audio"], r["mode"], r["quant"])
        cs = r.get("call_stats") or {}
        out[key] = {
            "engine": engine_label,
            "audio_seconds": r["audio_seconds"],
            "load_s": r["load_s"],
            "runs_s": r["runs_s"],
            "mean_run_s": r["mean_run_s"],
            "rt": r["realtime_factor"],
            "p50_ms": cs.get("p50_ms"),
            "p95_ms": cs.get("p95_ms"),
            "transcript_sample": r.get("transcript_sample", "")[:60],
            "wer": r.get("wer"),
        }
    return out


pre  = summarize(pre_results,  "crispasr-PRE")
post = summarize(post_results, "crispasr-POST")
onnx = summarize(onnx_results, "onnx-cuda")

# Print headline table.
print("\n\n=== HEADLINE: parakeet TDT 0.6B v3, CUDA, runs={}, chunked {}s ===\n"
      .format(RUNS, WINDOW_S))
print("| audio | engine | quant | mean_run | RT | p50 | p95 | runs_s |")
print("|---|---|---|---|---|---|---|---|")

all_keys = sorted(set(pre) | set(post) | set(onnx))
for key in all_keys:
    audio, mode, quant = key
    for label, table in (("PRE", pre), ("POST", post), ("onnx-CUDA", onnx)):
        if key not in table:
            continue
        r = table[key]
        print(
            f"| {audio:5s} | {label:10s} | {quant:5s} | "
            f"{r['mean_run_s']:.3f}s | {r['rt']:.1f}× | "
            f"{fmt_ms(r['p50_ms'] / 1000 if r['p50_ms'] else None)} | "
            f"{fmt_ms(r['p95_ms'] / 1000 if r['p95_ms'] else None)} | "
            f"{[round(x, 2) for x in r['runs_s']]} |"
        )

# Print A/B deltas: PRE → POST and POST → onnx-CUDA.
print("\n=== A/B deltas ===\n")
print("| audio | quant | PRE mean | POST mean | flash-attn gain | onnx mean | gap remaining |")
print("|---|---|---|---|---|---|---|")
for (audio, mode, quant) in all_keys:
    p = pre.get((audio, mode, quant))
    o = post.get((audio, mode, quant))
    # ONNX cell uses the int8 quant by convention; pair with GGUF q8_0.
    n = onnx.get((audio, mode, "int8")) or onnx.get((audio, mode, quant))
    if not (p and o):
        continue
    gain = (p["mean_run_s"] - o["mean_run_s"]) / p["mean_run_s"] * 100
    n_str = f"{n['mean_run_s']:.3f}s" if n else "—"
    gap_str = (f"{(o['mean_run_s'] / n['mean_run_s']):.2f}× slower"
               if n else "—")
    print(f"| {audio} | {quant} | {p['mean_run_s']:.3f}s | "
          f"{o['mean_run_s']:.3f}s | {gain:+.0f}% | {n_str} | {gap_str} |")

# Rising-runs pattern check (the issue #81 reporter saw runs[2] > runs[0]).
print("\n=== rising-runs check (CrispASR POST, all cells) ===\n")
for (audio, mode, quant), r in post.items():
    runs = r["runs_s"]
    if len(runs) < 3:
        continue
    rising = sum(runs[i + 1] > runs[i] for i in range(len(runs) - 1))
    trend = "↑" * rising + "↓" * (len(runs) - 1 - rising)
    print(f"  {audio} {mode} {quant}: "
          f"{[round(x, 2) for x in runs]}  trend {trend}  "
          f"(last/first = {runs[-1] / runs[0]:.2f}×)")

# ─────────────────────────── cell 10 (code) ───────────────────────────
step("cell_10_begin")
# ── Persist + (optional) gist ──────────────────────────────────────────────
combined = {
    "host": {
        "datetime_utc": datetime.utcnow().isoformat(),
        "has_gpu": HAS_GPU,
        "ort_version": ort.__version__,
        "ort_providers": avail,
        "pre_sha": PRE_SHA,
        "post_sha": POST_SHA,
    },
    "runs": RUNS,
    "warmups": WARMUPS,
    "window_s": WINDOW_S,
    "results": {
        "crispasr-pre":  load(j_pre),
        "crispasr-post": load(j_post),
        "onnx-cuda":     load(j_onnx),
    },
}
combined_path = RESULTS_DIR / "issue81-cuda-ab.json"
combined_path.write_text(json.dumps(combined, indent=2))
print(f"\n✓ wrote {combined_path}")

# Optional: push to a GitHub gist for sharing in the issue thread.
# GH_GIST_TOKEN was already pulled from Kaggle secrets in cell 1b.
if GH_GIST_TOKEN:
    import requests
    desc = (f"CrispASR issue #81 CUDA A/B — "
            f"pre={PRE_SHA} post={POST_SHA} runs={RUNS}")
    payload = {
        "description": desc,
        "public": False,
        "files": {"issue81-cuda-ab.json": {"content": combined_path.read_text()}},
    }
    r = requests.post(
        "https://api.github.com/gists",
        json=payload,
        headers={"Authorization": f"token {GH_GIST_TOKEN}",
                 "Accept": "application/vnd.github+json"},
    )
    if r.status_code in (200, 201):
        print(f"✓ gist: {r.json()['html_url']}")
    else:
        print(f"⚠ gist push failed: {r.status_code} {r.text[:200]}")
else:
    print("(no GH_GIST_TOKEN — skipping gist push; download "
          f"{combined_path} from /kaggle/working manually)")
