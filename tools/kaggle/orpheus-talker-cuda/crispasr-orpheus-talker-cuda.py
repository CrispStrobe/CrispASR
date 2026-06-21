"""
CrispASR -- orpheus talker CUDA pipeline: build + quantize + diff + upload.

Orpheus passes on M1 Metal/CPU but the sweep reported a 0-byte on CUDA
(PLAN §201). This kernel, in one CUDA run (resilient — a failing step is
recorded and the rest continue, so the run always commits its output):

  1. Build CrispASR (CUDA): crispasr-diff + crispasr-quantize, warm ccache.
  2. Download unsloth/orpheus-3b-0.1-ft (tara) + convert to F16 GGUF.
  3. Quantize with **crispasr-quantize** -> Q8_0 and Q4_K.
  4. Generate the talker reference (greedy codec stream; GPU+F32 on T4,
     CPU+bf16 on P100 since its sm_60 is unsupported by modern torch).
  5. Diff the talker AR decode: CPU vs GPU vs ground truth, + SNAC vocoder.
  6. Upload f16/q8_0/q4_k GGUFs + the talker ref to cstr/orpheus-3b-0.1-ft-GGUF.
  7. Save the warmed ccache to /kaggle/working/ccache.tar for dataset refresh.
"""

import json
import os
import subprocess
import sys
import time
import traceback
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
HF_DIR = WORK / "orpheus-3b-hf"
F16 = WORK / "orpheus-3b-0.1-ft-f16.gguf"
Q8 = WORK / "orpheus-3b-0.1-ft-q8_0.gguf"
Q4 = WORK / "orpheus-3b-0.1-ft-q4_k.gguf"
REF = WORK / "orpheus-talker-ref.gguf"

CRISPASR_REF = os.environ.get("CRISPASR_REF", "main")
CRISPASR_REPO = os.environ.get("CRISPASR_REPO", "https://github.com/CrispStrobe/CrispASR.git")
HF_MODEL = os.environ.get("ORPHEUS_HF_MODEL", "unsloth/orpheus-3b-0.1-ft")
HF_REPO = os.environ.get("ORPHEUS_HF_REPO", "cstr/orpheus-3b-0.1-ft-GGUF")
TEXT = os.environ.get("ORPHEUS_TEXT", "Hey there, my name is Tara.")
SPEAKER = os.environ.get("ORPHEUS_SPEAKER", "tara")
MAXGEN = os.environ.get("ORPHEUS_DIFF_MAXGEN", "48")
DO_UPLOAD = os.environ.get("ORPHEUS_UPLOAD", "1") != "0"

PROGRESS = RESULTS / "progress.jsonl"
SUMMARY = {"steps": {}, "errors": {}, "results": []}
_T0 = time.time()


def step(name, **kv):
    line = json.dumps({"t": round(time.time() - _T0, 2), "step": name, **kv})
    print(f"[step] {line}", flush=True)
    with open(PROGRESS, "a") as f:
        f.write(line + "\n")
    SUMMARY["steps"][name] = kv or True
    _write_summary()


def _write_summary():
    try:
        (RESULTS / "summary.json").write_text(json.dumps(SUMMARY, indent=2))
    except Exception:
        pass


def run(cmd, check=True, env=None, cwd=None, timeout=None):
    print(f"\n$ {' '.join(str(c) for c in cmd)}", flush=True)
    e = os.environ.copy()
    if env:
        e.update(env)
    r = subprocess.run(cmd, env=e, cwd=cwd, timeout=timeout)
    if check and r.returncode != 0:
        raise RuntimeError(f"command failed (rc={r.returncode}): {cmd}")
    return r


def safe(label, fn, *a, **k):
    """Run fn; on failure record the error and continue (never abort the run)."""
    try:
        return fn(*a, **k)
    except Exception as e:
        msg = f"{type(e).__name__}: {e}"
        print(f"[ERROR] {label}: {msg}\n{traceback.format_exc()}", flush=True)
        SUMMARY["errors"][label] = msg
        step(f"{label}_FAILED", err=msg)
        return None


# ── Clone + build (fatal — nothing works without the binaries) ─────────────
step("start", ref=CRISPASR_REF)
if REPO.exists():
    import shutil
    shutil.rmtree(REPO)
run(["git", "clone", "--depth", "1", "--branch", CRISPASR_REF, "--recursive", CRISPASR_REPO, str(REPO)])
sys.path.insert(0, os.path.join(str(REPO), "tools", "kaggle"))
import kaggle_harness as kh  # noqa: E402

kh.init_progress()
sha = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
SUMMARY["sha"] = sha
step("cloned", sha=sha)

run(["nvidia-smi", "-L"])
gpu_name = subprocess.check_output(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True).strip()
SUMMARY["gpu"] = gpu_name
step("gpu", gpu_name=gpu_name)

kh.install_build_toolchain()
arch = kh.detect_cuda_arch()
step("cuda_arch", arch=arch)
BUILD.mkdir(parents=True, exist_ok=True)
run(["cmake", "-S", str(REPO), "-B", str(BUILD), "-DCMAKE_BUILD_TYPE=Release",
     "-DBUILD_SHARED_LIBS=ON"] + kh.cuda_build_flags(arch) + kh.cache_and_link_flags())
step("cmake_done")
with kh.build_heartbeat("cmake.build"):
    kh.sh_with_progress(
        f"stdbuf -oL -eL cmake --build {BUILD} --target crispasr-diff crispasr-quantize "
        f"-j{kh.safe_build_jobs(gpu=True)}")
step("build_done")


def _find(name):
    p = BUILD / "bin" / name
    if p.exists():
        return p
    c = [x for x in BUILD.rglob(name) if x.is_file() and os.access(x, os.X_OK)]
    if not c:
        raise RuntimeError(f"{name} not found after build")
    return c[0]


DIFF = _find("crispasr-diff")
QUANT = _find("crispasr-quantize")
step("binaries", diff=str(DIFF), quant=str(QUANT))
os.environ["LD_LIBRARY_PATH"] = f"{BUILD / 'src'}:{os.environ.get('LD_LIBRARY_PATH', '')}"

# ── ccache save-back (snapshot the warm cache for dataset refresh) ──────────
def save_ccache():
    cc = WORK / ".ccache"
    if cc.exists():
        run(["tar", "cf", str(WORK / "ccache.tar"), "-C", str(WORK), ".ccache"], check=False)
        step("ccache_saved", mb=round((WORK / "ccache.tar").stat().st_size / 1e6, 1)
             if (WORK / "ccache.tar").exists() else 0)


# ── Download HF model + convert + quantize (each isolated) ─────────────────
hf_token = kh.resolve_hf_token()
step("hf_token", have=bool(hf_token))


def do_download():
    from huggingface_hub import snapshot_download
    snapshot_download(HF_MODEL, local_dir=str(HF_DIR), token=hf_token,
                      allow_patterns=["*.json", "*.safetensors", "*.model", "tokenizer*", "*.txt"])
    step("hf_downloaded")


safe("download", do_download)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "gguf"], check=False)


def do_convert():
    run([sys.executable, str(REPO / "models" / "convert-orpheus-to-gguf.py"),
         "--input", str(HF_DIR), "--output", str(F16), "--outtype", "f16", "--variant", "fixed_speaker"],
        env={"OMP_NUM_THREADS": "1", "PYTHONUNBUFFERED": "1"})
    step("converted_f16", mb=round(F16.stat().st_size / 1e6, 1))


safe("convert", do_convert)

offset = count = None
if F16.exists():
    import gguf as _g
    _r = _g.GGUFReader(str(F16))
    offset = int(_r.fields["orpheus.custom_token_offset"].parts[-1][0])
    count = int(_r.fields["orpheus.custom_token_count"].parts[-1][0])
    SUMMARY["codec_offset"], SUMMARY["codec_count"] = offset, count
    step("codec_range", offset=offset, count=count)
    for out, ftype in [(Q8, "q8_0"), (Q4, "q4_k")]:
        safe(f"quantize_{ftype}", lambda o=out, t=ftype: (
            run([str(QUANT), str(F16), str(o), t]),
            step(f"quantized_{t}", mb=round(o.stat().st_size / 1e6, 1))))

safe("ccache_save", save_ccache)


# ── Generate talker reference ──────────────────────────────────────────────
def do_ref():
    with kh.build_heartbeat("ref.gen"):
        run([sys.executable, str(REPO / "tools" / "dump_reference.py"),
             "--backend", "orpheus-talker", "--model-dir", str(HF_DIR),
             "--audio", str(REPO / "samples" / "jfk.wav"), "--output", str(REF),
             "--max-new-tokens", MAXGEN],
            env={"ORPHEUS_TEXT": TEXT, "ORPHEUS_SPEAKER": SPEAKER,
                 "ORPHEUS_CUSTOM_OFFSET": str(offset), "ORPHEUS_CUSTOM_COUNT": str(count),
                 # Kaggle workers have ~20 GB RAM (not the doc's 13): use F32 for
                 # a clean ground truth even on a P100 (sm_60, no torch GPU).
                 "ORPHEUS_REF_DTYPE": os.environ.get("ORPHEUS_REF_DTYPE", "float32"),
                 "OMP_NUM_THREADS": "1", "PYTHONUNBUFFERED": "1", "HF_HUB_OFFLINE": "1"})
    step("ref_generated", ref_kb=round(REF.stat().st_size / 1e3, 1))


if F16.exists():
    safe("ref_gen", do_ref)


# ── Diffs (each isolated) ──────────────────────────────────────────────────
def run_diff(label, backend, model, ref, extra_env, timeout=900):
    step(f"{label}_start")
    cmd = [str(DIFF), backend, str(model), str(ref), str(REPO / "samples" / "jfk.wav")]
    t0 = time.time()
    try:
        r = subprocess.run(cmd, timeout=timeout, stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, text=True, env={**os.environ, **extra_env})
        rc, out = r.returncode, r.stdout
    except subprocess.TimeoutExpired as ex:
        rc = -1
        out = ex.stdout.decode(errors="replace") if isinstance(ex.stdout, bytes) else (ex.stdout or "")
    (RESULTS / f"{label}.txt").write_text(out or "")
    print((out or "")[-3500:], flush=True)
    verdict = "PASS" if ("→ PASS" in (out or "") or "0 fail" in (out or "")) else \
        ("FAIL" if ("FAIL" in (out or "")) else "?")
    res = {"label": label, "rc": rc, "elapsed": round(time.time() - t0, 2), "verdict": verdict}
    SUMMARY["results"].append(res)
    step(f"{label}_done", **res)
    return res


if REF.exists():
    safe("talker_cpu", run_diff, "talker_cpu", "orpheus-talker", F16, REF, {"ORPHEUS_DIFF_MAXGEN": MAXGEN})
    safe("talker_gpu", run_diff, "talker_gpu", "orpheus-talker", F16, REF,
         {"ORPHEUS_DIFF_MAXGEN": MAXGEN, "ORPHEUS_DIFF_GPU": "1"})


def do_snac():
    from huggingface_hub import hf_hub_download
    hf_hub_download("cstr/snac-24khz-GGUF", "snac-24khz.gguf", local_dir=str(WORK), token=hf_token)
    hf_hub_download("cstr/snac-24khz-GGUF", "diff-harness-ref/orpheus-snac-ref.gguf", local_dir=str(WORK), token=hf_token)
    snac = WORK / "snac-24khz.gguf"
    snac_ref = WORK / "diff-harness-ref" / "orpheus-snac-ref.gguf"
    run_diff("snac_cpu", "orpheus", snac, snac_ref, {})
    run_diff("snac_gpu", "orpheus", snac, snac_ref, {"ORPHEUS_SNAC_GPU": "1"})


safe("snac", do_snac)


# ── Upload GGUFs + ref to HF (each file isolated) ──────────────────────────
def do_upload():
    from huggingface_hub import HfApi
    api = HfApi(token=hf_token)
    api.create_repo(HF_REPO, repo_type="model", exist_ok=True)
    for f, dst in [(F16, F16.name), (Q8, Q8.name), (Q4, Q4.name),
                   (REF, "diff-harness-ref/orpheus-talker-ref.gguf")]:
        if not f.exists():
            continue
        safe(f"upload_{f.name}", lambda f=f, dst=dst: (
            api.upload_file(path_or_fileobj=str(f), path_in_repo=dst, repo_id=HF_REPO, repo_type="model"),
            step("uploaded", file=dst)))


if DO_UPLOAD and hf_token:
    safe("upload", do_upload)

SUMMARY["hf_repo"] = HF_REPO
_write_summary()
step("done", n_results=len(SUMMARY["results"]), n_errors=len(SUMMARY["errors"]))
print("\n=== SUMMARY ===", flush=True)
for r in SUMMARY["results"]:
    print(f"  {r['label']:12s} rc={r['rc']} verdict={r['verdict']}", flush=True)
if SUMMARY["errors"]:
    print("  errors:", json.dumps(SUMMARY["errors"]), flush=True)
