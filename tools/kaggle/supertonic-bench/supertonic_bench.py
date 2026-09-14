#!/usr/bin/env python3
"""Supertonic-3: CrispASR (--backend supertonic) vs audio.cpp (--family supertonic).

Same text, same voice, same steps, same seed, same machine, same CUDA arch.

WHAT THIS ESTABLISHES
  * wall-clock + realtime-factor per arm, CPU and GPU, for both engines
  * CrispASR at f16 AND q4_k; audio.cpp at f16 AND q8_0 (it publishes no q4_k)
  * audio equivalence: cosine on the PCM PAYLOAD (the `data` chunk, parsed by
    hand -- never the WAV file, whose LIST/INFO + C2PA chunks carry timestamps),
    against each other AND against the upstream ONNX Runtime reference, plus an
    ASR roundtrip on every wav.

READOUTS THAT CAN REPORT FAILURE (house rules, not suggestions)
  * every arm carries `status`: ok | BUILD_FAILED | ARGUMENT_REJECTED |
    RUN_FAILED | NO_OUTPUT. A missing engine can never read as "it was slow":
    `verdict.audiocpp_built` is a separate boolean and the summary prints
    ===AUDIOCPP-BUILD-FAILED=== when it is false.
  * TIMING POSITIVE CONTROL: every arm is also run at 32 flow steps. 4x the
    denoising work MUST show up as a slower wall clock (>= 1.3x). An arm whose
    8-step and 32-step numbers come out the same has an instrument that cannot
    discriminate, and its timing is marked inconclusive rather than reported.
  * METRIC SELF-TEST: the cosine/magnitude comparator is run on known-answer
    inputs before it touches real audio -- identical (must be 1.0), half-scale
    (cosine must STILL be 1.0 while the magnitude ratio reports 2.0, which is
    the whole point of printing the norms), and white noise (must be < 0.2).
    A comparator that prints the same thing for all three is not measuring.

NOT MEASURED, AND WHY (stated, never estimated)
  * CrispASR TTS output is watermarked by default and that watermark is an
    STFT analyse-modify-resynthesise pass over the whole signal -- a real cost
    and a real change to the PCM. So CrispASR is timed BOTH as shipped and with
    --no-watermark, and both numbers are reported. Getting --no-watermark
    honoured at all requires the native C2PA signer compiled IN (the CLI forces
    the watermark back on for any output that cannot carry a manifest), so this
    kernel deliberately does NOT pass -DCRISPASR_NO_C2PA_NATIVE=ON.

SCRIPT_VERSION = v3
"""

import json
import math
import os
import re
import shutil
import struct
import subprocess
import sys
import time
from pathlib import Path

SCRIPT_VERSION = "v3"

WORK = Path("/kaggle/working")
TEMP = Path("/kaggle/temp/st-bench")
REPO = Path("/kaggle/temp/CrispASR")
ACPP = Path("/kaggle/temp/audio.cpp")
MODELS = TEMP / "models"
OUTS = TEMP / "wav"
RESULTS = WORK / "results.json"

for p in (WORK, TEMP, MODELS, OUTS):
    p.mkdir(parents=True, exist_ok=True)

CRISPASR_BRANCH = "main"
CRISPASR_URL = "https://github.com/CrispStrobe/CrispASR.git"
AUDIOCPP_URL = "https://github.com/0xShug0/audio.cpp.git"

LANG, VOICE, SEED = "en", "M1", 1234
STEPS, STEPS_CONTROL = 8, 32

# Both engines chunk at 300 chars for non-ko/ja (CrispASR: st_chunk_text(text,
# 300) in src/supertonic_tts.cpp; audio.cpp: --text-chunk-size default 300), so
# SHORT and LONG are single-chunk on both and XL chunks identically on both.
TEXT_SHORT = "The quick brown fox jumps over the lazy dog."
TEXT_LONG = (
    "The quick brown fox jumps over the lazy dog. Pack my box with five dozen "
    "liquor jugs. How vexingly quick daft zebras jump. The five boxing wizards "
    "jump quickly, and bright vixens jab at my lazy dog. Sphinx of black quartz, "
    "judge my vow."
)
TEXT_XL = " ".join([TEXT_LONG] * 4)

LONG_REPS = 3   # best-of, to take the noise out of the headline number
XL_REPS = 2


# ───────────────────────────── plumbing ────────────────────────────────────

def run(argv, *, cwd=None, env=None, timeout=7200, capture=True, check=False):
    merged = os.environ.copy()
    if env:
        merged.update({str(k): str(v) for k, v in env.items()})
    print("$ " + " ".join(map(str, argv)), flush=True)
    return subprocess.run([str(x) for x in argv], cwd=cwd, env=merged, check=check,
                          timeout=timeout, text=True, capture_output=capture,
                          errors="replace")


def out_tail(p, n=3000):
    return ((p.stdout or "")[-n:], (p.stderr or "")[-n:])


# ─────────────────────── environment provenance ────────────────────────────

print(f"=== SCRIPT_VERSION={SCRIPT_VERSION} ===", flush=True)
HW = {}
try:
    r = run(["nvidia-smi", "--query-gpu=name,compute_cap,memory.total,driver_version",
             "--format=csv,noheader"], timeout=120)
    HW["gpu_csv"] = (r.stdout or "").strip()
    print("GPU: " + HW["gpu_csv"], flush=True)
except Exception as e:
    HW["gpu_csv"] = f"nvidia-smi failed: {e}"
HW["nproc"] = os.cpu_count() or 4
try:
    HW["cpu_model"] = next(l.split(":", 1)[1].strip() for l in
                           open("/proc/cpuinfo") if l.startswith("model name"))
except Exception:
    HW["cpu_model"] = "unknown"
try:
    HW["mem_total_kb"] = int(next(l.split()[1] for l in open("/proc/meminfo")
                                  if l.startswith("MemTotal")))
except Exception:
    HW["mem_total_kb"] = 0
try:
    HW["nvcc"] = (run(["nvcc", "--version"], timeout=120).stdout or "").strip().splitlines()[-1]
except Exception:
    HW["nvcc"] = "unknown"
try:
    HW["gxx"] = (run(["g++", "--version"], timeout=120).stdout or "").splitlines()[0]
except Exception:
    HW["gxx"] = "unknown"
print(json.dumps(HW, indent=1), flush=True)

NTHREADS = str(HW["nproc"])

# ─────────────────────────── clone + harness ───────────────────────────────

if not REPO.exists():
    run(["git", "clone", "--depth", "1", "--branch", CRISPASR_BRANCH, "--recursive",
         CRISPASR_URL, str(REPO)], timeout=3600, capture=False, check=True)
sys.path.insert(0, str(REPO / "tools" / "kaggle"))
import kaggle_harness as kh  # noqa: E402

kh.init_progress()
kh.resolve_hf_token()
CRISPASR_SHA = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                                       text=True).strip()
kh.step("provenance", script_version=SCRIPT_VERSION, crispasr=CRISPASR_SHA)

kh.step("deps")
run([sys.executable, "-m", "pip", "install", "--quiet",
     "onnx", "onnxruntime", "gguf", "soundfile", "huggingface_hub"], timeout=1800)
import numpy as np  # noqa: E402
from huggingface_hub import hf_hub_download, snapshot_download  # noqa: E402

# ───────────────────────────── downloads ───────────────────────────────────

kh.step("download.models")
ST_UPSTREAM = MODELS / "supertonic-3-onnx"
snapshot_download("Supertone/supertonic-3", local_dir=str(ST_UPSTREAM),
                  allow_patterns=["onnx/*", "voice_styles/*", "LICENSE", "README.md"])

CA_F16 = Path(hf_hub_download("cstr/supertonic-3-GGUF", "supertonic3-f16.gguf",
                              local_dir=str(MODELS / "crispasr")))

# audio.cpp GGUF model dirs: the family spec (model_specs/supertonic.json) wants
# <dir>/*.gguf + <dir>/config/{tts,unicode_indexer}.json + <dir>/voice_styles/*.json.
AC_CONFIG_SRC = MODELS / "supertonic-3-mlx"
snapshot_download("mlx-community/supertonic-3-mlx", local_dir=str(AC_CONFIG_SRC),
                  allow_patterns=["config/*", "voice_styles/*"])


def audiocpp_model_dir(fname, tag):
    d = MODELS / f"audiocpp-{tag}"
    d.mkdir(parents=True, exist_ok=True)
    gguf = hf_hub_download("audio-cpp/audio.cpp-gguf", f"Supertonic-3-GGUF/{fname}",
                           local_dir=str(MODELS / "audiocpp-dl"))
    dst = d / fname
    if not dst.exists():
        shutil.copy2(gguf, dst)
    for sub in ("config", "voice_styles"):
        tgt = d / sub
        if not tgt.exists():
            shutil.copytree(AC_CONFIG_SRC / sub, tgt)
    return d


AC_F16_DIR = audiocpp_model_dir("supertonic-3-f16.gguf", "f16")
AC_Q8_DIR = audiocpp_model_dir("supertonic-3-q8_0.gguf", "q8_0")
kh.step("download.done",
        crispasr_f16_mb=round(CA_F16.stat().st_size / 1e6, 1),
        audiocpp_f16_mb=round((AC_F16_DIR / "supertonic-3-f16.gguf").stat().st_size / 1e6, 1),
        audiocpp_q8_mb=round((AC_Q8_DIR / "supertonic-3-q8_0.gguf").stat().st_size / 1e6, 1))


GGML_TYPE_NAMES = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1", 8: "Q8_0",
    9: "Q8_1", 10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K",
    15: "Q8_K", 16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS", 19: "IQ1_S",
    20: "IQ4_NL", 21: "IQ3_S", 22: "IQ2_S", 23: "IQ4_XS", 24: "I8", 25: "I16",
    26: "I32", 27: "I64", 28: "F64", 29: "IQ1_M", 30: "BF16", 39: "TQ1_0",
    40: "TQ2_0",
}


def gguf_dtype_histogram(path):
    """What is ACTUALLY in the file. A package labelled q8_0 that is byte-for-byte
    the size of the orig-dtype package has earned the question.

    Hand-rolled, and NOT gguf.GGUFReader, because that reader expands every KV
    array ELEMENT BY ELEMENT into its own numpy object (see _get_field_parts:
    `for idx in range(alen[0]): ... aparts += curr_parts`). audio.cpp's package
    carries its 37 companion resources inside the GGUF as one uint8 KV array of
    57,057,963 elements, so asking that reader for a tensor list allocated tens
    of gigabytes and the OOM killer took the whole run down -- after the build,
    after the quantize, for a diagnostic line. This walks the header, seeks past
    array payloads without materialising them, and stops at the tensor table."""
    SCALAR = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
    try:
        with open(path, "rb") as f:
            def rd(fmt):
                return struct.unpack(fmt, f.read(struct.calcsize(fmt)))[0]
            if f.read(4) != b"GGUF":
                return {"error": "not a GGUF file"}
            rd("<I")                      # version
            n_tensors = rd("<Q")
            n_kv = rd("<Q")

            def skip_str():
                f.seek(rd("<Q"), 1)

            def skip_val(t):
                if t == 8:
                    skip_str()
                elif t == 9:
                    et = rd("<I")
                    n = rd("<Q")
                    if et == 8:
                        for _ in range(n):
                            f.seek(rd("<Q"), 1)
                    else:
                        f.seek(SCALAR[et] * n, 1)   # seek, never allocate
                else:
                    f.seek(SCALAR[t], 1)

            kv_keys = []
            for _ in range(n_kv):
                n = rd("<Q")
                kv_keys.append(f.read(n).decode("utf-8", "replace"))
                skip_val(rd("<I"))

            hist, nelem = {}, 0
            for _ in range(n_tensors):
                skip_str()
                nd = rd("<I")
                dims = [rd("<Q") for _ in range(nd)]
                tt = rd("<I")
                rd("<Q")                  # offset
                name = GGML_TYPE_NAMES.get(tt, f"type_{tt}")
                hist[name] = hist.get(name, 0) + 1
                c = 1
                for d in dims:
                    c *= d
                nelem += c
        return {"file_mb": round(Path(path).stat().st_size / 1e6, 1),
                "n_tensors": int(n_tensors), "elements": int(nelem),
                "types": hist, "kv_keys": kv_keys}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


# ───────────────────────── toolchain + builds ──────────────────────────────

kh.install_build_toolchain()
ARCH = kh.detect_cuda_arch()
kh.step("toolchain", cuda_arch=ARCH, threads=NTHREADS)

BUILD_CA = TEMP / "build-crispasr"
BUILD_AC = TEMP / "build-audiocpp"
BUILD_LOG = {}

# NOTE: kh.cache_and_link_flags() folds in -DCRISPASR_NO_C2PA_NATIVE=ON. Dropped
# on purpose here: with C2PA compiled out, crispasr_output_carries_c2pa() is
# false for every path, crispasr_enforce_cli_watermark_floor() calls
# set_forced(true), and --no-watermark is OVERRIDDEN. Without the native signer
# there is no way to measure CrispASR's synthesis cost separately from its
# provenance cost. The clone above is --recursive, so third_party/c2pa-audio
# (2 files, no external deps) is present.
C2PA_SRC = REPO / "third_party/c2pa-audio/src/c2pa_native.cpp"
if not C2PA_SRC.is_file():
    # --recursive is supposed to have brought it; if the submodule fetch was
    # skipped or failed, cmake dies with "Cannot find source file". Try once
    # more, then give up on the watermark-off arm rather than on the whole run.
    run(["git", "-C", str(REPO), "submodule", "update", "--init", "--recursive",
         "third_party/c2pa-audio"], timeout=1800)
C2PA_NATIVE = C2PA_SRC.is_file()
ca_flags = [f for f in kh.cache_and_link_flags() if "NO_C2PA_NATIVE" not in f]
if not C2PA_NATIVE:
    ca_flags.append("-DCRISPASR_NO_C2PA_NATIVE=ON")
    print("===C2PA-SUBMODULE-MISSING=== building with the native signer OFF. "
          "The CLI will then FORCE the watermark on for every output, so the "
          "*_nowm arms will measure the watermarked path -- they are marked "
          "watermark_actually_disabled=false, not silently believed.", flush=True)
BUILD_LOG["c2pa_native"] = C2PA_NATIVE
ca_cfg = ["-G", "Ninja", "-DCMAKE_BUILD_TYPE=Release", "-DGGML_NATIVE=OFF",
          *kh.cuda_build_flags(ARCH), *ca_flags]

kh.step("build.crispasr", flags=" ".join(ca_cfg))
with kh.build_heartbeat("build.crispasr"):
    kh.sh_with_progress(f"cmake -S {REPO} -B {BUILD_CA} " + " ".join(ca_cfg))
    kh.sh_with_progress(f"cmake --build {BUILD_CA} -j {kh.safe_build_jobs(gpu=True)} "
                        f"--target crispasr-cli crispasr-quantize")

CLI = BUILD_CA / "bin/crispasr"
QUANT = BUILD_CA / "bin/crispasr-quantize"
# Proof-of-work, not an exit code: the binaries must exist.
if not CLI.exists() or not QUANT.exists():
    raise RuntimeError(f"CrispASR build produced no binaries: {CLI} {QUANT}")
BUILD_LOG["crispasr"] = "ok"

kh.step("quantize.q4_k")
CA_Q4K = MODELS / "crispasr" / "supertonic3-q4_k.gguf"
pq = run([QUANT, CA_F16, CA_Q4K, "q4_k"], timeout=3600)
print((pq.stdout or "")[-4000:], flush=True)
if not CA_Q4K.exists():
    raise RuntimeError("crispasr-quantize produced no q4_k file:\n" + (pq.stderr or "")[-4000:])
kh.step("quantize.done", q4k_mb=round(CA_Q4K.stat().st_size / 1e6, 1))

GGUF_HIST = {
    "crispasr_f16": gguf_dtype_histogram(CA_F16),
    "crispasr_q4_k": gguf_dtype_histogram(CA_Q4K),
    "audiocpp_f16": gguf_dtype_histogram(AC_F16_DIR / "supertonic-3-f16.gguf"),
    "audiocpp_q8_0": gguf_dtype_histogram(AC_Q8_DIR / "supertonic-3-q8_0.gguf"),
}
print("GGUF dtype histograms:\n" + json.dumps(GGUF_HIST, indent=1), flush=True)

# -- audio.cpp -------------------------------------------------------------
# NOT --recursive: the only submodule is external/audio.cpp-server-frontends over
# git@github.com (SSH), which cannot clone here. ggml is vendored in-tree at
# external/ggml, and audiocpp_cli does not need the server frontends.
ACLI = None
AUDIOCPP_SHA = None
AC_HAS_CUDA = False
AC_ATTEMPTS = []

AC_CCACHE = Path("/kaggle/temp/.ccache-audiocpp")
AC_CCACHE.mkdir(parents=True, exist_ok=True)


def audiocpp_base_cfg():
    cfg = ["-G", "Ninja", "-DCMAKE_BUILD_TYPE=Release",
           # Same CPU-ISA policy as CrispASR's -DGGML_NATIVE=OFF, so the CPU arms
           # are not decided by one side getting -march=native and the other not.
           "-DENGINE_ENABLE_NATIVE_CPU=OFF",
           "-DAUDIOCPP_BUILD_NATIVE_MODEL_MANAGER=OFF",
           "-DENGINE_BUILD_TESTS=OFF", "-DENGINE_BUILD_EXAMPLES=OFF",
           # keep the kernel log readable; warnings do not change codegen
           "-DCMAKE_CXX_FLAGS=-w", "-DCMAKE_CUDA_FLAGS=-w"]
    for f in kh.cache_and_link_flags():
        if "NO_C2PA_NATIVE" not in f and "LINKER_FLAGS" not in f:
            cfg.append(f)
    return cfg


def try_build_audiocpp(label, extra_cfg, build_dir):
    """Returns (path_or_None, seconds, error_or_None). Never raises."""
    t0 = time.time()
    cfg = audiocpp_base_cfg() + extra_cfg
    kh.step(f"build.audiocpp.{label}", flags=" ".join(cfg))
    old = os.environ.get("CCACHE_DIR")
    os.environ["CCACHE_DIR"] = str(AC_CCACHE)
    os.environ["CCACHE_MAXSIZE"] = "8G"
    try:
        with kh.build_heartbeat(f"build.audiocpp.{label}"):
            kh.sh_with_progress(f"cmake -S {ACPP} -B {build_dir} " + " ".join(cfg))
            kh.sh_with_progress(f"cmake --build {build_dir} -j "
                                f"{kh.safe_build_jobs(gpu=True)} --target audiocpp_cli")
        cand = [c for c in Path(build_dir).rglob("audiocpp_cli")
                if c.is_file() and os.access(c, os.X_OK)]
        if not cand:
            # proof-of-work, not an exit code
            return None, time.time() - t0, "build reported success but no audiocpp_cli binary exists"
        return cand[0], time.time() - t0, None
    except Exception as e:
        return None, time.time() - t0, f"{type(e).__name__}: {e}"
    finally:
        if old:
            os.environ["CCACHE_DIR"] = old


try:
    kh.step("clone.audiocpp")
    if not ACPP.exists():
        run(["git", "clone", "--depth", "1", AUDIOCPP_URL, str(ACPP)],
            timeout=3600, capture=False, check=True)
    AUDIOCPP_SHA = subprocess.check_output(["git", "-C", str(ACPP), "rev-parse", "HEAD"],
                                           text=True).strip()
except Exception as e:
    BUILD_LOG["audiocpp_clone"] = f"CLONE_FAILED: {type(e).__name__}: {e}"
    print("===AUDIOCPP-CLONE-FAILED=== " + BUILD_LOG["audiocpp_clone"], flush=True)

if AUDIOCPP_SHA:
    cuda_cfg = ["-DENGINE_ENABLE_CUDA=ON", f"-DCMAKE_CUDA_ARCHITECTURES={ARCH}"]
    nvcc = "/usr/local/cuda/bin/nvcc"
    if os.path.isfile(nvcc):
        cuda_cfg.append(f"-DCMAKE_CUDA_COMPILER={nvcc}")

    ACLI, secs, err = try_build_audiocpp("cuda", cuda_cfg, BUILD_AC)
    AC_ATTEMPTS.append({"label": "cuda", "seconds": round(secs, 1), "error": err})
    AC_HAS_CUDA = ACLI is not None

    # audio.cpp documents GCC >= 13. If the stock compiler was not it, a retry
    # with g++-13 for the C++ TUs (nvcc keeps the host compiler it already
    # accepts) is worth one shot -- but only when the first attempt died early
    # enough that a second build still fits the session.
    if ACLI is None and secs < 1500:
        # Ubuntu 22.04 (which this worker is) has no g++-13 in its default
        # archive -- jammy tops out at g++-12 -- so an `apt-get install g++-13`
        # on its own cannot succeed. The toolchain PPA is what makes the retry
        # a real retry instead of a second way to fail.
        subprocess.run("apt-get install -y --no-install-recommends "
                       "software-properties-common && "
                       "add-apt-repository -y ppa:ubuntu-toolchain-r/test && "
                       "apt-get update -qq", shell=True, capture_output=True)
        rc = subprocess.run("apt-get install -y --no-install-recommends g++-13",
                            shell=True, capture_output=True).returncode
        g13 = shutil.which("g++-13")
        if rc == 0 and g13:
            retry = cuda_cfg + [f"-DCMAKE_CXX_COMPILER={g13}"]
            ACLI, secs2, err2 = try_build_audiocpp("cuda-gcc13", retry,
                                                   TEMP / "build-audiocpp-g13")
            AC_ATTEMPTS.append({"label": "cuda-gcc13", "seconds": round(secs2, 1),
                                "error": err2})
            if ACLI is not None:
                # A confound, and it has to be stated: CrispASR is built with the
                # stock g++ and audio.cpp with g++-13, so a CPU-arm difference is
                # no longer purely an implementation difference.
                BUILD_LOG["compiler_asymmetry"] = (
                    "audio.cpp built with g++-13, CrispASR with the stock g++ "
                    "(" + HW.get("gxx", "?") + ") -- CPU arms carry a compiler confound")
                print("===COMPILER-ASYMMETRY=== " + BUILD_LOG["compiler_asymmetry"],
                      flush=True)
            AC_HAS_CUDA = ACLI is not None
            secs = secs2
        else:
            AC_ATTEMPTS.append({"label": "cuda-gcc13", "error": "g++-13 not installable"})

    # Last resort: a CPU-only audio.cpp still answers half the question honestly.
    # Its CUDA arms are then reported ABSENT, never as a slow measurement.
    if ACLI is None and secs < 1500:
        ACLI, secs3, err3 = try_build_audiocpp("cpu-only", [], TEMP / "build-audiocpp-cpu")
        AC_ATTEMPTS.append({"label": "cpu-only", "seconds": round(secs3, 1), "error": err3})
        AC_HAS_CUDA = False

BUILD_LOG["audiocpp_attempts"] = AC_ATTEMPTS
if ACLI is not None:
    BUILD_LOG["audiocpp"] = "ok" if AC_HAS_CUDA else "ok-cpu-only"
    print(f"audiocpp_cli = {ACLI}  (cuda={AC_HAS_CUDA})", flush=True)
    bi = run([ACLI, "--version"], timeout=300)
    BUILD_LOG["audiocpp_version"] = ((bi.stdout or "") + (bi.stderr or ""))[-1500:]
    print(BUILD_LOG["audiocpp_version"], flush=True)
    dv = run([ACLI, "--list-devices"], timeout=300)
    BUILD_LOG["audiocpp_devices"] = ((dv.stdout or "") + (dv.stderr or ""))[-1500:]
    print("audio.cpp devices:\n" + BUILD_LOG["audiocpp_devices"], flush=True)
    # A "cuda" arm that silently ran on CPU would be a lie dressed as a number.
    if AC_HAS_CUDA and "cuda" not in BUILD_LOG["audiocpp_devices"].lower():
        print("===AUDIOCPP-CUDA-DEVICE-NOT-LISTED=== "
              "the build enabled CUDA but --list-devices does not show one; "
              "the cuda arm is suspect", flush=True)
        BUILD_LOG["audiocpp_cuda_device_listed"] = False
    else:
        BUILD_LOG["audiocpp_cuda_device_listed"] = AC_HAS_CUDA
else:
    # LOUD and DISTINCT from "it was slow".
    BUILD_LOG["audiocpp"] = "BUILD_FAILED: " + json.dumps(AC_ATTEMPTS)[:1500]
    print("=" * 72, flush=True)
    print("===AUDIOCPP-BUILD-FAILED===", flush=True)
    print(BUILD_LOG["audiocpp"], flush=True)
    print("=" * 72, flush=True)
    kh.step("build.audiocpp.FAILED", attempts=AC_ATTEMPTS)

# ───────────────────── upstream ONNX reference (ground truth) ──────────────

REF_WAV = None
REF_ERR = None
try:
    kh.step("reference.onnx")
    ref_gguf = TEMP / "supertonic-ref.gguf"
    pr = run([sys.executable, str(REPO / "tools/reference_backends/supertonic_tts.py"),
              "--model-dir", str(ST_UPSTREAM), "--text", TEXT_LONG, "--lang", LANG,
              "--voice", VOICE, "--steps", str(STEPS), "--seed", str(SEED),
              "--output", str(ref_gguf)], timeout=5400)
    if not ref_gguf.exists():
        raise RuntimeError("ref dumper wrote nothing:\n" + (pr.stderr or "")[-3000:])
    from gguf import GGUFReader
    rr = GGUFReader(str(ref_gguf))
    audio = next((np.array(t.data, dtype=np.float32) for t in rr.tensors
                  if t.name == "audio"), None)
    if audio is None:
        raise RuntimeError("reference gguf has no 'audio' stage")
    import soundfile as sf
    REF_WAV = OUTS / "reference_onnx.wav"
    sf.write(str(REF_WAV), audio, 44100)
    kh.step("reference.ok", samples=int(audio.size), seconds=round(audio.size / 44100, 2))
except Exception as e:
    REF_ERR = f"{type(e).__name__}: {e}"
    print(f"===ONNX-REFERENCE-UNAVAILABLE=== {REF_ERR}", flush=True)
    kh.step("reference.FAILED", error=REF_ERR[:400])

# ─────────────────────── PCM payload comparator ────────────────────────────

def read_wav_payload(path):
    """Walk the RIFF chunks and return (float32 mono samples, sample_rate).

    Deliberately hand-rolled: the point is to read the `data` chunk and NOTHING
    else. A CrispASR wav carries a LIST/INFO provenance chunk and, in this
    build, a C2PA manifest -- both of which contain a timestamp, so any
    comparison done on the FILE would report a difference that is not audio."""
    raw = Path(path).read_bytes()
    if raw[:4] != b"RIFF" or raw[8:12] != b"WAVE":
        raise ValueError(f"{path}: not a RIFF/WAVE file")
    pos, fmt, data = 12, None, None
    while pos + 8 <= len(raw):
        cid = raw[pos:pos + 4]
        (size,) = struct.unpack("<I", raw[pos + 4:pos + 8])
        body = raw[pos + 8:pos + 8 + size]
        if cid == b"fmt ":
            fmt = struct.unpack("<HHIIHH", body[:16])
        elif cid == b"data":
            data = body
        pos += 8 + size + (size & 1)
    if fmt is None or data is None:
        raise ValueError(f"{path}: missing fmt/data chunk")
    tag, ch, sr, _, _, bits = fmt
    if tag == 0xFFFE:  # WAVE_FORMAT_EXTENSIBLE -> subformat GUID's first 2 bytes
        tag = 3 if bits == 32 else 1
    if tag == 3 and bits == 32:
        x = np.frombuffer(data, dtype="<f4").astype(np.float32)
    elif tag == 1 and bits == 16:
        x = np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0
    elif tag == 1 and bits == 32:
        x = np.frombuffer(data, dtype="<i4").astype(np.float32) / 2147483648.0
    elif tag == 1 and bits == 24:
        b = np.frombuffer(data, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
        v = (b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16))
        v = np.where(v & 0x800000, v - 0x1000000, v)
        x = v.astype(np.float32) / 8388608.0
    else:
        raise ValueError(f"{path}: unsupported fmt tag={tag} bits={bits}")
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)
    return x, sr


def _resample_linear(x, sr_from, sr_to):
    if sr_from == sr_to:
        return x
    n = int(round(len(x) * sr_to / sr_from))
    idx = np.arange(n, dtype=np.float64) * sr_from / sr_to
    i0 = np.clip(idx.astype(np.int64), 0, len(x) - 1)
    i1 = np.clip(i0 + 1, 0, len(x) - 1)
    f = (idx - i0).astype(np.float32)
    return (x[i0] * (1 - f) + x[i1] * f).astype(np.float32)


def compare_pcm(a, sr_a, b, sr_b, max_lag_s=0.05):
    """Cosine AND magnitude. Cosine is scale-blind -- a signal wrong by a
    uniform factor passes it -- so rms_ratio is reported next to it every time."""
    b = _resample_linear(b, sr_b, sr_a)
    n = min(len(a), len(b))
    if n < 16:
        return {"error": "too short"}
    a0, b0 = a[:n].astype(np.float64), b[:n].astype(np.float64)

    def cos(u, v):
        du, dv = math.sqrt(float(u @ u)), math.sqrt(float(v @ v))
        return float(u @ v) / (du * dv) if du > 0 and dv > 0 else 0.0

    raw = cos(a0, b0)
    # best alignment inside +-max_lag_s, via FFT cross-correlation
    L = int(max_lag_s * sr_a)
    nfft = 1 << int(math.ceil(math.log2(2 * n)))
    xc = np.fft.irfft(np.fft.rfft(a0, nfft) * np.conj(np.fft.rfft(b0, nfft)), nfft)
    lags = np.concatenate([np.arange(0, L + 1), np.arange(nfft - L, nfft)])
    best = int(lags[int(np.argmax(xc[lags]))])
    if best > nfft // 2:
        best -= nfft
    if best >= 0:
        u, v = a0[best:], b0[:n - best]
    else:
        u, v = a0[:n + best], b0[-best:]
    aligned = cos(u, v) if len(u) > 16 else raw
    rms_a = float(np.sqrt(np.mean(a0 ** 2)))
    rms_b = float(np.sqrt(np.mean(b0 ** 2)))
    return {"cos_raw": round(raw, 6), "cos_aligned": round(aligned, 6),
            "best_lag_samples": best,
            "rms_a": round(rms_a, 6), "rms_b": round(rms_b, 6),
            "rms_ratio_a_over_b": round(rms_a / rms_b, 4) if rms_b > 0 else None,
            "len_a": int(len(a)), "len_b": int(len(b)),
            "dur_a_s": round(len(a) / sr_a, 3), "dur_b_s": round(len(b) / sr_b, 3)}


def metric_self_test():
    """Prove the comparator can render more than one state BEFORE it is pointed
    at real audio. Known-answer inputs with three different right answers."""
    rng = np.random.default_rng(7)
    t = np.arange(44100 * 2) / 44100.0
    x = (0.3 * np.sin(2 * np.pi * 220 * t) + 0.1 * np.sin(2 * np.pi * 660 * t)).astype(np.float32)
    noise = rng.standard_normal(x.size).astype(np.float32) * 0.3
    ident = compare_pcm(x, 44100, x.copy(), 44100)
    half = compare_pcm(x, 44100, (x * 0.5).astype(np.float32), 44100)
    rand = compare_pcm(x, 44100, noise, 44100)
    checks = {
        "identical_cos_is_1": abs(ident["cos_aligned"] - 1.0) < 1e-6,
        "identical_ratio_is_1": abs(ident["rms_ratio_a_over_b"] - 1.0) < 1e-3,
        "half_scale_cos_still_1": abs(half["cos_aligned"] - 1.0) < 1e-6,
        "half_scale_ratio_catches_it": abs(half["rms_ratio_a_over_b"] - 2.0) < 1e-2,
        "noise_cos_below_0_2": abs(rand["cos_aligned"]) < 0.2,
    }
    return {"identical": ident, "half_scale": half, "noise": rand,
            "checks": checks, "passed": all(checks.values())}


METRIC = metric_self_test()
print("metric self-test:\n" + json.dumps(METRIC, indent=1), flush=True)
if not METRIC["passed"]:
    print("===COMPARATOR-SELF-TEST-FAILED=== every cosine below is untrustworthy",
          flush=True)

# ───────────────────────────── the arms ────────────────────────────────────

# Three different ways a run can produce no audio, and they must NOT collapse
# into one "it failed" (or worse, into a slow timing). The supertonic-434 kernel
# lost a run to a rejected `-o`: the CLI printed usage, exited in 0.13 s, and the
# roundtrip scored 0.00 -- which read as "the audio is wrong" while every stage
# had already passed at cos 0.999996.
REJECT_RE = re.compile(r"usage:|unknown option|unsupported backend|unrecognised|"
                       r"unknown argument|no lookup ever asked", re.I)
MISSING_BACKEND_RE = re.compile(r"unknown backend|not available in this build|"
                                r"unknown family|unsupported family", re.I)
LOAD_FAIL_RE = re.compile(r"failed to load model|could not load|no such file", re.I)


def crispasr_cmd(gguf, device, text, out, steps, no_watermark=False):
    cmd = [str(CLI), "--backend", "supertonic", "-m", str(gguf), "--tts", text,
           "-l", LANG, "--voice", VOICE, "--tts-steps", str(steps),
           "--seed", str(SEED), "-t", NTHREADS, "--tts-output", str(out)]
    if device == "cpu":
        cmd.append("--no-gpu")
    if no_watermark:
        cmd += ["--no-watermark", "--accept-marking-responsibility"]
    return cmd


def audiocpp_cmd(model_dir, device, text, out, steps):
    return [str(ACLI), "--task", "tts", "--family", "supertonic",
            "--model", str(model_dir),
            "--backend", "cuda" if device == "gpu" else "cpu",
            "--language", LANG, "--text", text, "--voice-id", VOICE,
            "--num-inference-steps", str(steps), "--seed", str(SEED),
            "--threads", NTHREADS, "--metrics", "--out", str(out)]


def one_run(cmd, out, timeout=3600):
    """One cold process. Returns wall seconds + a status that distinguishes a
    REJECTED ARGUMENT from a failed synthesis -- the supertonic-434 kernel lost
    a whole run to a rejected `-o` that scored as 'the audio is wrong'."""
    if out.exists():
        out.unlink()
    t0 = time.perf_counter()
    try:
        p = run(cmd, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"status": "TIMEOUT", "wall_s": timeout}
    wall = time.perf_counter() - t0
    blob = (p.stdout or "") + (p.stderr or "")
    rec = {"wall_s": round(wall, 4), "rc": p.returncode,
           "stdout_tail": (p.stdout or "")[-1200:], "stderr_tail": (p.stderr or "")[-1800:]}
    if MISSING_BACKEND_RE.search(blob):
        rec["status"] = "BACKEND_MISSING"
    elif REJECT_RE.search(blob):
        rec["status"] = "ARGUMENT_REJECTED"
    elif p.returncode != 0 and LOAD_FAIL_RE.search(blob):
        rec["status"] = "MODEL_LOAD_FAILED"
    elif p.returncode != 0:
        rec["status"] = "RUN_FAILED"
    elif not out.exists() or out.stat().st_size < 2048:
        rec["status"] = "NO_OUTPUT"
    else:
        rec["status"] = "ok"
        # The CLI overrides --no-watermark for any output that cannot carry a
        # C2PA manifest and says so on stderr. A "watermark off" arm that was
        # silently re-forced must not be reported as one.
        if "--no-watermark" in cmd:
            rec["watermark_actually_disabled"] = not re.search(
                r"--no-watermark is overridden|watermark is kept", blob, re.I)
        x, sr = read_wav_payload(out)
        rec["audio_s"] = round(len(x) / sr, 4)
        rec["sample_rate"] = sr
        rec["rtf"] = round(rec["audio_s"] / wall, 3) if wall > 0 else None
    return rec


ARMS = []
if CLI.exists():
    for tag, gguf in (("f16", CA_F16), ("q4_k", CA_Q4K)):
        for dev in ("cpu", "gpu"):
            ARMS.append({"id": f"crispasr_{tag}_{dev}", "engine": "crispasr",
                         "precision": tag, "device": dev, "model": gguf,
                         "no_watermark": False})
    # apples-to-apples arm: synthesis without the always-on provenance watermark
    for dev in ("cpu", "gpu"):
        ARMS.append({"id": f"crispasr_f16_{dev}_nowm", "engine": "crispasr",
                     "precision": "f16", "device": dev, "model": CA_F16,
                     "no_watermark": True})
if ACLI is not None:
    ac_devs = ("cpu", "gpu") if AC_HAS_CUDA else ("cpu",)
    for tag, d in (("f16", AC_F16_DIR), ("q8_0", AC_Q8_DIR)):
        for dev in ac_devs:
            ARMS.append({"id": f"audiocpp_{tag}_{dev}", "engine": "audiocpp",
                         "precision": tag, "device": dev, "model": d,
                         "no_watermark": False})


def build_cmd(arm, text, out, steps):
    if arm["engine"] == "crispasr":
        return crispasr_cmd(arm["model"], arm["device"], text, out, steps,
                            no_watermark=arm["no_watermark"])
    return audiocpp_cmd(arm["model"], arm["device"], text, out, steps)


kh.step("bench.start", arms=[a["id"] for a in ARMS],
        text_chars={"short": len(TEXT_SHORT), "long": len(TEXT_LONG), "xl": len(TEXT_XL)})

BENCH = {}
for arm in ARMS:
    aid = arm["id"]
    kh.step(f"bench.{aid}")
    rec = {"engine": arm["engine"], "precision": arm["precision"],
           "device": arm["device"], "no_watermark": arm["no_watermark"],
           "model": str(arm["model"])}

    rec["short"] = one_run(build_cmd(arm, TEXT_SHORT, OUTS / f"{aid}_short.wav", STEPS),
                           OUTS / f"{aid}_short.wav")
    rec["long"] = [one_run(build_cmd(arm, TEXT_LONG, OUTS / f"{aid}_long.wav", STEPS),
                           OUTS / f"{aid}_long.wav") for _ in range(LONG_REPS)]
    rec["xl"] = [one_run(build_cmd(arm, TEXT_XL, OUTS / f"{aid}_xl.wav", STEPS),
                         OUTS / f"{aid}_xl.wav", timeout=5400) for _ in range(XL_REPS)]
    # TIMING POSITIVE CONTROL: 4x the flow steps must cost measurably more.
    rec["long_steps32"] = one_run(
        build_cmd(arm, TEXT_LONG, OUTS / f"{aid}_long32.wav", STEPS_CONTROL),
        OUTS / f"{aid}_long32.wav")

    if arm["no_watermark"]:
        flags = [r.get("watermark_actually_disabled") for r in rec["long"]
                 if r["status"] == "ok"]
        rec["watermark_actually_disabled"] = bool(flags) and all(flags)
    ok = [r for r in rec["long"] if r["status"] == "ok"]
    okxl = [r for r in rec["xl"] if r["status"] == "ok"]
    rec["status"] = "ok" if ok else (rec["long"][0]["status"] if rec["long"] else "NO_RUNS")
    if ok:
        best = min(r["wall_s"] for r in ok)
        rec["long_best_wall_s"] = round(best, 4)
        rec["long_audio_s"] = ok[0]["audio_s"]
        rec["long_rtf"] = round(ok[0]["audio_s"] / best, 3)
        rec["long_wall_spread_s"] = round(max(r["wall_s"] for r in ok) - best, 4)
        s = rec["short"]
        if s["status"] == "ok" and s["audio_s"] < ok[0]["audio_s"]:
            # Constant per-process cost (model load, graph build, file IO) drops
            # out of a two-point slope, so this is synthesis throughput rather
            # than "how fast does the binary start".
            d_wall = best - s["wall_s"]
            d_aud = ok[0]["audio_s"] - s["audio_s"]
            rec["slope_d_wall_s"] = round(d_wall, 4)
            if d_wall > 0.02:
                rec["slope_rtf_excl_startup"] = round(d_aud / d_wall, 3)
                rec["implied_startup_s"] = round(s["wall_s"] - s["audio_s"] * d_wall / d_aud, 4)
            else:
                # Two points 20 ms apart cannot resolve a slope. Say so instead of
                # printing a number the data does not support.
                rec["slope_rtf_excl_startup"] = None
                rec["slope_note"] = ("long and short walls differ by less than 20 ms; "
                                     "the two-point slope is below the timer's resolution")
    if okxl:
        b = min(r["wall_s"] for r in okxl)
        rec["xl_best_wall_s"] = round(b, 4)
        rec["xl_audio_s"] = okxl[0]["audio_s"]
        rec["xl_rtf"] = round(okxl[0]["audio_s"] / b, 3)

    c = rec["long_steps32"]
    if ok and c["status"] == "ok":
        ratio = c["wall_s"] / rec["long_best_wall_s"]
        rec["control_steps32_wall_s"] = round(c["wall_s"], 4)
        rec["control_ratio_32_over_8"] = round(ratio, 3)
        rec["control_discriminates"] = bool(ratio >= 1.3)
    else:
        rec["control_discriminates"] = False
        rec["control_note"] = f"32-step control status={c['status']}"
    BENCH[aid] = rec
    print(f"--- {aid}: status={rec['status']} "
          f"long_rtf={rec.get('long_rtf')} xl_rtf={rec.get('xl_rtf')} "
          f"slope_rtf={rec.get('slope_rtf_excl_startup')} "
          f"control={rec.get('control_ratio_32_over_8')}x "
          f"discriminates={rec['control_discriminates']}", flush=True)

# Emit the timing table NOW, before equivalence and ASR. The v2 run lost a
# 29-minute build and a completed quantize to an OOM in a diagnostic that ran
# after them; anything already measured goes into the log the moment it exists.
print("===PARTIAL-BENCH-JSON-BEGIN===", flush=True)
print(json.dumps({"hardware": HW, "cuda_arch": ARCH, "build": BUILD_LOG,
                  "bench": BENCH}, ensure_ascii=False, indent=2), flush=True)
print("===PARTIAL-BENCH-JSON-END===", flush=True)

# ─────────────────────── audio equivalence + ASR ───────────────────────────

kh.step("equivalence")
EQUIV = {"reference_available": REF_WAV is not None, "reference_error": REF_ERR}
loaded = {}
for aid in BENCH:
    w = OUTS / f"{aid}_long.wav"
    if BENCH[aid]["status"] == "ok" and w.exists():
        try:
            loaded[aid] = read_wav_payload(w)
        except Exception as e:
            EQUIV.setdefault("read_errors", {})[aid] = str(e)

if REF_WAV is not None:
    ref = read_wav_payload(REF_WAV)
    EQUIV["vs_onnx_reference"] = {
        aid: compare_pcm(ref[0], ref[1], x, sr) for aid, (x, sr) in loaded.items()}

PAIRS = [
    ("crispasr_f16_cpu", "audiocpp_f16_cpu"),
    ("crispasr_f16_gpu", "audiocpp_f16_gpu"),
    ("crispasr_f16_cpu_nowm", "audiocpp_f16_cpu"),
    ("crispasr_f16_cpu", "crispasr_f16_cpu_nowm"),   # isolates the watermark
    ("crispasr_f16_cpu", "crispasr_q4_k_cpu"),       # isolates CrispASR quantization
    ("crispasr_f16_cpu", "crispasr_f16_gpu"),        # CPU/GPU agreement, CrispASR
    ("audiocpp_f16_cpu", "audiocpp_q8_0_cpu"),       # isolates audio.cpp quantization
    ("audiocpp_f16_cpu", "audiocpp_f16_gpu"),        # CPU/GPU agreement, audio.cpp
]
EQUIV["pairs"] = {}
for a, b in PAIRS:
    if a in loaded and b in loaded:
        EQUIV["pairs"][f"{a}__vs__{b}"] = compare_pcm(loaded[a][0], loaded[a][1],
                                                      loaded[b][0], loaded[b][1])
    else:
        EQUIV["pairs"][f"{a}__vs__{b}"] = {"status": "MISSING_ARM",
                                           "have_a": a in loaded, "have_b": b in loaded}

kh.step("asr.roundtrip")


def asr(wav):
    if wav is None or not Path(wav).exists():
        return ""
    p = run([str(CLI), "--backend", "whisper", "-m", "auto", "-f", str(wav),
             "-nt", "--no-gpu", "-l", "en"], timeout=3600)
    txt = (p.stdout or "") + " " + (p.stderr or "")
    return re.sub(r"\x1b\[[0-9;]*m", "", txt)


def overlap(blob, target):
    words = [re.sub(r"[^a-z']", "", w.lower()) for w in target.split()]
    words = [w for w in words if len(w) > 2]
    low = blob.lower()
    return round(sum(1 for w in words if w in low) / max(1, len(words)), 3)


ASR = {}
if REF_WAV is not None:
    t = asr(REF_WAV)
    ASR["reference_onnx"] = {"overlap": overlap(t, TEXT_LONG), "tail": t[-500:]}
for aid in BENCH:
    w = OUTS / f"{aid}_long.wav"
    if BENCH[aid]["status"] == "ok" and w.exists():
        t = asr(w)
        ASR[aid] = {"overlap": overlap(t, TEXT_LONG), "tail": t[-500:]}
    else:
        ASR[aid] = {"overlap": None, "status": BENCH[aid]["status"]}

# ─────────────────────────────── verdict ───────────────────────────────────

control_ok = {a: BENCH[a]["control_discriminates"] for a in BENCH}
asr_ctrl = ASR.get("reference_onnx", {}).get("overlap")
problems = []
if ACLI is None:
    problems.append("audio.cpp did NOT build -- its arms are ABSENT, not slow")
elif not AC_HAS_CUDA:
    problems.append("audio.cpp built CPU-only -- its GPU arms are ABSENT, not slow")
if not METRIC["passed"]:
    problems.append("comparator self-test failed -- cosines untrustworthy")
for aid, r in BENCH.items():
    if r.get("no_watermark") and r.get("watermark_actually_disabled") is False:
        problems.append(f"{aid}: --no-watermark was OVERRIDDEN by the CLI -- this arm "
                        "is NOT a watermark-free measurement")
for a, good in control_ok.items():
    if not good:
        problems.append(f"{a}: timing instrument did not discriminate 8 vs 32 steps")
if REF_WAV is None:
    problems.append(f"ONNX reference unavailable ({REF_ERR}) -- no ground truth")
elif asr_ctrl is not None and asr_ctrl < 0.8:
    problems.append(f"ASR CONTROL failed on the upstream ONNX wav ({asr_ctrl}) -- "
                    "the ASR arm is at fault, every roundtrip below is inconclusive")
else:
    for a, r in ASR.items():
        if a != "reference_onnx" and r.get("overlap") is not None and r["overlap"] < 0.8:
            problems.append(f"{a}: ASR roundtrip overlap {r['overlap']} < 0.8")

RESULT = {
    "script_version": SCRIPT_VERSION,
    "crispasr_commit": CRISPASR_SHA,
    "audiocpp_commit": AUDIOCPP_SHA,
    "hardware": HW,
    "cuda_arch": ARCH,
    "threads": NTHREADS,
    "build": BUILD_LOG,
    "c2pa_native_compiled_in": BUILD_LOG.get("c2pa_native"),
    "gguf_dtype_histograms": GGUF_HIST,
    "texts": {"short": TEXT_SHORT, "long": TEXT_LONG, "xl_chars": len(TEXT_XL)},
    "settings": {"lang": LANG, "voice": VOICE, "seed": SEED, "steps": STEPS,
                 "control_steps": STEPS_CONTROL, "long_reps": LONG_REPS,
                 "xl_reps": XL_REPS},
    "metric_self_test": METRIC,
    "bench": BENCH,
    "equivalence": EQUIV,
    "asr": ASR,
    "verdict": {
        "audiocpp_built": ACLI is not None,
        "audiocpp_cuda_arm_available": AC_HAS_CUDA,
        "audiocpp_build_attempts": AC_ATTEMPTS,
        "crispasr_built": BUILD_LOG.get("crispasr") == "ok",
        "comparator_ok": METRIC["passed"],
        "timing_control_ok": control_ok,
        "asr_control_overlap": asr_ctrl,
        "problems": problems,
        "clean": not problems,
    },
}
RESULTS.write_text(json.dumps(RESULT, ensure_ascii=False, indent=2) + "\n")

print("\n" + "=" * 72, flush=True)
print("===SUMMARY-TABLE===", flush=True)
hdr = f"{'arm':28} {'status':10} {'long_rtf':>9} {'xl_rtf':>8} {'slope_rtf':>10} {'32/8':>6} {'asr':>6}"
print(hdr, flush=True)
for aid, r in BENCH.items():
    print(f"{aid:28} {r['status']:10} {str(r.get('long_rtf')):>9} "
          f"{str(r.get('xl_rtf')):>8} {str(r.get('slope_rtf_excl_startup')):>10} "
          f"{str(r.get('control_ratio_32_over_8')):>6} "
          f"{str(ASR.get(aid, {}).get('overlap')):>6}", flush=True)
print("=" * 72, flush=True)
if ACLI is None:
    print("===AUDIOCPP-BUILD-FAILED=== " + str(BUILD_LOG.get("audiocpp"))[:800], flush=True)
elif not AC_HAS_CUDA:
    print("===AUDIOCPP-CUDA-ARM-ABSENT=== built CPU-only; no GPU number exists to report",
          flush=True)
print("===RESULTS-JSON-BEGIN===", flush=True)
print(json.dumps(RESULT, ensure_ascii=False, indent=2), flush=True)
print("===RESULTS-JSON-END===", flush=True)

kh.step("done", clean=not problems, problems=problems)
