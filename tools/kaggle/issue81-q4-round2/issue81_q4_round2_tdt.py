#!/usr/bin/env python3
"""Q4 FastConformer CUDA optimization round 2 (#81).

A/B tests scalar device-side selection and adaptive joint batches on P100.
Each arm uses the same varied audio and hard transcript stability/equality gates.
"""

import json
import io
import os
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

WORK = Path("/kaggle/working")
TEMP = Path("/kaggle/temp") if Path("/kaggle/temp").is_dir() else Path("/tmp")
REPO = TEMP / "CrispASR"
BUILD = TEMP / "build"
BRANCH = "bench/81-tdt-device-select"
SCRIPT_VERSION = "issue81-round2-v3"

subprocess.check_call([
    "git", "clone", "--depth", "1", "-b", BRANCH,
    "https://github.com/CrispStrobe/CrispASR", str(REPO),
])
subprocess.check_call(
    ["git", "submodule", "update", "--init", "ggml", "third_party/c2pa-audio"],
    cwd=REPO,
)
sys.path.insert(0, str(REPO / "tools" / "kaggle"))
import kaggle_harness as kh  # noqa: E402

kh.init_progress()
token = kh.resolve_hf_token("HF_TOKEN")
commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
arch = kh.detect_cuda_arch()
kh.step("provenance", commit=commit, branch=BRANCH, script_version=SCRIPT_VERSION, cuda_arch=arch)
print(f"SCRIPT_VERSION={SCRIPT_VERSION} COMMIT={commit} CUDA_ARCH={arch}", flush=True)

subprocess.check_call([
    sys.executable, "-m", "pip", "install", "--quiet",
    "huggingface_hub", "soundfile", "datasets",
])
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

kh.install_build_toolchain()
BUILD.mkdir(parents=True, exist_ok=True)
flags = kh.cuda_build_flags(arch) + kh.cache_and_link_flags() + ["-DGGML_CUDA_CRISPASR_FA_PERHEAD_MASK=ON"]
subprocess.check_call([
    "cmake", "-G", "Ninja", "-B", str(BUILD), "-S", str(REPO),
    "-DCMAKE_BUILD_TYPE=Release", "-DCRISPASR_NO_C2PA_NATIVE=ON", *flags,
])
with kh.build_heartbeat("cmake.build"):
    kh.sh_with_progress(
        f"cmake --build {BUILD} -j {kh.safe_build_jobs(True)} --target crispasr-lib"
    )
lib = BUILD / "src" / "libcrispasr.so"
if not lib.is_file():
    raise RuntimeError(f"missing built library: {lib}")
kh.step("build.done")

from huggingface_hub import hf_hub_download  # noqa: E402
import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

models_dir = TEMP / "models"
models_dir.mkdir(parents=True, exist_ok=True)
models = {
    "tdt": hf_hub_download(
        "cstr/parakeet-tdt-0.6b-v3-GGUF", "parakeet-tdt-0.6b-v3-q4_k.gguf",
        local_dir=str(models_dir), token=token,
    ),
}
kh.step("model.downloaded", size=Path(models["tdt"]).stat().st_size)

jfk, sample_rate = sf.read(str(REPO / "samples" / "jfk.wav"), dtype="float32")
if sample_rate != 16000:
    raise RuntimeError(f"unexpected JFK sample rate: {sample_rate}")

# Varied speech avoids the cache-friendly repeated-JFK artifact from early #81
# measurements. This is the same source and construction as the canonical
# issue81-onnx-bench comparison.
from datasets import Audio, load_dataset  # noqa: E402

dataset = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
# Current Kaggle Torch wheels no longer contain sm_60 kernels. Avoid the
# datasets/torchcodec decoder entirely; soundfile can decode the stored FLAC
# bytes on the CPU without initializing Torch or touching the P100.
dataset = dataset.cast_column("audio", Audio(decode=False))
gap = np.zeros(int(0.3 * sample_rate), dtype="float32")
parts = []
for row in dataset:
    source = row["audio"]
    audio, audio_rate = sf.read(
        io.BytesIO(source["bytes"]) if source.get("bytes") is not None else source["path"],
        dtype="float32",
    )
    if audio_rate != sample_rate:
        raise RuntimeError(f"unexpected LibriSpeech sample rate: {audio_rate}")
    audio = np.asarray(audio, dtype="float32")
    parts.extend((audio, gap))
    if sum(len(x) for x in parts) / sample_rate >= 120:
        break
long_audio = np.concatenate(parts)
clips_dir = TEMP / "clips"
clips_dir.mkdir(parents=True, exist_ok=True)
clip_paths = {"jfk": clips_dir / "jfk.wav", "varied": clips_dir / "varied.wav"}
sf.write(clip_paths["jfk"], jfk, sample_rate)
sf.write(clip_paths["varied"], long_audio, sample_rate)
durations = {name: len(audio) / sample_rate for name, audio in (("jfk", jfk), ("varied", long_audio))}
kh.step("audio.ready", durations=durations)

CHILD = r'''
import json, os, sys, time
import soundfile as sf
sys.path.insert(0, os.path.join(sys.argv[1], "python"))
os.environ["CRISPASR_LIB_PATH"] = sys.argv[2]
from crispasr import Session

model, kind, mode = sys.argv[3:6]
paths = json.loads(sys.argv[6])
session = Session(model, n_threads=4)
out = {}
for name, path in paths.items():
    pcm, sr = sf.read(path, dtype="float32")
    if sr != 16000:
        raise RuntimeError(f"unexpected sample rate {sr}")
    runs = 1 if mode == "profile" else (4 if name == "jfk" else 3)
    times, texts = [], []
    for _ in range(runs):
        t0 = time.perf_counter()
        segs = session.transcribe(pcm.copy(), language="en")
        times.append(time.perf_counter() - t0)
        texts.append(" ".join(s.text for s in segs).strip())
    out[name] = {"times_s": times, "texts": texts}
    if mode == "profile":
        break
print("RESULT::" + json.dumps(out), flush=True)
'''


def run_child(kind, mode, env_extra=None, model_key=None):
    env = dict(os.environ, **(env_extra or {}))
    env["CRISPASR_PARAKEET_BENCH" if kind == "tdt" else "CRISPASR_CANARY_CTC_BENCH"] = "1"
    if kind == "tdt":
        env["CRISPASR_PARAKEET_DECODE_TIMING"] = "1"
    if mode == "profile":
        env["CRISPASR_FC_PROFILE"] = "1"
    proc = subprocess.run(
        [sys.executable, "-c", CHILD, str(REPO), str(lib), models[model_key or kind], kind, mode,
         json.dumps({k: str(v) for k, v in clip_paths.items()})],
        env=env, text=True, capture_output=True, timeout=3600,
    )
    if proc.returncode:
        raise RuntimeError(f"{kind}/{mode} failed ({proc.returncode}):\n{proc.stderr[-4000:]}")
    payload = next((line[8:] for line in proc.stdout.splitlines() if line.startswith("RESULT::")), None)
    if payload is None:
        raise RuntimeError(f"{kind}/{mode} emitted no result: {proc.stdout[-1000:]}")
    data = json.loads(payload)
    stage_re = re.compile(
        r"(?:parakeet|canary_ctc)_bench:\s+([+a-z]+)\s+([0-9.]+) ms"
    )
    stages = {}
    for stage, value in stage_re.findall(proc.stderr):
        stages.setdefault(stage, []).append(float(value))
    profile_lines = [line.strip() for line in proc.stderr.splitlines() if "cc_profile:" in line]
    decode_lines = [line.strip() for line in proc.stderr.splitlines() if "parakeet: tdt_decode" in line]
    for clip, row in data.items():
        if not all(row["texts"]):
            raise RuntimeError(f"{kind}/{mode}/{clip} produced empty text")
        row["transcript_stable"] = len(set(row["texts"])) == 1
        row["median_s"] = statistics.median(row["times_s"][1:] or row["times_s"])
        row["rtf_x"] = durations[clip] / row["median_s"]
        row["word_count"] = len(row["texts"][-1].split())
    return {"clips": data, "stage_ms": stages, "profile_lines": profile_lines, "decode_lines": decode_lines}


results = {"commit": commit, "script_version": SCRIPT_VERSION, "cuda_arch": arch,
           "durations_s": durations, "models": {}}
tdt_arms = {}
tdt_specs = [
    ("baseline", {}),
    ("gpu-select", {"CRISPASR_RNNT_GPU_SELECT": "1"}),
    ("batch4", {"CRISPASR_RNNT_GPU_BATCH": "4"}),
    ("batch8", {"CRISPASR_RNNT_GPU_BATCH": "8"}),
    ("batch4-select", {"CRISPASR_RNNT_GPU_BATCH": "4", "CRISPASR_RNNT_GPU_SELECT": "1"}),
    ("batch8-select", {"CRISPASR_RNNT_GPU_BATCH": "8", "CRISPASR_RNNT_GPU_SELECT": "1"}),
]
for name, env in tdt_specs:
    kh.step(f"{name}.tdt")
    tdt_arms[name] = run_child("tdt", name, env)
results["models"]["tdt"] = tdt_arms

baseline = tdt_arms["baseline"]
stable = all(row["transcript_stable"] for arm in tdt_arms.values() for row in arm["clips"].values())
equal = all(
    tdt_arms[name]["clips"][clip]["texts"][-1] == baseline["clips"][clip]["texts"][-1]
    for name, _ in tdt_specs[1:] for clip in clip_paths
)
results["validation"] = {
    "passed": stable and equal,
    "all_arms_stable": stable,
    "implementation_equal": equal,
}

(WORK / "results.json").write_text(json.dumps(results, indent=2) + "\n")
kh.export_ccache_tar()
kh.step("done", validation=results["validation"])
print(json.dumps(results, indent=2), flush=True)
if not results["validation"]["passed"]:
    raise RuntimeError("an implementation arm changed the transcript or an arm was unstable")
