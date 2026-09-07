#!/usr/bin/env python3
"""#426: official reference -> F16 GGUF -> Q4 A/B -> persistent-stream parity."""

import json
import os
import shutil
import subprocess
import sys
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np

WORK = Path("/kaggle/working")
TEMP = Path("/kaggle/temp/vibevoice-streaming-426")
REPO = Path("/kaggle/temp/CrispASR")
UPSTREAM = Path("/kaggle/temp/VibeVoice")
MODEL_DIR = TEMP / "model"
for path in (WORK, TEMP, MODEL_DIR):
    path.mkdir(parents=True, exist_ok=True)

BRANCH = "feat/426-vibevoice-streaming-q4"


def run(argv, *, cwd=None, env=None, timeout=7200, capture=False):
    merged = os.environ.copy()
    if env:
        merged.update({str(k): str(v) for k, v in env.items()})
    print("$ " + " ".join(map(str, argv)), flush=True)
    return subprocess.run([str(x) for x in argv], cwd=cwd, env=merged, check=True, timeout=timeout,
                          text=True, capture_output=capture)


if not REPO.exists():
    run(["git", "clone", "--depth", "1", "--branch", BRANCH, "--recursive",
         "https://github.com/CrispStrobe/CrispASR.git", REPO], timeout=2400)
sys.path.insert(0, str(REPO / "tools" / "kaggle"))
import kaggle_harness as kh  # noqa: E402

kh.init_progress()
kh.resolve_hf_token()
commit = subprocess.check_output(["git", "-C", REPO, "rev-parse", "HEAD"], text=True).strip()
kh.step("provenance", commit=commit)

kh.step("dependencies")
run([sys.executable, "-m", "pip", "install", "--quiet", "transformers>=4.51.3,<5", "accelerate",
     "safetensors", "huggingface_hub", "hf_transfer", "librosa", "soundfile", "ml-collections", "absl-py"])
if not UPSTREAM.exists():
    run(["git", "clone", "--depth", "1", "https://github.com/microsoft/VibeVoice.git", UPSTREAM], timeout=1200)
run([sys.executable, "-m", "pip", "install", "--quiet", "--no-deps", "-e", UPSTREAM])
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

from huggingface_hub import HfApi, snapshot_download  # noqa: E402

kh.step("model.download")
snapshot = Path(snapshot_download("microsoft/VibeVoice-ASR-Streaming-1.5B", local_dir=MODEL_DIR))

kh.step("reference.mean")
ref_dir = TEMP / "reference-mean"
run([sys.executable, REPO / "tools/vibevoice_asr_streaming_ref.py", "--model", snapshot,
     "--audio", REPO / "samples/jfk.wav", "--output-dir", ref_dir, "--mean-acoustic"], timeout=7200)

kh.step("convert.f16")
f16 = TEMP / "vibevoice-asr-streaming-1.5b-f16.gguf"
run([sys.executable, REPO / "models/convert-vibevoice-to-gguf.py", "--input", snapshot, "--output", f16],
    timeout=7200)

kh.install_build_toolchain()
build = TEMP / "build-cuda"
arch = kh.detect_cuda_arch()
flags = ["-G", "Ninja", "-DCMAKE_BUILD_TYPE=Release", "-DCRISPASR_NO_C2PA_NATIVE=ON",
         "-DGGML_NATIVE=OFF", *kh.cuda_build_flags(arch), *kh.cache_and_link_flags()]
kh.step("build", arch=arch)
run(["cmake", "-S", REPO, "-B", build, *flags], timeout=2400)
run(["cmake", "--build", build, "-j", str(kh.safe_build_jobs(gpu=True)), "--target",
     "crispasr-cli", "crispasr-quantize"], timeout=7200)
cli = build / "bin/crispasr"
quant = build / "bin/crispasr-quantize"

q4_plain = WORK / "vibevoice-asr-streaming-1.5b-q4_k-plain.gguf"
q4_front = WORK / "vibevoice-asr-streaming-1.5b-q4_k-frontend-f16.gguf"
kh.step("quant.plain")
run([quant, f16, q4_plain, "q4_k"], timeout=7200)
kh.step("quant.frontend-f16")
run([quant, f16, q4_front, "q4_k"], env={"CRISPASR_VIBEVOICE_ASR_FRONTEND_F16": "1"}, timeout=7200)


def native(model, label, mean=True):
    dump = TEMP / f"native-{label}"
    dump.mkdir(exist_ok=True)
    env = {"CRISPASR_VIBEVOICE_DUMP_DIR": dump}
    if mean:
        env["CRISPASR_VIBEVOICE_ASR_SAMPLE"] = "0"
    proc = run([cli, "--backend", "vibevoice", "-m", model, "-f", REPO / "samples/jfk.wav", "-nt"],
               env=env, timeout=7200, capture=True)
    return dump, "\n".join(x.strip() for x in proc.stdout.splitlines() if x.strip())


def compare(ref, got):
    rows = {}
    for rp in sorted(ref.glob("*.bin")):
        gp = got / rp.name
        if not gp.exists():
            continue
        dtype = np.int32 if "ids" in rp.name else np.float32
        a, b = np.fromfile(rp, dtype=dtype), np.fromfile(gp, dtype=dtype)
        if dtype == np.int32:
            rows[rp.stem] = {"exact": bool(np.array_equal(a, b)), "n_ref": len(a), "n_got": len(b)}
        elif len(a) == len(b) and len(a):
            af, bf = a.astype(np.float64), b.astype(np.float64)
            denom = np.linalg.norm(af) * np.linalg.norm(bf)
            rows[rp.stem] = {"cos": float(np.dot(af, bf) / denom) if denom else 1.0,
                             "max_abs": float(np.max(np.abs(af - bf)))}
    return rows


results = {"commit": commit, "cuda_arch": arch, "models": {}}
for model, label in ((f16, "f16"), (q4_plain, "q4-plain"), (q4_front, "q4-frontend-f16")):
    kh.step(f"native.{label}")
    dump, text = native(model, label)
    results["models"][label] = {"bytes": model.stat().st_size, "text": text, "diff": compare(ref_dir, dump)}

f16_text = results["models"]["f16"]["text"]
for label in ("q4-plain", "q4-frontend-f16"):
    results["models"][label]["text_similarity_to_f16"] = SequenceMatcher(
        None, f16_text, results["models"][label]["text"]
    ).ratio()

# The frontend-F16 arm pays a size cost only if it improves transcript parity.
plain_score = results["models"]["q4-plain"]["text_similarity_to_f16"]
front_score = results["models"]["q4-frontend-f16"]["text_similarity_to_f16"]
selected = q4_front if front_score > plain_score else q4_plain
results["selected"] = selected.name

kh.step("native.selected.sampled")
_, sampled_text = native(selected, "selected-sampled", mean=False)
results["sampled_text"] = sampled_text

result_path = WORK / "results.json"
result_path.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(results, ensure_ascii=False, indent=2), flush=True)

token = os.environ.get("HF_TOKEN")
if token:
    api = HfApi(token=token)
    repo_id = "cstr/vibevoice-asr-streaming-1.5b-GGUF"
    api.create_repo(repo_id, repo_type="model", private=False, exist_ok=True)
    kh.step("publish", selected=selected.name)
    api.upload_file(path_or_fileobj=str(selected), path_in_repo="vibevoice-asr-streaming-1.5b-q4_k.gguf",
                    repo_id=repo_id, repo_type="model", commit_message=f"Add validated Q4_K from {commit[:12]}")
    api.upload_file(path_or_fileobj=str(result_path), path_in_repo="validation/results.json",
                    repo_id=repo_id, repo_type="model", commit_message="Add official/native streaming validation")

# Avoid publishing both experimental arms as Kaggle output; the selected file
# remains downloadable even if HF upload was unavailable.
for candidate in (q4_plain, q4_front):
    if candidate != selected and candidate.exists():
        candidate.unlink()
if selected.name != "vibevoice-asr-streaming-1.5b-q4_k.gguf":
    selected.rename(WORK / "vibevoice-asr-streaming-1.5b-q4_k.gguf")
kh.step("done", selected=results["selected"])
