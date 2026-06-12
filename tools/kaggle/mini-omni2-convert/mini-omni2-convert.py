#!/usr/bin/env python3
"""CrispASR — Mini-Omni2 GGUF conversion + quantization bench.

Converts gpt-omni/mini-omni2 (Whisper-small + Qwen2-0.5B) to F16 GGUF,
quantizes to Q8_0 and Q4_K with selective tensor protection
(encoder/adapter/embedding at F16, only LLM layers quantized), runs
crispasr-diff on all 3 quants, and uploads to cstr/mini-omni2-GGUF.
"""

import os
import subprocess
import sys
import shutil
from pathlib import Path

WORK = Path("/kaggle/working")
REPO = WORK / "CrispASR"
BUILD = WORK / "build"
BRANCH = os.environ.get("CRISPASR_REF", "main")

HF_REPO = "cstr/mini-omni2-GGUF"
HF_MODEL = "gpt-omni/mini-omni2"
GH_MINI_OMNI2 = "https://github.com/gpt-omni/mini-omni2.git"

OUT_F16 = WORK / "mini-omni2-f16.gguf"
OUT_Q8  = WORK / "mini-omni2-q8_0.gguf"
OUT_Q4K = WORK / "mini-omni2-q4_k.gguf"

# ===========================================================================
# 1. Clone CrispASR + setup
# ===========================================================================
print(f"[1] cloning CrispASR ({BRANCH})", flush=True)
if REPO.exists():
    shutil.rmtree(REPO)
subprocess.check_call([
    "git", "clone", "--depth", "1", "--branch", BRANCH,
    "https://github.com/CrispStrobe/CrispASR.git", str(REPO),
])

sys.path.insert(0, str(REPO / "tools" / "kaggle"))
try:
    import kaggle_harness as kh
    kh.init_progress()
    hf_token = kh.resolve_hf_token()
    kh.step("cloned", branch=BRANCH, hf_token_ok=bool(hf_token))
except Exception as e:
    print(f"[1] kaggle_harness: {e} — running without harness", flush=True)
    hf_token = os.environ.get("HF_TOKEN", "")
    class kh:
        @staticmethod
        def step(*a, **kw): pass
        @staticmethod
        def install_build_toolchain(): pass
        @staticmethod
        def sh_with_progress(cmd): subprocess.check_call(cmd, shell=True)
        @staticmethod
        def safe_build_jobs(gpu=False): return 4
        @staticmethod
        def cache_and_link_flags(): return []
        class build_heartbeat:
            def __init__(self, n): pass
            def __enter__(self): return self
            def __exit__(self, *a): pass

# ===========================================================================
# 2. Install dependencies
# ===========================================================================
print("[2] installing deps", flush=True)
subprocess.check_call([
    sys.executable, "-m", "pip", "install", "--quiet",
    "safetensors", "gguf", "huggingface_hub", "hf_transfer",
    "openai-whisper", "torch", "torchaudio",
])
kh.step("deps_installed")

# ===========================================================================
# 3. Download model weights
# ===========================================================================
print("[3] downloading model", flush=True)
from huggingface_hub import snapshot_download

for candidate in ("/kaggle/temp", "/tmp"):
    if os.path.isdir(candidate):
        scratch = Path(candidate) / "mini-omni2-src"
        break
scratch.mkdir(parents=True, exist_ok=True)
print(f"  scratch: {scratch} (free: {shutil.disk_usage(scratch).free / (1024**3):.1f} GiB)", flush=True)

# Download HF weights (lit_model.pth + small.pt + tokenizer.json + model_config.yaml)
src = snapshot_download(
    repo_id=HF_MODEL,
    cache_dir=str(scratch),
    ignore_patterns=["ViT-B-32.pt", "data/*", "*.mp4"],
)
print(f"  source: {src}", flush=True)

# Clone mini-omni2 repo for litgpt module (needed by reference backend)
mo2_repo = scratch / "mini-omni2-repo"
if not mo2_repo.exists():
    subprocess.check_call([
        "git", "clone", "--depth", "1", GH_MINI_OMNI2, str(mo2_repo),
    ])
print(f"  litgpt repo: {mo2_repo}", flush=True)
kh.step("model_downloaded")

# ===========================================================================
# 4. Convert to F16 GGUF
# ===========================================================================
print("[4] converting to F16 GGUF", flush=True)
subprocess.check_call([
    sys.executable, str(REPO / "models" / "convert-mini-omni2-to-gguf.py"),
    "--input", src,
    "--output", str(OUT_F16),
])
print(f"  F16: {OUT_F16.stat().st_size / (1024**3):.2f} GiB", flush=True)
kh.step("f16_done", size_gb=round(OUT_F16.stat().st_size / (1024**3), 2))

# ===========================================================================
# 5. Build CrispASR (quantizer + diff harness)
# ===========================================================================
print("[5] building crispasr-quantize + crispasr-diff", flush=True)
kh.install_build_toolchain()
BUILD.mkdir(exist_ok=True)

cmake_cfg = (
    f"cmake -G Ninja -S {REPO} -B {BUILD} "
    f"-DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=OFF -DCRISPASR_BUILD_TESTS=OFF "
    + " ".join(kh.cache_and_link_flags())
)
kh.sh_with_progress(cmake_cfg)

with kh.build_heartbeat("cmake.build"):
    kh.sh_with_progress(
        f"cmake --build {BUILD} --target crispasr-quantize crispasr-diff "
        f"-j{kh.safe_build_jobs(gpu=False)}"
    )
kh.step("build_done")

# ===========================================================================
# 6. Quantize to Q8_0 and Q4_K
# ===========================================================================
QUANTIZE = BUILD / "bin" / "crispasr-quantize"

print("[6a] quantizing F16 -> Q8_0", flush=True)
subprocess.check_call([str(QUANTIZE), str(OUT_F16), str(OUT_Q8), "q8_0"])
print(f"  Q8_0: {OUT_Q8.stat().st_size / (1024**3):.2f} GiB", flush=True)

print("[6b] quantizing F16 -> Q4_K", flush=True)
subprocess.check_call([str(QUANTIZE), str(OUT_F16), str(OUT_Q4K), "q4_k"])
print(f"  Q4_K: {OUT_Q4K.stat().st_size / (1024**3):.2f} GiB", flush=True)
kh.step("quantize_done")

# ===========================================================================
# 7. Generate Python reference dump for diff testing
# ===========================================================================
print("[7] generating reference dump", flush=True)
AUDIO = REPO / "samples" / "jfk.wav"
REF = WORK / "mini-omni2-ref.gguf"

env = os.environ.copy()
env["MINI_OMNI2_REPO"] = str(mo2_repo)

subprocess.check_call([
    sys.executable, str(REPO / "tools" / "dump_reference.py"),
    "--backend", "mini-omni2",
    "--model-dir", src,
    "--audio", str(AUDIO),
    "--stages", "mel_spectrogram,whisper_encoder_output,adapter_output",
    "--output", str(REF),
], env=env)
kh.step("ref_dump_done")

# ===========================================================================
# 8. Run crispasr-diff on all 3 quants
# ===========================================================================
DIFF = BUILD / "bin" / "crispasr-diff"
results = {}

for label, gguf in [("F16", OUT_F16), ("Q8_0", OUT_Q8), ("Q4_K", OUT_Q4K)]:
    if not gguf.exists():
        print(f"[8] {label}: SKIP (file missing)", flush=True)
        continue
    print(f"\n[8] testing {label}: {gguf.name}", flush=True)
    r = subprocess.run(
        [str(DIFF), "mini-omni2", str(gguf), str(REF), str(AUDIO)],
        capture_output=True, text=True, timeout=600,
    )
    output = r.stdout + r.stderr
    print(output, flush=True)
    results[label] = output

print("\n" + "=" * 60, flush=True)
print("SUMMARY", flush=True)
print("=" * 60, flush=True)
for label in ["F16", "Q8_0", "Q4_K"]:
    if label in results:
        lines = [l for l in results[label].split("\n")
                 if "PASS" in l or "FAIL" in l or "transcribe" in l]
        print(f"\n--- {label} ---")
        for l in lines:
            print(f"  {l}")
kh.step("diff_done")

# ===========================================================================
# 9. Upload to HuggingFace
# ===========================================================================
if hf_token:
    print("[9] uploading to HuggingFace", flush=True)
    os.environ["HF_TOKEN"] = hf_token
    from huggingface_hub import HfApi
    api = HfApi(token=hf_token)

    try:
        api.create_repo(HF_REPO, repo_type="model", exist_ok=True)
    except Exception as e:
        print(f"  repo create: {e}", flush=True)

    for label, gguf, fname in [
        ("F16", OUT_F16, "mini-omni2-f16.gguf"),
        ("Q8_0", OUT_Q8, "mini-omni2-q8_0.gguf"),
        ("Q4_K", OUT_Q4K, "mini-omni2-q4_k.gguf"),
    ]:
        if not gguf.exists():
            continue
        sz = gguf.stat().st_size / (1024**3)
        print(f"  uploading {label} ({sz:.2f} GiB): {fname}", flush=True)
        api.upload_file(
            path_or_fileobj=str(gguf),
            path_in_repo=fname,
            repo_id=HF_REPO,
            repo_type="model",
            commit_message=f"Add {label} GGUF (Whisper-small + Qwen2-0.5B)",
        )
        print(f"  uploaded {label}", flush=True)

    # Upload a README
    readme = f"""---
license: mit
language: en
tags:
- asr
- tts
- speech-to-speech
- mini-omni2
- gguf
base_model: gpt-omni/mini-omni2
---

# Mini-Omni2 GGUF

GGUF conversion of [gpt-omni/mini-omni2](https://huggingface.co/gpt-omni/mini-omni2)
for use with [CrispASR](https://github.com/CrispStrobe/CrispASR).

Architecture: Whisper-small encoder (80 mel, 12L, 768d) + whisperMLP adapter
(SwiGLU 768→4864→896) + Qwen2-0.5B LLM (896d, 24L, GQA 14/2).

Supports ASR (audio→text), TTS (text→audio), and speech-to-speech (audio→audio).
TTS/S2S require the SNAC 24kHz codec companion
([cstr/snac-24khz-GGUF](https://huggingface.co/cstr/snac-24khz-GGUF)).

## Files

| File | Quant | Size | Notes |
|------|-------|------|-------|
| `mini-omni2-f16.gguf` | F16 | ~1.5 GB | Full precision |
| `mini-omni2-q8_0.gguf` | Q8_0 | ~1.2 GB | Encoder/adapter at F16, LLM at Q8_0 |
| `mini-omni2-q4_k.gguf` | Q4_K | ~1.0 GB | Encoder/adapter at F16, LLM at Q4_K |

## Usage

```bash
# ASR
crispasr -m mini-omni2-q4_k.gguf -f audio.wav --backend mini-omni2

# TTS (needs SNAC codec)
crispasr -m mini-omni2-q4_k.gguf --tts "Hello world" \\
    --codec-model snac-24khz.gguf --tts-output out.wav --backend mini-omni2
```
"""
    api.upload_file(
        path_or_fileobj=readme.encode(),
        path_in_repo="README.md",
        repo_id=HF_REPO,
        repo_type="model",
        commit_message="Add README",
    )
    print("[9] uploaded README", flush=True)
    kh.step("uploaded")
else:
    print("[9] no HF_TOKEN — skipping upload", flush=True)

kh.step("done")
print("\n[DONE]", flush=True)
