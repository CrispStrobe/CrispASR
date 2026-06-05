# %% [markdown]
# # CrispASR — TADA-3B-ML reference dump + GGUF conversion
#
# Downloads HumeAI/tada-3b-ml + HumeAI/tada-codec on Kaggle's
# 30 GB-RAM CPU notebook, runs the reference backend to dump
# intermediate tensors for the diff harness, converts both models
# to GGUF F16, and uploads everything to HuggingFace.
#
# Outputs:
#   - tada-ref.gguf: reference activation dump for crispasr-diff
#   - tada-tts-3b-ml-f16.gguf: main model GGUF
#   - tada-codec-f16.gguf: codec decoder GGUF
#
# Triggered from Kaggle UI ("Save Version → Run All").

# %% [code]
# ── Cell 1: env setup ──
import os
import sys

# Prevent transformers from importing tensorflow (protobuf clash on Kaggle)
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# %% [code]
# ── Cell 2: clone CrispASR + install deps ──
import subprocess
from pathlib import Path

WORK = Path("/kaggle/working")
REPO = WORK / "CrispASR"

if not REPO.exists():
    subprocess.check_call([
        "git", "clone", "--depth", "1", "--branch", "feature/tada-tts",
        "https://github.com/CrispStrobe/CrispASR.git", str(REPO),
    ])

# Import harness
sys.path.insert(0, str(REPO / "tools" / "kaggle"))
import kaggle_harness as kh
kh.init_progress()

# Nuke tensorflow to avoid protobuf version clash that kills transformers import
subprocess.check_call([
    sys.executable, "-m", "pip", "uninstall", "-y", "tensorflow", "tf-keras",
], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
print("[cell 2] tensorflow removed")

# Install deps (upgrade protobuf to avoid any remaining issues)
subprocess.check_call([
    sys.executable, "-m", "pip", "install", "--quiet", "--upgrade",
    "protobuf",
])
subprocess.check_call([
    sys.executable, "-m", "pip", "install", "--quiet",
    "torch", "torchaudio", "transformers", "safetensors", "gguf",
    "huggingface_hub", "hf_transfer",
    "hume-tada",  # TADA Python package (pulls dac etc.)
])
print("[cell 2] deps installed")

# Create a minimal 16 kHz WAV for the dump harness (audio is unused for TTS
# but dump_reference.py requires it). 1 second of silence.
import wave, struct
dummy_wav = REPO / "samples" / "jfk.wav"
if not dummy_wav.exists():
    dummy_wav.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(dummy_wav), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(struct.pack("<" + "h" * 16000, *([0] * 16000)))
    print("[cell 2] created dummy jfk.wav")

# %% [code]
# ── Cell 3: download source models ──
import shutil
from huggingface_hub import snapshot_download

token = kh.resolve_hf_token()

for candidate in ("/kaggle/temp", "/tmp"):
    if os.path.isdir(candidate):
        scratch = Path(candidate) / "tada-models"
        break
scratch.mkdir(parents=True, exist_ok=True)
print(f"[cell 3] scratch: {scratch}  "
      f"(free: {shutil.disk_usage(scratch).free / (1024**3):.1f} GiB)")

kh.step("downloading tada-3b-ml")
model_dir = Path(snapshot_download(
    repo_id="HumeAI/tada-3b-ml",
    cache_dir=str(scratch),
    token=token,
))
print(f"[cell 3] model_dir: {model_dir}")

kh.step("downloading tada-codec")
codec_dir = Path(snapshot_download(
    repo_id="HumeAI/tada-codec",
    cache_dir=str(scratch),
    token=token,
    allow_patterns=["decoder/*"],
))
print(f"[cell 3] codec_dir: {codec_dir}")

# %% [code]
# ── Cell 4: run reference dump ──
kh.step("running reference dump")
os.environ["TADA_SYN_TEXT"] = "Hello world."
os.environ["TADA_NUM_FM_STEPS"] = "10"
os.environ["TADA_CFG_SCALE"] = "1.0"
os.environ["TADA_NOISE_TEMP"] = "0.0"
os.environ["TADA_SEED"] = "42"
os.environ["TADA_DEVICE"] = "cpu"
os.environ["TADA_CODEC_DIR"] = str(codec_dir)

ref_output = WORK / "tada-ref.gguf"
subprocess.check_call([
    sys.executable, "tools/dump_reference.py",
    "--backend", "tada-tts",
    "--model-dir", str(model_dir),
    "--audio", "samples/jfk.wav",
    "--output", str(ref_output),
], cwd=str(REPO))
print(f"[cell 4] ref dump: {ref_output} ({ref_output.stat().st_size / 1e6:.1f} MB)")

# %% [code]
# ── Cell 5: convert main model to GGUF ──
kh.step("converting tada-3b-ml to GGUF")
model_gguf = WORK / "tada-tts-3b-ml-f16.gguf"
try:
    subprocess.check_call([
        sys.executable, "models/convert-tada-to-gguf.py",
        "--input", str(model_dir),
        "--output", str(model_gguf),
    ], cwd=str(REPO))
    print(f"[cell 5] model GGUF: {model_gguf} ({model_gguf.stat().st_size / 1e9:.2f} GB)")
except subprocess.CalledProcessError as e:
    print(f"[cell 5] WARN: model GGUF conversion failed: {e}")

# %% [code]
# ── Cell 6: convert codec to GGUF ──
kh.step("converting tada-codec to GGUF")
codec_gguf = WORK / "tada-codec-f16.gguf"
try:
    subprocess.check_call([
        sys.executable, "models/convert-tada-codec-to-gguf.py",
        "--input", str(codec_dir),
        "--output", str(codec_gguf),
    ], cwd=str(REPO))
    print(f"[cell 6] codec GGUF: {codec_gguf} ({codec_gguf.stat().st_size / 1e9:.2f} GB)")
except subprocess.CalledProcessError as e:
    print(f"[cell 6] WARN: codec GGUF conversion failed: {e}")

# %% [code]
# ── Cell 7: quantize to Q4_K and Q8_0 ──
kh.step("quantizing model GGUF")
model_q4k = WORK / "tada-tts-3b-ml-q4_k.gguf"
model_q8  = WORK / "tada-tts-3b-ml-q8_0.gguf"

if model_gguf.exists() and model_gguf.stat().st_size > 0:
    # Build the quantize binary from CrispASR
    build_dir = WORK / "quant-build"
    kh.install_build_toolchain()
    try:
        kh.sh_with_progress(
            f"cmake -G Ninja -B {build_dir} -S {REPO}"
            f" -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=OFF"
            f" -DCMAKE_C_COMPILER_LAUNCHER=ccache"
            f" -DCMAKE_CXX_COMPILER_LAUNCHER=ccache"
        )
        kh.sh_with_progress(
            f"cmake --build {build_dir} -j{kh.safe_build_jobs(gpu=False)}"
            f" --target quantize"
        )
        quantize_bin = build_dir / "bin" / "quantize"
        if quantize_bin.exists():
            kh.sh_with_progress(f"{quantize_bin} {model_gguf} {model_q8} Q8_0")
            print(f"[cell 7] Q8_0: {model_q8} ({model_q8.stat().st_size / 1e9:.2f} GB)")
            kh.sh_with_progress(f"{quantize_bin} {model_gguf} {model_q4k} Q4_K")
            print(f"[cell 7] Q4_K: {model_q4k} ({model_q4k.stat().st_size / 1e9:.2f} GB)")
        else:
            print(f"[cell 7] quantize binary not found at {quantize_bin}")
    except Exception as e:
        print(f"[cell 7] quantization failed: {e}")
else:
    print("[cell 7] skipping quantization — F16 GGUF not found")

# %% [code]
# ── Cell 8: write HF README ──
kh.step("writing HF README")
readme_content = """\
---
license: llama3.2
language:
- en
- es
- ja
- zh
- de
- fr
- it
- pt
- ko
- ar
base_model:
- HumeAI/tada-3b-ml
- meta-llama/Llama-3.2-3B
pipeline_tag: text-to-speech
tags:
- tts
- text-to-speech
- tada
- llama
- flow-matching
- gguf
- crispasr
library_name: ggml
---

# TADA-3B-ML — GGUF (ggml-quantised)

GGUF / ggml conversion of [`HumeAI/tada-3b-ml`](https://huggingface.co/HumeAI/tada-3b-ml) for use with **[CrispStrobe/CrispASR](https://github.com/CrispStrobe/CrispASR)**.

TADA-3B-ML is a 4B-param text-to-speech model built on Meta Llama 3.2 3B with a flow-matching (FM) speech decoder and custom Hume codec. Key innovation: **1:1 token alignment** — every text token maps to exactly one speech vector (no 7:1 expansion like Orpheus/SNAC), eliminating transcript hallucination. 10 languages (en, es, ja, zh, de, fr, it, pt, ko, ar). 24 kHz mono output.

**License:** Apache-2.0 / Llama 3.2 Community License (\"Built with Llama\").

Pair this with the TADA codec decoder at `tada-codec-f16.gguf` (included in this repo) — the talker outputs continuous acoustic vectors that the codec converts to audio.

## Files

| File | Quant | Size | Notes |
|---|---|---:|---|
| `tada-tts-3b-ml-f16.gguf` | F16 | ~8.2 GB | Reference quality (LLM + FM head) |
| `tada-tts-3b-ml-q4_k.gguf` | Q4_K | ~2.5 GB | **Recommended** — good quality, fits 8 GB RAM |
| `tada-tts-3b-ml-q8_0.gguf` | Q8_0 | ~4.5 GB | Near-lossless |
| `tada-codec-f16.gguf` | F16 | ~1.1 GB | Codec decoder (required companion) |
| `tada-ref.gguf` | F32 | ~200 KB | Reference activations for diff harness |

## Architecture

```
Text Input
  |
BPE Tokenize (Llama-3.2 128K vocab)
  |
Llama-3.2-3B AR Forward (28L, 3072d, 24 heads / 8 KV)
  + acoustic embedding (512d) + time embedding (gray code)
  |-- Each position outputs: hidden state for FM head
  |
VibeVoice Diffusion Head (6L SwiGLU + AdaLN, flow matching)
  |-- Sinusoidal timestep embedding
  |-- 10 Euler ODE steps: noise -> speech vector (528d)
  |
TADA Codec Decoder (6L local-attention + DAC upsampler)
  |-- speech vectors -> 24 kHz PCM
  |
Output: float32 mono @ 24 kHz
```

## Quick start

```bash
# 1. Build CrispASR
git clone https://github.com/CrispStrobe/CrispASR
cd CrispASR
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j --target crispasr-cli

# 2. Pull the model + codec
huggingface-cli download cstr/tada-tts-3b-ml-GGUF tada-tts-3b-ml-q4_k.gguf --local-dir .
huggingface-cli download cstr/tada-tts-3b-ml-GGUF tada-codec-f16.gguf --local-dir .

# 3. Synthesise
./build/bin/crispasr --backend tada \\
    -m tada-tts-3b-ml-q4_k.gguf \\
    --codec-model tada-codec-f16.gguf \\
    --tts "Hello, this is a test of the TADA speech synthesis system." \\
    --tts-output hello.wav
```

## Source model

- **Upstream:** [HumeAI/tada-3b-ml](https://huggingface.co/HumeAI/tada-3b-ml) (safetensors, ~6.6 GB BF16)
- **Codec:** [HumeAI/tada-codec](https://huggingface.co/HumeAI/tada-codec) (encoder + decoder)
- **Paper:** [arXiv:2602.23068](https://arxiv.org/abs/2602.23068)
- **Converted with:** `models/convert-tada-to-gguf.py` + `models/convert-tada-codec-to-gguf.py`
"""

readme_path = WORK / "README.md"
readme_path.write_text(readme_content)
print(f"[cell 8] README written ({len(readme_content)} bytes)")

# %% [code]
# ── Cell 9: upload to HuggingFace ──
kh.step("uploading to HuggingFace")
try:
    from huggingface_hub import HfApi
    api = HfApi(token=token)
    repo_id = "cstr/tada-tts-3b-ml-GGUF"

    api.create_repo(repo_id=repo_id, exist_ok=True, repo_type="model")

    upload_files = [ref_output, model_gguf, codec_gguf, readme_path]
    if model_q4k.exists():
        upload_files.append(model_q4k)
    if model_q8.exists():
        upload_files.append(model_q8)

    for fpath in upload_files:
        if fpath.exists() and fpath.stat().st_size > 0:
            print(f"  uploading {fpath.name}...")
            api.upload_file(
                path_or_fileobj=str(fpath),
                path_in_repo=fpath.name,
                repo_id=repo_id,
                repo_type="model",
            )
            print(f"  ✓ {fpath.name}")

    print(f"[cell 9] uploaded to {repo_id}")
except Exception as exc:
    print(f"[cell 9] upload failed: {exc}")
    print("  Files staged at /kaggle/working/ for manual pickup")

# Clean up the cloned repo from /kaggle/working/ so kernel output
# only contains the GGUF files (not 2600+ repo files that bloat the
# output and prevent `kaggle kernels output` from reaching the GGUFs).
import shutil
if REPO.exists():
    shutil.rmtree(str(REPO), ignore_errors=True)
    print("[cleanup] removed CrispASR clone from working dir")

kh.step("done")
print("\n=== ALL DONE ===")
for fpath in [ref_output, model_gguf, codec_gguf]:
    if fpath.exists():
        print(f"  {fpath.name}: {fpath.stat().st_size / 1e9:.2f} GB")
