#!/usr/bin/env python3
"""
Voxtral-Mini-4B-Realtime ground truth dump — designed for Kaggle (16GB+ RAM).

Dumps all intermediate activations to compare with C++ runtime.
Uploads results as a GitHub Gist automatically.

Setup in Kaggle:
  1. Add secret: GH_TOKEN = your GitHub personal access token (gist scope)
  2. Enable internet access
  3. Select GPU T4 or CPU (16GB RAM minimum)
  4. Paste this entire script into a notebook cell and run

What it dumps:
  - Mel features (our Whisper-style + correct VoxtralRealtime-style)
  - t_cond (time embedding for delay=6)
  - Ada-norm scales per layer (from actual model weights)
  - Encoder output (after projector)
  - Conv stem output (after conv1, conv2)
  - First token logits from prefill
  - Full generation output (50 tokens)
  - Token IDs for the prompt
"""

import json
import math
import os
import subprocess
import sys


# ============================================================
# Install dependencies
# ============================================================
def install(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])


# Install ALL dependencies FIRST, before any transformers import.
# mistral-common MUST be installed before transformers tries to load
# VoxtralRealtimeProcessor, otherwise you get ImportError.
install("mistral-common[audio]")
install("librosa")
install("safetensors")
install("scipy")
install("huggingface_hub")
install("transformers>=5.2.0")

# Force reimport after installs (Kaggle/Jupyter caches old import state)
for mod_name in list(sys.modules.keys()):
    if "mistral" in mod_name or "transformers" in mod_name:
        del sys.modules[mod_name]

# Try to get GH_TOKEN from Kaggle secrets
GH_TOKEN = None
try:
    from kaggle_secrets import UserSecretsClient

    GH_TOKEN = UserSecretsClient().get_secret("GH_TOKEN")
except Exception:
    GH_TOKEN = os.environ.get("GH_TOKEN", None)

if not GH_TOKEN:
    print("WARNING: GH_TOKEN not found — will save locally only, no gist upload")

import numpy as np  # noqa: E402
import torch  # noqa: E402
import requests  # noqa: E402

# ============================================================
# Download model + audio
# ============================================================
print("=" * 60)
print("Downloading model and audio...")
print("=" * 60)

from huggingface_hub import snapshot_download  # noqa: E402

model_dir = snapshot_download(
    "mistralai/Voxtral-Mini-4B-Realtime-2602",
    local_dir="/tmp/voxtral-4b",
    ignore_patterns=["consolidated.safetensors"],  # skip the duplicate
)
print(f"Model downloaded to: {model_dir}")

# Download jfk.wav (the standard Whisper test audio)
audio_url = "https://raw.githubusercontent.com/CrispStrobe/CrispASR/main/samples/jfk.wav"
audio_path = "/tmp/jfk.wav"
if not os.path.exists(audio_path):
    import urllib.request

    urllib.request.urlretrieve(audio_url, audio_path)
print(f"Audio: {audio_path}")

# ============================================================
# Load audio
# ============================================================
import scipy.io.wavfile as wavfile  # noqa: E402

sr, data = wavfile.read(audio_path)
assert sr == 16000, f"Expected 16kHz, got {sr}"
audio = (
    data.astype(np.float32) / 32768.0
    if data.dtype == np.int16
    else data.astype(np.float32)
)
print(f"Audio: {len(audio)} samples, {len(audio)/16000:.2f}s")

results = {}  # name -> {"shape": ..., "dtype": ..., "data": base64, "stats": ...}


def to_np(x):
    """Convert tensor or array to numpy float32, handling GPU tensors."""
    if hasattr(x, "cpu"):
        x = x.cpu()
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "float"):
        x = x.float()
    if hasattr(x, "numpy"):
        x = x.numpy()
    return np.asarray(x, dtype=np.float32)


def save_result(name, arr, desc=""):
    """Save a numpy array to results dict."""
    arr = to_np(arr)
    results[name] = {
        "shape": list(arr.shape),
        "desc": desc,
        "min": float(arr.min()),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "first_8": arr.flatten()[:8].tolist(),
        "last_8": arr.flatten()[-8:].tolist(),
    }
    np.save(f"/tmp/v4b-{name}.npy", arr)
    print(
        f"  {name}: shape={arr.shape} min={arr.min():.6f} max={arr.max():.6f} mean={arr.mean():.6f}"
    )


# ============================================================
# 1. Time embedding (t_cond) — pure math
# ============================================================
print("\n" + "=" * 60)
print("1. Time embedding (t_cond)")
print("=" * 60)

dim = 3072
delay_tokens = 6
theta = 10000.0
inv_freq = np.exp(-math.log(theta) * np.arange(dim // 2, dtype=np.float32) / (dim // 2))
emb = delay_tokens * inv_freq
t_cond_np = np.concatenate([np.cos(emb), np.sin(emb)]).astype(np.float32)
save_result("t_cond", t_cond_np, "sinusoidal time embedding for delay=6")

# ============================================================
# 2. Mel features (both Whisper-style and VoxtralRealtime-style)
# ============================================================
print("\n" + "=" * 60)
print("2. Mel features")
print("=" * 60)

try:
    from librosa.filters import mel as librosa_mel

    mel_filters = librosa_mel(sr=16000, n_fft=400, n_mels=128).T  # (201, 128)
except ImportError:
    install("librosa")
    from librosa.filters import mel as librosa_mel

    mel_filters = librosa_mel(sr=16000, n_fft=400, n_mels=128).T

from scipy.signal import get_window  # noqa: E402
from scipy.fft import rfft  # noqa: E402

n_fft, hop, n_mels, n_freqs = 400, 160, 128, 201
hann = get_window("hann", n_fft, fftbins=True).astype(np.float32)

pad = n_fft // 2
padded = np.pad(audio, (pad, pad), mode="constant")
T = (len(padded) - n_fft) // hop + 1 - 1  # Whisper drops last frame

power = np.zeros((n_freqs, T), dtype=np.float32)
for t in range(T):
    frame = padded[t * hop : t * hop + n_fft] * hann
    spectrum = rfft(frame)
    power[:, t] = np.abs(spectrum) ** 2

mel = np.zeros((n_mels, T), dtype=np.float32)
for m in range(n_mels):
    for t in range(T):
        mel[m, t] = np.log10(max(np.dot(mel_filters[:, m], power[:, t]), 1e-10))

# VoxtralRealtime normalization (global_log_mel_max=1.5)
global_log_mel_max = 1.5
floor_v = global_log_mel_max - 8.0
mel_fixed = np.clip(mel, floor_v, None)
mel_fixed = (mel_fixed + 4.0) / 4.0

# Pad to 3000
T_out = 3000
mel_padded = np.full((n_mels, T_out), (floor_v + 4.0) / 4.0, dtype=np.float32)
mel_padded[:, : min(T, T_out)] = mel_fixed[:, : min(T, T_out)]

save_result(
    "mel_voxtral", mel_padded, "mel with global_log_mel_max=1.5, padded to 3000"
)

# ============================================================
# 3. Load model and dump intermediates
# ============================================================
print("\n" + "=" * 60)
print("3. Loading model...")
print("=" * 60)

from transformers import VoxtralRealtimeForConditionalGeneration  # noqa: E402

# Load processor — try AutoProcessor first, fall back to manual construction
try:
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(model_dir)
    print("  Loaded processor via AutoProcessor")
except (ImportError, Exception) as e:
    print(f"  AutoProcessor failed ({e}), building processor manually...")
    # Manually build feature extractor + tokenizer
    from transformers import AutoFeatureExtractor, AutoTokenizer

    feature_extractor = AutoFeatureExtractor.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    # Create a minimal processor-like object
    class MinimalProcessor:
        def __init__(self, fe, tok):
            self.feature_extractor = fe
            self.tokenizer = tok

        def __call__(self, audio, **kwargs):
            feats = self.feature_extractor(audio, sampling_rate=16000, **kwargs)
            # Build default input_ids: BOS + STREAMING_PAD * 38
            input_ids = torch.tensor([[1] + [32] * 38])
            feats["input_ids"] = input_ids
            return feats

        def batch_decode(self, ids, **kwargs):
            return self.tokenizer.batch_decode(ids, **kwargs)

        def decode(self, ids, **kwargs):
            return self.tokenizer.decode(ids, **kwargs)

    processor = MinimalProcessor(feature_extractor, tokenizer)
    print("  Built minimal processor")

_device = "cuda" if torch.cuda.is_available() else "cpu"
_dtype = torch.float16 if _device == "cuda" else torch.float32
print(f"  Loading model on {_device} with {_dtype}")
model = VoxtralRealtimeForConditionalGeneration.from_pretrained(
    model_dir, torch_dtype=_dtype, device_map=_device
)
model.eval()
print("Model loaded!")

# ============================================================
# 4. Processor mel (ground truth from HF)
# ============================================================
print("\n" + "=" * 60)
print("4. HF Processor mel features")
print("=" * 60)

inputs = processor(audio, return_tensors="pt")


# Move inputs to model device AND cast float tensors to model dtype
def move_inputs(d, device, dtype):
    out = {}
    for k, v in d.items():
        if hasattr(v, "to"):
            if v.is_floating_point():
                out[k] = v.to(device=device, dtype=dtype)
            else:
                out[k] = v.to(device=device)
        else:
            out[k] = v
    return out


inputs = move_inputs(inputs, _device, _dtype)

# Debug: print all input shapes
print("  Input keys and shapes:")
for k, v in inputs.items():
    if hasattr(v, "shape"):
        print(f"    {k}: {v.shape} dtype={v.dtype}")

# Mel features — handle various possible shapes
raw_mel = inputs["input_features"]
print(f"  raw input_features shape: {raw_mel.shape}")
if raw_mel.dim() == 4:
    hf_mel = raw_mel[0, 0]  # (batch, channel, n_mels, T) -> (n_mels, T)
elif raw_mel.dim() == 3:
    hf_mel = raw_mel[0]  # (batch, n_mels, T) -> (n_mels, T)
elif raw_mel.dim() == 2:
    hf_mel = raw_mel  # already (n_mels, T) or (T, n_mels)
else:
    hf_mel = raw_mel.squeeze()
print(f"  hf_mel shape after squeeze: {hf_mel.shape}")
save_result(
    "mel_hf",
    to_np(hf_mel),
    f"mel from HF processor, raw shape was {list(raw_mel.shape)}",
)

# Also save the mel transposed if it looks like (T, n_mels)
if hf_mel.shape[0] > hf_mel.shape[-1] if hf_mel.dim() >= 2 else False:
    save_result(
        "mel_hf_transposed",
        hf_mel.T.numpy(),
        "mel_hf transposed (might be correct orientation)",
    )

hf_input_ids = inputs["input_ids"][0]
save_result(
    "input_ids_hf",
    to_np(hf_input_ids),
    f"HF processor input_ids (first 20: {hf_input_ids[:20].tolist()})",
)

# ============================================================
# 5. Model t_cond (from model's time_embedding module)
# ============================================================
print("\n" + "=" * 60)
print("5. Model time embedding")
print("=" * 60)

with torch.no_grad():
    time_tensor = torch.full((1,), float(delay_tokens), device=_device, dtype=_dtype)
    t_cond_model = model.time_embedding(time_tensor)
    save_result(
        "t_cond_model", to_np(t_cond_model), "t_cond from model.time_embedding(6)"
    )

# ============================================================
# 6. Ada-norm scales per layer
# ============================================================
print("\n" + "=" * 60)
print("6. Ada-norm scales")
print("=" * 60)

with torch.no_grad():
    t_cond_torch = (
        torch.from_numpy(t_cond_np).unsqueeze(0).to(device=_device, dtype=_dtype)
    )
    for il in range(min(3, len(model.language_model.model.layers))):
        layer = model.language_model.model.layers[il]
        ada_out = layer.ada_rms_norm(t_cond_torch)
        one_plus = to_np((1 + ada_out).squeeze(0))
        save_result(
            f"ada_scale_layer{il}", one_plus, f"1 + ada_rms_norm(t_cond) for layer {il}"
        )


def gpu_cleanup():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    import gc

    gc.collect()


# ============================================================
# 7. Encoder output (after projector)
# ============================================================
print("\n" + "=" * 60)
print("7. Encoder output")
print("=" * 60)

try:
    with torch.no_grad():
        audio_outputs = model.get_audio_features(
            input_features=inputs["input_features"],
            return_dict=True,
        )
        encoder_out = audio_outputs.pooler_output[0]
        save_result(
            "encoder_out",
            to_np(encoder_out),
            f"encoder output after projector, shape {encoder_out.shape}",
        )
        encoder_hidden = audio_outputs.last_hidden_state
        save_result(
            "encoder_hidden_pre_proj",
            to_np(encoder_hidden[0]),
            f"encoder hidden before projector, shape {encoder_hidden[0].shape}",
        )
        del audio_outputs, encoder_out, encoder_hidden
        gpu_cleanup()
except Exception as e:
    print(f"  Encoder output failed: {e}")
    import traceback

    traceback.print_exc()

# ============================================================
# 8. Full generation
# ============================================================
print("\n" + "=" * 60)
print("8. Full generation (50 tokens)")
print("=" * 60)

try:
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=50)
        gen_ids = outputs[0].tolist()
        decoded = processor.batch_decode(outputs, skip_special_tokens=True)
        save_result(
            "gen_ids",
            np.array(gen_ids, dtype=np.float32),
            f"generated token IDs: {gen_ids[:30]}...",
        )
        results["gen_text"] = decoded[0]
        print(f"  Generated text: {decoded[0]!r}")
        print(f"  Token IDs: {gen_ids[:30]}...")
        del outputs
        gpu_cleanup()
except Exception as e:
    print(f"  Generation failed: {e}")
    import traceback

    traceback.print_exc()

# ============================================================
# 9. First-token logits from prefill
# ============================================================
print("\n" + "=" * 60)
print("9. First-token logits")
print("=" * 60)

try:
    with torch.no_grad():
        fwd_out = model(**inputs)
        logits = fwd_out.logits[0, -1]
        top_k = torch.topk(logits, 10)
        save_result(
            "prefill_logits_top10_vals",
            to_np(top_k.values),
            "top-10 logit values from prefill",
        )
        save_result(
            "prefill_logits_top10_ids",
            to_np(top_k.indices),
            f"top-10 token IDs: {top_k.indices.tolist()}",
        )
        print(
            f"  Top-10 tokens: {[(processor.decode([tid]), float(val)) for tid, val in zip(top_k.indices.tolist(), top_k.values.tolist())]}"
        )
except Exception as e:
    print(f"  Prefill logits failed: {e}")
    gpu_cleanup()

# ============================================================
# 10. Conv stem output (first few frames)
# ============================================================
print("\n" + "=" * 60)
print("10. Conv stem output")
print("=" * 60)

try:
    with torch.no_grad():
        conv_out = model.audio_tower.embedder(inputs["input_features"])
        save_result(
            "conv_stem_out",
            to_np(conv_out[0, :10, :]),
            f"conv stem output (first 10 frames), full shape {conv_out.shape}",
        )
        del conv_out
        gpu_cleanup()
except Exception as e:
    print(f"  Conv stem failed: {e}")
    import traceback

    traceback.print_exc()

# ============================================================
# Upload to GitHub Gist
# ============================================================
print("\n" + "=" * 60)
print("Uploading to GitHub Gist...")
print("=" * 60)

# Build a summary JSON
summary = json.dumps(results, indent=2, default=str)

# Also save the raw numpy arrays in a compact format
npy_files = {}
for name in results:
    npy_path = f"/tmp/v4b-{name}.npy"
    if os.path.exists(npy_path):
        import base64

        with open(npy_path, "rb") as f:
            npy_files[f"v4b-{name}.npy.b64"] = base64.b64encode(f.read()).decode(
                "ascii"
            )

if GH_TOKEN:
    gist_files = {
        "voxtral4b-groundtruth.json": {"content": summary},
    }
    # Add npy files as base64 (gist has a size limit, so only add the small ones)
    for fname, b64data in npy_files.items():
        if len(b64data) < 500000:  # skip files > 500KB
            gist_files[fname] = {"content": b64data}
        else:
            print(f"  Skipping {fname} ({len(b64data)//1024}KB > 500KB limit)")

    resp = requests.post(
        "https://api.github.com/gists",
        headers={
            "Authorization": f"token {GH_TOKEN}",
            "Accept": "application/vnd.github.v3+json",
        },
        json={
            "description": "Voxtral-Mini-4B-Realtime ground truth activations (jfk.wav, delay=6)",
            "public": False,
            "files": gist_files,
        },
    )
    if resp.status_code == 201:
        gist_url = resp.json()["html_url"]
        print(f"\n  *** GIST CREATED: {gist_url} ***\n")
    else:
        print(f"  Gist creation failed: {resp.status_code} {resp.text[:500]}")
else:
    # Save locally
    with open("/tmp/voxtral4b-groundtruth.json", "w") as f:
        f.write(summary)
    print("  Saved to /tmp/voxtral4b-groundtruth.json")

print("\n" + "=" * 60)
print("DONE! All ground truth activations dumped.")
print("=" * 60)
print("\nKey results:")
print(f"  Generated text: {results.get('gen_text', 'N/A')!r}")
print(f"  Encoder out shape: {results.get('encoder_out', {}).get('shape', 'N/A')}")
print(f"  HF mel shape: {results.get('mel_hf', {}).get('shape', 'N/A')}")
