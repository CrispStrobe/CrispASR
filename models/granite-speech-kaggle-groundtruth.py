#!/usr/bin/env python3

# LEGACY: kept to feed the examples/{qwen3,voxtral}-test-*/main.cpp
# differential-test drivers which still consume .npy files with the
# specific names this script produces. New backend ports should use
# tools/dump_reference.py + tools/reference_backends/<name>.py, which
# write a single GGUF tensor archive that crispasr-diff can load via
# core_gguf::load_weights. See ARCHITECTURE.md for the overall picture.

"""
Granite Speech 4.0-1B ground truth dump — for Kaggle (16GB+ RAM).
Dumps intermediate activations for comparing with C++ runtime.
"""

import json, math, os, subprocess, sys

def install(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

install("transformers>=4.52.1")
install("safetensors")
install("scipy")
install("huggingface_hub")
install("torchaudio")
install("soundfile")
install("peft")

import numpy as np
import torch
import requests

GH_TOKEN = None
try:
    from kaggle_secrets import UserSecretsClient
    GH_TOKEN = UserSecretsClient().get_secret("GH_TOKEN")
except:
    GH_TOKEN = os.environ.get("GH_TOKEN")

# Download model + audio
from huggingface_hub import snapshot_download

hf_token = None
try:
    from kaggle_secrets import UserSecretsClient
    hf_token = UserSecretsClient().get_secret("HF_TOKEN")
except:
    hf_token = os.environ.get("HF_TOKEN")

model_dir = snapshot_download(
    "ibm-granite/granite-4.0-1b-speech",
    local_dir="/tmp/granite-speech",
    token=hf_token,
)

audio_url = "https://github.com/ggerganov/whisper.cpp/raw/master/samples/jfk.wav"
audio_path = "/tmp/jfk.wav"
if not os.path.exists(audio_path):
    import urllib.request
    urllib.request.urlretrieve(audio_url, audio_path)

import scipy.io.wavfile as wavfile
sr, data = wavfile.read(audio_path)
audio = data.astype(np.float32) / 32768.0 if data.dtype == np.int16 else data.astype(np.float32)
print(f"Audio: {len(audio)} samples, {len(audio)/16000:.2f}s")

results = {}


def to_np(x):
    if hasattr(x, "cpu"):
        x = x.cpu()
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "float"):
        x = x.float()
    if hasattr(x, "numpy"):
        x = x.numpy()
    return np.asarray(x, dtype=np.float32)


def save(name, arr, desc=""):
    arr = to_np(arr)
    results[name] = {
        "shape": list(arr.shape),
        "desc": desc,
        "min": float(arr.min()),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "first_8": arr.flatten()[:8].tolist(),
    }
    np.save(f"/tmp/granite-{name}.npy", arr)
    print(f"  {name}: shape={arr.shape} min={arr.min():.6f} max={arr.max():.6f} mean={arr.mean():.6f}")


# ── Load model ──────────────────────────────────────────────────────
print("\nLoading model...")
from transformers import GraniteSpeechForConditionalGeneration, AutoProcessor

processor = AutoProcessor.from_pretrained(model_dir)
model = GraniteSpeechForConditionalGeneration.from_pretrained(
    model_dir, torch_dtype=torch.float32, device_map="cpu"
)
model.eval()
print("Model loaded!")

# ── Inspect model structure ─────────────────────────────────────────
print("\n=== Model structure (top-level) ===")
for name, child in model.named_children():
    print(f"  model.{name}: {type(child).__name__}")

encoder = model.encoder
print(f"\n=== Encoder structure ===")
for name, child in encoder.named_children():
    nparams = sum(p.numel() for p in child.parameters())
    print(f"  encoder.{name}: {type(child).__name__}  ({nparams} params)")

print(f"  encoder.num_layers = {getattr(encoder, 'num_layers', 'N/A')}")

# ── Process audio ───────────────────────────────────────────────────
print("\n=== Processor output ===")
import torchaudio

wav, sr_wav = torchaudio.load(audio_path, normalize=True)
assert wav.shape[0] == 1 and sr_wav == 16000

tokenizer = processor.tokenizer
user_prompt = "<|audio|>can you transcribe the speech into a written format?"
chat = [{"role": "user", "content": user_prompt}]
prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
print(f"  Prompt: {prompt!r}")

inputs = processor(prompt, wav, device="cpu", return_tensors="pt")
print(f"Input keys: {list(inputs.keys())}")
for k, v in inputs.items():
    if hasattr(v, "shape"):
        print(f"  {k}: {v.shape} {v.dtype}")

if "input_features" in inputs:
    save("input_features", inputs["input_features"][0], "mel features from processor")
if "input_ids" in inputs:
    ids = inputs["input_ids"][0].tolist()
    save("input_ids", np.array(ids[:50], dtype=np.float32), f"first 50 input_ids: {ids[:50]}")
    n_audio = sum(1 for i in ids if i == 100352)
    print(f"  audio tokens in prompt: {n_audio}")
    print(f"  total tokens: {len(ids)}")

# ── Encoder with hooks for per-layer intermediates ──────────────────
print("\n=== Encoder output (via hooks) ===")
layer_outputs = {}


def make_hook(layer_idx):
    def hook_fn(module, input, output):
        # output may be a tensor or tuple; grab the tensor
        out = output[0] if isinstance(output, tuple) else output
        layer_outputs[layer_idx] = out.detach().clone()
    return hook_fn


# Register hooks on encoder layers
hooks = []
for idx, layer in enumerate(encoder.layers):
    hooks.append(layer.register_forward_hook(make_hook(idx)))

# Also hook into input_linear
input_linear_output = {}
def input_linear_hook(module, input, output):
    out = output[0] if isinstance(output, tuple) else output
    input_linear_output["val"] = out.detach().clone()

hooks.append(encoder.input_linear.register_forward_hook(input_linear_hook))

with torch.no_grad():
    feats = inputs.get("input_features", inputs.get("input_values"))
    if feats is not None:
        # Run the full encoder forward (handles attention_dists, mid-layer
        # self-conditioned CTC, etc. correctly)
        enc_out = encoder(feats)
        # enc_out may be a BaseModelOutput or a plain tensor
        if hasattr(enc_out, "last_hidden_state"):
            enc_hidden = enc_out.last_hidden_state
        elif isinstance(enc_out, tuple):
            enc_hidden = enc_out[0]
        else:
            enc_hidden = enc_out

        # Save input_linear output
        if "val" in input_linear_output:
            save("enc_after_input_linear", input_linear_output["val"][0],
                 "after input_linear")

        # Save selected layer outputs
        num_layers = len(encoder.layers)
        for idx in sorted(layer_outputs.keys()):
            layer_num = idx + 1  # 1-based
            if layer_num in [1, 4, 8, num_layers // 2, num_layers]:
                save(f"enc_after_layer{layer_num}", layer_outputs[idx][0],
                     f"encoder hidden after layer {layer_num}")

        save("encoder_out", enc_hidden[0], "final encoder output")

# Remove hooks
for h in hooks:
    h.remove()

# ── Projector output ────────────────────────────────────────────────
print("\n=== Projector output ===")
with torch.no_grad():
    try:
        # The projector (q-former) takes encoder hidden states
        # and produces downsampled audio tokens for the LLM.
        projector = model.projector
        print(f"  Projector type: {type(projector).__name__}")

        mask = inputs.get("input_features_mask")
        try:
            proj_out = projector(enc_hidden, encoder_attention_mask=mask)
        except TypeError:
            try:
                proj_out = projector(enc_hidden, mask)
            except TypeError:
                proj_out = projector(enc_hidden)

        # proj_out may be a tensor or a BaseModelOutput
        if hasattr(proj_out, "last_hidden_state"):
            proj_tensor = proj_out.last_hidden_state
        elif isinstance(proj_out, tuple):
            proj_tensor = proj_out[0]
        else:
            proj_tensor = proj_out

        save("projector_out", proj_tensor[0],
             f"projector output shape={proj_tensor.shape}")
    except Exception as e:
        print(f"  Projector call failed: {e}")
        import traceback; traceback.print_exc()

        # Fallback: try to get projector output via a full forward pass
        print("  Trying alternative: running model forward to capture projector output...")
        proj_hook_out = {}
        def proj_hook(module, input, output):
            out = output[0] if isinstance(output, tuple) else output
            if hasattr(output, "last_hidden_state"):
                out = output.last_hidden_state
            proj_hook_out["val"] = out.detach().clone()

        h = model.projector.register_forward_hook(proj_hook)
        try:
            with torch.no_grad():
                _ = model.forward(**inputs, return_dict=True)
            if "val" in proj_hook_out:
                save("projector_out", proj_hook_out["val"][0], "projector output (via forward hook)")
        except Exception as e2:
            print(f"  Forward hook fallback also failed: {e2}")
            import traceback; traceback.print_exc()
        finally:
            h.remove()

# ── Full generation ─────────────────────────────────────────────────
print("\n=== Generation ===")
try:
    with torch.no_grad():
        model_outputs = model.generate(
            **inputs, max_new_tokens=200, do_sample=False, num_beams=1
        )
        num_input_tokens = inputs["input_ids"].shape[-1]
        new_tokens = model_outputs[0, num_input_tokens:].unsqueeze(0)
        output_text = tokenizer.batch_decode(
            new_tokens, add_special_tokens=False, skip_special_tokens=True
        )
        gen_ids = model_outputs[0].tolist()
        save("gen_ids", np.array(gen_ids[:50], dtype=np.float32), f"first 50 gen_ids")
        results["gen_text"] = output_text[0] if output_text else ""
        print(f"  Generated: {output_text[0]!r}")
        print(
            f"  Input tokens: {num_input_tokens}, "
            f"Output tokens: {len(gen_ids) - num_input_tokens}"
        )
except Exception as e:
    print(f"  Generation failed: {e}")
    import traceback; traceback.print_exc()

# ── Upload to gist ──────────────────────────────────────────────────
print("\n=== Uploading ===")
summary = json.dumps(results, indent=2, default=str)
if GH_TOKEN:
    gist_files = {"granite-speech-groundtruth.json": {"content": summary}}
    resp = requests.post(
        "https://api.github.com/gists",
        headers={
            "Authorization": f"token {GH_TOKEN}",
            "Accept": "application/vnd.github.v3+json",
        },
        json={
            "description": "Granite Speech 4.0-1B ground truth (jfk.wav)",
            "public": False,
            "files": gist_files,
        },
    )
    if resp.status_code == 201:
        print(f"  GIST: {resp.json()['html_url']}")
    else:
        print(f"  Gist failed: {resp.status_code} {resp.text[:200]}")
else:
    with open("/tmp/granite-groundtruth.json", "w") as f:
        f.write(summary)
    print("  Saved to /tmp/granite-groundtruth.json")

print("\nDone!")
