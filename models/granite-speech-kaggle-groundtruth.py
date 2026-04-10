#!/usr/bin/env python3
"""
Granite Speech 4.0-1B ground truth dump — for Kaggle (16GB+ RAM).
Dumps intermediate activations for comparing with C++ runtime.
"""

import json, math, os, subprocess, sys

def install(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

install("transformers>=4.50")
install("safetensors")
install("scipy")
install("huggingface_hub")
install("torchaudio")
install("soundfile")

import numpy as np
import torch
import requests

GH_TOKEN = None
try:
    from kaggle_secrets import UserSecretsClient
    GH_TOKEN = UserSecretsClient().get_secret("GH_TOKEN")
except: GH_TOKEN = os.environ.get("GH_TOKEN")

# Download model + audio
from huggingface_hub import snapshot_download

# Use HF token for faster downloads (add HF_TOKEN secret in Kaggle)
hf_token = None
try:
    from kaggle_secrets import UserSecretsClient
    hf_token = UserSecretsClient().get_secret("HF_TOKEN")
except: hf_token = os.environ.get("HF_TOKEN")

model_dir = snapshot_download("ibm-granite/granite-4.0-1b-speech", local_dir="/tmp/granite-speech",
                               token=hf_token)
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
    if hasattr(x, 'cpu'): x = x.cpu()
    if hasattr(x, 'detach'): x = x.detach()
    if hasattr(x, 'float'): x = x.float()
    if hasattr(x, 'numpy'): x = x.numpy()
    return np.asarray(x, dtype=np.float32)

def save(name, arr, desc=""):
    arr = to_np(arr)
    results[name] = {
        "shape": list(arr.shape), "desc": desc,
        "min": float(arr.min()), "max": float(arr.max()),
        "mean": float(arr.mean()), "std": float(arr.std()),
        "first_8": arr.flatten()[:8].tolist(),
    }
    np.save(f"/tmp/granite-{name}.npy", arr)
    print(f"  {name}: shape={arr.shape} min={arr.min():.6f} max={arr.max():.6f} mean={arr.mean():.6f}")

# Load model
print("\nLoading model...")
from transformers import GraniteSpeechForConditionalGeneration, AutoProcessor
processor = AutoProcessor.from_pretrained(model_dir)
model = GraniteSpeechForConditionalGeneration.from_pretrained(model_dir, torch_dtype=torch.float32, device_map="cpu")
model.eval()
print("Model loaded!")

# Process audio through the processor (following the official README)
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
    if hasattr(v, 'shape'):
        print(f"  {k}: {v.shape} {v.dtype}")

if "input_features" in inputs:
    save("input_features", inputs["input_features"][0], "mel features from processor")
if "input_ids" in inputs:
    ids = inputs["input_ids"][0].tolist()
    save("input_ids", np.array(ids[:50], dtype=np.float32), f"first 50 input_ids: {ids[:50]}")
    n_audio = sum(1 for i in ids if i == 100352)
    print(f"  audio tokens in prompt: {n_audio}")
    print(f"  total tokens: {len(ids)}")

# Encoder output
print("\n=== Encoder output ===")
with torch.no_grad():
    # Run just the encoder
    feats = inputs.get("input_features", inputs.get("input_values"))
    if feats is not None:
        enc_out = model.encoder(feats)
        save("encoder_out", enc_out.last_hidden_state[0], f"encoder output")

        # Projector output
        print("\n=== Projector output ===")
        proj_out = model.projector(enc_out.last_hidden_state)
        save("projector_out", proj_out[0], f"projector output (audio tokens for LLM)")

# Full generation (following README pattern)
print("\n=== Generation ===")
try:
    with torch.no_grad():
        model_outputs = model.generate(**inputs, max_new_tokens=200, do_sample=False, num_beams=1)
        num_input_tokens = inputs["input_ids"].shape[-1]
        new_tokens = model_outputs[0, num_input_tokens:].unsqueeze(0)
        output_text = tokenizer.batch_decode(new_tokens, add_special_tokens=False, skip_special_tokens=True)
        gen_ids = model_outputs[0].tolist()
        save("gen_ids", np.array(gen_ids[:50], dtype=np.float32), f"first 50 gen_ids")
        results["gen_text"] = output_text[0] if output_text else ""
        print(f"  Generated: {output_text[0]!r}")
        print(f"  Input tokens: {num_input_tokens}, Output tokens: {len(gen_ids) - num_input_tokens}")
except Exception as e:
    print(f"  Generation failed: {e}")
    import traceback; traceback.print_exc()

# Upload to gist
print("\n=== Uploading ===")
summary = json.dumps(results, indent=2, default=str)
if GH_TOKEN:
    gist_files = {"granite-speech-groundtruth.json": {"content": summary}}
    resp = requests.post("https://api.github.com/gists",
        headers={"Authorization": f"token {GH_TOKEN}", "Accept": "application/vnd.github.v3+json"},
        json={"description": "Granite Speech 4.0-1B ground truth (jfk.wav)", "public": False, "files": gist_files})
    if resp.status_code == 201:
        print(f"  GIST: {resp.json()['html_url']}")
    else:
        print(f"  Gist failed: {resp.status_code}")
else:
    with open("/tmp/granite-groundtruth.json", "w") as f: f.write(summary)
    print("  Saved to /tmp/granite-groundtruth.json")

print("\nDone!")
