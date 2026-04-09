#!/usr/bin/env python3
"""Dump intermediate activations from the Voxtral 4B Realtime model for
comparison with the C++ runtime."""

import sys
import numpy as np
import torch
import torchaudio

model_dir = "/mnt/akademie_storage/voxtral-4b-realtime"
audio_path = "samples/jfk.wav"

print("Loading model...")
try:
    from transformers import VoxtralRealtimeForConditionalGeneration, AutoProcessor
except ImportError:
    sys.exit("pip install transformers>=5.2.0")

processor = AutoProcessor.from_pretrained(model_dir)
model = VoxtralRealtimeForConditionalGeneration.from_pretrained(
    model_dir, torch_dtype=torch.float32, device_map="cpu")
model.eval()

# Load audio
print(f"Loading audio: {audio_path}")
waveform, sr = torchaudio.load(audio_path)
if sr != 16000:
    waveform = torchaudio.functional.resample(waveform, sr, 16000)
if waveform.shape[0] > 1:
    waveform = waveform.mean(0, keepdim=True)
audio_array = waveform.squeeze().numpy()
print(f"  audio: {len(audio_array)} samples, {len(audio_array)/16000:.2f}s")

# Process through processor
inputs = processor(audio_array, return_tensors="pt")
print(f"  input_features shape: {inputs['input_features'].shape}")
print(f"  input_ids shape: {inputs['input_ids'].shape}")
print(f"  input_ids: {inputs['input_ids'][0,:20].tolist()}...")

# Mel features
mel = inputs['input_features'][0, 0]  # (n_mels, T)
print(f"  mel shape: {mel.shape}")
np.save("/tmp/voxtral4b-ref-mel.npy", mel.numpy())
print(f"  mel stats: min={mel.min():.4f} max={mel.max():.4f} mean={mel.mean():.4f}")

# Run encoder
with torch.no_grad():
    # Get audio features through the full pipeline
    audio_outputs = model.get_audio_features(
        input_features=inputs['input_features'],
        return_dict=True,
    )
    encoder_out = audio_outputs.pooler_output  # After projector
    print(f"\n  encoder_out (after projector) shape: {encoder_out.shape}")
    print(f"  encoder_out stats: min={encoder_out.min():.4f} max={encoder_out.max():.4f} mean={encoder_out.mean():.4f}")
    np.save("/tmp/voxtral4b-ref-encoder-out.npy", encoder_out[0].numpy())

    # Time embedding
    time_tensor = torch.full((1,), 6.0)  # delay_tokens=6
    t_cond = model.time_embedding(time_tensor)
    print(f"\n  t_cond shape: {t_cond.shape}")
    print(f"  t_cond[:8]: {t_cond[:8].tolist()}")
    np.save("/tmp/voxtral4b-ref-t_cond.npy", t_cond.numpy())

    # Full generation
    print("\nRunning generation...")
    outputs = model.generate(**inputs, max_new_tokens=50)
    decoded = processor.batch_decode(outputs, skip_special_tokens=True)
    print(f"  output: {decoded[0]!r}")
    print(f"  output tokens: {outputs[0].tolist()}")

print("\nDone. Reference files saved to /tmp/voxtral4b-ref-*.npy")
