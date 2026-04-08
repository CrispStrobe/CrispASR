#!/usr/bin/env python3
"""
Dump PyTorch reference activations for Qwen3-ASR-0.6B at architectural
boundaries — used as ground truth for the C++ ggml port.

Hooks the HF model and writes one .npy per checkpoint:

  mel_input.npy             (1, 128, T)        log-mel spectrogram fed to encoder
  conv1_out.npy             (1, 480, F1, T1)   after conv2d1 + activation
  conv2_out.npy             (1, 480, F2, T2)
  conv3_out.npy             (1, 480, F3, T3)
  conv_out.npy              (1, T_enc, 896)    after flatten + conv_out linear
  enc_blk00_out.npy         (1, T_enc, 896)    after encoder block 0
  enc_blk17_out.npy         (1, T_enc, 896)    after encoder block 17 (last)
  ln_post_out.npy           (1, T_enc, 896)
  proj1_out.npy             (1, T_enc, 896)
  proj2_out.npy             (1, T_enc, 1024)   final audio embeddings → LLM
  llm_logits_first10.npy    (1, 10, 151936)    LLM logits for first 10 generated tokens
  generated_tokens.npy      (N,)                token IDs produced by greedy decode

Usage:
  python models/qwen3-asr-reference-dump.py \\
      --model-dir /tmp/qwen3-asr-inspect \\
      --audio samples/jfk.wav \\
      --out-dir /tmp/qwen3-asr-ref/jfk
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def load_audio_16k_mono(path: Path) -> np.ndarray:
    """Load 16 kHz mono WAV using stdlib wave (no torchaudio needed)."""
    import wave
    with wave.open(str(path), "rb") as w:
        sr = w.getframerate()
        nchan = w.getnchannels()
        sampw = w.getsampwidth()
        nframes = w.getnframes()
        raw = w.readframes(nframes)
    if sampw != 2:
        raise SystemExit(f"only 16-bit PCM supported, got {sampw*8}-bit")
    pcm = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if nchan > 1:
        pcm = pcm.reshape(-1, nchan).mean(axis=1)
    if sr != 16000:
        raise SystemExit(f"expected 16 kHz, got {sr} — pre-convert with ffmpeg")
    return pcm


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", required=True, type=Path)
    p.add_argument("--audio",     required=True, type=Path)
    p.add_argument("--out-dir",   required=True, type=Path)
    p.add_argument("--max-new-tokens", type=int, default=20)
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model from {args.model_dir} ...")
    import torch
    from qwen_asr import Qwen3ASRModel

    wrapper = Qwen3ASRModel.from_pretrained(
        args.model_dir, dtype="float32", device_map="cpu",
    )
    processor = wrapper.processor
    model = wrapper.model  # Qwen3ASRForConditionalGeneration
    model.eval()

    thinker = model.thinker
    audio_tower = thinker.audio_tower
    text_model  = thinker.model
    print(f"  audio_tower: {type(audio_tower).__name__}")
    print(f"  text_model:  {type(text_model).__name__}")

    # ----- Audio preprocessing -----
    print(f"Loading audio: {args.audio}")
    audio = load_audio_16k_mono(args.audio)
    print(f"  samples: {len(audio)}  ({len(audio)/16000:.2f} s)")

    # Use the feature_extractor (WhisperFeatureExtractor) directly — the
    # processor's __call__ requires text input as well.
    feat = processor.feature_extractor(audio, sampling_rate=16000,
                                       return_tensors="pt", padding=True,
                                       truncation=False)
    print("  feature_extractor outputs:")
    for k, v in feat.items():
        if hasattr(v, "shape"):
            print(f"    {k}: {tuple(v.shape)} {v.dtype}")
    mel = feat["input_features"]
    feat_mask = feat.get("attention_mask", None)
    print(f"  mel: {tuple(mel.shape)} {mel.dtype}")
    np.save(args.out_dir / "mel_input.npy", mel.detach().cpu().float().numpy())
    if feat_mask is not None:
        np.save(args.out_dir / "feat_mask.npy", feat_mask.detach().cpu().numpy())

    # ----- Hook the audio tower -----
    captures: dict[str, np.ndarray] = {}

    def cap(name):
        def hook(_mod, _inp, out):
            t = out[0] if isinstance(out, tuple) else out
            captures[name] = t.detach().cpu().float().numpy()
        return hook

    handles = []
    handles.append(audio_tower.conv2d1.register_forward_hook(cap("conv1_out")))
    handles.append(audio_tower.conv2d2.register_forward_hook(cap("conv2_out")))
    handles.append(audio_tower.conv2d3.register_forward_hook(cap("conv3_out")))
    handles.append(audio_tower.conv_out.register_forward_hook(cap("conv_out")))
    handles.append(audio_tower.layers[0].register_forward_hook(cap("enc_blk00_out")))
    handles.append(audio_tower.layers[-1].register_forward_hook(cap("enc_blk17_out")))
    handles.append(audio_tower.ln_post.register_forward_hook(cap("ln_post_out")))
    handles.append(audio_tower.proj1.register_forward_hook(cap("proj1_out")))
    handles.append(audio_tower.proj2.register_forward_hook(cap("proj2_out")))

    # ----- Run encoder -----
    # Encoder needs feature_lens = number of valid mel frames per batch element.
    # Our mel is (1, 128, T) so feature_lens = [T].
    # Encoder's forward uses `input_features.T.split(...)` which only works
    # if the input is 2D (128, T), not (B, 128, T). Squeeze the batch dim.
    mel_2d = mel.squeeze(0)  # (128, T)
    feature_lens = torch.tensor([mel_2d.shape[-1]], dtype=torch.long)
    print(f"  feature_lens: {feature_lens.tolist()}  mel_2d: {tuple(mel_2d.shape)}")
    print("Running audio encoder ...")
    with torch.no_grad():
        enc_out = audio_tower(mel_2d, feature_lens=feature_lens)
    final = enc_out.last_hidden_state if hasattr(enc_out, "last_hidden_state") else enc_out
    print(f"  encoder final output: {tuple(final.shape)} {final.dtype}")
    np.save(args.out_dir / "encoder_final.npy", final.detach().cpu().float().numpy())

    for h in handles:
        h.remove()

    print(f"  captured: {sorted(captures.keys())}")
    for name, arr in captures.items():
        path = args.out_dir / f"{name}.npy"
        np.save(path, arr)
        print(f"    {name}: {arr.shape}  {arr.dtype}  → {path.name}")

    # ----- End-to-end via wrapper.transcribe -----
    print("Running full ASR transcription via wrapper ...")
    try:
        result = wrapper.transcribe(audio=str(args.audio))
        if isinstance(result, list):
            result = result[0]
        text = getattr(result, "text", str(result))
        lang = getattr(result, "language", "?")
        print(f"  language: {lang}")
        print(f"  text: {text}")
        (args.out_dir / "generated_text.txt").write_text(f"{lang}\n{text}\n")
    except Exception as e:
        print(f"  transcribe failed (non-fatal): {e}")

    print(f"\nDone. Reference dump in {args.out_dir}")


if __name__ == "__main__":
    main()
