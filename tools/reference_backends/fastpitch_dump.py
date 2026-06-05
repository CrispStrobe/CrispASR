#!/usr/bin/env python3
"""
FastPitch TTS reference dumper — generates per-stage intermediate dumps
for diff-testing against the C++ fastpitch_tts runtime.

Loads the model directly from .nemo archives (no NeMo model class needed,
just torch + yaml). Patches NeMo's overrides compatibility issue if NeMo
is available for the model.parse() phoneme tokenizer.

Usage:
    python tools/reference_backends/fastpitch_dump.py \
        --hf-model nvidia/tts_en_fastpitch \
        --hf-vocoder nvidia/tts_hifigan \
        --text "Hello there, how are you doing today?" \
        --output /mnt/storage/fastpitch/ref_dump
"""

from __future__ import annotations

import argparse
import os
import sys
import struct
import wave
from pathlib import Path

import numpy as np
import torch


def patch_overrides():
    """Patch overrides library to work around NeMo compatibility issues."""
    try:
        import overrides.signature
        for fn_name in [
            "ensure_all_kwargs_defined_in_sub",
            "ensure_all_positional_args_defined_in_sub",
            "ensure_signature_is_compatible",
        ]:
            orig = getattr(overrides.signature, fn_name, None)
            if orig is None:
                continue
            def make_patched(o):
                def patched(*a, **kw):
                    try:
                        return o(*a, **kw)
                    except TypeError:
                        pass
                return patched
            setattr(overrides.signature, fn_name, make_patched(orig))
    except ImportError:
        pass


def load_nemo_model(hf_repo: str):
    """Load NeMo model from HF using patched NeMo API."""
    patch_overrides()
    from nemo.collections.tts.models import FastPitchModel
    model = FastPitchModel.from_pretrained(hf_repo)
    return model.eval().cpu()


def load_nemo_vocoder(hf_repo: str):
    """Load NeMo HiFi-GAN vocoder from HF."""
    patch_overrides()
    from nemo.collections.tts.models import HifiGanModel
    model = HifiGanModel.from_pretrained(hf_repo)
    return model.eval().cpu()


def dump_array(arr: np.ndarray, path: str, name: str):
    """Save array as raw F32 binary (for crispasr-diff) and .npy."""
    # Raw F32 for crispasr-diff
    f32_path = os.path.join(path, f"{name}.f32")
    arr_f32 = arr.astype(np.float32).flatten()
    arr_f32.tofile(f32_path)
    # Also npy for easy inspection
    npy_path = os.path.join(path, f"{name}.npy")
    np.save(npy_path, arr)
    print(f"  {name:30s}  shape={list(arr.shape):20s}  "
          f"min={arr.min():.6f}  max={arr.max():.6f}")


def dump_wav(audio: np.ndarray, path: str, sr: int = 22050):
    """Write PCM float array to WAV."""
    audio = audio.flatten().clip(-1.0, 1.0)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        pcm16 = (audio * 32767).astype(np.int16)
        wf.writeframes(pcm16.tobytes())
    print(f"  WAV: {path}  ({len(audio)} samples @ {sr} Hz, {len(audio)/sr:.2f}s)")


def main():
    parser = argparse.ArgumentParser(description="FastPitch reference dumper")
    parser.add_argument("--hf-model", default="nvidia/tts_en_fastpitch")
    parser.add_argument("--hf-vocoder", default="nvidia/tts_hifigan")
    parser.add_argument("--text", default="Hello there, how are you doing today?")
    parser.add_argument("--speaker", type=int, default=0)
    parser.add_argument("--pace", type=float, default=1.0)
    parser.add_argument("--output", required=True, help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # ── Load models ──
    print(f"Loading FastPitch: {args.hf_model}")
    fp_model = load_nemo_model(args.hf_model)

    print(f"Loading HiFi-GAN: {args.hf_vocoder}")
    voc_model = load_nemo_vocoder(args.hf_vocoder)

    # ── Get the FastPitch internal module ──
    fp = fp_model.fastpitch if hasattr(fp_model, "fastpitch") else fp_model

    # ── Tokenize ──
    print(f"\nText: '{args.text}'")
    tokens = fp_model.parse(args.text)  # (1, T)
    print(f"Tokens: {tokens.shape} = {tokens[0].tolist()}")
    dump_array(tokens.cpu().numpy(), args.output, "tokens")

    # ── Hook intermediate outputs ──
    intermediates = {}

    def make_hook(name):
        def hook_fn(module, inp, out):
            if isinstance(out, tuple):
                for i, o in enumerate(out):
                    if isinstance(o, torch.Tensor):
                        intermediates[f"{name}.{i}"] = o.detach().cpu()
            elif isinstance(out, torch.Tensor):
                intermediates[name] = out.detach().cpu()
        return hook_fn

    hooks = []
    hooks.append(fp.encoder.register_forward_hook(make_hook("encoder")))
    hooks.append(fp.duration_predictor.register_forward_hook(make_hook("duration_predictor")))
    hooks.append(fp.pitch_predictor.register_forward_hook(make_hook("pitch_predictor")))
    hooks.append(fp.pitch_emb.register_forward_hook(make_hook("pitch_emb")))
    hooks.append(fp.decoder.register_forward_hook(make_hook("decoder")))
    hooks.append(fp.proj.register_forward_hook(make_hook("proj")))

    # ── Run FastPitch forward ──
    speaker = torch.tensor([args.speaker], dtype=torch.long)
    with torch.no_grad():
        mel = fp_model.generate_spectrogram(
            tokens=tokens, speaker=speaker, pace=args.pace
        )

    # Remove hooks
    for h in hooks:
        h.remove()

    # ── Dump intermediates ──
    print(f"\nDumping intermediates to {args.output}/")
    for name, tensor in sorted(intermediates.items()):
        dump_array(tensor.numpy(), args.output, name)

    dump_array(mel.cpu().numpy(), args.output, "mel_output")

    # ── Run vocoder ──
    print("\nRunning HiFi-GAN vocoder...")
    with torch.no_grad():
        audio = voc_model.convert_spectrogram_to_audio(spec=mel)
    audio_np = audio.cpu().numpy()
    dump_array(audio_np, args.output, "vocoder_output")
    dump_wav(audio_np, os.path.join(args.output, "output.wav"), sr=22050)

    # ── Also dump the generate_spectrogram internals we need ──
    # We need to manually trace to get durations and expanded features
    print("\nRunning manual forward pass for detailed dumps...")
    with torch.no_grad():
        # Encoder
        enc_out = fp.encoder(tokens, conditioning=None)
        if isinstance(enc_out, tuple):
            enc_out = enc_out[0]
        dump_array(enc_out.cpu().numpy(), args.output, "encoder_output")

        # Duration predictor
        dur_pred = fp.duration_predictor(enc_out, conditioning=None)
        if isinstance(dur_pred, tuple):
            dur_pred = dur_pred[0]
        dump_array(dur_pred.cpu().numpy(), args.output, "dur_pred_output")

        # Pitch predictor
        pitch_pred = fp.pitch_predictor(enc_out, conditioning=None)
        if isinstance(pitch_pred, tuple):
            pitch_pred = pitch_pred[0]
        dump_array(pitch_pred.cpu().numpy(), args.output, "pitch_pred_output")

        # Compute durations (same as NeMo's generate_spectrogram)
        log_durs = dur_pred.squeeze(-1)
        durs = torch.clamp(torch.round(torch.exp(log_durs) - 1), min=0)
        durs_int = durs.long()
        dump_array(durs_int.cpu().numpy(), args.output, "durations")

        print(f"\n  Total frames: {durs_int.sum().item()}")
        print(f"  Duration range: [{durs_int.min().item()}, {durs_int.max().item()}]")

    print("\nDone.")


if __name__ == "__main__":
    main()
