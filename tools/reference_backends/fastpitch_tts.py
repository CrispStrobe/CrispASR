#!/usr/bin/env python3
"""
Reference backend for NVIDIA FastPitch TTS — dumps intermediate stage outputs
for diff-testing against the C++ fastpitch_tts runtime.

Loads the NeMo FastPitch model from HuggingFace (inOXcrm/German_multispeaker_FastPitch_nemo)
and runs inference, dumping encoder output, durations, pitch, expanded frames,
mel spectrogram, and vocoder stages.

Usage:
    python tools/reference_backends/fastpitch_tts.py \
        --nemo /mnt/storage/fastpitch/German_multispeaker_FastPitch_nemo.nemo \
        --vocoder-nemo /mnt/storage/fastpitch/tts_de_hui_hifigan_ft_fastpitch_multispeaker_5.nemo \
        --text "Hallo Welt, wie geht es Ihnen?" \
        --speaker 0 \
        --output /mnt/storage/fastpitch/ref_dump

Or with HuggingFace auto-download:
    python tools/reference_backends/fastpitch_tts.py \
        --hf-model inOXcrm/German_multispeaker_FastPitch_nemo \
        --hf-vocoder nvidia/tts_de_hui_hifigan_ft_fastpitch_multispeaker_5 \
        --text "Hallo Welt" \
        --output /mnt/storage/fastpitch/ref_dump
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Lazy imports — NeMo is heavy
# ---------------------------------------------------------------------------


def _import_nemo():
    """Import NeMo TTS models. Returns (FastPitchModel, HifiGanModel)."""
    try:
        from nemo.collections.tts.models import FastPitchModel, HifiGanModel
        return FastPitchModel, HifiGanModel
    except ImportError:
        sys.exit(
            "NeMo not found. Install with:\n"
            "  pip install nemo_toolkit[tts]"
        )


def _import_torch():
    try:
        import torch
        return torch
    except ImportError:
        sys.exit("pip install torch")


# ---------------------------------------------------------------------------
# Hook into FastPitch internals
# ---------------------------------------------------------------------------


def register_hooks(model, intermediates: dict):
    """Register forward hooks on FastPitch submodules to capture intermediates."""
    torch = _import_torch()

    def make_hook(name):
        def hook_fn(module, inp, out):
            if isinstance(out, tuple):
                for i, o in enumerate(out):
                    if isinstance(o, torch.Tensor):
                        intermediates[f"{name}.{i}"] = o.detach().cpu().numpy()
            elif isinstance(out, torch.Tensor):
                intermediates[name] = out.detach().cpu().numpy()
        return hook_fn

    fp = model.fastpitch if hasattr(model, "fastpitch") else model

    # Encoder
    if hasattr(fp, "encoder"):
        fp.encoder.register_forward_hook(make_hook("encoder"))
    # Duration predictor
    if hasattr(fp, "duration_predictor"):
        fp.duration_predictor.register_forward_hook(make_hook("duration_predictor"))
    # Pitch predictor
    if hasattr(fp, "pitch_predictor"):
        fp.pitch_predictor.register_forward_hook(make_hook("pitch_predictor"))
    # Pitch embedding
    if hasattr(fp, "pitch_emb"):
        fp.pitch_emb.register_forward_hook(make_hook("pitch_emb"))
    # Decoder
    if hasattr(fp, "decoder"):
        fp.decoder.register_forward_hook(make_hook("decoder"))
    # Output projection
    if hasattr(fp, "proj"):
        fp.proj.register_forward_hook(make_hook("proj"))


def run_fastpitch_inference(model, text: str, speaker_id: int = 0, pace: float = 1.0):
    """Run FastPitch inference and return mel + intermediates."""
    torch = _import_torch()

    intermediates = {}
    register_hooks(model, intermediates)

    # Parse text to tokens
    tokens = model.parse(text)

    # Speaker tensor
    speaker = torch.tensor([speaker_id], dtype=torch.long).to(tokens.device)

    # Generate spectrogram
    with torch.no_grad():
        spect = model.generate_spectrogram(tokens=tokens, speaker=speaker, pace=pace)

    intermediates["tokens"] = tokens.cpu().numpy()
    intermediates["mel_output"] = spect.cpu().numpy()

    return spect, intermediates


def run_vocoder(vocoder_model, mel):
    """Run HiFi-GAN vocoder on mel spectrogram."""
    torch = _import_torch()

    with torch.no_grad():
        audio = vocoder_model.convert_spectrogram_to_audio(spec=mel)

    return audio.cpu().numpy()


# ---------------------------------------------------------------------------
# Dump helpers
# ---------------------------------------------------------------------------


def dump_intermediates(intermediates: dict, output_dir: str):
    """Save all intermediate arrays as .npy files."""
    os.makedirs(output_dir, exist_ok=True)
    for name, arr in intermediates.items():
        safe_name = name.replace(".", "_").replace("/", "_")
        path = os.path.join(output_dir, f"{safe_name}.npy")
        np.save(path, arr)
        print(f"  saved {safe_name}.npy  shape={arr.shape}  dtype={arr.dtype}")


def dump_wav(audio: np.ndarray, path: str, sr: int = 22050):
    """Write PCM float array to WAV."""
    import wave
    import struct

    audio = audio.flatten()
    # Clamp to [-1, 1]
    audio = np.clip(audio, -1.0, 1.0)

    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sr)
        pcm16 = (audio * 32767).astype(np.int16)
        wf.writeframes(pcm16.tobytes())

    print(f"  saved {path}  ({len(audio)} samples @ {sr} Hz, {len(audio)/sr:.2f}s)")


def dump_model_weights(model, output_dir: str, prefix: str = ""):
    """Dump model state_dict as .npy for weight inspection."""
    os.makedirs(output_dir, exist_ok=True)
    sd = model.state_dict()
    for name, param in sd.items():
        arr = param.cpu().numpy()
        safe_name = (prefix + name).replace(".", "_").replace("/", "_")
        path = os.path.join(output_dir, f"{safe_name}.npy")
        np.save(path, arr)
    print(f"  dumped {len(sd)} weight tensors to {output_dir}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="FastPitch TTS reference dumper")
    parser.add_argument("--nemo", help="Path to FastPitch .nemo file")
    parser.add_argument("--vocoder-nemo", help="Path to HiFi-GAN vocoder .nemo file")
    parser.add_argument("--hf-model", default="inOXcrm/German_multispeaker_FastPitch_nemo",
                        help="HuggingFace model name for FastPitch")
    parser.add_argument("--hf-vocoder",
                        default="nvidia/tts_de_hui_hifigan_ft_fastpitch_multispeaker_5",
                        help="HuggingFace model name for HiFi-GAN vocoder")
    parser.add_argument("--text", default="Hallo Welt, wie geht es Ihnen?",
                        help="Text to synthesize")
    parser.add_argument("--speaker", type=int, default=0,
                        help="Speaker ID (0-4 for multi-speaker)")
    parser.add_argument("--pace", type=float, default=1.0,
                        help="Speed factor (>1 = faster)")
    parser.add_argument("--output", required=True,
                        help="Output directory for dumps")
    parser.add_argument("--dump-weights", action="store_true",
                        help="Also dump model weights as .npy")
    parser.add_argument("--sample-rate", type=int, default=22050,
                        help="Output sample rate (22050 for this model)")
    args = parser.parse_args()

    torch = _import_torch()
    FastPitchModel, HifiGanModel = _import_nemo()

    # ── Load FastPitch ──
    if args.nemo:
        print(f"Loading FastPitch from .nemo: {args.nemo}")
        fp_model = FastPitchModel.restore_from(args.nemo)
    else:
        print(f"Loading FastPitch from HuggingFace: {args.hf_model}")
        fp_model = FastPitchModel.from_pretrained(args.hf_model)

    fp_model = fp_model.eval().cpu()

    # ── Load Vocoder ──
    if args.vocoder_nemo:
        print(f"Loading HiFi-GAN from .nemo: {args.vocoder_nemo}")
        voc_model = HifiGanModel.restore_from(args.vocoder_nemo)
    else:
        print(f"Loading HiFi-GAN from HuggingFace: {args.hf_vocoder}")
        voc_model = HifiGanModel.from_pretrained(args.hf_vocoder)

    voc_model = voc_model.eval().cpu()

    # ── Print model info ──
    fp_inner = fp_model.fastpitch if hasattr(fp_model, "fastpitch") else fp_model
    print(f"\nFastPitch architecture:")
    for name, param in fp_inner.named_parameters():
        print(f"  {name:60s}  {list(param.shape)}")

    print(f"\nHiFi-GAN vocoder:")
    for name, param in voc_model.named_parameters():
        print(f"  {name:60s}  {list(param.shape)}")

    # ── Run FastPitch inference ──
    print(f"\nSynthesizing: '{args.text}' (speaker={args.speaker}, pace={args.pace})")
    mel, intermediates = run_fastpitch_inference(
        fp_model, args.text, speaker_id=args.speaker, pace=args.pace
    )

    # ── Run HiFi-GAN vocoder ──
    print("Running HiFi-GAN vocoder...")
    audio = run_vocoder(voc_model, mel)

    intermediates["vocoder_output"] = audio

    # ── Dump everything ──
    print(f"\nDumping intermediates to {args.output}/")
    dump_intermediates(intermediates, args.output)

    wav_path = os.path.join(args.output, "output.wav")
    dump_wav(audio, wav_path, sr=args.sample_rate)

    if args.dump_weights:
        print("\nDumping model weights...")
        dump_model_weights(fp_model, os.path.join(args.output, "weights_fastpitch"), "fp.")
        dump_model_weights(voc_model, os.path.join(args.output, "weights_vocoder"), "voc.")

    print("\nDone.")


if __name__ == "__main__":
    main()
