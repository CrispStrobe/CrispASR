#!/usr/bin/env python3
"""
Reference backend for Zonos TTS -- runs the upstream Zyphra/Zonos model
and dumps per-layer intermediate activations + output codes/PCM for
diff-testing against the C++ zonos_tts runtime.

Architecture (Zyphra/Zonos-v0.1-transformer):
  - Text encoder: character-level eSpeak phonemizer -> embedding (PLAN #130)
  - Prefix conditioner: concatenated conditioning embeddings
    (espeak, speaker, emotion, fmax, pitch_std, speaking_rate, language_id)
  - AR backbone: 26-layer transformer (d_model=2048, n_heads=16, n_kv=4, GQA)
    generating DAC audio codes (9 codebooks x 1024 entries, ~86 Hz)
  - DAC codec decoder: descript/dac_44khz (9 codebooks, 44.1 kHz output)
  - Speaker conditioning: ResNet293 + ECAPA-TDNN -> 128-d LDA embedding

Usage:
  # Full pipeline with activation dump:
  python tools/reference_backends/zonos_tts_reference.py \\
      --text "Hello world" \\
      --output /mnt/storage/zonos-tts/ref_out.wav \\
      --dump-dir /mnt/storage/zonos-tts/ref_activations/ \\
      --seed 42

  # Just generate audio (no activation dump):
  python tools/reference_backends/zonos_tts_reference.py \\
      --text "Hello world" \\
      --output /mnt/storage/zonos-tts/ref_out.wav

  # With reference speaker audio:
  python tools/reference_backends/zonos_tts_reference.py \\
      --text "Hello world" \\
      --output /mnt/storage/zonos-tts/ref_out.wav \\
      --ref-audio /path/to/speaker.wav

Prerequisites:
  pip install zonos torch torchaudio numpy soundfile
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np


def _hook_factory(dump_dir: str, layer_name: str) -> callable:
    """Create a forward hook that saves output activations as .npy."""
    def hook(module, input, output):
        if isinstance(output, tuple):
            out = output[0]
        else:
            out = output
        if hasattr(out, 'detach'):
            arr = out.detach().cpu().float().numpy()
        else:
            arr = np.array(out)
        path = os.path.join(dump_dir, f"{layer_name}.npy")
        np.save(path, arr)
    return hook


def main():
    ap = argparse.ArgumentParser(description="Zonos TTS reference backend for diff testing")
    ap.add_argument("--text", required=True, help="Text to synthesize")
    ap.add_argument("--output", required=True, help="Output WAV path")
    ap.add_argument("--model", default="Zyphra/Zonos-v0.1-transformer",
                    help="HF model ID or local path")
    ap.add_argument("--ref-audio", default="",
                    help="Reference audio for speaker cloning")
    ap.add_argument("--dump-dir", default="",
                    help="Directory to dump per-layer activations as .npy")
    ap.add_argument("--dump-codes", default="",
                    help="Dump raw DAC codes to text file")
    ap.add_argument("--language", default="en-us",
                    help="Language code (eSpeak)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-tokens", type=int, default=86 * 30,
                    help="Maximum number of AR decode steps")
    ap.add_argument("--cfg-scale", type=float, default=2.0,
                    help="Classifier-free guidance scale")

    # Conditioning parameters
    ap.add_argument("--pitch-std", type=float, default=20.0)
    ap.add_argument("--speaking-rate", type=float, default=15.0)
    ap.add_argument("--fmax", type=float, default=22050.0)
    ap.add_argument("--emotion", type=float, nargs=8,
                    default=[0.3077, 0.0256, 0.0256, 0.0256, 0.0256, 0.0256, 0.2564, 0.3077],
                    help="8-element emotion vector (Happiness, Sadness, Disgust, Fear, "
                         "Surprise, Anger, Other, Neutral)")
    args = ap.parse_args()

    try:
        import torch
        import torchaudio
        import soundfile as sf
    except ImportError:
        sys.exit("pip install torch torchaudio soundfile")

    # Import Zonos
    try:
        from zonos.model import Zonos
        from zonos.conditioning import make_cond_dict
    except ImportError:
        sys.exit("pip install zonos  (or: pip install git+https://github.com/Zyphra/Zonos.git)")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading {args.model} on {device}...", file=sys.stderr)
    model = Zonos.from_pretrained(args.model, device=device)

    # Speaker embedding
    speaker = None
    if args.ref_audio:
        print(f"Extracting speaker embedding from {args.ref_audio}...", file=sys.stderr)
        wav, sr = torchaudio.load(args.ref_audio)
        speaker = model.make_speaker_embedding(wav, sr)
    else:
        print("No ref audio; using random speaker embedding", file=sys.stderr)
        speaker = torch.randn(1, 1, 128, device=device, dtype=torch.bfloat16)

    # Build conditioning
    cond_dict = make_cond_dict(
        text=args.text,
        language=args.language,
        speaker=speaker,
        emotion=args.emotion,
        fmax=args.fmax,
        pitch_std=args.pitch_std,
        speaking_rate=args.speaking_rate,
        device=device,
    )

    # Set up activation hooks if dump_dir is requested
    hooks = []
    if args.dump_dir:
        os.makedirs(args.dump_dir, exist_ok=True)

        # Hook the prefix conditioner output
        hooks.append(model.prefix_conditioner.register_forward_hook(
            _hook_factory(args.dump_dir, "prefix_conditioner_output")))

        # Hook each backbone transformer layer
        for i, layer in enumerate(model.backbone.layers):
            hooks.append(layer.register_forward_hook(
                _hook_factory(args.dump_dir, f"backbone_layer_{i:02d}")))

        # Hook the backbone final norm
        hooks.append(model.backbone.norm_f.register_forward_hook(
            _hook_factory(args.dump_dir, "backbone_norm_f")))

        # Hook individual attention and MLP sub-modules in each layer
        for i, layer in enumerate(model.backbone.layers):
            hooks.append(layer.mixer.register_forward_hook(
                _hook_factory(args.dump_dir, f"backbone_layer_{i:02d}_attn")))
            hooks.append(layer.mlp.register_forward_hook(
                _hook_factory(args.dump_dir, f"backbone_layer_{i:02d}_mlp")))

    # Prepare conditioning
    conditioning = model.prepare_conditioning(cond_dict)

    if args.dump_dir:
        np.save(os.path.join(args.dump_dir, "conditioning_prefix.npy"),
                conditioning.detach().cpu().float().numpy())

    # Generate
    torch.manual_seed(args.seed)
    print(f"Generating (max_tokens={args.max_tokens}, cfg={args.cfg_scale})...",
          file=sys.stderr)
    codes = model.generate(
        conditioning,
        max_new_tokens=args.max_tokens,
        cfg_scale=args.cfg_scale,
        progress_bar=True,
    )

    # Remove hooks
    for h in hooks:
        h.remove()

    if args.dump_dir:
        np.save(os.path.join(args.dump_dir, "output_codes.npy"),
                codes.detach().cpu().numpy())

    # Dump codes as text
    if args.dump_codes:
        codes_np = codes.detach().cpu().numpy()
        with open(args.dump_codes, "w") as f:
            # Shape: (batch, n_codebooks, seq_len)
            for cb in range(codes_np.shape[1]):
                f.write(f"codebook_{cb}: ")
                f.write(" ".join(str(int(c)) for c in codes_np[0, cb]))
                f.write("\n")
        print(f"Wrote codes to {args.dump_codes}", file=sys.stderr)

    # Decode to audio
    print("Decoding codes to audio...", file=sys.stderr)
    wavs = model.autoencoder.decode(codes).cpu()
    wav_np = wavs[0, 0].numpy()
    sr = model.autoencoder.sampling_rate  # 44100

    if args.dump_dir:
        np.save(os.path.join(args.dump_dir, "output_pcm.npy"), wav_np)

    # Save output WAV
    sf.write(args.output, wav_np, sr)
    print(f"Wrote {args.output} ({len(wav_np)/sr:.2f}s, {sr} Hz)", file=sys.stderr)

    # Print summary
    print(f"\nSummary:", file=sys.stderr)
    print(f"  Model: {args.model}", file=sys.stderr)
    print(f"  Text: {args.text!r}", file=sys.stderr)
    print(f"  Language: {args.language}", file=sys.stderr)
    print(f"  Codes shape: {list(codes.shape)}", file=sys.stderr)
    print(f"  Audio: {len(wav_np)} samples @ {sr} Hz = {len(wav_np)/sr:.2f}s",
          file=sys.stderr)
    if args.dump_dir:
        n_files = len([f for f in os.listdir(args.dump_dir) if f.endswith('.npy')])
        print(f"  Activation dumps: {n_files} files in {args.dump_dir}", file=sys.stderr)


if __name__ == "__main__":
    main()
