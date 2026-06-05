#!/usr/bin/env python3
"""
CSM TTS reference backend — loads sesame/csm-1b from HuggingFace,
runs inference, and optionally dumps per-layer intermediates to .npy
files for diff-testing against the C++ runtime.

Usage:
    # Basic synthesis (saves WAV):
    python tools/reference_backends/csm_tts_reference.py \
        --text "Hello, this is a test." \
        --output /mnt/storage/csm-test.wav

    # Dump intermediates for diff testing:
    python tools/reference_backends/csm_tts_reference.py \
        --text "Hello" \
        --dump-dir /mnt/storage/csm-ref-dumps \
        --max-frames 5 \
        --seed 42

    # With speaker conditioning (voice cloning):
    python tools/reference_backends/csm_tts_reference.py \
        --text "Hello, this is a test." \
        --reference-audio /mnt/storage/ref-speaker.wav \
        --reference-text "This is the reference speaker." \
        --output /mnt/storage/csm-cloned.wav

Requirements:
    pip install torch torchaudio transformers huggingface_hub
    # For the original CSM repo generator:
    # pip install torchtune moshi

Notes:
    CSM-1B uses Mimi for audio codec (32 codebooks, 24 kHz).
    The backbone is Llama-3.2-1B and depth decoder is Llama-3.2-100M.
    Text tokenizer is Llama-3.2's BPE tokenizer.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

try:
    import torch
    import torchaudio
except ImportError:
    sys.exit("pip install torch torchaudio")


def load_csm_transformers(device: str = "cpu"):
    """Load CSM-1B using the HuggingFace transformers integration."""
    try:
        from transformers import CsmForConditionalGeneration, AutoTokenizer
        print("Loading CSM-1B via transformers...", file=sys.stderr)
        model = CsmForConditionalGeneration.from_pretrained(
            "sesame/csm-1b",
            cache_dir="/mnt/storage/cache",
            torch_dtype=torch.float32,
        ).to(device)
        tokenizer = AutoTokenizer.from_pretrained(
            "sesame/csm-1b",
            cache_dir="/mnt/storage/cache",
        )
        return model, tokenizer
    except Exception as e:
        print(f"transformers load failed: {e}", file=sys.stderr)
        return None, None


def load_csm_original(device: str = "cpu"):
    """Load CSM-1B using the original SesameAILabs/csm repo."""
    try:
        # The original repo uses torchtune + moshi
        sys.path.insert(0, str(Path(__file__).parent))
        from models import Model, ModelArgs
        from generator import Generator, Segment

        print("Loading CSM-1B via original repo...", file=sys.stderr)
        model = Model.from_pretrained("sesame/csm-1b")
        model = model.to(device=device, dtype=torch.float32)
        model.eval()

        generator = Generator(model, device=device)
        return generator, None
    except Exception as e:
        print(f"original repo load failed: {e}", file=sys.stderr)
        return None, None


def dump_backbone_intermediates(model, tokens, tokens_mask, input_pos, dump_dir: Path):
    """Hook into backbone layers and dump hidden states."""
    dump_dir.mkdir(parents=True, exist_ok=True)
    intermediates = {}

    def make_hook(name):
        def hook(module, input, output):
            if isinstance(output, tuple):
                intermediates[name] = output[0].detach().cpu().numpy()
            else:
                intermediates[name] = output.detach().cpu().numpy()
        return hook

    handles = []
    # Hook backbone layers
    if hasattr(model, 'backbone_model'):
        # transformers format
        for i, layer in enumerate(model.backbone_model.layers):
            h = layer.register_forward_hook(make_hook(f"backbone.layer.{i}"))
            handles.append(h)
    elif hasattr(model, 'backbone'):
        # original format
        for i, layer in enumerate(model.backbone.layers):
            h = layer.register_forward_hook(make_hook(f"backbone.layer.{i}"))
            handles.append(h)

    return handles, intermediates


def run_inference_transformers(model, tokenizer, text: str, args):
    """Run CSM inference using transformers API."""
    print(f"Synthesizing: '{text}'", file=sys.stderr)

    # Prepare inputs
    input_ids = tokenizer(text, return_tensors="pt").input_ids.to(model.device)

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            max_new_tokens=args.max_frames * 32 if args.max_frames else 2048,
            do_sample=True,
            temperature=args.temperature,
            top_k=args.topk,
        )

    # Decode audio
    if hasattr(outputs, 'audio_values'):
        audio = outputs.audio_values.squeeze().cpu().numpy()
    else:
        audio = outputs.squeeze().cpu().numpy()

    return audio


def run_inference_original(generator, text: str, args):
    """Run CSM inference using the original generator."""
    from generator import Segment

    context = []

    # Add reference audio context if provided
    if args.reference_audio and args.reference_text:
        ref_audio, sr = torchaudio.load(args.reference_audio)
        if sr != 24000:
            ref_audio = torchaudio.functional.resample(ref_audio, sr, 24000)
        ref_audio = ref_audio.squeeze()
        context.append(Segment(
            speaker=0,
            text=args.reference_text,
            audio=ref_audio,
        ))

    print(f"Synthesizing: '{text}'", file=sys.stderr)
    with torch.no_grad():
        audio = generator.generate(
            text=text,
            speaker=0,
            context=context,
            max_audio_length_ms=args.max_audio_length_ms,
            temperature=args.temperature,
            topk=args.topk,
        )

    return audio.cpu().numpy()


def main():
    ap = argparse.ArgumentParser(description="CSM TTS reference backend")
    ap.add_argument("--text", required=True, help="Text to synthesize")
    ap.add_argument("--output", default=None, help="Output WAV path")
    ap.add_argument("--dump-dir", default=None,
                    help="Directory for per-layer .npy dumps")
    ap.add_argument("--max-frames", type=int, default=0,
                    help="Max audio frames to generate (0=unlimited)")
    ap.add_argument("--max-audio-length-ms", type=int, default=10000,
                    help="Max audio length in ms")
    ap.add_argument("--temperature", type=float, default=0.9,
                    help="Sampling temperature")
    ap.add_argument("--topk", type=int, default=50,
                    help="Top-k sampling")
    ap.add_argument("--seed", type=int, default=0,
                    help="Random seed (0=nondeterministic)")
    ap.add_argument("--device", default="cpu",
                    help="Device (cpu, cuda, mps)")
    ap.add_argument("--reference-audio", default=None,
                    help="Reference audio WAV for speaker conditioning")
    ap.add_argument("--reference-text", default=None,
                    help="Transcript of reference audio")
    ap.add_argument("--backend", default="auto",
                    choices=["auto", "transformers", "original"],
                    help="Which backend to use")
    args = ap.parse_args()

    if args.seed:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    # Try to load model
    model = None
    tokenizer = None
    generator = None
    backend_used = None

    if args.backend in ("auto", "transformers"):
        model, tokenizer = load_csm_transformers(args.device)
        if model is not None:
            backend_used = "transformers"

    if model is None and args.backend in ("auto", "original"):
        generator, _ = load_csm_original(args.device)
        if generator is not None:
            backend_used = "original"

    if model is None and generator is None:
        sys.exit("Failed to load CSM-1B. Install transformers>=4.52 or "
                 "the original csm repo (torchtune + moshi).")

    print(f"Using backend: {backend_used}", file=sys.stderr)

    # Run inference
    if backend_used == "transformers":
        audio = run_inference_transformers(model, tokenizer, args.text, args)
    else:
        audio = run_inference_original(generator, args.text, args)

    # Save output
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if isinstance(audio, np.ndarray):
            audio_tensor = torch.from_numpy(audio).float()
        else:
            audio_tensor = audio.float()

        if audio_tensor.dim() == 1:
            audio_tensor = audio_tensor.unsqueeze(0)

        torchaudio.save(str(out_path), audio_tensor, 24000)
        print(f"Saved: {out_path} ({audio_tensor.shape[-1] / 24000:.2f}s)", file=sys.stderr)

    # Dump intermediates
    if args.dump_dir:
        dump_dir = Path(args.dump_dir)
        dump_dir.mkdir(parents=True, exist_ok=True)

        if isinstance(audio, np.ndarray):
            np.save(dump_dir / "output_pcm.npy", audio)
        else:
            np.save(dump_dir / "output_pcm.npy", audio.cpu().numpy())
        print(f"Dumped output_pcm.npy to {dump_dir}", file=sys.stderr)

    # Print stats
    if isinstance(audio, np.ndarray):
        n_samples = audio.shape[-1] if audio.ndim > 1 else len(audio)
    else:
        n_samples = audio.shape[-1] if audio.dim() > 1 else len(audio)
    print(f"Generated {n_samples} samples ({n_samples / 24000:.2f}s at 24 kHz)",
          file=sys.stderr)


if __name__ == "__main__":
    main()
