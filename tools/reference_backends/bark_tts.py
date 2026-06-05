#!/usr/bin/env python3
"""
Reference backend for Suno Bark TTS -- runs the upstream model and dumps
intermediates at each stage for diff-testing against the C++ bark_tts runtime.

3-stage pipeline:
  Stage 1 (semantic):   text -> semantic tokens   (GPT-2, ~100M)
  Stage 2 (coarse):     semantic -> coarse EnCodec tokens (2 codebooks)
  Stage 3 (fine):       coarse -> fine EnCodec tokens     (8 codebooks)
  Decode:               fine tokens -> 24 kHz PCM via EnCodec decoder

Usage:
  # Generate reference audio + dump intermediates:
  python tools/reference_backends/bark_tts.py \
      --text "Hello, this is a test." \
      --output /mnt/storage/bark-tts/ref_out.wav \
      --dump-dir /mnt/storage/bark-tts/ref_dump/

  # With speaker prompt:
  python tools/reference_backends/bark_tts.py \
      --text "Hallo, das ist ein Test." \
      --speaker v2/de_speaker_0 \
      --output /mnt/storage/bark-tts/ref_de.wav

Prerequisites:
  pip install bark torch scipy soundfile numpy
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser(description="Bark TTS reference backend for diff testing")
    ap.add_argument("--text", required=True, help="Text to synthesize")
    ap.add_argument("--output", required=True, help="Output WAV path")
    ap.add_argument("--speaker", default=None,
                    help="Speaker preset (e.g. v2/de_speaker_0) or .npz path")
    ap.add_argument("--dump-dir", default="",
                    help="Directory to dump intermediate tensors for diff testing")
    ap.add_argument("--temperature-semantic", type=float, default=0.7,
                    help="Temperature for semantic stage")
    ap.add_argument("--temperature-coarse", type=float, default=0.7,
                    help="Temperature for coarse stage")
    ap.add_argument("--temperature-fine", type=float, default=0.5,
                    help="Temperature for fine stage")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--use-small", action="store_true",
                    help="Use bark-small models")
    ap.add_argument("--use-kv-caching", action="store_true",
                    help="Enable KV caching for semantic/coarse stages")
    args = ap.parse_args()

    if args.use_small:
        os.environ["SUNO_USE_SMALL_MODELS"] = "True"

    try:
        import torch
        import soundfile as sf
    except ImportError:
        sys.exit("pip install torch soundfile")

    if args.seed:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    # Import bark after setting env vars
    from bark.generation import (
        generate_text_semantic,
        generate_coarse,
        generate_fine,
        codec_decode,
        preload_models,
        SAMPLE_RATE,
    )

    print(f"Preloading models (small={args.use_small})...", file=sys.stderr)
    preload_models(
        text_use_gpu=torch.cuda.is_available(),
        coarse_use_gpu=torch.cuda.is_available(),
        fine_use_gpu=torch.cuda.is_available(),
        codec_use_gpu=torch.cuda.is_available(),
    )

    # Stage 1: text -> semantic tokens
    print(f"Stage 1: generating semantic tokens for: {args.text!r}", file=sys.stderr)
    semantic_tokens = generate_text_semantic(
        args.text,
        history_prompt=args.speaker,
        temp=args.temperature_semantic,
        use_kv_caching=args.use_kv_caching,
    )
    print(f"  -> {len(semantic_tokens)} semantic tokens", file=sys.stderr)

    # Stage 2: semantic -> coarse
    print("Stage 2: generating coarse tokens...", file=sys.stderr)
    coarse_tokens = generate_coarse(
        semantic_tokens,
        history_prompt=args.speaker,
        temp=args.temperature_coarse,
        use_kv_caching=args.use_kv_caching,
    )
    print(f"  -> coarse shape: {coarse_tokens.shape}", file=sys.stderr)

    # Stage 3: coarse -> fine
    print("Stage 3: generating fine tokens...", file=sys.stderr)
    fine_tokens = generate_fine(
        coarse_tokens,
        history_prompt=args.speaker,
        temp=args.temperature_fine,
    )
    print(f"  -> fine shape: {fine_tokens.shape}", file=sys.stderr)

    # Decode: fine tokens -> PCM
    print("Decoding with EnCodec...", file=sys.stderr)
    audio_arr = codec_decode(fine_tokens)
    print(f"  -> {len(audio_arr)} samples ({len(audio_arr)/SAMPLE_RATE:.2f}s "
          f"at {SAMPLE_RATE} Hz)", file=sys.stderr)

    # Save output
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), audio_arr, SAMPLE_RATE)
    print(f"Saved: {out_path}", file=sys.stderr)

    # Dump intermediates
    if args.dump_dir:
        dump = Path(args.dump_dir)
        dump.mkdir(parents=True, exist_ok=True)

        np.save(str(dump / "semantic_tokens.npy"), semantic_tokens)
        np.save(str(dump / "coarse_tokens.npy"), coarse_tokens)
        np.save(str(dump / "fine_tokens.npy"), fine_tokens)
        np.save(str(dump / "audio_pcm.npy"), audio_arr)

        # Also save as text for easy diffing
        with open(dump / "semantic_tokens.txt", "w") as f:
            for t in semantic_tokens:
                f.write(f"{t}\n")
        with open(dump / "coarse_tokens.txt", "w") as f:
            for cb in range(coarse_tokens.shape[0]):
                f.write(f"cb{cb}: " + " ".join(str(x) for x in coarse_tokens[cb]) + "\n")
        with open(dump / "fine_tokens.txt", "w") as f:
            for cb in range(fine_tokens.shape[0]):
                f.write(f"cb{cb}: " + " ".join(str(x) for x in fine_tokens[cb]) + "\n")

        print(f"Dumped intermediates to: {dump}", file=sys.stderr)


if __name__ == "__main__":
    main()
