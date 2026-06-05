#!/usr/bin/env python3
"""
Reference backend for OuteTTS — runs the upstream model and dumps
output codes + PCM for diff-testing against the C++ outetts runtime.

Usage:
  # Generate reference audio:
  python tools/reference_backends/outetts_tts.py \
      --text "Hello world" \
      --output /mnt/storage/outetts/ref_out.wav \
      --dump-codes /mnt/storage/outetts/ref_codes.txt

  # Compare codes from C++ vs Python:
  diff <(sort /mnt/storage/outetts/ref_codes.txt) \
       <(sort /mnt/storage/outetts/cpp_codes.txt)

Prerequisites:
  pip install outetts torch
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser(description="OuteTTS reference backend for diff testing")
    ap.add_argument("--text", required=True, help="Text to synthesize")
    ap.add_argument("--output", required=True, help="Output WAV path")
    ap.add_argument("--model", default="OuteAI/OuteTTS-0.3-1B",
                    help="HF model ID")
    ap.add_argument("--dump-codes", default="",
                    help="Dump raw audio codes to file (one per line)")
    ap.add_argument("--temperature", type=float, default=0.4)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    try:
        import outetts
    except ImportError:
        sys.exit("pip install outetts")
    try:
        import torch
        import soundfile as sf
    except ImportError:
        sys.exit("pip install torch soundfile")

    print(f"Loading {args.model}...", file=sys.stderr)
    interface = outetts.InterfaceHF(model_version=args.model)

    # Generate
    print(f"Synthesizing: {args.text!r}", file=sys.stderr)
    gen = interface.generate(
        text=args.text,
        temperature=args.temperature,
        repetition_penalty=1.1,
        max_length=4096,
    )

    if gen is None or gen.audio is None:
        sys.exit("Generation failed — no audio produced")

    # Save WAV
    audio = gen.audio.cpu().numpy()
    if audio.ndim > 1:
        audio = audio.squeeze()
    sr = gen.sr if hasattr(gen, "sr") else 24000
    sf.write(args.output, audio, sr)
    print(f"Saved {args.output} ({len(audio)} samples, {sr} Hz, "
          f"{len(audio)/sr:.2f} s)", file=sys.stderr)

    # Dump codes if requested
    if args.dump_codes and hasattr(gen, "codes"):
        codes = gen.codes
        if hasattr(codes, "cpu"):
            codes = codes.cpu().numpy()
        with open(args.dump_codes, "w") as f:
            for c in codes.flatten():
                f.write(f"{int(c)}\n")
        print(f"Dumped {len(codes.flatten())} codes to {args.dump_codes}",
              file=sys.stderr)


if __name__ == "__main__":
    main()
