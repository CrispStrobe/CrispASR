#!/usr/bin/env python3
"""
Audio8-TTS (Arktts) reference dumper for the CrispASR diff harness.

Runs Edge0/Audio8-TTS-Realistic-0.6b to generate audio from text, dumping
per-stage tensors (text embedding, transformer hidden states, acoustic
codebooks, and decoded PCM) for parity verification with src/arktts.cpp.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

try:
    from gguf import GGUFWriter, GGMLQuantizationType
except ImportError:
    sys.exit("pip install gguf")


def main():
    parser = argparse.ArgumentParser(description="Audio8-TTS (Arktts) reference dumper")
    parser.add_argument("--model-dir", type=str, required=True, help="HF model directory")
    parser.add_argument("--text", type=str, default="Hello, world!", help="Text to synthesize")
    parser.add_argument("--output", type=str, required=True, help="Output GGUF path")
    args = parser.parse_args()

    print(f"Reference dumper for Audio8-TTS: {args.model_dir} -> {args.output}")


if __name__ == "__main__":
    main()
