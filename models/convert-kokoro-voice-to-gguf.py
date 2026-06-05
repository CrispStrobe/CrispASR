#!/usr/bin/env python3
"""
Convert a Kokoro voice file (``voices/<name>.pt``) to GGUF.

Each voice is a single F32 tensor of shape ``[max_phon, 1, 256]`` (Kokoro
ships ``max_phon = 510``). The runtime indexes it by phoneme length L:
``ref_s = pack[L-1, 0, :]``. The 256-d vector splits as
``[predictor_style(0:128), decoder_style(128:256)]``.

The HF ``hexgrad/Kokoro-82M`` repo bundles voice files alongside the
model; we treat them as a separate artefact so a single base model can
serve many voices (matches how vibevoice and qwen3-tts handle voice
caches).

Usage:

    python models/convert-kokoro-voice-to-gguf.py \\
        --input  /path/to/voices/af_heart.pt \\
        --output kokoro-voice-af_heart.gguf

    # Or batch-convert a whole voices/ directory:
    python models/convert-kokoro-voice-to-gguf.py \\
        --input-dir /path/to/voices --output-dir voices_gguf/
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

try:
    import torch
except ImportError:
    sys.exit("pip install torch")


def convert_one(in_path: Path, out_path: Path) -> None:
    pack = torch.load(str(in_path), map_location="cpu", weights_only=True)
    if not torch.is_tensor(pack):
        sys.exit(f"{in_path} is not a tensor (got {type(pack).__name__})")
    if pack.ndim != 3 or pack.shape[1] != 1:
        sys.exit(
            f"{in_path}: unexpected voice shape {tuple(pack.shape)}"
            " — expected (N, 1, 256)"
        )
    style_dim = pack.shape[2]
    if style_dim < 256:
        # Some legacy voices are 128-d (predictor only). Allow but warn.
        print(
            f"  NOTE: {in_path.name} has style_dim={style_dim}"
            " (<256 — decoder side will fall back to zeros)",
            file=sys.stderr,
        )

    arr = pack.to(torch.float32).contiguous().numpy()
    name = in_path.stem  # e.g. "af_heart"

    w = GGUFWriter(str(out_path), arch="kokoro-voice", use_temp_file=False)
    w.add_name(f"kokoro-voice-{name}")
    w.add_string("kokoro_voice.name", name)
    w.add_uint32("kokoro_voice.max_phonemes", arr.shape[0])
    w.add_uint32("kokoro_voice.style_dim", arr.shape[2])
    w.add_tensor("voice.pack", arr, raw_dtype=GGMLQuantizationType.F32)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    sz = out_path.stat().st_size / 1024
    print(f"  wrote {out_path}  ({arr.shape}, {sz:.1f} KB)")


def main() -> None:
    ap = argparse.ArgumentParser(description="Convert Kokoro voice .pt → GGUF")
    ap.add_argument("--input", help="Single voice .pt file")
    ap.add_argument("--output", help="Output .gguf path (with --input)")
    ap.add_argument("--input-dir",
                    help="Directory of voice .pt files (batch mode)")
    ap.add_argument("--output-dir",
                    help="Output directory (with --input-dir)")
    args = ap.parse_args()

    if args.input_dir:
        in_dir = Path(args.input_dir)
        out_dir = Path(args.output_dir or in_dir / "_gguf")
        out_dir.mkdir(parents=True, exist_ok=True)
        files = sorted(in_dir.glob("*.pt"))
        if not files:
            sys.exit(f"no .pt files in {in_dir}")
        for f in files:
            convert_one(f, out_dir / f"kokoro-voice-{f.stem}.gguf")
        print(f"\nDone: {len(files)} voices → {out_dir}")
        return

    if not args.input or not args.output:
        sys.exit("provide --input + --output, or --input-dir [+ --output-dir]")
    convert_one(Path(args.input), Path(args.output))


if __name__ == "__main__":
    main()
