#!/usr/bin/env python3
"""Pack CSM-1B manual-reference .npy dumps into a single GGUF archive.

The CSM reference (`csm_reference_manual.py`) writes per-stage activations as
loose .npy files (it predates the unified `tools/dump_reference.py` plug-in
API and runs a manual safetensors forward because HF transformers' dynamo
path is broken for CSM). This script collects those .npy files into one GGUF
tensor archive that the C++ diff harness (`crispasr-diff csm`) loads via
`crispasr_diff::Ref`.

No torch required — pure numpy + gguf.

Stage names are kept verbatim from the dump dir so the C++ branch can
`ref.compare("backbone_layer0_output", ...)` etc. Leading unit axes are
squeezed: a (1, T, D) activation becomes (T, D) so the harness treats the
last dim (D) as the cosine row. Integer code tensors (all_codes,
text_tokens) are stored as F32 — every value is a small exact integer.

Usage:
    python tools/pack_csm_ref_gguf.py \
        --dump-dir /Volumes/backups/ai/crispasr/ref-dumps/csm \
        --output   /Volumes/backups/ai/crispasr/ref-dumps/csm-ref.gguf
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _squeeze_leading_units(a: np.ndarray) -> np.ndarray:
    # Drop leading size-1 axes (e.g. batch): (1, T, D) -> (T, D). Keep at
    # least 1 dim. GGUF tensors are max 4D; these dumps are <= 3D after this.
    while a.ndim > 1 and a.shape[0] == 1:
        a = a[0]
    return np.ascontiguousarray(a)


def main() -> None:
    ap = argparse.ArgumentParser(description="Pack CSM reference .npy dumps into a GGUF archive")
    ap.add_argument("--dump-dir", type=Path, required=True, help="Directory of .npy reference dumps")
    ap.add_argument("--output", type=Path, required=True, help="Output GGUF archive path")
    ap.add_argument("--text", default="Hello, how are you?", help="Synthesis text (stored as metadata)")
    args = ap.parse_args()

    try:
        import gguf
    except ImportError as e:
        raise SystemExit("pip install gguf") from e

    npy_files = sorted(args.dump_dir.glob("*.npy"))
    if not npy_files:
        raise SystemExit(f"no .npy files in {args.dump_dir}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    w = gguf.GGUFWriter(str(args.output), arch="crispasr.reference")
    w.add_description("CSM-1B manual reference activation dump (packed from .npy)")
    w.add_string("crispasr.ref.backend", "csm")
    w.add_string("crispasr.ref.text", args.text)

    packed = []
    for f in npy_files:
        name = f.stem
        a = np.load(f, allow_pickle=False)
        a = _squeeze_leading_units(a)
        if a.dtype != np.float32 and a.dtype != np.int32:
            a = a.astype(np.float32)
        a = np.ascontiguousarray(a)
        w.add_tensor(name, a)
        packed.append((name, tuple(a.shape), str(a.dtype)))

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()

    print(f"Wrote {args.output}  ({args.output.stat().st_size/1024:.1f} KiB)")
    for name, shape, dtype in packed:
        print(f"  {name:34s} {str(shape):18s} {dtype}")


if __name__ == "__main__":
    main()
