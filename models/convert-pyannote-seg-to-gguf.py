#!/usr/bin/env python3
"""Convert pyannote-segmentation-3.0 ONNX to GGUF.

Source: sherpa-onnx speaker-segmentation-models release
Architecture: SincNet → MaxPool × 3 → LSTM × 4 → Linear × 3 → LogSoftmax
Input: (N, 1, T) raw 16 kHz audio
Output: (N, T', 7) per-frame speaker activity log-probabilities
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path
import numpy as np

try:
    import onnx
except ImportError:
    sys.exit("pip install onnx")
try:
    import gguf
except ImportError:
    sys.exit("pip install gguf")


def convert(onnx_path: Path, out_path: Path) -> None:
    print(f"Loading: {onnx_path}")
    model = onnx.load(str(onnx_path))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = gguf.GGUFWriter(str(out_path), arch="pyannote_seg")

    writer.add_uint32("pyannote_seg.sample_rate", 16000)
    writer.add_uint32("pyannote_seg.n_classes", 7)

    n_written = 0
    for w in model.graph.initializer:
        arr = onnx.numpy_helper.to_array(w).astype(np.float32)
        # Skip scalar constants (shape (), (1,), int64 types)
        if arr.size <= 4 and arr.dtype in (np.int64, np.int32):
            continue
        if arr.size <= 1 and "ortshared" in w.name:
            # Small float constants — keep only meaningful ones
            if arr.dtype == np.float32:
                name = f"pyannote.const.{w.name}"
            else:
                continue
        else:
            name = f"pyannote.{w.name}"
        # Clean up ONNX export names
        name = name.replace("/", ".")
        writer.add_tensor(name, arr)
        n_written += 1
        print(f"  {name:55s} {str(arr.shape):20s}")

    print(f"\n  total: {n_written} tensors")
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"Done: {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    convert(args.input, args.output)
