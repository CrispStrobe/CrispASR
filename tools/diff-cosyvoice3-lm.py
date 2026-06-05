#!/usr/bin/env python3
"""Diff CosyVoice3 LLM C++ runtime vs PyTorch reference.

Workflow:
  1. Load the PyTorch reference .npz (from tools/reference_backends/cosyvoice3_tts.py).
  2. Write `input_embeds` to a raw float32 .bin file.
  3. Invoke the C++ smoke-test binary in diff mode (--embeds-bin/--logits-bin).
  4. Read the C++ step0 logits back.
  5. Compute cosine + max-abs-diff vs the reference step0_logits.

Exit code: 0 if cos >= 0.99, 1 otherwise.
"""

from __future__ import annotations

import argparse
import os
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    af = a.astype(np.float64).ravel()
    bf = b.astype(np.float64).ravel()
    na = float(np.linalg.norm(af))
    nb = float(np.linalg.norm(bf))
    if na == 0 or nb == 0:
        return float("nan")
    return float(np.dot(af, bf) / (na * nb))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ref", required=True, help="PyTorch reference .npz (from cosyvoice3_tts.py)")
    ap.add_argument("--gguf", required=True, help="cosyvoice3-llm-f16.gguf")
    ap.add_argument("--cpp-bin", required=True, help="Path to test-cv3-lm-smoke binary")
    ap.add_argument("--tmp-dir", default="/tmp/cv3-diff",
                    help="Scratch dir (used for embeds.bin / logits.bin)")
    ap.add_argument("--threshold", type=float, default=0.99,
                    help="Minimum cosine similarity required to pass")
    args = ap.parse_args()

    ref = np.load(args.ref)
    embeds = ref["input_embeds"].astype(np.float32)
    ref_logits = ref["step0_logits"].astype(np.float32)
    T, d = embeds.shape
    print(f"ref: input_embeds={embeds.shape}, step0_logits={ref_logits.shape}")

    tmp = Path(args.tmp_dir)
    tmp.mkdir(parents=True, exist_ok=True)
    embeds_bin = tmp / "input_embeds.bin"
    logits_bin = tmp / "step0_logits.bin"
    # Row-major flat: T * d float32. Our C++ side reads it as [d_model, T]
    # column-major (ggml's layout), so we need to transpose before writing.
    # ggml stores tensor ne=[d, T] with rows along dim 0 (d), and tensor_set
    # writes byte-flat so the first d floats are column 0 (first token's d
    # values), etc. Python ref row-major has dim 0=T, dim 1=d → element
    # [t][k] at offset t*d + k. To make that look like ggml col-major [d, T],
    # we need element [d_idx, t_idx] at offset t_idx*d + d_idx — which IS
    # the row-major order of the python array. So embeds.tobytes() is correct.
    embeds_bin.write_bytes(embeds.tobytes())
    print(f"wrote {embeds.size * 4} bytes to {embeds_bin}")

    env = os.environ.copy()
    env["DYLD_LIBRARY_PATH"] = ":".join([
        str(Path(args.cpp_bin).parent / "ggml" / "src"),
        str(Path(args.cpp_bin).parent / "ggml" / "src" / "ggml-blas"),
        str(Path(args.cpp_bin).parent / "ggml" / "src" / "ggml-metal"),
        env.get("DYLD_LIBRARY_PATH", ""),
    ])
    cmd = [
        args.cpp_bin, args.gguf,
        "--embeds-bin", str(embeds_bin),
        "--n-tokens", str(T),
        "--logits-bin", str(logits_bin),
    ]
    print(f"running: {' '.join(cmd)}")
    r = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if r.returncode != 0:
        print("FAIL: C++ runner returned non-zero")
        print(r.stdout)
        print(r.stderr)
        return r.returncode
    print(r.stderr)

    cpp_logits_bytes = logits_bin.read_bytes()
    cpp_logits = np.frombuffer(cpp_logits_bytes, dtype=np.float32)
    if cpp_logits.size != ref_logits.size:
        print(f"FAIL: cpp logits size {cpp_logits.size} != ref {ref_logits.size}")
        return 1

    cos = cosine(cpp_logits, ref_logits)
    diff = cpp_logits - ref_logits
    max_abs = float(np.max(np.abs(diff)))
    rel = max_abs / max(1e-9, float(np.max(np.abs(ref_logits))))
    print(f"cosine = {cos:.6f}")
    print(f"max |Δ| = {max_abs:.6f}  (rel to max ref: {rel:.4%})")

    # Top-K agreement.
    k = 5
    ref_top = np.argsort(-ref_logits)[:k]
    cpp_top = np.argsort(-cpp_logits)[:k]
    print(f"ref top-{k}: {ref_top.tolist()}  vals: {ref_logits[ref_top].tolist()}")
    print(f"cpp top-{k}: {cpp_top.tolist()}  vals: {cpp_logits[cpp_top].tolist()}")

    ok = cos >= args.threshold
    print(f"\n{'PASS' if ok else 'FAIL'}: cos={cos:.4f} (threshold {args.threshold})")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
