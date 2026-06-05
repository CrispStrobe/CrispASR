"""Compare C++ vs Python per-layer hidden state dumps for chatterbox-turbo's
GPT-2 backbone at the first AR step (issue #94 follow-up bisect).

Inputs:
    /tmp/cb_gpt2_step_<n_past>_LNN_post_{attn,ffn}.bin   from C++
    /tmp/py_gpt2_step_<n_past>_LNN_post_{attn,ffn}.bin   from Python

Looks for the smallest n_past >= 1 (= the first AR step) where both sides
have dumps. Prints per-layer cosine similarity, RMS, and max |diff|.
The first layer where cos drops noticeably below 0.99 is the divergence
locus.
"""
import glob
import os
import re
import sys
from typing import Optional

import numpy as np


def _load(path: str) -> Optional[np.ndarray]:
    return np.fromfile(path, dtype=np.float32) if os.path.exists(path) else None


def cos(a: np.ndarray, b: np.ndarray) -> float:
    n = min(len(a), len(b))
    a = a[:n]
    b = b[:n]
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    return float(np.dot(a, b) / (na * nb)) if na > 0 and nb > 0 else float("nan")


def rms(a: np.ndarray) -> float:
    return float(np.sqrt(np.mean(a * a))) if a.size else float("nan")


def main():
    # Find shared n_past values present in both C++ and Python dumps.
    cpp_files = glob.glob("/tmp/cb_gpt2_step_*_L00_post_attn.bin")
    py_files = glob.glob("/tmp/py_gpt2_step_*_L00_post_attn.bin")
    rx = re.compile(r"_step_(\d+)_")
    cpp_steps = {int(m.group(1)) for f in cpp_files if (m := rx.search(f))}
    py_steps = {int(m.group(1)) for f in py_files if (m := rx.search(f))}
    shared = sorted(cpp_steps & py_steps)
    if not shared:
        print("no shared n_past between C++ and Python dumps")
        sys.exit(1)

    step = shared[0]
    print(f"step={step}  (first AR step shared between C++ and Python dumps)")
    print(f"{'name':<22s} {'cos':>8s} {'rms_cpp':>10s} {'rms_py':>10s} {'rms_diff':>10s} {'max_abs_diff':>13s}")
    print("-" * 80)

    names = (
        ["inputs_embeds"]
        + [f"L{il:02d}_post_attn" for il in range(48)]
        + [f"L{il:02d}_post_ffn" for il in range(48)]
    )
    for name in names:
        c = _load(f"/tmp/cb_gpt2_step_{step}_{name}.bin")
        p = _load(f"/tmp/py_gpt2_step_{step}_{name}.bin")
        if c is None and p is None:
            continue
        if c is None or p is None:
            print(f"{name:<22s} (skip — missing {'cpp' if c is None else 'py'})")
            continue
        diff = c[: min(len(c), len(p))] - p[: min(len(c), len(p))]
        print(
            f"{name:<22s} {cos(c, p):8.5f} {rms(c):10.5f} {rms(p):10.5f}"
            f" {rms(diff):10.5f} {float(np.max(np.abs(diff))):13.5f}"
        )


if __name__ == "__main__":
    main()
