#!/usr/bin/env python
"""
Compare SpeechT5 reference (npy) vs C++ (f32) dumps per decoder step.
Computes cosine similarity and max absolute error for each stage.

Usage:
    python tools/compare_speecht5_dumps.py \
        --ref /mnt/storage/speecht5/ref_dec \
        --cpp /mnt/storage/speecht5/cpp_dec
"""

import argparse
import sys
from pathlib import Path

import numpy as np


def load_ref(path: Path) -> np.ndarray:
    """Load a .npy reference dump."""
    return np.load(str(path)).flatten().astype(np.float32)


def load_cpp(path: Path) -> np.ndarray:
    """Load a raw f32 C++ dump."""
    return np.fromfile(str(path), dtype=np.float32)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def compare(name: str, ref: np.ndarray, cpp: np.ndarray):
    if ref.shape != cpp.shape:
        print(f"  {name}: SHAPE MISMATCH ref={ref.shape} cpp={cpp.shape}")
        minlen = min(len(ref), len(cpp))
        ref = ref[:minlen]
        cpp = cpp[:minlen]
    cos = cosine_sim(ref, cpp)
    maxabs = float(np.max(np.abs(ref - cpp)))
    meanabs = float(np.mean(np.abs(ref - cpp)))
    status = "OK" if cos > 0.999 else ("WARN" if cos > 0.99 else "FAIL")
    print(f"  {name}: cos={cos:.6f} maxabs={maxabs:.6f} meanabs={meanabs:.6f} [{status}]")
    return cos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, help="Reference dump dir (npy files)")
    ap.add_argument("--cpp", required=True, help="C++ dump dir (f32 files)")
    ap.add_argument("--max-steps", type=int, default=5)
    args = ap.parse_args()

    ref_dir = Path(args.ref)
    cpp_dir = Path(args.cpp)

    # Compare per-step intermediates
    for step in range(args.max_steps):
        print(f"\n=== Step {step} ===")
        for stage in ["prenet", "hidden", "mel"]:
            ref_path = ref_dir / f"step{step}_{stage}.npy"
            cpp_path = cpp_dir / f"step{step}_{stage}.f32"
            if ref_path.exists() and cpp_path.exists():
                ref_data = load_ref(ref_path)
                cpp_data = load_cpp(cpp_path)
                compare(stage, ref_data, cpp_data)
            elif ref_path.exists():
                print(f"  {stage}: C++ dump missing")
            elif cpp_path.exists():
                print(f"  {stage}: ref dump missing")

        # Self-attn K/V layer 0
        for kv in ["self_k_L0", "self_v_L0"]:
            ref_path = ref_dir / f"step{step}_{kv}.npy"
            cpp_path = cpp_dir / f"step{step}_{kv}.f32"
            if ref_path.exists() and cpp_path.exists():
                ref_data = load_ref(ref_path)
                cpp_data = load_cpp(cpp_path)
                # Note: ref KV shape is (1, n_heads, T_kv, head_dim)
                # C++ KV shape for cur_k is just (hidden_size,) for current step only
                # Need to extract just the current step's K from the ref
                # Ref at step N has shape (1, 12, N+1, 64)
                # We want the last time step: ref[:, :, -1, :].flatten()
                ref_full = np.load(str(ref_path))  # (1, 12, N+1, 64)
                ref_cur = ref_full[:, :, -1, :].flatten()
                compare(kv, ref_cur, cpp_data)

    # Compare final mel
    for tag in ["mel_pre_postnet", "mel_post_postnet"]:
        ref_path = ref_dir / f"{tag}.npy"
        cpp_path = cpp_dir / f"{tag}.f32"
        if ref_path.exists() and cpp_path.exists():
            print(f"\n=== {tag} ===")
            compare(tag, load_ref(ref_path), load_cpp(cpp_path))


if __name__ == "__main__":
    main()
