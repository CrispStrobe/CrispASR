#!/usr/bin/env python3
"""Compare OmniASR C++ dumps against Python reference.

Usage:
  python tools/compare_omniasr_ref.py \
      --ref /tmp/omniasr-ref/ \
      --cpp /tmp/omniasr-cpp/
"""

import argparse
import os
import numpy as np

def load_ref(path):
    """Load .npy reference file."""
    return np.load(path).astype(np.float32)

def load_cpp(path, shape=None):
    """Load .bin C++ dump (raw f32)."""
    data = np.fromfile(path, dtype=np.float32)
    if shape:
        data = data.reshape(shape)
    return data

def compare(name, ref, cpp, max_show=5):
    """Compare two arrays, print stats."""
    if ref.size != cpp.size:
        print(f"  {name}: SIZE MISMATCH ref={ref.shape}({ref.size}) cpp={cpp.shape}({cpp.size})")
        # Try to compare overlap
        n = min(ref.size, cpp.size)
        ref = ref.ravel()[:n]
        cpp = cpp.ravel()[:n]
    else:
        ref = ref.ravel()
        cpp = cpp.ravel()

    diff = np.abs(ref - cpp)
    max_diff = diff.max()
    mean_diff = diff.mean()

    # Cosine similarity
    dot = np.dot(ref, cpp)
    norm_ref = np.linalg.norm(ref)
    norm_cpp = np.linalg.norm(cpp)
    cos_sim = dot / (norm_ref * norm_cpp + 1e-12)

    status = "OK" if cos_sim > 0.99 else "WARN" if cos_sim > 0.9 else "FAIL"
    print(f"  {name:25s} cos={cos_sim:.6f} max_diff={max_diff:.6f} mean_diff={mean_diff:.6f} [{status}]")

    if cos_sim < 0.99:
        print(f"    ref[:5] = {ref[:max_show].tolist()}")
        print(f"    cpp[:5] = {cpp[:max_show].tolist()}")
        # Find worst position
        worst = np.argmax(diff)
        print(f"    worst@{worst}: ref={ref[worst]:.6f} cpp={cpp[worst]:.6f} diff={diff[worst]:.6f}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ref', required=True, help='Python reference dir')
    parser.add_argument('--cpp', required=True, help='C++ dump dir')
    args = parser.parse_args()

    stages = [
        # (name, ref_file, cpp_file, ref_layout_note)
        ('pcm_norm', 'pcm_norm.npy', 'pcm_norm.bin', None),
        ('proj_out', 'proj_out.npy', 'proj_out.bin', 'ref=[T,d] cpp=[d,T]_colmaj'),
        ('pos_conv_out', 'pos_conv_out.npy', 'pos_conv_out.bin', 'ref=[T,d] cpp=[d,T]_colmaj'),
        ('encoder_output', 'encoder_output.npy', 'encoder_output.bin', 'ref=[T,d] cpp=[d,T]_colmaj'),
        ('enc_proj_output', 'enc_proj_output.npy', 'enc_proj_output.bin', 'ref=[T,d] cpp=[d,T]_colmaj'),
    ]

    print(f"Comparing {args.ref} vs {args.cpp}\n")

    for name, ref_file, cpp_file, note in stages:
        ref_path = os.path.join(args.ref, ref_file)
        cpp_path = os.path.join(args.cpp, cpp_file)

        if not os.path.exists(ref_path):
            print(f"  {name:25s} SKIP (no ref)")
            continue
        if not os.path.exists(cpp_path):
            print(f"  {name:25s} SKIP (no cpp)")
            continue

        ref = load_ref(ref_path)
        cpp = load_cpp(cpp_path)

        # Handle layout differences: Python is [T, d] row-major, C++ ggml is [d, T] col-major
        # But ggml col-major [d, T] data[t*d + c] = row-major [T, d] data[t*d + c] — SAME!
        # So for 2D arrays, the flat data should match directly.
        compare(name, ref, cpp)

if __name__ == '__main__':
    main()
