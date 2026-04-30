#!/usr/bin/env python3
"""
Compare per-step qwen3-tts code-predictor dumps produced via
QWEN3_TTS_DUMP_DIR.

Usage:
  python tools/compare_qwen3_tts_cp_dumps.py /tmp/qwen3-main /tmp/qwen3-cpu-f16

Reports:
  - generated_codes length / first mismatch
  - per-step embed/logit max_abs + rms
  - first sampled-id divergence

This is intentionally additive to the existing crispasr-diff flow: it
consumes the extra debug dumps from src/qwen3_tts.cpp without changing
any public baseline format.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable

import numpy as np


def load_f32(path: Path) -> np.ndarray:
    return np.fromfile(path, dtype=np.float32)


def load_i32(path: Path) -> np.ndarray:
    return np.fromfile(path, dtype=np.int32)


def rms(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64), dtype=np.float64)))


def fmt_metric(a: np.ndarray, b: np.ndarray) -> str:
    n = min(a.size, b.size)
    if n == 0:
        return "n=0"
    d = (a[:n].astype(np.float64) - b[:n].astype(np.float64))
    return f"n={n} max_abs={float(np.max(np.abs(d))):.6g} rms={rms(d):.6g}"


def existing_steps(base: Path, frame: int) -> Iterable[int]:
    for step in range(15):
        p = base / f"cp_f{frame:03d}_step{step:02d}_id.bin"
        if p.exists():
            yield step


def compare_generated(a_dir: Path, b_dir: Path) -> None:
    a_path = a_dir / "generated_codes.bin"
    b_path = b_dir / "generated_codes.bin"
    if not a_path.exists() or not b_path.exists():
        print("generated_codes: missing in one or both dirs")
        return

    a = load_i32(a_path)
    b = load_i32(b_path)
    print(f"generated_codes: len_a={a.size} len_b={b.size}")
    n = min(a.size, b.size)
    first = next((i for i in range(n) if a[i] != b[i]), None)
    if first is None:
        if a.size == b.size:
            print("generated_codes: identical")
        else:
            print(f"generated_codes: common prefix identical through {n} ids")
    else:
        print(f"generated_codes: first mismatch at idx={first} a={int(a[first])} b={int(b[first])}")


def compare_frame(a_dir: Path, b_dir: Path, frame: int) -> bool:
    any_step = False
    first_id_div = None
    for step in existing_steps(a_dir, frame):
        any_step = True
        a_id_p = a_dir / f"cp_f{frame:03d}_step{step:02d}_id.bin"
        b_id_p = b_dir / f"cp_f{frame:03d}_step{step:02d}_id.bin"
        a_emb_p = a_dir / f"cp_f{frame:03d}_step{step:02d}_embed.bin"
        b_emb_p = b_dir / f"cp_f{frame:03d}_step{step:02d}_embed.bin"
        a_log_p = a_dir / f"cp_f{frame:03d}_step{step:02d}_logits.bin"
        b_log_p = b_dir / f"cp_f{frame:03d}_step{step:02d}_logits.bin"

        if not (b_id_p.exists() and b_emb_p.exists() and b_log_p.exists()):
            print(f"frame {frame:03d} step {step:02d}: missing in rhs")
            continue

        a_id = load_i32(a_id_p)
        b_id = load_i32(b_id_p)
        a_emb = load_f32(a_emb_p)
        b_emb = load_f32(b_emb_p)
        a_log = load_f32(a_log_p)
        b_log = load_f32(b_log_p)

        id_note = ""
        if a_id.size and b_id.size and a_id[0] != b_id[0] and first_id_div is None:
            first_id_div = (step, int(a_id[0]), int(b_id[0]))
            id_note = f" id_mismatch={int(a_id[0])}!={int(b_id[0])}"

        print(
            f"frame {frame:03d} step {step:02d}: "
            f"embed[{fmt_metric(a_emb, b_emb)}] "
            f"logits[{fmt_metric(a_log, b_log)}]{id_note}"
        )

    if first_id_div is not None:
        step, a_id, b_id = first_id_div
        print(f"frame {frame:03d}: first sampled-id divergence at step {step:02d} ({a_id} vs {b_id})")
    return any_step


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("lhs", type=Path, help="reference dump dir")
    ap.add_argument("rhs", type=Path, help="comparison dump dir")
    ap.add_argument("--max-frames", type=int, default=4, help="how many frames to scan")
    args = ap.parse_args()

    compare_generated(args.lhs, args.rhs)
    print()
    for frame in range(args.max_frames):
        if not compare_frame(args.lhs, args.rhs, frame):
            break
        print()


if __name__ == "__main__":
    main()
