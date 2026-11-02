#!/usr/bin/env python3
"""Per-stage and end-to-end parity of the native FRCRN_SE_16K port.

Runs the native enhancer through libcrispasr's C ABI
(``crispasr_enhance_audio_spec``, the same entry the CLI ``--enhance`` and the
Session ABI use) with ``CRISPASR_FRCRN_SE_DUMP_DIR`` set, then compares every
stage it dumped with a reference ``.npz`` from
``tools/dump_frcrn_se_reference.py`` (same clip, same ``--seconds``/
``--offset``/``--force-segment``).

    python tools/frcrn_se_parity.py --model frcrn-se-16k-f16.gguf \
        --ref ref.npz --lib build/src/libcrispasr.so [--min-cos 0.999]

Exits non-zero when any stage or the end-to-end waveform falls below
``--min-cos``.
"""

import argparse
import ctypes
import os
import sys
import tempfile

import numpy as np


def cosine(a, b):
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    if a.shape != b.shape:
        return float("nan")
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 and nb == 0:
        return 1.0
    return float(a @ b / (na * nb + 1e-300))


def run_native(lib_path, model, audio, threads, force_segment, dump_dir):
    os.environ["CRISPASR_FRCRN_SE_DUMP_DIR"] = dump_dir
    if force_segment:
        os.environ["CRISPASR_FRCRN_SE_FORCE_SEGMENT"] = "1"
    else:
        os.environ.pop("CRISPASR_FRCRN_SE_FORCE_SEGMENT", None)
    lib = ctypes.CDLL(lib_path)
    fn = lib.crispasr_enhance_audio_spec
    fn.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
                   ctypes.POINTER(ctypes.c_float), ctypes.c_int32]
    fn.restype = ctypes.c_int
    x = np.ascontiguousarray(audio, dtype=np.float32)
    y = np.zeros_like(x)
    rc = fn(model.encode(), x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), x.size, 16000, threads,
            y.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), y.size)
    if rc != 0:
        raise SystemExit(f"crispasr_enhance_audio_spec failed rc={rc}")
    return y


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--lib", required=True, help="path to libcrispasr.so")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--force-segment", action="store_true")
    ap.add_argument("--min-cos", type=float, default=0.999)
    ap.add_argument("--dump-dir", default="")
    args = ap.parse_args()

    ref = np.load(args.ref)
    dump = args.dump_dir or tempfile.mkdtemp(prefix="frcrn_se_dump_")
    os.makedirs(dump, exist_ok=True)
    y = run_native(args.lib, args.model, ref["audio"], args.threads, args.force_segment, dump)

    worst = 1.0
    rows = []
    for name in ref.files:
        if name in ("audio", "output", "scalar"):
            continue
        path = os.path.join(dump, name + ".npy")
        if not os.path.exists(path):
            rows.append((name, None, "not dumped"))
            continue
        a = np.load(path)
        b = ref[name]
        if name == "wav":
            n = min(a.size, b.size)
            a, b = a[:n], b[:n]
        c = cosine(a, b)
        err = float(np.max(np.abs(a.astype(np.float64) - b))) if a.shape == b.shape else float("nan")
        rows.append((name, c, f"max|d|={err:.3e} shape={tuple(b.shape)}"))
        worst = min(worst, c) if not np.isnan(c) else -1.0
    out = ref["output"]
    c_e2e = cosine(y, out)
    snr_vs_ref = 10 * np.log10(np.sum(out.astype(np.float64) ** 2) / (np.sum((y - out.astype(np.float64)) ** 2) + 1e-30))
    for name, c, note in rows:
        print(f"{name:16s} {('%.7f' % c) if c is not None else '    -    '}  {note}")
    print(f"{'END-TO-END':16s} {c_e2e:.7f}  agreement SNR {snr_vs_ref:.1f} dB, max|d|={np.max(np.abs(y - out)):.3e}")
    ok = c_e2e >= args.min_cos and worst >= args.min_cos and all(c is not None for _, c, _ in rows)
    print("PASS" if ok else "FAIL", f"(min stage cos {worst:.7f}, threshold {args.min_cos})")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
