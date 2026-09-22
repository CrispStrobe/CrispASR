#!/usr/bin/env python3
"""
hFT-Transformer under native onnxruntime, timed the way the ggml arm is timed.

The point of this script is that it is a WHOLE PROCESS doing the WHOLE JOB —
front end, every window of inference, and the note decoder — so that
`/usr/bin/time` around it and around `hft-parity-dump` measure the same thing.
§36.4 of the flutter_tuner benchmark records what happens otherwise: the
Onsets & Frames ggml arm's whole process was first compared against ORT's
inference call alone, and CPU-seconds against wall-clock, and the gap was
published as "an order of magnitude" when it was 5.5–7×.

It also prints the same fields `hft-parity-dump` prints, so the two can be
read side by side without a unit conversion in between.

  usage:
    python tools/hft_ort_cost.py --wav input16k.wav [--threads 4]
                                 [--onnx /path/hft_transformer.pruned.onnx]
"""

from __future__ import annotations

import argparse
import math
import resource
import time
from pathlib import Path

import numpy as np

SR = 16000
HOP = 256
N_FFT = 2048
N_MELS = 256
N_CLASSES = 88
N_FRAME = 128
N_MARGIN = 32
MEL_EPS = 1e-8


def cpu_seconds() -> float:
    ru = resource.getrusage(resource.RUSAGE_SELF)
    return ru.ru_utime + ru.ru_stime


def peak_rss_mib() -> float:
    # Linux reports ru_maxrss in kilobytes.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", required=True, help="16 kHz mono WAV")
    ap.add_argument("--onnx", default="/mnt/storage/tuner-bench/onnx/hft_transformer.pruned.onnx")
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()

    import librosa
    import onnxruntime as ort
    import soundfile as sf

    c0 = cpu_seconds()
    t0 = time.time()

    x, sr = sf.read(args.wav, dtype="float32", always_2d=True)
    assert sr == SR, sr
    a = x.mean(axis=1)
    audio_sec = len(a) / SR

    S = np.abs(librosa.stft(a, n_fft=N_FFT, hop_length=HOP, win_length=N_FFT,
                            window="hann", center=True, pad_mode="constant")) ** 2.0
    fb = librosa.filters.mel(sr=SR, n_fft=N_FFT, n_mels=N_MELS, fmin=0.0, fmax=8000.0,
                             htk=True, norm="slaney")
    feat = np.log((fb @ S).T + MEL_EPS)

    so = ort.SessionOptions()
    so.intra_op_num_threads = args.threads
    so.inter_op_num_threads = 1
    sess = ort.InferenceSession(args.onnx, so, providers=["CPUExecutionProvider"])

    frames = feat.shape[0]
    minv = math.log(MEL_EPS)
    total = int(np.ceil(frames / N_FRAME) * N_FRAME)
    padded = np.concatenate([
        np.full((N_MARGIN, N_MELS), minv, np.float32),
        feat.astype(np.float32),
        np.full((total - frames + N_MARGIN, N_MELS), minv, np.float32)])
    heads = {h: np.zeros((total, N_CLASSES), np.float32) for h in ("onset", "offset", "mpe", "velocity")}
    sig = lambda z: 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=np.float32)))
    for i in range(0, total, N_FRAME):
        w = padded[i:i + N_FRAME + 2 * N_MARGIN].T[None, :, :].astype(np.float32)
        on, off, mp, vel = sess.run(["onset_B", "offset_B", "mpe_B", "velocity_B"], {"spec": w})
        heads["onset"][i:i + N_FRAME] = sig(on[0])
        heads["offset"][i:i + N_FRAME] = sig(off[0])
        heads["mpe"][i:i + N_FRAME] = sig(mp[0])
        heads["velocity"][i:i + N_FRAME] = vel[0].argmax(-1)

    # The decoder, so both sides of the comparison do the same job. Imported
    # rather than re-implemented, for the same reason tools/hft_musicnet_f1.py
    # is the only place it lives.
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from hft_musicnet_f1 import hft_notes  # noqa: E402

    notes = hft_notes({k: v.astype(np.float64) for k, v in heads.items()})

    wall = time.time() - t0
    cpu = cpu_seconds() - c0
    print(f"frames={total} notes={len(notes)} {wall * 1000:.1f} ms wall / {cpu * 1000:.1f} ms cpu "
          f"for {audio_sec:.2f} s audio ({wall / audio_sec:.4f} x real time, "
          f"{cpu / audio_sec:.4f} cpu-s per audio-s, {args.threads} threads, "
          f"peak RSS {peak_rss_mib():.0f} MiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
