#!/usr/bin/env python3
"""
Numerical parity for the hFT-Transformer ggml port, against native onnxruntime.

Two comparisons, in this order, because a front-end difference does not raise —
it just scores worse, and §12.1 of the flutter_tuner benchmark is that project's
record of how long that can go unnoticed:

  1. the log-mel the C++ runtime computes vs librosa's, on the same PCM;
  2. every head's post-sigmoid activation, with BOTH runtimes handed the SAME
     mel, so a model difference cannot hide behind a front-end one.

The ONNX side reproduces `infer.py`'s window arithmetic exactly: 32 frames of
log(1e-8) margin padding at each end, the tail padded up to a multiple of 128,
a 192-frame window advancing 128 frames, and the model answering for the middle
128. The velocity head is [1, 128, 88, 128] of logits and is decoded by argmax
over the last axis — that argmax, not a probability, is what the note decoder's
`ignore_zero` gate reads.

USE THE PRUNED GRAPH. The full export keeps all fifteen forward outputs, two of
which nobody decodes (`enc_vector` is 11.5 M floats), and could not be loaded
and run at all on the box this port was measured on.

  usage:
    python tools/hft_parity.py --model build/.../hft-transformer-f32.gguf \\
                               --audio some.wav \\
                               --dump  build/bin/hft-parity-dump \\
                               --onnx  /path/hft_transformer.pruned.onnx
"""

from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

N_MELS = 256
N_CLASSES = 88
SR = 16000
HOP = 256
N_FFT = 2048
N_FRAME = 128
N_MARGIN = 32
MEL_EPS = 1e-8

HEADS = ("onset", "offset", "mpe", "velocity")


def to_16k_mono_pcm16(src: Path, dst: Path) -> None:
    """One canonical 16 kHz 16-bit mono file, so both runtimes see identical PCM."""
    import soundfile as sf
    import torch
    import torchaudio

    x, sr = sf.read(str(src), dtype="float32", always_2d=True)
    mono = torch.from_numpy(x.mean(axis=1))
    if sr != SR:
        mono = torchaudio.functional.resample(mono, sr, SR, resampling_method="sinc_interp_hann")
    pcm = np.clip(mono.numpy(), -1.0, 1.0)
    sf.write(str(dst), (pcm * 32767.0).astype(np.int16), SR, subtype="PCM_16")


def librosa_log_mel(wav: Path) -> np.ndarray:
    """[T, 256]. Three of these settings are non-default somewhere and each is
    individually fatal: a periodic Hann window, the HTK mel scale, and slaney
    filter normalisation. power 2.0, CONSTANT padding, log(mel + 1e-8)."""
    import librosa
    import soundfile as sf

    x, sr = sf.read(str(wav), dtype="float32", always_2d=True)
    assert sr == SR, sr
    a = x.mean(axis=1)
    S = np.abs(librosa.stft(a, n_fft=N_FFT, hop_length=HOP, win_length=N_FFT,
                            window="hann", center=True, pad_mode="constant")) ** 2.0
    fb = librosa.filters.mel(sr=SR, n_fft=N_FFT, n_mels=N_MELS, fmin=0.0, fmax=8000.0,
                             htk=True, norm="slaney")
    return np.log((fb @ S).T + MEL_EPS)


def onnx_heads(sess, mel: np.ndarray) -> dict[str, np.ndarray]:
    """`infer.py`'s stitched windows, post-sigmoid (post-argmax for velocity)."""
    frames = mel.shape[0]
    minv = math.log(MEL_EPS)
    total = int(np.ceil(frames / N_FRAME) * N_FRAME)
    padded = np.concatenate([
        np.full((N_MARGIN, N_MELS), minv, np.float32),
        mel.astype(np.float32),
        np.full((total - frames + N_MARGIN, N_MELS), minv, np.float32)])
    out = {h: np.zeros((total, N_CLASSES), np.float64) for h in HEADS}
    sig = lambda z: 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=np.float64)))
    for i in range(0, total, N_FRAME):
        x = padded[i:i + N_FRAME + 2 * N_MARGIN].T[None, :, :].astype(np.float32)
        on, off, mp, vel = sess.run(["onset_B", "offset_B", "mpe_B", "velocity_B"], {"spec": x})
        out["onset"][i:i + N_FRAME] = sig(on[0])
        out["offset"][i:i + N_FRAME] = sig(off[0])
        out["mpe"][i:i + N_FRAME] = sig(mp[0])
        # [1, 128, 88, 128] logits; the argmax bin is what `ignore_zero` reads.
        out["velocity"][i:i + N_FRAME] = vel[0].argmax(-1).astype(np.float64)
    return out


def blas_safe_env() -> dict:
    """MKL_NUM_THREADS=1, and here is why it is not superstition.

    `core_mel`'s mel projection is a `cblas_sgemm`, and when CMake finds
    Debian's MKL it links `libmkl_intel_thread` into a process that already
    carries libgomp. On this box that combination returns a mel spectrogram
    whose upper bins are multiplied by the THREAD COUNT — exactly ln(4) of
    error at 4 threads. Commit 00822744 prefers OpenBLAS and adds
    tests/test-mel-blas-parity.cpp as a guard; issue #453 covers the hosts that
    fix does not reach. Pinning MKL here removes the variable either way.
    """
    env = dict(os.environ)
    env.setdefault("MKL_NUM_THREADS", "1")
    return env


def report(label: str, a: np.ndarray, b: np.ndarray) -> dict:
    """a = ours, b = reference. Norms are printed alongside every correlation,
    because cosine is scale-blind (crispasr-crispembed-dev.md rule 2b)."""
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    n = min(a.size, b.size)
    a, b = a[:n], b[:n]
    d = a - b
    cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))
    out = {
        "max_abs": float(np.abs(d).max()),
        "mean_abs": float(np.abs(d).mean()),
        "rms": float(np.sqrt((d ** 2).mean())),
        "cos": cos,
        "norm_mine": float(np.linalg.norm(a)),
        "norm_ref": float(np.linalg.norm(b)),
    }
    print(f"  {label:12s} max {out['max_abs']:.3e}  mean {out['mean_abs']:.3e} "
          f" rms {out['rms']:.3e}  cos {cos:.8f}  |mine| {out['norm_mine']:.4g} "
          f" |ref| {out['norm_ref']:.4g}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="GGUF under test")
    ap.add_argument("--audio", required=True)
    ap.add_argument("--dump", default="build/bin/hft-parity-dump")
    ap.add_argument("--onnx", default="/mnt/storage/tuner-bench/onnx/hft_transformer.pruned.onnx")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--seconds", type=float, default=0.0, help="truncate the audio (0 = whole file)")
    ap.add_argument("--workdir", default=None)
    args = ap.parse_args()

    work = Path(args.workdir) if args.workdir else Path(tempfile.mkdtemp(prefix="hft-parity-"))
    work.mkdir(parents=True, exist_ok=True)
    wav16 = work / "input16k.wav"
    to_16k_mono_pcm16(Path(args.audio), wav16)

    if args.seconds > 0:
        import soundfile as sf
        x, sr = sf.read(str(wav16), dtype="int16")
        sf.write(str(wav16), x[: int(args.seconds * sr)], sr, subtype="PCM_16")

    prefix = work / "ggml"
    cmd = [args.dump, args.model, str(wav16), str(prefix), str(args.threads)]
    print("$", " ".join(cmd))
    r = subprocess.run(cmd, capture_output=True, text=True, env=blas_safe_env())
    sys.stdout.write(r.stdout)
    if r.returncode != 0:
        sys.stderr.write(r.stderr)
        return r.returncode

    meta = dict(line.split() for line in (prefix.with_suffix(".meta.txt")).read_text().split("\n") if line)
    T = int(meta["frames"])
    print(f"\n  T = {T} frames, {float(meta['audio_seconds']):.2f} s audio, "
          f"{float(meta['realtime_factor']):.4f}x real time / "
          f"{float(meta['cpu_factor']):.4f} cpu-s per audio-s on {meta['threads']} threads, "
          f"peak RSS {float(meta['peak_rss_mib']):.0f} MiB\n")

    mel_mine = np.fromfile(str(prefix) + ".mel.f32", dtype=np.float32).reshape(-1, N_MELS)

    print("front end (C++ core_mel vs librosa):")
    mel_ref = librosa_log_mel(wav16)
    if mel_ref.shape[0] != mel_mine.shape[0]:
        print(f"  !! frame count differs: ours {mel_mine.shape[0]}, librosa {mel_ref.shape[0]}")
    k = min(mel_ref.shape[0], mel_mine.shape[0])
    mel_stats = report("log-mel", mel_mine[:k], mel_ref[:k])

    print("\nheads (both runtimes on OUR mel, so only the model differs):")
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.intra_op_num_threads = args.threads
    so.inter_op_num_threads = 1
    sess = ort.InferenceSession(args.onnx, so, providers=["CPUExecutionProvider"])
    ref = onnx_heads(sess, mel_mine)

    stats = {"mel": mel_stats}
    for meaning in HEADS:
        mine = np.fromfile(f"{prefix}.{meaning}.f32", dtype=np.float32).reshape(-1, N_CLASSES)
        stats[meaning] = report(meaning, mine[:T], ref[meaning][:T])

    # The number that matters for a transcriber is not the RMS over all
    # 88 x T values (mostly near-zero) but whether the DECISIONS agree. For
    # hFT there are two decisions: the onset threshold, and the velocity gate
    # — and §35.4 measured the gate as the one that actually filters.
    print("\ndecision agreement:")
    for meaning, thr in (("onset", 0.5), ("offset", 0.5), ("mpe", 0.5)):
        mine = np.fromfile(f"{prefix}.{meaning}.f32", dtype=np.float32).reshape(-1, N_CLASSES)[:T]
        r_ = ref[meaning][:T]
        agree = float(((mine > thr) == (r_ > thr)).mean())
        print(f"  {meaning:12s} {agree * 100:.4f}% of {mine.size} cells agree "
              f"({int((mine > thr).sum())} above vs {int((r_ > thr).sum())} in the reference)")
        stats[meaning]["agree"] = agree

    vel_mine = np.fromfile(f"{prefix}.velocity.f32", dtype=np.float32).reshape(-1, N_CLASSES)[:T]
    vel_ref = ref["velocity"][:T]
    gate = float(((vel_mine > 0) == (vel_ref > 0)).mean())
    exact = float((np.rint(vel_mine) == np.rint(vel_ref)).mean())
    print(f"  {'velocity':12s} {gate * 100:.4f}% of cells agree on the ignore_zero GATE, "
          f"{exact * 100:.4f}% on the exact argmax bin")
    stats["velocity"]["agree"] = gate
    stats["velocity"]["argmax_exact"] = exact

    print(f"\n  artefacts in {work}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
