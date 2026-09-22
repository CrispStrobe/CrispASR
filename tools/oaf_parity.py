#!/usr/bin/env python3
"""
Numerical parity for the Onsets & Frames ggml port, against native onnxruntime.

Two comparisons, in this order, because a front-end difference does not raise —
it just scores worse, and §12.1 of the flutter_tuner benchmark is that project's
record of how long that can go unnoticed:

  1. the log-mel the C++ runtime computes vs torchaudio's, on the same PCM;
  2. every head's post-sigmoid activation, with BOTH runtimes handed the SAME
     mel, so a model difference cannot hide behind a front-end one.

Head naming follows the MEANINGS, not the ONNX export's output names — the
export was given four `output_names` for a five-output forward and every name
slid down one slot. The map is asserted structurally by
models/convert-onsets-and-frames-to-gguf.py; it is repeated here so this script
stands alone.

  usage:
    python tools/oaf_parity.py --model  build/.../onsets-and-frames-f32.gguf \\
                               --audio  some.wav \\
                               --dump   build/bin/oaf-parity-dump \\
                               --onnx   /path/onsets_and_frames.onnx
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

N_MELS = 229
N_CLASSES = 88
SR = 16000
HOP = 512
N_FFT = 2048

# meaning -> ONNX graph output name (the shift is real; see the converter).
HEADS = {
    "onset": "onset",
    "offset": "offset",
    "activation": "frame",
    "frame": "velocity",
    "velocity": "679",
}


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


def torchaudio_log_mel(wav: Path) -> np.ndarray:
    import soundfile as sf
    import torch
    import torchaudio

    x, sr = sf.read(str(wav), dtype="float32", always_2d=True)
    assert sr == SR, sr
    a = torch.from_numpy(x.mean(axis=1))
    a = a[:-1]  # the reference forward's x[:, :-1]
    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=SR, n_fft=N_FFT, win_length=N_FFT, hop_length=HOP,
        f_min=30.0, f_max=8000.0, n_mels=N_MELS, power=1.0,
        norm="slaney", mel_scale="htk", center=True, pad_mode="reflect",
    )(a)
    return torch.log(torch.clamp(mel, min=1e-5)).T.numpy()  # [T, 229]


def blas_safe_env() -> dict:
    """MKL_NUM_THREADS=1, and here is why it is not superstition.

    `core_mel`'s mel projection is a `cblas_sgemm`, and when CMake finds
    Debian's MKL it links `libmkl_intel_thread` into a process that already
    carries libgomp. On this box that combination returns a mel spectrogram
    whose upper bins are multiplied by the THREAD COUNT — exactly ln(4) of
    error at 4 threads, ln(3) at 3 — while the lower bins are correct. The
    same object file linked against OpenBLAS is exact. It is a host defect,
    not a model one, but it is silent: nothing raises, the spectrum just has
    the wrong shape, and §35.1's warning applies in full.

    Pinning MKL to one thread costs nothing here (the projection is a rounding
    error next to the convolutions) and removes the variable from every number
    this script prints.
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
    ap.add_argument("--dump", default="build/bin/oaf-parity-dump")
    ap.add_argument("--onnx", default="/mnt/storage/tuner-bench/onnx/onsets_and_frames.onnx")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--seconds", type=float, default=0.0, help="truncate the audio (0 = whole file)")
    ap.add_argument("--workdir", default=None)
    args = ap.parse_args()

    work = Path(args.workdir) if args.workdir else Path(tempfile.mkdtemp(prefix="oaf-parity-"))
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
          f"{float(meta['realtime_factor']):.4f}x real time on {meta['threads']} threads\n")

    mel_mine = np.fromfile(str(prefix) + ".mel.f32", dtype=np.float32).reshape(-1, N_MELS)

    print("front end (C++ core_mel vs torchaudio):")
    mel_ref = torchaudio_log_mel(wav16)
    if mel_ref.shape[0] != mel_mine.shape[0]:
        print(f"  !! frame count differs: ours {mel_mine.shape[0]}, torchaudio {mel_ref.shape[0]}")
    k = min(mel_ref.shape[0], mel_mine.shape[0])
    mel_stats = report("log-mel", mel_mine[:k], mel_ref[:k])

    print("\nheads (both runtimes on OUR mel, so only the model differs):")
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.intra_op_num_threads = args.threads
    sess = ort.InferenceSession(args.onnx, so, providers=["CPUExecutionProvider"])
    want = [HEADS[m] for m in ("onset", "offset", "activation", "frame", "velocity")]
    outs = sess.run(want, {"mel": mel_mine[None, :, :].astype(np.float32)})
    ref = {m: 1.0 / (1.0 + np.exp(-np.asarray(o, dtype=np.float64).reshape(-1, N_CLASSES)))
           for m, o in zip(("onset", "offset", "activation", "frame", "velocity"), outs)}

    stats = {"mel": mel_stats}
    for meaning in ("onset", "offset", "activation", "frame", "velocity"):
        mine = np.fromfile(f"{prefix}.{meaning}.f32", dtype=np.float32).reshape(-1, N_CLASSES)
        stats[meaning] = report(meaning, mine[:T], ref[meaning][:T])

    # The number that matters for a transcriber is not the RMS over all
    # 88 x T values (mostly near-zero) but whether the DECISIONS agree.
    print("\ndecision agreement at the shipped thresholds (onset 0.5, frame 0.5):")
    for meaning, thr in (("onset", 0.5), ("frame", 0.5)):
        mine = np.fromfile(f"{prefix}.{meaning}.f32", dtype=np.float32).reshape(-1, N_CLASSES)[:T]
        r_ = ref[meaning][:T]
        agree = float(((mine > thr) == (r_ > thr)).mean())
        n_mine = int((mine > thr).sum())
        n_ref = int((r_ > thr).sum())
        print(f"  {meaning:12s} {agree * 100:.4f}% of {mine.size} cells agree "
              f"({n_mine} above vs {n_ref} in the reference)")
        stats[meaning]["agree"] = agree

    print(f"\n  artefacts in {work}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
