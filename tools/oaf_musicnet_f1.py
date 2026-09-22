#!/usr/bin/env python3
"""
Note-level F1 for Onsets & Frames on MusicNet's test split.

Scores any number of arms — the ONNX export under native onnxruntime, and one
or more GGUF quantisations under the ggml runtime — through the SAME decoder
and the SAME `mir_eval.transcription` call, so the rows are comparable to each
other. The ONNX arm exists precisely so the comparison does not depend on
another project's table.

MUSICNET'S LABEL TIMES ARE SAMPLE INDICES AT 44100 Hz, NOT SECONDS. That has
already cost this project's sibling benchmark one wrong table; the conversion
is `start_time / 44100.0` and it is done once, here.

  usage:
    python tools/oaf_musicnet_f1.py \\
      --musicnet /mnt/storage/tuner-bench/datasets/musicnet/musicnet \\
      --arm onnx=/mnt/storage/tuner-bench/onnx/onsets_and_frames.onnx \\
      --arm f32=/path/onsets-and-frames-f32.gguf \\
      --arm q4_0=/path/onsets-and-frames-q4_0.gguf \\
      --dump build/bin/oaf-parity-dump --workdir /mnt/storage/.../f1
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

SR = 16000
HOP = 512
N_FFT = 2048
N_MELS = 229
N_CLASSES = 88
MIN_MIDI = 21
MUSICNET_LABEL_RATE = 44100.0   # sample indices, not seconds

# meaning -> ONNX graph output name (the export's names are shifted by one).
HEAD_ONSET = "onset"
HEAD_FRAME = "velocity"          # yes: the real frame head is called "velocity"

# The three piano-only pieces of the ten, so the solo-piano row can be split
# out the way §35.3 of the flutter_tuner report does.
SOLO_PIANO = {"1759", "2303", "2556"}


def blas_safe_env() -> dict:
    """See tools/oaf_parity.py: MKL's threaded sgemm corrupts core_mel's mel
    projection on a host where it is linked alongside libgomp."""
    env = dict(os.environ)
    env.setdefault("MKL_NUM_THREADS", "1")
    return env


def to_16k_wav(src: Path, dst: Path) -> None:
    import soundfile as sf
    import torch
    import torchaudio

    if dst.exists():
        return
    x, sr = sf.read(str(src), dtype="float32", always_2d=True)
    mono = torch.from_numpy(x.mean(axis=1))
    if sr != SR:
        mono = torchaudio.functional.resample(mono, sr, SR, resampling_method="sinc_interp_hann")
    pcm = np.clip(mono.numpy(), -1.0, 1.0)
    dst.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(dst), (pcm * 32767.0).astype(np.int16), SR, subtype="PCM_16")


def reference_notes(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    intervals, pitches = [], []
    with csv_path.open() as f:
        for row in csv.DictReader(f):
            t0 = float(row["start_time"]) / MUSICNET_LABEL_RATE
            t1 = float(row["end_time"]) / MUSICNET_LABEL_RATE
            if t1 <= t0:
                continue
            intervals.append((t0, t1))
            pitches.append(440.0 * 2.0 ** ((int(row["note"]) - 69) / 12.0))
    return np.array(intervals), np.array(pitches)


def extract_notes(onset: np.ndarray, frame: np.ndarray, onset_th: float, frame_th: float):
    """`modules/decoding.py: extract_notes` — the reference decoder, verbatim
    in behaviour. Kept in Python so the ONNX arm and the ggml arm are scored by
    the same code path; the C++ runtime's own copy is checked against this by
    the ggml arm agreeing with it note for note at f32."""
    n = onset.shape[0]
    intervals, pitches = [], []
    on = onset > onset_th
    fr = frame > frame_th
    for p in range(N_CLASSES):
        for t in range(n):
            if not on[t, p] or (t > 0 and on[t - 1, p]):
                continue
            off = t
            while off < n and (on[off, p] or fr[off, p]):
                off += 1
            if off > t:
                scale = HOP / SR
                intervals.append((t * scale, off * scale))
                pitches.append(440.0 * 2.0 ** ((p + MIN_MIDI - 69) / 12.0))
    order = np.argsort([i[0] for i in intervals]) if intervals else []
    intervals = np.array([intervals[i] for i in order]) if intervals else np.zeros((0, 2))
    pitches = np.array([pitches[i] for i in order]) if len(order) else np.zeros(0)
    return intervals, pitches


def onnx_heads(sess, wav: Path) -> tuple[np.ndarray, np.ndarray]:
    import soundfile as sf
    import torch
    import torchaudio

    x, sr = sf.read(str(wav), dtype="float32", always_2d=True)
    a = torch.from_numpy(x.mean(axis=1))[:-1]
    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=SR, n_fft=N_FFT, win_length=N_FFT, hop_length=HOP,
        f_min=30.0, f_max=8000.0, n_mels=N_MELS, power=1.0,
        norm="slaney", mel_scale="htk", center=True, pad_mode="reflect",
    )(a)
    feat = torch.log(torch.clamp(mel, min=1e-5)).T.numpy()[None, :, :].astype(np.float32)
    o, f = sess.run([HEAD_ONSET, HEAD_FRAME], {"mel": feat})
    sig = lambda z: 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=np.float64).reshape(-1, N_CLASSES)))
    return sig(o), sig(f)


def ggml_heads(dump: str, model: str, wav: Path, work: Path,
               threads: int) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    work.mkdir(parents=True, exist_ok=True)
    prefix = work / "run"
    t0 = time.time()
    r = subprocess.run([dump, model, str(wav), str(prefix), str(threads)],
                       capture_output=True, text=True, env=blas_safe_env())
    if r.returncode != 0:
        sys.stderr.write(r.stdout + r.stderr)
        raise RuntimeError(f"oaf-parity-dump failed ({r.returncode})")
    meta = dict(line.split() for line in (prefix.with_suffix(".meta.txt")).read_text().split("\n") if line)
    on = np.fromfile(f"{prefix}.onset.f32", dtype=np.float32).reshape(-1, N_CLASSES)
    fr = np.fromfile(f"{prefix}.frame.f32", dtype=np.float32).reshape(-1, N_CLASSES)
    # The C++ decoder's own note list. Everything after the logits — this
    # decoder, the MIDI numbering, the seconds-per-frame — is invisible to the
    # activation diff (crispasr-crispembed-dev.md rule 3b), so it is checked
    # separately against the Python decoder below.
    cpp = np.loadtxt(f"{prefix}.notes.tsv", ndmin=2) if Path(f"{prefix}.notes.tsv").stat().st_size else np.zeros((0, 4))
    return (on.astype(np.float64), fr.astype(np.float64),
            (float(meta["realtime_factor"]), float(meta.get("cpu_factor", 0.0))), cpp)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--musicnet", default="/mnt/storage/tuner-bench/datasets/musicnet/musicnet")
    ap.add_argument("--arm", action="append", required=True,
                    help="NAME=PATH; a .onnx path runs under onnxruntime, a .gguf under ggml")
    ap.add_argument("--dump", default="build/bin/oaf-parity-dump")
    ap.add_argument("--workdir", default="/tmp/oaf-f1")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--onset-threshold", type=float, default=0.5)
    ap.add_argument("--frame-threshold", type=float, default=0.5)
    ap.add_argument("--pieces", default=None, help="comma-separated ids (default: all ten)")
    ap.add_argument("--out", default=None, help="write the per-piece table as JSON")
    args = ap.parse_args()

    import mir_eval

    root = Path(args.musicnet)
    work = Path(args.workdir)
    work.mkdir(parents=True, exist_ok=True)

    ids = sorted(p.stem for p in (root / "test_data").glob("*.wav"))
    if args.pieces:
        want = set(args.pieces.split(","))
        ids = [i for i in ids if i in want]

    wavs = {}
    for pid in ids:
        dst = work / "audio16k" / f"{pid}.wav"
        to_16k_wav(root / "test_data" / f"{pid}.wav", dst)
        wavs[pid] = dst

    results: dict[str, dict] = {}
    for spec in args.arm:
        name, _, path = spec.partition("=")
        is_onnx = path.endswith(".onnx")
        sess = None
        if is_onnx:
            import onnxruntime as ort
            so = ort.SessionOptions()
            so.intra_op_num_threads = args.threads
            sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])

        per_piece = {}
        for pid in ids:
            ref_int, ref_pitch = reference_notes(root / "test_labels" / f"{pid}.csv")
            if is_onnx:
                t0 = time.time()
                on, fr = onnx_heads(sess, wavs[pid])
                import soundfile as sf
                dur = sf.info(str(wavs[pid])).duration
                rtf = ((time.time() - t0) / dur, 0.0)
                cpp_notes = None
            else:
                on, fr, rtf, cpp_notes = ggml_heads(args.dump, path, wavs[pid], work / name / pid, args.threads)
            est_int, est_pitch = extract_notes(on, fr, args.onset_threshold, args.frame_threshold)

            decoder_note = ""
            if cpp_notes is not None:
                # Same heads, two decoders. They must agree note for note; if
                # they do not, the F1 below is measuring this script rather
                # than the runtime the CLI ships.
                py = sorted((round(a, 4), round(b, 4), int(round(69 + 12 * np.log2(p_ / 440.0))))
                            for (a, b), p_ in zip(est_int, est_pitch))
                cx = sorted((round(r[0], 4), round(r[1], 4), int(r[2])) for r in cpp_notes)
                if py != cx:
                    decoder_note = f"  [!! C++ decoder {len(cx)} notes vs python {len(py)}]"

            p, r, f1, _ = mir_eval.transcription.precision_recall_f1_overlap(
                ref_int, ref_pitch, est_int, est_pitch, offset_ratio=None)
            po, ro, fo, _ = mir_eval.transcription.precision_recall_f1_overlap(
                ref_int, ref_pitch, est_int, est_pitch)
            per_piece[pid] = {"precision": p, "recall": r, "f1": f1,
                              "precision_with_offsets": po, "recall_with_offsets": ro,
                              "f1_with_offsets": fo,
                              "n_ref": len(ref_pitch), "n_est": len(est_pitch),
                              "rtf": rtf[0], "cpu_factor": rtf[1]}
            print(f"  [{name}] {pid}: P {p*100:5.1f}  R {r*100:5.1f}  F1 {f1*100:5.1f}  "
                  f"F1+off {fo*100:5.1f}  ({len(est_pitch)} est / {len(ref_pitch)} ref, "
                  f"{rtf[0]:.3f}x RT, {rtf[1]:.3f} cpu-s/s)"
                  f"{decoder_note}", flush=True)

        # Aggregate by POOLING the counts across pieces — one precision and one
        # recall over every note in the corpus — not by averaging per-piece F1.
        # The two differ by about two points here (47.5% vs 49.6% for the ONNX
        # arm), and pooling is what §35.3 of the flutter_tuner report used, so
        # pooling is what makes these rows comparable to its table.
        def agg(keys):
            tp = sum(per_piece[k]["recall"] * per_piece[k]["n_ref"] for k in keys)
            tp_off = sum(per_piece[k]["recall_with_offsets"] * per_piece[k]["n_ref"] for k in keys)
            n_ref = sum(per_piece[k]["n_ref"] for k in keys)
            n_est = sum(per_piece[k]["n_est"] for k in keys)
            if n_ref == 0 or n_est == 0:
                return {}
            def prf(hits):
                pr = hits / n_est
                rc = hits / n_ref
                return pr, rc, (2 * pr * rc / (pr + rc) if pr + rc > 0 else 0.0)
            pr, rc, f1 = prf(tp)
            _, _, f1o = prf(tp_off)
            return {"precision": pr, "recall": rc, "f1": f1, "f1_with_offsets": f1o,
                    "n_ref": n_ref, "n_est": n_est}

        results[name] = {
            "per_piece": per_piece,
            "all": agg(ids),
            "solo_piano": agg([i for i in ids if i in SOLO_PIANO]),
            "rest": agg([i for i in ids if i not in SOLO_PIANO]),
            "mean_rtf": float(np.mean([per_piece[k]["rtf"] for k in ids])),
            "mean_cpu_factor": float(np.mean([per_piece[k]["cpu_factor"] for k in ids])),
        }

    print(f"\n{'arm':10s} {'P':>7s} {'R':>7s} {'F1':>7s} {'F1+off':>7s} {'solo piano':>11s} {'rest':>7s} {'xRT':>7s} {'cpu-s/s':>8s}")
    for name, v in results.items():
        a, sp, rest = v["all"], v["solo_piano"], v["rest"]
        print(f"{name:10s} {a['precision']*100:6.1f}% {a['recall']*100:6.1f}% {a['f1']*100:6.1f}% "
              f"{a['f1_with_offsets']*100:6.1f}% {sp.get('f1', 0)*100:10.1f}% {rest.get('f1', 0)*100:6.1f}% "
              f"{v['mean_rtf']:7.3f} {v['mean_cpu_factor']:8.3f}")

    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2))
        print(f"\n  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
