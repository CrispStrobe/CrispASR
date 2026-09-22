#!/usr/bin/env python3
"""
Note-level F1 for hFT-Transformer on MusicNet's test split.

Scores any number of arms — the ONNX export under native onnxruntime, and one
or more GGUF quantisations under the ggml runtime — through the SAME decoder
and the SAME `mir_eval.transcription` call, so the rows are comparable to each
other. The ONNX arm exists precisely so the comparison does not depend on
another project's table: §35.3 of the flutter_tuner benchmark published 52.2%
overall and 70.5% on solo piano, and reproducing that row here is what
calibrates the harness before anything else is claimed.

MUSICNET'S LABEL TIMES ARE SAMPLE INDICES AT 44100 Hz, NOT SECONDS. That has
already cost this project's sibling benchmark one wrong table; the conversion
is `start_time / 44100.0` and it is done once, here.

THE DECODER IS `preprocess/midi.py`, INCLUDING THE VELOCITY GATE.
`mode_velocity='ignore_zero'` drops any note whose velocity head reads zero at
the onset frame, and §35.4 measured that gate as a better precision filter
than the onset threshold — 52.2% with it against 52.1% for the best
thresholded arm without it, while answering more often. `--no-velocity-gate`
reproduces the ungated arm, which is the evidence for that claim.

  usage:
    python tools/hft_musicnet_f1.py \\
      --musicnet /mnt/storage/tuner-bench/datasets/musicnet/musicnet \\
      --arm onnx=/mnt/storage/tuner-bench/onnx/hft_transformer.pruned.onnx \\
      --arm f32=/path/hft-transformer-f32.gguf \\
      --arm q8_0=/path/hft-transformer-q8_0.gguf \\
      --dump build/bin/hft-parity-dump --workdir /mnt/storage/.../f1
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

SR = 16000
HOP = 256
N_FFT = 2048
N_MELS = 256
N_CLASSES = 88
MIN_MIDI = 21
N_FRAME = 128
N_MARGIN = 32
MEL_EPS = 1e-8
HOP_SEC = HOP / SR
MUSICNET_LABEL_RATE = 44100.0   # sample indices, not seconds

# How far the C++ and Python decoders may disagree on a note time before the
# cross-check complains. They run the same algorithm on the same heads; the
# only difference is float32 against float64 in the sub-frame refinement,
# which is worth tens of microseconds.
DECODER_TOL_S = 1e-3

# The three piano-only pieces of the ten, so the solo-piano row can be split
# out the way §35.3 of the flutter_tuner report does.
SOLO_PIANO = {"1759", "2303", "2556"}


def blas_safe_env() -> dict:
    """See tools/hft_parity.py: MKL's threaded sgemm corrupts core_mel's mel
    projection on a host where it is linked alongside libgomp (commit
    00822744, issue #453)."""
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


# --- the model, on either runtime ------------------------------------------

def onnx_heads(sess, wav: Path) -> tuple[dict[str, np.ndarray], float]:
    import librosa
    import soundfile as sf

    x, sr = sf.read(str(wav), dtype="float32", always_2d=True)
    a = x.mean(axis=1)
    S = np.abs(librosa.stft(a, n_fft=N_FFT, hop_length=HOP, win_length=N_FFT,
                            window="hann", center=True, pad_mode="constant")) ** 2.0
    fb = librosa.filters.mel(sr=SR, n_fft=N_FFT, n_mels=N_MELS, fmin=0.0, fmax=8000.0,
                             htk=True, norm="slaney")
    feat = np.log((fb @ S).T + MEL_EPS)

    frames = feat.shape[0]
    minv = math.log(MEL_EPS)
    total = int(np.ceil(frames / N_FRAME) * N_FRAME)
    padded = np.concatenate([
        np.full((N_MARGIN, N_MELS), minv, np.float32),
        feat.astype(np.float32),
        np.full((total - frames + N_MARGIN, N_MELS), minv, np.float32)])
    out = {h: np.zeros((total, N_CLASSES), np.float64) for h in ("onset", "offset", "mpe", "velocity")}
    sig = lambda z: 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=np.float64)))
    t0 = time.time()
    for i in range(0, total, N_FRAME):
        w = padded[i:i + N_FRAME + 2 * N_MARGIN].T[None, :, :].astype(np.float32)
        on, off, mp, vel = sess.run(["onset_B", "offset_B", "mpe_B", "velocity_B"], {"spec": w})
        out["onset"][i:i + N_FRAME] = sig(on[0])
        out["offset"][i:i + N_FRAME] = sig(off[0])
        out["mpe"][i:i + N_FRAME] = sig(mp[0])
        out["velocity"][i:i + N_FRAME] = vel[0].argmax(-1).astype(np.float64)
    return out, time.time() - t0


def ggml_heads(dump: str, model: str, wav: Path, work: Path, threads: int):
    work.mkdir(parents=True, exist_ok=True)
    prefix = work / "run"
    r = subprocess.run([dump, model, str(wav), str(prefix), str(threads)],
                       capture_output=True, text=True, env=blas_safe_env())
    if r.returncode != 0:
        sys.stderr.write(r.stdout + r.stderr)
        raise RuntimeError(f"hft-parity-dump failed ({r.returncode})")
    meta = dict(line.split() for line in (prefix.with_suffix(".meta.txt")).read_text().split("\n") if line)
    heads = {h: np.fromfile(f"{prefix}.{h}.f32", dtype=np.float32).reshape(-1, N_CLASSES).astype(np.float64)
             for h in ("onset", "offset", "mpe", "velocity")}
    cpp = (np.loadtxt(f"{prefix}.notes.tsv", ndmin=2)
           if Path(f"{prefix}.notes.tsv").exists() and Path(f"{prefix}.notes.tsv").stat().st_size
           else np.zeros((0, 4)))
    return heads, meta, cpp


# --- the decoder, ported from `preprocess/midi.py` --------------------------

def detect_event(data: np.ndarray, idx: int, threshold: float):
    """A frame is an event when it is at or above the threshold and is a local
    maximum in the WEAK sense — scanning outward in each direction, the first
    strictly different neighbour is smaller. The time is then refined between
    the neighbours, which is where hFT gets onset resolution finer than its
    16 ms frame."""
    col = data[:, idx]
    n = len(col)
    out = []
    for i in np.flatnonzero(col >= threshold):
        v = col[i]
        left = True
        for ii in range(i - 1, -1, -1):
            if v > col[ii]:
                break
            if v < col[ii]:
                left = False
                break
        if not left:
            continue
        right = True
        for ii in range(i + 1, n):
            if v > col[ii]:
                break
            if v < col[ii]:
                right = False
                break
        if not right:
            continue
        if i == 0 or i == n - 1:
            t = i * HOP_SEC
        else:
            l, r = col[i - 1], col[i + 1]
            if l == r:
                t = i * HOP_SEC
            elif l > r:
                t = i * HOP_SEC - HOP_SEC * 0.5 * (l - r) / (v - r)
            else:
                t = i * HOP_SEC + HOP_SEC * 0.5 * (r - l) / (v - l)
        out.append((int(i), float(t)))
    return out


def process_label(pitch, onsets, offsets, mpe, thred_mpe, velocity):
    """`process_label`, `mode_offset='shorter'`."""
    out = []
    n_mpe = mpe.shape[0]
    for k, (loc_onset, time_onset) in enumerate(onsets):
        if k + 1 < len(onsets):
            loc_next, time_next = onsets[k + 1]
        else:
            loc_next = n_mpe
            time_next = (loc_next - 1) * HOP_SEC
        loc_offset, time_offset, flag_offset = loc_onset + 1, 0.0, False
        for loc, t in offsets:
            if loc_onset < loc:
                loc_offset, time_offset, flag_offset = loc, t, True
                break
        if loc_offset > loc_next:
            loc_offset, time_offset = loc_next, time_next
        loc_mpe, time_mpe, flag_mpe = loc_onset + 1, 0.0, False
        for ii in range(loc_onset + 1, min(loc_next, n_mpe)):
            if mpe[ii, pitch] < thred_mpe:
                loc_mpe, flag_mpe = ii, True
                time_mpe = loc_mpe * HOP_SEC
                break
        if not flag_offset and not flag_mpe:
            offset_value = time_next
        elif flag_offset and not flag_mpe:
            offset_value = time_offset
        elif not flag_offset and flag_mpe:
            offset_value = time_mpe
        else:
            offset_value = time_offset if loc_offset <= loc_mpe else time_mpe
        # `mode_velocity='ignore_zero'`.
        if velocity is not None and velocity[loc_onset, pitch] <= 0:
            continue
        out.append((time_onset, offset_value, float(pitch + MIN_MIDI)))
    return out


def hft_notes(acts, onset=0.5, offset=0.5, mpe=0.5, ignore_zero=True):
    velocity = np.rint(acts["velocity"]).astype(np.int32) if ignore_zero else None
    notes = []
    for pitch in range(N_CLASSES):
        on = detect_event(acts["onset"], pitch, onset)
        if not on:
            continue
        off = detect_event(acts["offset"], pitch, offset)
        for note in process_label(pitch, on, off, acts["mpe"], mpe, velocity):
            # "a re-onset of the same pitch ends the previous note".
            if notes and notes[-1][2] == note[2] and note[0] < notes[-1][1]:
                prev = notes.pop()
                notes.append((prev[0], note[0], prev[2]))
            notes.append(note)
    notes.sort(key=lambda n: n[0])
    return notes


def to_arrays(notes):
    if not notes:
        return np.zeros((0, 2)), np.zeros(0)
    a = np.array([[n[0], n[1]] for n in notes], dtype=float)
    # mir_eval rejects zero-length intervals; a note whose decoded offset
    # lands on its onset is widened by one frame rather than dropped, which
    # is what the reference's MIDI writer does in effect.
    bad = a[:, 1] <= a[:, 0]
    a[bad, 1] = a[bad, 0] + 1e-3
    return a, np.array([440.0 * 2 ** ((n[2] - 69) / 12) for n in notes])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--musicnet", default="/mnt/storage/tuner-bench/datasets/musicnet/musicnet")
    ap.add_argument("--arm", action="append", required=True,
                    help="NAME=PATH; a .onnx path runs under onnxruntime, a .gguf under ggml")
    ap.add_argument("--dump", default="build/bin/hft-parity-dump")
    ap.add_argument("--workdir", default="/tmp/hft-f1")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--onset-threshold", type=float, default=0.5)
    ap.add_argument("--offset-threshold", type=float, default=0.5)
    ap.add_argument("--mpe-threshold", type=float, default=0.5)
    ap.add_argument("--no-velocity-gate", action="store_true",
                    help="turn `mode_velocity='ignore_zero'` off — the evidence for §35.4")
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

    gate = not args.no_velocity_gate
    results: dict[str, dict] = {}
    for spec in args.arm:
        name, _, path = spec.partition("=")
        is_onnx = path.endswith(".onnx")
        sess = None
        if is_onnx:
            import onnxruntime as ort
            so = ort.SessionOptions()
            so.intra_op_num_threads = args.threads
            so.inter_op_num_threads = 1
            sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])

        per_piece = {}
        for pid in ids:
            ref_int, ref_pitch = reference_notes(root / "test_labels" / f"{pid}.csv")
            import soundfile as sf
            dur = sf.info(str(wavs[pid])).duration
            if is_onnx:
                acts, secs = onnx_heads(sess, wavs[pid])
                rtf, cpuf, rss = secs / dur, 0.0, 0.0
                cpp_notes = None
            else:
                acts, meta, cpp_notes = ggml_heads(args.dump, path, wavs[pid], work / name / pid, args.threads)
                rtf = float(meta["realtime_factor"])
                cpuf = float(meta.get("cpu_factor", 0.0))
                rss = float(meta.get("peak_rss_mib", 0.0))

            notes = hft_notes(acts, args.onset_threshold, args.offset_threshold,
                              args.mpe_threshold, ignore_zero=gate)
            est_int, est_pitch = to_arrays(notes)

            decoder_note = ""
            if cpp_notes is not None and gate:
                # Same heads, two decoders. They must agree note for note; if
                # they do not, the F1 below is measuring this script rather
                # than the runtime the CLI ships.
                #
                # Matched with a tolerance, not by exact equality of rounded
                # times. The sub-frame onset refinement is float32 in C++ and
                # float64 here, which moves a time by tens of MICROseconds —
                # and an exact comparison of values rounded to a millisecond
                # then disagrees on every note that happens to sit near a
                # rounding boundary. That produced a "1374 of 1457 shared"
                # warning on a pair of decoders whose worst disagreement was
                # 68 us. The count must match exactly; the times must agree to
                # DECODER_TOL_S, which is two orders of magnitude inside
                # mir_eval's 50 ms onset tolerance.
                cxn = [(float(r[0]), float(r[1]), int(r[2])) for r in cpp_notes]
                pyn = [(float(a), float(b), int(round(69 + 12 * np.log2(p_ / 440.0))))
                       for (a, b), p_ in zip(est_int, est_pitch)]
                worst = 0.0
                by_pitch: dict[int, list] = {}
                for a_, b_, p_ in pyn:
                    by_pitch.setdefault(p_, []).append((a_, b_))
                unmatched = 0
                for a_, b_, p_ in cxn:
                    cands = by_pitch.get(p_)
                    if not cands:
                        unmatched += 1
                        continue
                    j = min(range(len(cands)), key=lambda i: abs(cands[i][0] - a_))
                    worst = max(worst, abs(cands[j][0] - a_), abs(cands[j][1] - b_))
                if len(cxn) != len(pyn) or unmatched or worst > DECODER_TOL_S:
                    decoder_note = (f"  [!! C++ decoder {len(cxn)} notes vs python {len(pyn)}, "
                                    f"{unmatched} unmatched, worst |dt| {worst * 1e3:.3f} ms]")

            p, r, f1, _ = mir_eval.transcription.precision_recall_f1_overlap(
                ref_int, ref_pitch, est_int, est_pitch, offset_ratio=None)
            po, ro, fo, _ = mir_eval.transcription.precision_recall_f1_overlap(
                ref_int, ref_pitch, est_int, est_pitch)
            per_piece[pid] = {"precision": p, "recall": r, "f1": f1,
                              "precision_with_offsets": po, "recall_with_offsets": ro,
                              "f1_with_offsets": fo,
                              "n_ref": len(ref_pitch), "n_est": len(est_pitch),
                              "rtf": rtf, "cpu_factor": cpuf, "peak_rss_mib": rss,
                              "audio_seconds": dur}
            print(f"  [{name}] {pid}: P {p*100:5.1f}  R {r*100:5.1f}  F1 {f1*100:5.1f}  "
                  f"F1+off {fo*100:5.1f}  ({len(est_pitch)} est / {len(ref_pitch)} ref, "
                  f"{rtf:.3f}x RT, {cpuf:.3f} cpu-s/s, {rss:.0f} MiB)"
                  f"{decoder_note}", flush=True)

        # Aggregate by POOLING the counts across pieces — one precision and one
        # recall over every note in the corpus — not by averaging per-piece F1.
        # Pooling is what §35.3 of the flutter_tuner report used, so pooling is
        # what makes these rows comparable to its table.
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

        total_audio = sum(per_piece[k]["audio_seconds"] for k in ids)
        results[name] = {
            "per_piece": per_piece,
            "all": agg(ids),
            "solo_piano": agg([i for i in ids if i in SOLO_PIANO]),
            "rest": agg([i for i in ids if i not in SOLO_PIANO]),
            "velocity_gate": gate,
            # Audio-weighted, not a mean of per-piece ratios: the pieces differ
            # in length by more than 3x and the short ones would otherwise
            # dominate a figure that is meant to answer "what would
            # transcribing a piece cost".
            "rtf": sum(per_piece[k]["rtf"] * per_piece[k]["audio_seconds"] for k in ids) / total_audio,
            "cpu_factor": sum(per_piece[k]["cpu_factor"] * per_piece[k]["audio_seconds"] for k in ids) / total_audio,
            "peak_rss_mib": max(per_piece[k]["peak_rss_mib"] for k in ids),
        }

    print(f"\n{'arm':10s} {'P':>7s} {'R':>7s} {'F1':>7s} {'F1+off':>7s} {'solo piano':>11s} "
          f"{'rest':>7s} {'xRT':>7s} {'cpu-s/s':>8s} {'RSS MiB':>8s}")
    for name, v in results.items():
        a, sp, rest = v["all"], v["solo_piano"], v["rest"]
        print(f"{name:10s} {a['precision']*100:6.1f}% {a['recall']*100:6.1f}% {a['f1']*100:6.1f}% "
              f"{a['f1_with_offsets']*100:6.1f}% {sp.get('f1', 0)*100:10.1f}% {rest.get('f1', 0)*100:6.1f}% "
              f"{v['rtf']:7.3f} {v['cpu_factor']:8.3f} {v['peak_rss_mib']:8.0f}")

    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2))
        print(f"\n  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
