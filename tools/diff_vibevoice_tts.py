#!/usr/bin/env python3
"""Stage-by-stage diff between python ref and C++ vibevoice TTS dumps.

Reads paired float32/int32 .bin files from the two dump dirs, prints a
cosine + RMS table. The first row with cos < ~0.999 is the first divergent
stage — that is where the bug lives.

Workflow:

  # 1. Python reference (manual fallback path is used for these stages)
  python tools/vibevoice_tts_ref_realtime.py \\
      --text "Hello, how are you today?" \\
      --output-dir /tmp/vv_rt_ref

  # 2. C++ run with the SAME init noise (and dump dir)
  VIBEVOICE_TTS_DUMP=/tmp/vv_cpp_dump \\
  VIBEVOICE_TTS_NOISE=/tmp/vv_rt_ref/noise.bin \\
      build/bin/crispasr --tts "Hello, how are you today?" \\
          -m vibevoice-realtime-0.5b-tts-f16.gguf \\
          --voice vibevoice-voice-emma.gguf \\
          --tts-output /tmp/cpp.wav -ng

  # 3. Diff
  python tools/diff_vibevoice_tts.py /tmp/vv_rt_ref /tmp/vv_cpp_dump

For base-model voice conditioning, use --voice-only with the output of
vibevoice_tts_ref_voice_clone.py and matching native tts_voice_* dumps. This
requires effective-input PCM plus every voice stage and gates magnitude error
as well as cosine; use identical weights/PCM to isolate runtime arithmetic.

NOTE: the python reference's manual-fallback path does not update the TTS-LM
KV cache per frame, so frames > 0 from `all_latents.bin` are NOT a valid
reference. Only frame-0 stages (text, prefill, frame-0 prediction, frame-0
latent, frame-0 connector output) are trustworthy.
"""

import argparse
import sys
from pathlib import Path

import numpy as np


# (python_ref_name, cpp_dump_name, dtype, label)
VOICE_STAGES = [
    ("tts_voice_original_normalized_pcm", "tts_voice_encoder_input", "f32", "upstream vs native preprocessing PCM"),
    ("tts_voice_encoder_input", "tts_voice_encoder_input", "f32", "voice effective encoder PCM"),
    ("tts_voice_acoustic_mean", "tts_voice_acoustic_mean", "f32", "voice acoustic encoder mean"),
    ("tts_voice_acoustic_scaled", "tts_voice_acoustic_scaled", "f32", "voice acoustic scaled"),
    ("tts_voice_acoustic_connected", "tts_voice_acoustic_connected", "f32", "voice acoustic connector"),
    ("tts_voice_embeds", "tts_voice_embeds", "f32", "voice conditioning prompt"),
]

STAGES = [
    ("text_token_ids",       "tts_token_ids",              "i32", "tokenizer ids"),
    ("base_lm_hidden",       "tts_base_lm_hidden_voice",   "f32", "base LM hidden (voice mode)"),
    ("base_lm_hidden",       "tts_base_lm_hidden",         "f32", "base LM hidden (no voice)"),
    ("noise_frame0",         "tts_noise_frame0",           "f32", "init noise frame 0"),
    ("tts_lm_hidden_frame0", "tts_prefill_hidden",         "f32", "TTS LM prefill hidden (last token)"),
    ("tts_neg_condition_frame0","tts_neg_condition_frame0","f32", "TTS LM neg condition (frame 0)"),
    ("pred_v_step0_frame0",  "tts_v_cfg_step0",            "f32", "v-prediction step 0 frame 0"),
    ("speech_latent_frame0", "tts_latent_frame0",          "f32", "denoised latent frame 0"),
    ("acoustic_embed_frame0","tts_acoustic_embed_frame0",  "f32", "acoustic connector frame 0"),
    ("all_latents",          None,                         "f32", "all denoised latents (python: ref AR is simplified after frame 0)"),
    ("audio_output",         None,                         "f32", "final audio (python only)"),
]

PERFRAME_STAGES = [
    ("pos_cond",       "TTS LM positive condition"),
    ("neg_cond",       "TTS LM negative condition"),
    ("noise",          "diffusion init noise"),
    ("v_cfg_step0",    "CFG prediction step 0"),
    ("latent",         "denoised latent"),
    ("acoustic_embed", "acoustic connector output"),
    ("eos_logit",      "EOS stop-check logit"),
    ("eos_prob",       "EOS stop-check probability"),
]


def load_bin(path: Path, dtype: str):
    if not path.exists():
        return None
    if dtype == "i32":
        return np.fromfile(path, dtype=np.int32)
    return np.fromfile(path, dtype=np.float32)


def cos_rms(a: np.ndarray, b: np.ndarray):
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    if a.size != b.size or not a.size:
        rms_a = float(np.sqrt(np.mean(a * a))) if a.size else 0.0
        rms_b = float(np.sqrt(np.mean(b * b))) if b.size else 0.0
        return -1.0, rms_a, rms_b, float("inf")
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    cos = float(a @ b / (na * nb + 1e-12))
    rms_a = float(np.sqrt((a * a).mean()))
    rms_b = float(np.sqrt((b * b).mean()))
    rms_diff = float(np.sqrt(((a - b) ** 2).mean()))
    return cos, rms_a, rms_b, rms_diff


def fmt_status(cos):
    if cos >= 0.999:
        return "PASS"
    if cos >= 0.99:
        return "WARN"
    return "FAIL"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ref_dir", help="python reference dump dir")
    ap.add_argument("cpp_dir", help="C++ VIBEVOICE_TTS_DUMP dir")
    ap.add_argument("--threshold", type=float, default=0.999,
                    help="cos threshold below which a stage is flagged (default 0.999)")
    ap.add_argument("--voice-only", action="store_true",
                    help="compare original/effective PCM and four voice stages; all are required")
    ap.add_argument("--relative-rms-max", type=float, default=0.02,
                    help="voice-only RMS error / reference RMS gate (default 0.02)")
    args = ap.parse_args()

    ref = Path(args.ref_dir)
    cpp = Path(args.cpp_dir)

    if not ref.exists():
        print(f"ref dir does not exist: {ref}", file=sys.stderr)
        return 2
    if not cpp.exists():
        print(f"cpp dir does not exist: {cpp}", file=sys.stderr)
        return 2

    print(f"# python ref: {ref}")
    print(f"# C++ dump:   {cpp}")
    print()

    if not args.voice_only and (ref / "perframe_noise_f000.bin").exists():
        first_fail = None
        for stage, label in PERFRAME_STAGES:
            ref_files = sorted(ref.glob(f"perframe_{stage}_f*.bin"))
            cpp_files = sorted(cpp.glob(f"perframe_{stage}_f*.bin"))
            n_frames = min(len(ref_files), len(cpp_files))
            if n_frames == 0:
                print(f"{label}: SKIP ref_frames={len(ref_files)} cpp_frames={len(cpp_files)}")
                if ref_files and first_fail is None:
                    first_fail = label
                continue

            print(f"{label} ({n_frames} paired frames; ref={len(ref_files)} cpp={len(cpp_files)})")
            worst = (1.0, -1, 0.0)
            for i in range(n_frames):
                ref_arr = load_bin(ref / f"perframe_{stage}_f{i:03d}.bin", "f32")
                cpp_arr = load_bin(cpp / f"perframe_{stage}_f{i:03d}.bin", "f32")
                cos, ra, rb, rd = cos_rms(ref_arr, cpp_arr)
                if cos < worst[0]:
                    worst = (cos, i, rd)
                tiny_diff = rd < 1e-6
                if cos < args.threshold and not tiny_diff and first_fail is None:
                    first_fail = f"{label} frame {i}"
                status = "PASS" if tiny_diff else fmt_status(cos)
                print(f"  f{i:03d} cos={cos:8.5f} rms_ref={ra:9.4f} rms_cpp={rb:9.4f} "
                      f"rms_diff={rd:9.4f} {status}")
            print(f"  worst: frame {worst[1]} cos={worst[0]:.5f} rms_diff={worst[2]:.4f}")
            print()

        if first_fail:
            print(f"first divergent stage: {first_fail}")
            return 1
        print("all per-frame stages within threshold")
        return 0

    header = f"{'stage':<42} {'cos':>8} {'rms_ref':>9} {'rms_cpp':>9} {'rms_diff':>9}  status  shape"
    print(header)
    print("-" * len(header))

    first_fail = None
    for ref_name, cpp_name, dtype, label in (VOICE_STAGES if args.voice_only else STAGES):
        ref_path = ref / f"{ref_name}.bin"
        ref_arr = load_bin(ref_path, dtype)
        if ref_arr is None:
            if args.voice_only and first_fail is None:
                first_fail = label + " (reference missing)"
            print(f"{label:<42} {'-':>8} {'-':>9} {'-':>9} {'-':>9}  SKIP   ref missing ({ref_name}.bin)")
            continue
        if cpp_name is None:
            # Python-only — just print summary stats.
            if dtype == "i32":
                print(f"{label:<42} {'(py)':>8} {ref_arr.size:>9}d {'-':>9} {'-':>9}  --     ids head={ref_arr[:8].tolist()}")
            else:
                rms = float(np.sqrt((ref_arr.astype(np.float64) ** 2).mean()))
                print(f"{label:<42} {'(py)':>8} {rms:>9.4f} {'-':>9} {'-':>9}  --     n={ref_arr.size}")
            continue
        cpp_path = cpp / f"{cpp_name}.bin"
        cpp_arr = load_bin(cpp_path, dtype)
        if cpp_arr is None:
            if args.voice_only and first_fail is None:
                first_fail = label + " (native missing)"
            print(f"{label:<42} {'-':>8} {'-':>9} {'-':>9} {'-':>9}  SKIP   cpp missing ({cpp_name}.bin)")
            continue
        if dtype == "i32":
            n = min(ref_arr.size, cpp_arr.size)
            match = (ref_arr[:n] == cpp_arr[:n]).all()
            status = "PASS" if match and ref_arr.size == cpp_arr.size else "FAIL"
            extra = f"ref={ref_arr.size} cpp={cpp_arr.size} head_eq={int(match)}"
            print(f"{label:<42} {'-':>8} {'-':>9} {'-':>9} {'-':>9}  {status:<6} {extra}")
            if status == "FAIL" and first_fail is None:
                first_fail = label
            continue
        cos, ra, rb, rd = cos_rms(ref_arr, cpp_arr)
        status = fmt_status(cos)
        if args.voice_only:
            status = ("PASS" if np.isfinite(cos) and cos >= args.threshold
                      and rd <= args.relative_rms_max * max(ra, 1e-12) else "FAIL")
        if status == "FAIL" and first_fail is None:
            first_fail = label
        if status == "WARN" and first_fail is None and cos < args.threshold:
            first_fail = label
        shape = f"ref={ref_arr.size} cpp={cpp_arr.size}"
        print(f"{label:<42} {cos:>8.5f} {ra:>9.4f} {rb:>9.4f} {rd:>9.4f}  {status:<6} {shape}")

    print()
    if first_fail:
        print(f"first divergent stage: {first_fail}")
        return 1
    print("all stages within threshold")
    return 0


if __name__ == "__main__":
    sys.exit(main())
