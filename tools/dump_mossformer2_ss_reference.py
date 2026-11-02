#!/usr/bin/env python3
"""Dump ORIGINAL ClearerVoice-Studio MossFormer2_SS_16K stages for native diffs.

Runs the unmodified upstream PyTorch model (``models/mossformer2_ss`` of
modelscope/ClearerVoice-Studio, pinned below) on a 16-kHz mono clip and
captures named intermediate stages with forward hooks.  Nothing in the
upstream forward methods is replaced; the hooks only copy tensors out.

Two outputs, either or both:

* ``--npy-dir DIR``  one ``<stage>.npy`` per stage (NumPy row-major);
* ``--gguf FILE``    the same stages as a GGUF tensor archive that
  ``crispasr-diff mossformer2-ss <model.gguf> FILE <audio>`` compares stage by
  stage against the native ggml graph.

Layouts are NumPy row-major with time OUTER and channels INNER, i.e. the
reverse of ggml's ``ne`` order ([T, C] here == ggml ne [C, T]):

  input_audio            [L]
  enc_out                [T, 512]   Encoder Conv1d(k16, s8) + ReLU
  mn_norm                [T, 512]   MaskNet GroupNorm(1, 512)
  mn_in                  [T, 512]   conv1d_encoder + scaled sinusoidal PE
  l0_to_hidden           [T, 2048]  layer-0 FLASH to_hidden (FFConvM)
  l0_to_qk               [T, 128]   layer-0 FLASH to_qk (FFConvM)
  l0_fsmn_conv1          [T, 256]   layer-0 Gated-FSMN conv1 + PReLU
  l0_fsmn_ddn            [T, 256]   layer-0 dilated dense net output
  l0_fsmn_gated          [T, 256]   layer-0 Gated_FSMN_dilated output
  l{i}_flash, l{i}_fsmn  [T, 512]   FLASH layer / Gated-FSMN block outputs
  intra_out              [T, 512]   MossFormerM output (after final LayerNorm)
  mdl_out                [T, 512]   Computation_Block output (GroupNorm + skip)
  masks                  [2, T, 512]
  dec_spk1, dec_spk2     [L]        raw model outputs (one forward, no RMS)
  e2e_spk1, e2e_spk2     [L]        upstream decode_one_audio_mossformer2_ss_16k
                                    (segmentation + RMS renormalisation),
                                    truncated to L as ClearVoice.decode() does

``--e2e-only`` skips the single-forward stages (use it for long clips that
exercise the segmented path).  The ClearVoice data loader additionally applies
``audio_norm`` before decoding; that only rescales the input (the network is
scale-invariant up to GroupNorm's eps) and is deliberately NOT applied here, so
the e2e reference is exactly ``decode_one_audio_mossformer2_ss_16k(clip)``.

Requires torch, numpy, soundfile and the upstream model dependencies
``einops``, ``rotary-embedding-torch`` and ``torchinfo``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from reference_backends.mossformer2_ss import (  # noqa: E402
    CHECKPOINT_SHA256, DEFAULT_LAYERS, HF_REPO, HF_REV, UPSTREAM_REPO, UPSTREAM_REV, read_config,
    resolve_upstream, run, sha256_file,
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--upstream", type=Path, required=True, help="ClearerVoice-Studio checkout @ " + UPSTREAM_REV[:8])
    ap.add_argument("--checkpoint", type=Path, required=True, help="MossFormer2_SS_16K last_best_checkpoint.pt")
    ap.add_argument("--audio", type=Path, required=True, help="16-kHz mono WAV")
    ap.add_argument("--npy-dir", type=Path)
    ap.add_argument("--gguf", type=Path)
    ap.add_argument("--start", type=float, default=0.0, help="clip start (seconds)")
    ap.add_argument("--seconds", type=float, help="clip length (seconds); default: to the end")
    ap.add_argument("--layers", default=",".join(map(str, DEFAULT_LAYERS)), help="layers whose outputs to capture")
    ap.add_argument("--e2e-only", action="store_true", help="only the decode-pipeline outputs")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--allow-source-mismatch", action="store_true")
    args = ap.parse_args()
    if not args.npy_dir and not args.gguf:
        ap.error("pass --npy-dir and/or --gguf")

    import soundfile as sf
    import torch

    torch.set_num_threads(args.threads)
    root = resolve_upstream(args.upstream)
    ck_sha = sha256_file(args.checkpoint)
    if ck_sha != CHECKPOINT_SHA256:
        msg = f"checkpoint sha256 {ck_sha} != pinned {CHECKPOINT_SHA256}"
        if not args.allow_source_mismatch:
            raise SystemExit(msg)
        print("warning:", msg, file=sys.stderr)
    cfg = read_config(root)

    audio, sr = sf.read(str(args.audio), dtype="float32", always_2d=True)
    if sr != cfg.sampling_rate:
        raise SystemExit(f"{args.audio}: {sr} Hz, model needs {cfg.sampling_rate} Hz (resample explicitly first)")
    audio = audio.mean(axis=1) if audio.shape[1] > 1 else audio[:, 0]
    s0 = int(round(args.start * sr))
    audio = audio[s0:] if args.seconds is None else audio[s0:s0 + int(round(args.seconds * sr))]
    audio = np.ascontiguousarray(audio, dtype=np.float32)
    if len(audio) < 16:
        raise SystemExit("clip too short")

    layers = [int(x) for x in args.layers.split(",") if x.strip()]
    caps = run(root, args.checkpoint, audio, layers, args.e2e_only, args.allow_source_mismatch)

    meta = {
        "backend": "mossformer2-ss",
        "upstream_repo": UPSTREAM_REPO,
        "upstream_rev": UPSTREAM_REV,
        "hf_repo": HF_REPO,
        "hf_rev": HF_REV,
        "checkpoint_sha256": ck_sha,
        "sample_rate": int(sr),
        "n_samples": int(len(audio)),
        "segmented": int(len(audio) > cfg.sampling_rate * cfg.one_time_decode_length),
        "input_pcm_sha256": hashlib.sha256(audio.astype("<f4").tobytes()).hexdigest(),
        "stages_json": json.dumps(sorted(caps)),
    }
    if args.npy_dir:
        args.npy_dir.mkdir(parents=True, exist_ok=True)
        for name, arr in caps.items():
            np.save(args.npy_dir / f"{name}.npy", arr)
        (args.npy_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    if args.gguf:
        from dump_reference import write_gguf_archive

        write_gguf_archive(caps, meta, args.gguf)
    print(f"{len(audio)} samples ({len(audio) / sr:.2f} s), {len(caps)} stages")
    for name in sorted(caps):
        print(f"  {name:16s} {list(caps[name].shape)}")


if __name__ == "__main__":
    main()
