#!/usr/bin/env python
"""Synthesise the Kim mel-band-roformer config YAML that the checkpoint omits.

WHY THIS EXISTS. `KimberleyJSN/melbandroformer` ships ONLY
`MelBandRoformer.ckpt` (913 MB) — no YAML. But the band layout is not
recoverable from the weights alone, so BOTH
`models/convert-mel-band-roformer-to-gguf.py::_load_config` and
`tools/reference_backends/mel_band_roformer.py::_load` glob for `*.yaml`/`*.yml`
and exit with:

    error: no *.yaml config in <dir> (band layout is not recoverable from the
           weights alone)

Anyone converting or reference-dumping that checkpoint hits this wall. Two
Kaggle runs died on it before the cause was written down anywhere but a chat
log, which is what this file fixes.

LICENCE POSITION. The values below are transcribed from
`docs/mel-band-roformer/PLAN.md` (the "Target checkpoint + config (Kim vocals)"
block), where they were recorded deliberately. They are hyperparameters —
facts, not creative code — and they are NOT copied from Kim's inference repo,
which ships no licence. The weights are MIT; the lucidrains implementation the
model is built from is MIT. See the clean-room note in PLAN.md.

USE:
    python tools/melband_kim_config.py --model-dir <dir with the .ckpt>
    python tools/melband_kim_config.py --model-dir <dir> --verify   # strict load

or from Python / a Kaggle kernel:
    from melband_kim_config import write_config, verify_against_checkpoint
    cfg_path = write_config(ckpt_dir)
    verify_against_checkpoint(ckpt_dir)      # proves cfg matches the weights
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Kim vocals. These differ from lucidrains defaults — notably stft_hop_length
# 441 rather than 512 — which is exactly why the config cannot be guessed.
# Source: docs/mel-band-roformer/PLAN.md, "Target checkpoint + config".
KIM_VOCALS_AUDIO = {
    "sample_rate": 44100,
    "chunk_size": 352800,   # 8.0 s @ 44.1 kHz — the TRAINED window; the runtime
                            # defaults segmentation to it (PR #422 review).
    "n_fft": 2048,
    "hop_length": 441,
}

KIM_VOCALS_MODEL = {
    "dim": 384,
    "depth": 6,
    "stereo": True,
    "num_stems": 1,                  # vocals; "other" is the residual
    "time_transformer_depth": 1,
    "freq_transformer_depth": 1,
    "heads": 8,
    "dim_head": 64,
    "num_bands": 60,
    "dim_freqs_in": 1024,
    "mask_estimator_depth": 2,
    "attn_dropout": 0,
    "ff_dropout": 0,
    "flash_attn": True,
    "stft_n_fft": 2048,
    "stft_hop_length": 441,          # NOT the lucidrains default 512
    "stft_win_length": 2048,
}


def config_dict() -> dict:
    """The full two-section config both loaders accept.

    `_load_config` merges audio -> model via setdefault for sample_rate /
    stft_n_fft / stft_hop_length / stft_win_length / chunk_size, so carrying the
    STFT values in BOTH sections is harmless and removes any guess about which
    section a given loader reads.
    """
    return {"audio": dict(KIM_VOCALS_AUDIO), "model": dict(KIM_VOCALS_MODEL)}


def write_config(model_dir, name: str = "config_melband_roformer_vocals_kim.yaml") -> Path:
    """Write the config next to the checkpoint. Returns the path.

    A pre-existing *.yaml/*.yml in the directory is left alone and returned —
    a real upstream config always wins over this reconstruction.
    """
    import yaml

    d = Path(model_dir)
    existing = sorted(list(d.glob("*.yaml")) + list(d.glob("*.yml")))
    if existing:
        print(f"melband_kim_config: existing config found, leaving it: {existing[0].name}")
        return existing[0]
    dest = d / name
    dest.write_text(yaml.safe_dump(config_dict(), sort_keys=False), encoding="utf-8")
    print(f"melband_kim_config: wrote {dest}")
    return dest


def verify_against_checkpoint(model_dir) -> None:
    """Prove the config matches the weights with a strict load, or raise.

    A synthesised config is exactly the kind of thing that is silently
    almost-right: a wrong dim / depth / num_bands / dim_freqs_in yields a model
    that BUILDS and RUNS and emits subtly wrong audio. `strict=True` turns that
    into a shape mismatch in seconds. Cheap check before expensive work — run
    this before a CUDA build or a reference forward, not after.
    """
    import torch

    # Same import path the reference loader uses
    # (tools/reference_backends/mel_band_roformer.py). Getting this wrong would
    # make the CHECK fail on a healthy system — the inverted-guard bug this
    # whole exercise has been about — so it is copied from the working code
    # rather than guessed.
    from bs_roformer.mel_band_roformer import MelBandRoformer

    d = Path(model_dir)
    ckpts = sorted(list(d.glob("*.ckpt")) + list(d.glob("*.pt")) + list(d.glob("*.bin")))
    if not ckpts:
        raise SystemExit(f"melband_kim_config: no checkpoint in {d}")

    # mmap keeps the ~870 MB checkpoint on disk instead of peaking ~2 GB
    # alongside the constructed model — same discipline as the reference loader.
    try:
        sd = torch.load(str(ckpts[0]), map_location="cpu", mmap=True, weights_only=False)
    except Exception:
        sd = torch.load(str(ckpts[0]), map_location="cpu", weights_only=False)
    if isinstance(sd, dict):
        for key in ("state_dict", "model", "model_state_dict"):
            if key in sd and isinstance(sd[key], dict):
                sd = sd[key]
                break
    sd = {(k[len("module."):] if k.startswith("module.") else k): v for k, v in sd.items()}

    model = MelBandRoformer(**KIM_VOCALS_MODEL)
    # strict=True is the point: the reference loader uses strict=False and only
    # WARNS on missing/unexpected keys, which is right for it but would let a
    # wrong config through here. This is the check, so it must be strict.
    model.load_state_dict(sd, strict=True)
    n = sum(p.numel() for p in model.parameters())
    print(f"melband_kim_config: strict load OK against {ckpts[0].name} "
          f"({n/1e6:.1f}M params) — config matches the weights")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model-dir", type=Path, required=True,
                    help="directory holding MelBandRoformer.ckpt")
    ap.add_argument("--verify", action="store_true",
                    help="strict-load the checkpoint against the config to prove they match")
    a = ap.parse_args()
    write_config(a.model_dir)
    if a.verify:
        verify_against_checkpoint(a.model_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
