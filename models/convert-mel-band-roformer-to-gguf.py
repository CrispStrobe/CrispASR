#!/usr/bin/env python3
"""Convert a Mel-Band RoFormer checkpoint (§248 source separation) to GGUF.

Usage:
    python models/convert-mel-band-roformer-to-gguf.py \
        --model /path/to/dir_with_ckpt_and_yaml \
        --output mel-band-roformer-vocals.gguf [--dtype f16]

Weights: KimberleyJSN/melbandroformer (MIT). Architecture blueprint: MIT
lucidrains/BS-RoFormer (NOT Kim's unlicensed inference repo). Full trace +
tensor map in docs/mel-band-roformer/PLAN.md.

Two design decisions baked in here (see the PLAN):
  * `freq_indices` (int32) and `num_bands_per_freq` (int32) are computed once
    with librosa — INCLUDING the two hand-tweaks (mel[0,0], mel[-1,-1] *= 0.25)
    without which DC belongs to no band — and stored as tensors. C++ never
    reimplements the mel edge placement.
  * per-band linear widths (dim_in_b) are recovered from the checkpoint's own
    band_split Linear shapes and stored as an int32 array, so the runtime knows
    each band's width without recomputing membership.

GGUF tensor names keep the torch key verbatim (dots intact) so the C++ loader
maps 1:1 against the map in the PLAN — no renaming ambiguity.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def _load_config(model_dir: Path):
    import yaml

    cands = list(model_dir.glob("*.yaml")) + list(model_dir.glob("*.yml"))
    if not cands:
        sys.exit(f"error: no *.yaml config in {model_dir} (band layout is not "
                 f"recoverable from the weights alone)")

    # The upstream configs use `!!python/tuple`; teach SafeLoader that one tag
    # rather than dropping to the unsafe loader.
    class _Loader(yaml.SafeLoader):
        pass

    _Loader.add_constructor(
        "tag:yaml.org,2002:python/tuple",
        lambda loader, node: tuple(loader.construct_sequence(node)),
    )
    with open(cands[0], encoding="utf-8") as f:
        cfg_all = yaml.load(f, Loader=_Loader)
    m = dict(cfg_all.get("model", {}))
    a = cfg_all.get("audio", {}) or {}
    m.setdefault("sample_rate", a.get("sample_rate", 44100))
    m.setdefault("stft_n_fft", a.get("n_fft", 2048))
    m.setdefault("stft_hop_length", a.get("hop_length", 441))
    m.setdefault("stft_win_length", a.get("n_fft", 2048))
    # chunk_size = the inference window the checkpoint was TRAINED on (Kim
    # vocals: 352800 = 8.0 s @ 44100). The runtime defaults segmentation to it;
    # a 10 s default would extrapolate RoPE 25% past training (PR #422 review).
    # Some configs keep it under `audio`; default 0 = omit the KV entirely so
    # old runtimes fall back to their own 8 s default.
    m.setdefault("chunk_size", a.get("chunk_size", 0))
    return m


def _band_layout(sample_rate, n_fft, num_bands, stereo):
    """Reproduce the reference band membership EXACTLY (librosa + the two
    hand-tweaks). Returns (freq_indices int32, num_bands_per_freq int32,
    num_freqs_per_band int32)."""
    import torch
    from librosa import filters
    from einops import repeat, reduce

    mel = torch.from_numpy(filters.mel(sr=sample_rate, n_fft=n_fft, n_mels=num_bands))
    mel[0, 0] = mel[0, 1] * 0.25       # LOAD-BEARING: else DC has no band
    mel[-1, -1] = mel[-1, -2] * 0.25
    fpb = mel > 0                       # binary membership; weights discarded
    assert bool(fpb.any(dim=0).all()), "some frequency has no band — bad config"

    freqs = n_fft // 2 + 1
    rep = repeat(torch.arange(freqs), "f -> b f", b=num_bands)
    freq_indices = rep[fpb]
    if stereo:
        fi = repeat(freq_indices, "f -> f s", s=2)
        fi = fi * 2 + torch.arange(2)
        freq_indices = fi.reshape(-1)

    nfpb = reduce(fpb, "b f -> b", "sum")
    nbpf = reduce(fpb, "b f -> f", "sum")
    return (freq_indices.to(torch.int32).numpy(),
            nbpf.to(torch.int32).numpy(),
            nfpb.to(torch.int32).numpy())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True,
                    help="dir with the .ckpt and its .yaml config")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--dtype", choices=["f16", "f32"], default="f16")
    args = ap.parse_args()

    import torch
    from gguf import GGUFWriter, GGMLQuantizationType

    model_dir = args.model
    ckpts = sorted(list(model_dir.glob("*.ckpt")) + list(model_dir.glob("*.pt"))
                   + list(model_dir.glob("*.bin")))
    if not ckpts:
        sys.exit(f"error: no checkpoint in {model_dir}")
    cfg = _load_config(model_dir)

    sr = int(cfg["sample_rate"])
    n_fft = int(cfg["stft_n_fft"])
    hop = int(cfg["stft_hop_length"])
    win = int(cfg.get("stft_win_length", n_fft))
    num_bands = int(cfg["num_bands"])
    stereo = bool(cfg.get("stereo", True))
    channels = 2 if stereo else 1

    # Lazy per-tensor read; mmap keeps the ~870 MB fp32 ckpt off the heap.
    sd = torch.load(str(ckpts[0]), map_location="cpu", mmap=True, weights_only=False)
    for k in ("state_dict", "model", "model_state_dict"):
        if isinstance(sd, dict) and k in sd and isinstance(sd[k], dict):
            sd = sd[k]
            break
    sd = {(k[len("module."):] if k.startswith("module.") else k): v for k, v in sd.items()}

    n_stems = 1 + max(int(kk.split(".")[1]) for kk in sd if kk.startswith("mask_estimators."))
    depth = 1 + max(int(kk.split(".")[1]) for kk in sd if kk.startswith("layers."))

    freq_indices, num_bands_per_freq, num_freqs_per_band = _band_layout(sr, n_fft, num_bands, stereo)
    # Sanity: band_split widths must equal 2 * num_freqs_per_band * channels.
    dim_in = [2 * int(f) * channels for f in num_freqs_per_band]
    for b, w in enumerate(dim_in):
        got = tuple(sd[f"band_split.to_features.{b}.1.weight"].shape)
        assert got[1] == w, f"band {b}: computed dim_in {w} != ckpt {got[1]} — band layout mismatch"

    print(f"mel-band-roformer: dim={int(cfg['dim'])} depth={depth} bands={num_bands} "
          f"stems={n_stems} n_fft={n_fft} hop={hop} sr={sr} tensors={len(sd)}")

    w = GGUFWriter(str(args.output), "mel-band-roformer", use_temp_file=True)
    w.add_name(f"Mel-Band RoFormer ({model_dir.name})")
    w.add_uint32("mel-band-roformer.dim", int(cfg["dim"]))
    w.add_uint32("mel-band-roformer.depth", depth)
    w.add_uint32("mel-band-roformer.heads", int(cfg["heads"]))
    w.add_uint32("mel-band-roformer.dim_head", int(cfg["dim_head"]))
    w.add_uint32("mel-band-roformer.num_bands", num_bands)
    w.add_uint32("mel-band-roformer.num_stems", n_stems)
    w.add_uint32("mel-band-roformer.time_transformer_depth", int(cfg.get("time_transformer_depth", 1)))
    w.add_uint32("mel-band-roformer.freq_transformer_depth", int(cfg.get("freq_transformer_depth", 1)))
    w.add_uint32("mel-band-roformer.mask_estimator_depth", int(cfg.get("mask_estimator_depth", 2)))
    w.add_uint32("mel-band-roformer.stereo", 1 if stereo else 0)
    w.add_uint32("mel-band-roformer.audio_channels", channels)
    w.add_uint32("mel-band-roformer.sample_rate", sr)
    w.add_uint32("mel-band-roformer.stft_n_fft", n_fft)
    w.add_uint32("mel-band-roformer.stft_hop_length", hop)
    w.add_uint32("mel-band-roformer.stft_win_length", win)
    w.add_uint32("mel-band-roformer.stft_normalized", 1 if cfg.get("stft_normalized", False) else 0)
    if cfg.get("chunk_size"):
        w.add_uint32("mel-band-roformer.chunk_size", int(cfg["chunk_size"]))
    w.add_string("mel-band-roformer.instruments_json",
                 json.dumps(cfg.get("training", {}).get("instruments", ["vocals", "other"])
                            if isinstance(cfg.get("training"), dict) else ["vocals", "other"]))

    # Baked band-layout arrays (int32) — the runtime never recomputes librosa.
    f32 = GGMLQuantizationType.F32
    i32 = GGMLQuantizationType.I32
    f16 = GGMLQuantizationType.F16
    w.add_tensor("aux.freq_indices", freq_indices.astype(np.int32), raw_dtype=i32)
    w.add_tensor("aux.num_bands_per_freq", num_bands_per_freq.astype(np.int32), raw_dtype=i32)
    w.add_tensor("aux.num_freqs_per_band", num_freqs_per_band.astype(np.int32), raw_dtype=i32)

    # Model weights. Norms/biases/1-D and any *rotary* buffer stay F32 (small +
    # precision-sensitive); 2-D matmul weights honor --dtype.
    n_w = 0
    for name, t in sd.items():
        arr = t.detach().to(torch.float32).cpu().numpy()
        keep_f32 = (arr.ndim < 2) or name.endswith("gamma") or ("rotary" in name) or name.endswith("bias")
        if args.dtype == "f16" and not keep_f32:
            w.add_tensor(name, arr.astype(np.float16), raw_dtype=f16)
        else:
            w.add_tensor(name, arr.astype(np.float32), raw_dtype=f32)
        n_w += 1

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"wrote {args.output}  ({n_w} weight tensors + 3 aux, dtype={args.dtype})")


if __name__ == "__main__":
    main()
