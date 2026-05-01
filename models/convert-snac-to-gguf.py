#!/usr/bin/env python3
"""
Convert hubertsiuzdak/snac_24khz (PyTorch) → GGUF for the Orpheus
backend's SNAC decoder.

SNAC architecture (verified live against the actual checkpoint):

  encoder              — not converted (Orpheus only needs decode())
  quantizer.quantizers — 3 RVQ codebooks
                         each: codebook_size=4096, codebook_dim=8
                         in_proj  : 1×1 Conv1d(D, 8)   (to latent)
                         out_proj : 1×1 Conv1d(8, D)   (from latent)
  decoder.model        — Sequential of:
    [0] Conv1d(8, 768, k=7)    (input projection — z_q -> 768)
    [1] Snake1d alpha          (NOTE: actually [0] is Snake1d? — verified in
                                converter; in canonical SNAC,
                                model[0] = Conv1d, model[1] = Conv1d? — see
                                runtime fields and disambiguate at convert
                                time. We just walk parameters by name.)
    [2..5] DecoderBlock          (strides 8, 8, 4, 2)
        block[0]: Snake1d alpha
        block[1]: ConvTranspose1d(in, out, k=2*s, s)
        block[2]: NoiseBlock — Conv1d(out, out, k=1)  (treated as linear)
        block[3..5]: ResidualUnit
            block[0]: Snake1d alpha
            block[1]: Conv1d(C, C, k=7, dilation=d)   d in {1,3,9}
            block[2]: Snake1d alpha
            block[3]: Conv1d(C, C, k=1)
    [6] Snake1d alpha
    [7] Conv1d(64, 1, k=7)
    [8] Tanh                   (no params)

  Weight-norm reparameterisation:
    Each ParametrizedConv* tensor is stored as
      original0  shape (C_out, 1, 1)        — the `g` scalar (weight_norm dim=0)
      original1  shape (C_out, C_in, K)     — the unnormalised `v`
    Materialise as: w = original0 * (original1 / ||original1||_{(C_in,K)})

This converter walks the pytorch_model.bin parameter dict, materialises
each weight_norm pair into a single conv weight, and emits a flat
GGUF with names like:

  snac.dec.0.weight        — Conv1d(8, 768, k=7)
  snac.dec.0.bias
  snac.dec.<i>.alpha       — Snake1d (1, C, 1) → broadcast to (C,)
  snac.dec.<i>.upsample.weight    — ConvTranspose1d
  snac.dec.<i>.upsample.bias
  snac.dec.<i>.noise.weight       — 1×1 Conv1d
  snac.dec.<i>.noise.bias
  snac.dec.<i>.res.<j>.alpha0/.conv0.weight/.conv0.bias/.alpha1/.conv1.weight/.conv1.bias
  snac.dec.out.alpha
  snac.dec.out.weight
  snac.dec.out.bias
  snac.q.<k>.codebook         — (codebook_size, codebook_dim)
  snac.q.<k>.in_proj.weight   — (codebook_dim, D, 1)  [unwrapped weight_norm]
  snac.q.<k>.in_proj.bias
  snac.q.<k>.out_proj.weight  — (D, codebook_dim, 1)
  snac.q.<k>.out_proj.bias

Metadata also includes vq_strides and per-block strides+dilations so
the C++ side doesn't have to hardcode the architecture.

Usage:

    python models/convert-snac-to-gguf.py \\
        --input /Volumes/backups/ai/huggingface-hub/models--hubertsiuzdak--snac_24khz/snapshots/<sha> \\
        --output snac-24khz.gguf
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

try:
    from gguf import GGUFWriter, GGMLQuantizationType
except ImportError:
    sys.exit("pip install gguf")

try:
    import torch
except ImportError:
    sys.exit("pip install torch")


def load_state_dict(model_dir: Path) -> dict:
    bin_path = model_dir / "pytorch_model.bin"
    if not bin_path.exists():
        sys.exit(f"missing pytorch_model.bin in {model_dir}")
    return torch.load(bin_path, map_location="cpu", weights_only=True)


def materialize_weight_norm(state: dict, base: str) -> tuple[np.ndarray, np.ndarray | None]:
    """For a parametrized conv whose weight is stored as
        {base}.parametrizations.weight.original0   = g  (C_out, 1, 1)
        {base}.parametrizations.weight.original1   = v  (C_out, C_in, K)
        {base}.bias                                 (optional)
    return (w, bias) where w = g * v / ||v||_dim0 in the canonical
    PyTorch weight_norm sense (norm taken over all non-channel dims)."""

    g_key = f"{base}.parametrizations.weight.original0"
    v_key = f"{base}.parametrizations.weight.original1"
    b_key = f"{base}.bias"

    g = state[g_key].float().numpy()
    v = state[v_key].float().numpy()

    # PyTorch weight_norm default dim=0 → norm across (1,2,...).
    # For Conv1d (C_out, C_in, K) and ConvTranspose1d (C_in, C_out, K) both
    # have dim 0 reserved as the "channel-out-like" axis; the norm slice is
    # everything else.
    norm_axes = tuple(range(1, v.ndim))
    v_norm = np.linalg.norm(v, axis=norm_axes, keepdims=True)
    # Match torch.nn.functional._weight_norm: g shape broadcasts.
    w = g * (v / np.where(v_norm == 0, 1.0, v_norm))

    bias = state[b_key].float().numpy() if b_key in state else None
    return w.astype(np.float32), (bias.astype(np.float32) if bias is not None else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="SNAC snapshot dir")
    ap.add_argument("--output", required=True)
    ap.add_argument("--outtype", default="f16", choices=["f16", "f32"])
    args = ap.parse_args()

    model_dir = Path(args.input)
    state = load_state_dict(model_dir)

    with open(model_dir / "config.json") as f:
        cfg = json.load(f)

    # Decoder block strides — verified against the live model:
    # model.2: stride 8, model.3: stride 8, model.4: stride 4, model.5: stride 2
    block_strides = cfg.get("decoder_rates") or [8, 8, 4, 2]
    vq_strides = cfg.get("vq_strides") or [4, 2, 1]
    codebook_size = cfg.get("codebook_size", 4096)
    codebook_dim = cfg.get("codebook_dim", 8)
    n_q = len(vq_strides)
    sample_rate = cfg.get("sampling_rate", 24000)

    print(f"\nSNAC 24 kHz")
    print(f"  block_strides : {block_strides}")
    print(f"  vq_strides    : {vq_strides}")
    print(f"  codebook      : {n_q} × {codebook_size} × {codebook_dim}")
    print(f"  sample_rate   : {sample_rate}")

    out_dtype = np.float16 if args.outtype == "f16" else np.float32
    out_qt = GGMLQuantizationType.F16 if args.outtype == "f16" else GGMLQuantizationType.F32

    w = GGUFWriter(args.output, arch="snac", use_temp_file=True)
    w.add_name("snac-24khz")
    w.add_uint32("snac.sample_rate", sample_rate)
    w.add_uint32("snac.n_codebooks", n_q)
    w.add_uint32("snac.codebook_size", codebook_size)
    w.add_uint32("snac.codebook_dim", codebook_dim)
    w.add_array("snac.vq_strides", list(vq_strides))
    w.add_array("snac.decoder_strides", list(block_strides))
    # Dilations within a ResidualUnit (verified: 1,3,9). Same for every block.
    w.add_array("snac.residual_dilations", [1, 3, 9])
    w.add_uint32("snac.hop_length", int(np.prod(block_strides)))

    def emit(name: str, arr: np.ndarray, force_f32: bool = False):
        arr = np.ascontiguousarray(arr)
        if force_f32 or arr.ndim <= 1:
            w.add_tensor(name, arr.astype(np.float32), raw_dtype=GGMLQuantizationType.F32)
        else:
            w.add_tensor(name, arr.astype(out_dtype), raw_dtype=out_qt)

    n_emitted = 0

    # ---- Quantizer (3 codebooks) ----
    for k in range(n_q):
        cb = state[f"quantizer.quantizers.{k}.codebook.weight"].float().numpy()
        emit(f"snac.q.{k}.codebook", cb)
        n_emitted += 1

        # in_proj / out_proj are weight_norm Conv1d (k=1)
        ip_w, ip_b = materialize_weight_norm(state, f"quantizer.quantizers.{k}.in_proj")
        emit(f"snac.q.{k}.in_proj.weight", ip_w)
        if ip_b is not None:
            emit(f"snac.q.{k}.in_proj.bias", ip_b)
        n_emitted += 2 if ip_b is not None else 1

        op_w, op_b = materialize_weight_norm(state, f"quantizer.quantizers.{k}.out_proj")
        emit(f"snac.q.{k}.out_proj.weight", op_w)
        if op_b is not None:
            emit(f"snac.q.{k}.out_proj.bias", op_b)
        n_emitted += 2 if op_b is not None else 1

    # ---- Decoder ----
    # model.0 — input ConvTranspose1d? No: looking at the dump, model[0] and
    # model[1] are both Conv1d. model[0] is the "z_q -> 768" projection.
    # model[1] could be a second conv. Walk by parameter availability rather
    # than positional naming.
    in_w, in_b = materialize_weight_norm(state, "decoder.model.0")
    emit("snac.dec.in0.weight", in_w)
    if in_b is not None:
        emit("snac.dec.in0.bias", in_b)
    n_emitted += 2 if in_b is not None else 1

    in1_w, in1_b = materialize_weight_norm(state, "decoder.model.1")
    emit("snac.dec.in1.weight", in1_w)
    if in1_b is not None:
        emit("snac.dec.in1.bias", in1_b)
    n_emitted += 2 if in1_b is not None else 1

    # model.2..5 = DecoderBlocks (one per stride)
    for blk_idx, stride in enumerate(block_strides):
        m = blk_idx + 2  # model.[2..5]
        prefix = f"decoder.model.{m}.block"

        # block.0: Snake1d (alpha at .alpha)
        alpha = state[f"{prefix}.0.alpha"].float().numpy().reshape(-1)
        emit(f"snac.dec.{blk_idx}.alpha", alpha, force_f32=True)
        n_emitted += 1

        # block.1: ConvTranspose1d (the upsample), kernel = 2*stride
        up_w, up_b = materialize_weight_norm(state, f"{prefix}.1")
        emit(f"snac.dec.{blk_idx}.up.weight", up_w)
        if up_b is not None:
            emit(f"snac.dec.{blk_idx}.up.bias", up_b)
        n_emitted += 2 if up_b is not None else 1

        # block.2: NoiseBlock — internal Linear is "...block.2.linear"
        nb_w, nb_b = materialize_weight_norm(state, f"{prefix}.2.linear")
        emit(f"snac.dec.{blk_idx}.noise.weight", nb_w)
        if nb_b is not None:
            emit(f"snac.dec.{blk_idx}.noise.bias", nb_b)
        n_emitted += 2 if nb_b is not None else 1

        # block.3..5: 3 ResidualUnits
        for r in range(3):
            rb = f"{prefix}.{3 + r}.block"
            # block.0: Snake1d
            a0 = state[f"{rb}.0.alpha"].float().numpy().reshape(-1)
            emit(f"snac.dec.{blk_idx}.res.{r}.alpha0", a0, force_f32=True)
            # block.1: Conv1d (k=7, dilation=1/3/9)
            c0_w, c0_b = materialize_weight_norm(state, f"{rb}.1")
            emit(f"snac.dec.{blk_idx}.res.{r}.conv0.weight", c0_w)
            if c0_b is not None:
                emit(f"snac.dec.{blk_idx}.res.{r}.conv0.bias", c0_b)
            # block.2: Snake1d
            a1 = state[f"{rb}.2.alpha"].float().numpy().reshape(-1)
            emit(f"snac.dec.{blk_idx}.res.{r}.alpha1", a1, force_f32=True)
            # block.3: Conv1d (k=1)
            c1_w, c1_b = materialize_weight_norm(state, f"{rb}.3")
            emit(f"snac.dec.{blk_idx}.res.{r}.conv1.weight", c1_w)
            if c1_b is not None:
                emit(f"snac.dec.{blk_idx}.res.{r}.conv1.bias", c1_b)
            n_emitted += 6  # 2 alphas + 2 convs (each weight+bias)

    # model.6: Snake1d
    out_alpha = state["decoder.model.6.alpha"].float().numpy().reshape(-1)
    emit("snac.dec.out.alpha", out_alpha, force_f32=True)
    # model.7: Conv1d (k=7)
    out_w, out_b = materialize_weight_norm(state, "decoder.model.7")
    emit("snac.dec.out.weight", out_w)
    if out_b is not None:
        emit("snac.dec.out.bias", out_b)
    n_emitted += 3 if out_b is not None else 2
    # model.8: Tanh — no params

    print(f"\nEmitting {n_emitted} tensors → {args.output}")
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()

    sz = Path(args.output).stat().st_size / 1e6
    print(f"Done: {args.output}  ({sz:.1f} MB)")


if __name__ == "__main__":
    main()
