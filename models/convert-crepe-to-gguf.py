#!/usr/bin/env python3
"""Convert a CREPE pitch-estimation checkpoint to GGUF.

Usage:
    python models/convert-crepe-to-gguf.py --capacity tiny \
        --output crepe-tiny-f16.gguf [--dtype f16]
    python models/convert-crepe-to-gguf.py --model /path/to/full.pth \
        --output crepe-full-f16.gguf

Weights: torchcrepe (MIT, maxrmorrison) — a direct port of the original CREPE
Keras weights (MIT, Kim et al. 2018, "CREPE: A Convolutional Representation for
Pitch Estimation"). Blueprint trace + geometry table in
docs/music-transcription/PLAN.md.

Two decisions baked in here (see the PLAN):

  * **BatchNorm is emitted as a standalone per-channel affine — it does NOT
    fold into the conv.** The blueprint's layer order is

        F.pad -> conv -> F.relu -> batch_norm -> max_pool2d

    i.e. **ReLU sits between the conv and the BN**, so the usual conv+BN fold is
    invalid here. (Getting this backwards produces a plausible-looking net whose
    layer-1 output already sits at cos=0.83 with ~2x the reference magnitude,
    because fitting an affine through a rectified signal recovers roughly half
    the true scale. Read the blueprint, not a summary of it.)
    So we precompute the affine in **float64** and ship it as two vectors:
        scale  = gamma / sqrt(var + eps)
        offset = beta - mean * scale
    and the runtime graph is conv -> +bias -> relu -> x*scale + offset -> pool.
    float64 matters: computing `1/sqrt(var+eps)` in f32 (let alone f16) shifts
    the small-variance channels, which is the NeMo canary BN-fold bug (702db9af).

  * **Conv weights are emitted (out, in, K)** so ggml reads ne=(K, in, out),
    which is `ggml_conv_1d`'s expected [K, IC, OC] layout — no runtime permute.
    The torch weight is (out, in, K, 1); the trailing width-1 axis (CREPE uses
    conv2d over an Nx1 image) is squeezed here, not in C++.

GGUF tensor names keep the torch key verbatim so the C++ loader maps 1:1
against the table in the PLAN.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

# Geometry, from torchcrepe/model.py. Padding is (left, right) on the time axis.
# conv1 is symmetric; conv2..6 are ASYMMETRIC (31, 32) — the C++ side gets this
# with a symmetric p=32 conv and a leading-column crop (Metal rejects an
# asymmetric GGML_OP_PAD).
LAYERS = [
    # name,  kernel, stride, pad_l, pad_r
    ("conv1", 512, 4, 254, 254),
    ("conv2", 64, 1, 31, 32),
    ("conv3", 64, 1, 31, 32),
    ("conv4", 64, 1, 31, 32),
    ("conv5", 64, 1, 31, 32),
    ("conv6", 64, 1, 31, 32),
]

CAPACITY_CHANNELS = {
    # capacity: (out_channels per layer, classifier in_features)
    "full": ([1024, 128, 128, 128, 256, 512], 2048),
    "tiny": ([128, 16, 16, 16, 32, 64], 256),
}

BN_EPS = 1e-3  # torchcrepe: nn.BatchNorm2d(..., eps=0.001, momentum=0.0)

# torchcrepe/core.py + convert.py
SAMPLE_RATE = 16000
WINDOW_SIZE = 1024
PITCH_BINS = 360
CENTS_PER_BIN = 20.0
CENTS_OFFSET = 1997.3794084376191  # cents of bin 0 (= 10 * 2**(c/1200) Hz)


def _bn_affine(gamma, beta, mean, var, eps=BN_EPS):
    """Inference-mode BatchNorm as a per-channel affine. All math in float64.

    y = (x - mean)/sqrt(var + eps) * gamma + beta
      = x * scale + offset
    NOT folded into the conv — ReLU sits between them (see module docstring).
    """
    scale = gamma.astype(np.float64) / np.sqrt(var.astype(np.float64) + eps)
    offset = beta.astype(np.float64) - mean.astype(np.float64) * scale
    return scale, offset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path,
                    help="path to a torchcrepe .pth (default: the bundled asset "
                         "for --capacity)")
    ap.add_argument("--capacity", choices=sorted(CAPACITY_CHANNELS), default="full")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--dtype", choices=["f16", "f32"], default="f16")
    args = ap.parse_args()

    import torch
    from gguf import GGUFWriter, GGMLQuantizationType

    path = args.model
    if path is None:
        try:
            import torchcrepe
        except ImportError:
            sys.exit("error: --model not given and torchcrepe is not installed "
                     "(pip install torchcrepe), so no bundled asset to read")
        path = Path(torchcrepe.__file__).parent / "assets" / f"{args.capacity}.pth"
    if not path.exists():
        sys.exit(f"error: no checkpoint at {path}")

    sd = torch.load(str(path), map_location="cpu", weights_only=True)
    sd = {k: v.detach().cpu().numpy() for k, v in sd.items()
          if not k.endswith("num_batches_tracked")}

    out_ch, in_features = CAPACITY_CHANNELS[args.capacity]
    # The classifier width is the unambiguous capacity fingerprint — check the
    # checkpoint really is the capacity that was asked for before folding.
    got = sd["classifier.weight"].shape
    if got != (PITCH_BINS, in_features):
        sys.exit(f"error: classifier.weight is {got}, expected "
                 f"{(PITCH_BINS, in_features)} for --capacity {args.capacity}; "
                 f"wrong checkpoint/capacity pairing")

    print(f"crepe: capacity={args.capacity} channels={out_ch} "
          f"in_features={in_features} tensors={len(sd)} src={path}")

    w = GGUFWriter(str(args.output), "crepe", use_temp_file=True)
    w.add_name(f"CREPE ({args.capacity})")
    w.add_uint32("crepe.pitch_bins", PITCH_BINS)
    w.add_uint32("crepe.window_size", WINDOW_SIZE)
    w.add_uint32("crepe.sample_rate", SAMPLE_RATE)
    w.add_uint32("crepe.n_layers", len(LAYERS))
    w.add_uint32("crepe.in_features", in_features)
    w.add_float32("crepe.cents_per_bin", CENTS_PER_BIN)
    w.add_float32("crepe.cents_offset", CENTS_OFFSET)
    w.add_string("crepe.capacity", args.capacity)

    f32 = GGMLQuantizationType.F32
    f16 = GGMLQuantizationType.F16
    wdt = f16 if args.dtype == "f16" else f32
    nwdt = np.float16 if args.dtype == "f16" else np.float32

    for i, (name, k, stride, pad_l, pad_r) in enumerate(LAYERS):
        cw = sd[f"{name}.weight"]           # (out, in, K, 1)
        cb = sd[f"{name}.bias"]             # (out,)
        assert cw.ndim == 4 and cw.shape[2] == k and cw.shape[3] == 1, \
            f"{name}.weight is {cw.shape}, expected (out, in, {k}, 1)"
        assert cw.shape[0] == out_ch[i], \
            f"{name} has {cw.shape[0]} out-channels, expected {out_ch[i]}"
        cw = cw[:, :, :, 0]                 # -> (out, in, K)

        # (out, in, K) row-major -> ggml ne = (K, in, out) = conv_1d's [K, IC, OC].
        w.add_tensor(f"{name}.weight", np.ascontiguousarray(cw.astype(nwdt)),
                     raw_dtype=wdt)
        w.add_tensor(f"{name}.bias", cb.astype(np.float32), raw_dtype=f32)

        # BN as a standalone affine, applied AFTER the relu.
        scale, offset = _bn_affine(sd[f"{name}_BN.weight"], sd[f"{name}_BN.bias"],
                                   sd[f"{name}_BN.running_mean"],
                                   sd[f"{name}_BN.running_var"])
        w.add_tensor(f"{name}_BN.scale", scale.astype(np.float32), raw_dtype=f32)
        w.add_tensor(f"{name}_BN.offset", offset.astype(np.float32), raw_dtype=f32)

    # classifier: torch Linear weight is (out, in) = (360, in_features); ggml
    # mul_mat(A, B) = B x A^T wants ne = (in, out), which is this array as-is.
    w.add_tensor("classifier.weight",
                 np.ascontiguousarray(sd["classifier.weight"].astype(nwdt)),
                 raw_dtype=wdt)
    w.add_tensor("classifier.bias",
                 sd["classifier.bias"].astype(np.float32), raw_dtype=f32)

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"wrote {args.output} ({args.output.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
