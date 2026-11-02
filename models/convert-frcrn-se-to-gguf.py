#!/usr/bin/env python3
"""Convert ClearerVoice-Studio FRCRN_SE_16K to a CrispASR enhancement GGUF.

Source: modelscope/ClearerVoice-Studio (Apache-2.0), checkpoint
``alibabasglab/FRCRN_SE_16K`` (``last_best_checkpoint.pt``). Only the model
weights are converted; the checkpoint is read with ``weights_only=True`` and
no upstream code is imported or executed.

    python models/convert-frcrn-se-to-gguf.py \
        --checkpoint FRCRN_SE_16K/last_best_checkpoint.pt \
        --output frcrn-se-16k-f16.gguf --dtype f16

Architecture ``frcrn-se``. The graph is DCCRN-style: ConvSTFT (640/320/640,
sqrt periodic Hann) -> two cascaded complex UNets (7 encoders, 7 decoders,
frequency FSMN memories, complex SE layers, time-causal bottleneck FSMN) ->
complex mask ``tanh(u2) + tanh(u1)`` -> ConviSTFT with window-power OLA
normalisation.

Layout decisions made here (all exact rewrites, no approximation):

* ConvSTFT / ConviSTFT kernels are computed exactly like upstream's
  ``init_kernels`` (numpy rfft basis, ``pinv`` for the inverse, scipy window)
  and stored as F32 tensors ``stft.fwd`` (642 x 640) and ``stft.inv``
  (640 x 642) plus ``stft.window`` (640), so the transform matches bit-for-bit
  in layout and to float rounding in value.
* Every complex conv keeps its two real kernels (``w_re``, ``w_im``) flattened
  to 2-D ``(C_out, C_in*k_f*k_t)`` rows so they quantise along the reduction
  axis. Encoders keep the PyTorch kernel order.
* Complex transposed convs (stride 2 in frequency, padding 1 in time) are
  rewritten as an ordinary stride-1 polyphase conv: output channel
  ``r*C_out + c`` holds frequency phase ``r`` (rows 2m+r), the time taps are
  flipped and the frequency taps re-indexed, with ``ceil(k_f/2)-1`` rows of
  zero padding on both frequency edges. The runtime interleaves the phases and
  crops to ``2*(F_in-1)+k_f`` rows.
* Conv bias and eval-mode BatchNorm are folded into a per-(part, channel)
  affine ``scale``/``shift`` of shape (2, C): row 0 = real, row 1 = imaginary.
  Complex conv bias contributes ``b_re - b_im`` to the real part and
  ``b_re + b_im`` to the imaginary part.
* FSMN depthwise memory kernels are stored as (C, order) for ggml_ssm_conv.

Dtypes: ``f32`` keeps everything F32; ``f16`` stores the large conv and FSMN
matrices F16. STFT kernels, folded affines, SE layers, biases and FSMN memory
kernels are always F32. There is deliberately no Q8_0: FRCRN's cascaded UNets
amplify weight error, and even weight-only Q8_0 (round-tripped to F32) drops
the second UNet's output to cosine 0.23 against PyTorch (docs/frcrn-se.md).
"""

from __future__ import annotations

import argparse
import hashlib
import math
import sys
from pathlib import Path

import numpy as np

ARCH = "frcrn-se"
PREFIX = "frcrn_se"
UPSTREAM_REPO = "modelscope/ClearerVoice-Studio"
UPSTREAM_COMMIT = "6b3774dc"
HF_REPO = "alibabasglab/FRCRN_SE_16K"
HF_REVISION = "3766e6a6"

# UNet(model_depth=14) of clearvoice/models/frcrn_se/unet.py
N_LEVELS = 7
CHANNELS = 128
ENC_KF = [5, 5, 5, 5, 5, 5, 2]
DEC_KF = [2, 5, 5, 5, 6, 5, 5]
FSMN_ORDER = 20
SE_REDUCTION = 8
BN_EPS = 1e-5


def init_kernels(win_len, win_inc, fft_len, win_type, invers=False):
    """Verbatim numpy port of conv_stft.init_kernels (returns float32 arrays)."""
    from scipy.signal import get_window

    if win_type == "None" or win_type is None:
        window = np.ones(win_len)
    else:
        if win_type == "hanning":
            win_type = "hann"
        window = get_window(win_type, win_len, fftbins=True) ** 0.5
    fourier_basis = np.fft.rfft(np.eye(fft_len))[:win_len]
    kernel = np.concatenate([np.real(fourier_basis), np.imag(fourier_basis)], 1).T
    if invers:
        kernel = np.linalg.pinv(kernel).T
    kernel = kernel * window
    return kernel.astype(np.float32), window.astype(np.float32)


def fold_affine(sd, conv, bn):
    """Fold complex conv bias + eval BatchNorm into (2, C) scale/shift."""
    b_re = sd[f"{conv}_re.bias"].astype(np.float64)
    b_im = sd[f"{conv}_im.bias"].astype(np.float64)
    bias = [b_re - b_im, b_re + b_im]
    scale, shift = [], []
    for part, b in zip(("re", "im"), bias):
        g = sd[f"{bn}.bn_{part}.weight"].astype(np.float64)
        beta = sd[f"{bn}.bn_{part}.bias"].astype(np.float64)
        mean = sd[f"{bn}.bn_{part}.running_mean"].astype(np.float64)
        var = sd[f"{bn}.bn_{part}.running_var"].astype(np.float64)
        s = g / np.sqrt(var + BN_EPS)
        scale.append(s)
        shift.append(beta + (b - mean) * s)
    return np.stack(scale).astype(np.float32), np.stack(shift).astype(np.float32)


def polyphase_tconv(w):
    """ConvTranspose2d weight (Cin, Cout, kf, kt=2), stride (2,1), pad (0,1)
    -> Conv2d weight (2*Cout, Cin, Q, 2), Q = ceil(kf/2), for a stride-1 conv
    over the input zero-padded by Q-1 rows on both frequency edges."""
    cin, cout, kf, kt = w.shape
    assert kt == 2, w.shape
    q = (kf + 1) // 2
    p = q - 1
    out = np.zeros((2, cout, cin, q, 2), dtype=w.dtype)
    for r in range(2):
        for u in range(q):
            k = r + 2 * (p - u)
            if k >= kf:
                continue
            for m in range(2):
                # time: out[t] = in[t+1]*W[...,0] + in[t]*W[...,1]
                out[r, :, :, u, m] = w[:, :, k, 1 - m].T
    return out.reshape(2 * cout, cin, q, 2)


def build_tensors(sd):
    """Map an FRCRN state dict (numpy arrays, DCCRN keys) to GGUF tensors.

    Returns a list of (name, array, kind) where kind is 'big' (eligible for
    F16) or 'f32'.
    """
    # The UNet registers each layer twice (encoder{i} and encoders.{i}); the
    # values are identical. Use the ModuleList names, check the aliases.
    for k in list(sd.keys()):
        for single, multi in (("encoder", "encoders."), ("decoder", "decoders."), ("se_layer_enc", "se_layers_enc."),
                              ("se_layer_dec", "se_layers_dec."), ("fsmn_enc", "fsmn_enc."), ("fsmn_dec", "fsmn_dec.")):
            for unet in ("unet.", "unet2."):
                for i in range(N_LEVELS):
                    a = f"{unet}{single}{i}."
                    if k.startswith(a):
                        b = f"{unet}{multi}{i}." + k[len(a):]
                        if b in sd and not np.array_equal(sd[k], sd[b]):
                            raise ValueError(f"alias mismatch {k} vs {b}")
    out = []

    def add(name, arr, kind):
        out.append((name, np.ascontiguousarray(arr, dtype=np.float32), kind))

    def fsmn_unit(dst, src):
        add(f"{dst}.lin.w", sd[f"{src}.linear.weight"], "big")
        add(f"{dst}.lin.b", sd[f"{src}.linear.bias"], "f32")
        add(f"{dst}.proj.w", sd[f"{src}.project.weight"], "big")
        conv = sd[f"{src}.conv1.weight"]  # (C, 1, order, 1)
        add(f"{dst}.mem", conv.reshape(conv.shape[0], conv.shape[2]), "f32")

    def se(dst, src):
        for part, tag in (("r", "fc_r"), ("i", "fc_i")):
            add(f"{dst}.{part}1.w", sd[f"{src}.{tag}.0.weight"], "f32")
            add(f"{dst}.{part}1.b", sd[f"{src}.{tag}.0.bias"], "f32")
            add(f"{dst}.{part}2.w", sd[f"{src}.{tag}.2.weight"], "f32")
            add(f"{dst}.{part}2.b", sd[f"{src}.{tag}.2.bias"], "f32")

    for tag, unet in (("u1", "unet"), ("u2", "unet2")):
        for i in range(N_LEVELS):
            src = f"{unet}.encoders.{i}"
            for part in ("re", "im"):
                w = sd[f"{src}.conv.conv_{part}.weight"]
                add(f"{tag}.enc.{i}.w_{part}", w.reshape(w.shape[0], -1), "big")
            scale, shift = fold_affine(sd, f"{src}.conv.conv", f"{src}.bn")
            add(f"{tag}.enc.{i}.scale", scale, "f32")
            add(f"{tag}.enc.{i}.shift", shift, "f32")
            se(f"{tag}.se_enc.{i}", f"{unet}.se_layers_enc.{i}")
            if i > 0:
                for part in ("re", "im"):
                    fsmn_unit(f"{tag}.fsmn_enc.{i}.{part}", f"{unet}.fsmn_enc.{i}.fsmn_{part}_L1")

            src = f"{unet}.decoders.{i}"
            for part in ("re", "im"):
                w = polyphase_tconv(sd[f"{src}.transconv.tconv_{part}.weight"])
                add(f"{tag}.dec.{i}.w_{part}", w.reshape(w.shape[0], -1), "big")
            scale, shift = fold_affine(sd, f"{src}.transconv.tconv", f"{src}.bn")
            add(f"{tag}.dec.{i}.scale", scale, "f32")
            add(f"{tag}.dec.{i}.shift", shift, "f32")
            if i < N_LEVELS - 1:
                for part in ("re", "im"):
                    fsmn_unit(f"{tag}.fsmn_dec.{i}.{part}", f"{unet}.fsmn_dec.{i}.fsmn_{part}_L1")
            if i < N_LEVELS - 2:
                se(f"{tag}.se_dec.{i}", f"{unet}.se_layers_dec.{i}")
        for unit in ("re_L1", "im_L1", "re_L2", "im_L2"):
            fsmn_unit(f"{tag}.fsmn.{unit.lower().replace('_', '')}", f"{unet}.fsmn.fsmn_{unit}")
        lin = np.array([
            sd[f"{unet}.linear.conv_re.weight"].reshape(()), sd[f"{unet}.linear.conv_re.bias"].reshape(()),
            sd[f"{unet}.linear.conv_im.weight"].reshape(()), sd[f"{unet}.linear.conv_im.bias"].reshape(()),
        ])
        add(f"{tag}.out.w", lin, "f32")
    return out


def load_state_dict(path):
    import torch

    ck = torch.load(path, map_location="cpu", weights_only=True)
    sd = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    res = {}
    for k, v in sd.items():
        k = k[len("module."):] if k.startswith("module.") else k
        if k.endswith("num_batches_tracked"):
            continue
        res[k] = v.detach().cpu().numpy()
    return res


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_gguf(sd, output, dtype, source_sha=""):
    from gguf import GGMLQuantizationType, GGUFWriter

    win_len, win_inc, fft_len = 640, 320, 640
    fwd, window = init_kernels(win_len, win_inc, fft_len, "hanning")
    inv, _ = init_kernels(win_len, win_inc, fft_len, "hanning", invers=True)

    w = GGUFWriter(str(output), ARCH)
    w.add_name("FRCRN_SE_16K")
    w.add_description("ClearerVoice-Studio FRCRN speech enhancement (16 kHz), converted for CrispASR")
    w.add_string("general.license", "apache-2.0")
    w.add_string("general.source.url", f"https://github.com/{UPSTREAM_REPO}")
    w.add_string(f"{PREFIX}.upstream_commit", UPSTREAM_COMMIT)
    w.add_string(f"{PREFIX}.hf_repo", HF_REPO)
    w.add_string(f"{PREFIX}.hf_revision", HF_REVISION)
    if source_sha:
        w.add_string(f"{PREFIX}.checkpoint_sha256", source_sha)
    w.add_uint32(f"{PREFIX}.sample_rate", 16000)
    w.add_uint32(f"{PREFIX}.win_len", win_len)
    w.add_uint32(f"{PREFIX}.win_inc", win_inc)
    w.add_uint32(f"{PREFIX}.fft_len", fft_len)
    w.add_string(f"{PREFIX}.win_type", "hann-sqrt-periodic")
    w.add_uint32(f"{PREFIX}.n_levels", N_LEVELS)
    w.add_uint32(f"{PREFIX}.channels", CHANNELS)
    w.add_array(f"{PREFIX}.enc_kernel_f", ENC_KF)
    w.add_array(f"{PREFIX}.dec_kernel_f", DEC_KF)
    w.add_uint32(f"{PREFIX}.fsmn_order", FSMN_ORDER)
    w.add_uint32(f"{PREFIX}.se_reduction", SE_REDUCTION)
    w.add_float32(f"{PREFIX}.leaky_slope", 0.01)
    # decode_one_audio_frcrn_se_16k / FRCRN_SE_16K.yaml
    w.add_float32(f"{PREFIX}.decode_window_s", 1.0)
    w.add_float32(f"{PREFIX}.decode_stride_ratio", 0.75)
    w.add_float32(f"{PREFIX}.one_time_decode_length_s", 120.0)
    w.add_float32(f"{PREFIX}.norm_target_db", -25.0)
    ftype = {"f32": GGMLQuantizationType.F32, "f16": GGMLQuantizationType.F16}[dtype]
    w.add_file_type(int(ftype))

    w.add_tensor("stft.fwd", fwd)  # (642, 640)
    w.add_tensor("stft.inv", np.ascontiguousarray(inv.T))  # (640, 642)
    w.add_tensor("stft.window", window)
    n = 0
    for name, arr, kind in build_tensors(sd):
        if kind == "big" and dtype == "f16":
            w.add_tensor(name, arr.astype(np.float16))
        else:
            w.add_tensor(name, arr)
        n += 1
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return n


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True, help="FRCRN_SE_16K last_best_checkpoint.pt")
    ap.add_argument("--output", required=True)
    ap.add_argument("--dtype", choices=("f32", "f16"), default="f16")
    args = ap.parse_args(argv)
    sd = load_state_dict(args.checkpoint)
    n = write_gguf(sd, args.output, args.dtype, sha256(args.checkpoint))
    size = Path(args.output).stat().st_size
    print(f"wrote {args.output} ({size / 1e6:.1f} MB, {n + 3} tensors, dtype={args.dtype})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
