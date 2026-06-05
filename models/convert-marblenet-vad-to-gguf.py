#!/usr/bin/env python3
"""Convert NVIDIA MarbleNet VAD (.nemo) to GGUF for CrispASR.

Architecture: 1D time-channel separable CNN (6 Jasper blocks, 91.5K params).
Input: 80-bin mel at 16kHz. Output: per-frame speech probability (20ms).

Usage:
    python models/convert-marblenet-vad-to-gguf.py \
        --input nvidia/Frame_VAD_Multilingual_MarbleNet_v2.0 \
        --output marblenet-vad.gguf
"""

import argparse
import sys
import tarfile
from pathlib import Path

import numpy as np
import torch
import yaml

try:
    from gguf import GGUFWriter, GGMLQuantizationType
except ImportError:
    print("Error: gguf package not found. Install with: pip install gguf")
    sys.exit(1)


def load_nemo(path: str):
    """Load .nemo file (tarball with model_weights.ckpt + model_config.yaml)."""
    p = Path(path)
    if p.is_dir():
        ckpt = p / "model_weights.ckpt"
        cfg_path = p / "model_config.yaml"
    elif p.suffix == ".nemo":
        import tempfile
        tmpdir = tempfile.mkdtemp()
        with tarfile.open(str(p)) as t:
            t.extractall(tmpdir)
        ckpt = Path(tmpdir) / "model_weights.ckpt"
        cfg_path = Path(tmpdir) / "model_config.yaml"
    else:
        # Try HF download
        from huggingface_hub import hf_hub_download
        nemo_path = hf_hub_download(path, "frame_vad_multilingual_marblenet_v2.0.nemo",
                                     local_dir="/tmp/marblenet")
        import tempfile
        tmpdir = tempfile.mkdtemp()
        with tarfile.open(nemo_path) as t:
            t.extractall(tmpdir)
        ckpt = Path(tmpdir) / "model_weights.ckpt"
        cfg_path = Path(tmpdir) / "model_config.yaml"

    sd = torch.load(str(ckpt), map_location="cpu", weights_only=True)
    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return sd, cfg


def fuse_bn(conv_w, bn_weight, bn_bias, bn_mean, bn_var, eps=1e-5):
    """Fuse BatchNorm into convolution weights: W_fused = W * gamma/sqrt(var+eps),
    b_fused = gamma * (0 - mean) / sqrt(var+eps) + beta."""
    std = torch.sqrt(bn_var + eps)
    scale = bn_weight / std
    # conv_w shape: [C_out, C_in, K] or [C_out, 1, K] (depthwise)
    if conv_w.ndim == 3:
        w_fused = conv_w * scale.view(-1, 1, 1)
    else:
        w_fused = conv_w * scale.view(-1, 1)
    b_fused = -bn_mean * scale + bn_bias
    return w_fused, b_fused


def main():
    parser = argparse.ArgumentParser(description="Convert MarbleNet VAD to GGUF")
    parser.add_argument("--input", required=True, help=".nemo file, extracted dir, or HF model ID")
    parser.add_argument("--output", required=True, help="Output GGUF path")
    args = parser.parse_args()

    sd, cfg = load_nemo(args.input)

    # Extract config
    prep = cfg.get("preprocessor", {})
    jasper = cfg.get("encoder", {}).get("jasper", [])
    dec = cfg.get("decoder", {})

    print(f"MarbleNet VAD")
    print(f"  Preprocessor: {prep.get('features', 80)} mels, {prep.get('n_fft', 512)} FFT, "
          f"{prep.get('window_size', 0.025)*1000:.0f}ms window, {prep.get('window_stride', 0.01)*1000:.0f}ms stride")
    print(f"  Encoder: {len(jasper)} blocks")
    for i, j in enumerate(jasper):
        print(f"    [{i}] k={j['kernel']}, f={j['filters']}, r={j.get('repeat',1)}, "
              f"s={j['stride']}, d={j['dilation']}, res={j.get('residual',False)}, sep={j.get('separable',False)}")
    print(f"  Decoder: {dec.get('hidden_size', 128)} → {dec.get('num_classes', 2)}")
    print(f"  Tensors: {len(sd)}, Params: {sum(v.numel() for v in sd.values()):,}")

    # Write GGUF
    outfile = Path(args.output)
    writer = GGUFWriter(str(outfile), "marblenet_vad", use_temp_file=True)

    writer.add_name("marblenet-vad-multilingual-v2")
    sr = prep.get("sample_rate", 16000)
    n_mels = prep.get("features", 80)
    n_fft = prep.get("n_fft", 512)
    win_len = int(prep.get("window_size", 0.025) * sr)
    hop_len = int(prep.get("window_stride", 0.01) * sr)
    writer.add_uint32("marblenet.sample_rate", sr)
    writer.add_uint32("marblenet.n_mels", n_mels)
    writer.add_uint32("marblenet.n_fft", n_fft)
    writer.add_uint32("marblenet.win_length", win_len)
    writer.add_uint32("marblenet.hop_length", hop_len)
    writer.add_uint32("marblenet.n_blocks", len(jasper))
    writer.add_uint32("marblenet.num_classes", dec.get("num_classes", 2))

    # Store block configs as metadata
    for i, j in enumerate(jasper):
        writer.add_uint32(f"marblenet.block.{i}.filters", j["filters"])
        writer.add_uint32(f"marblenet.block.{i}.kernel", j["kernel"][0])
        writer.add_uint32(f"marblenet.block.{i}.stride", j["stride"][0])
        writer.add_uint32(f"marblenet.block.{i}.dilation", j["dilation"][0])
        writer.add_uint32(f"marblenet.block.{i}.repeat", j.get("repeat", 1))
        writer.add_uint32(f"marblenet.block.{i}.residual", 1 if j.get("residual", False) else 0)
        writer.add_uint32(f"marblenet.block.{i}.separable", 1 if j.get("separable", False) else 0)

    # Preprocessor: mel filterbank + window (stored in checkpoint)
    fb = sd.get("preprocessor.featurizer.fb")
    win = sd.get("preprocessor.featurizer.window")
    if fb is not None:
        writer.add_tensor("mel_filters", fb.squeeze(0).numpy().astype(np.float32),
                          raw_dtype=GGMLQuantizationType.F32)
    if win is not None:
        writer.add_tensor("mel_window", win.numpy().astype(np.float32),
                          raw_dtype=GGMLQuantizationType.F32)

    mapped = 2  # mel_filters + mel_window

    # Encoder blocks: fuse BatchNorm into conv weights
    for blk_idx in range(len(jasper)):
        prefix = f"encoder.encoder.{blk_idx}"
        j = jasper[blk_idx]
        repeat = j.get("repeat", 1)
        separable = j.get("separable", False)
        residual = j.get("residual", False)

        for sub in range(repeat):
            # Separable conv = depthwise (groups=C_in) + pointwise (1x1)
            # mconv indices: sub*3+0=depthwise, sub*3+1=pointwise, sub*3+2=batchnorm
            # For non-separable (block 5): mconv.0=conv, mconv.1=batchnorm
            if separable:
                dw_idx = sub * 5  # 0, 5
                pw_idx = sub * 5 + 1
                bn_idx = sub * 5 + 2
            else:
                # block 5 (1x1 conv, not separable): mconv.0=conv, mconv.1=bn
                dw_idx = None
                pw_idx = sub * 3  # 0
                bn_idx = sub * 3 + 1

            # Get BN tensors
            bn_w = sd.get(f"{prefix}.mconv.{bn_idx}.weight")
            bn_b = sd.get(f"{prefix}.mconv.{bn_idx}.bias")
            bn_m = sd.get(f"{prefix}.mconv.{bn_idx}.running_mean")
            bn_v = sd.get(f"{prefix}.mconv.{bn_idx}.running_var")

            if separable and dw_idx is not None:
                # Depthwise conv
                dw_w = sd.get(f"{prefix}.mconv.{dw_idx}.conv.weight")
                if dw_w is not None:
                    gguf_name = f"enc.{blk_idx}.sub.{sub}.dw.weight"
                    writer.add_tensor(gguf_name, dw_w.numpy().astype(np.float32),
                                      raw_dtype=GGMLQuantizationType.F32)
                    mapped += 1

            # Pointwise conv (or regular conv for non-separable)
            pw_key = f"{prefix}.mconv.{pw_idx}.conv.weight"
            pw_w = sd.get(pw_key)
            if pw_w is None:
                pw_key = f"{prefix}.mconv.{pw_idx}.weight"  # non-separable might not have .conv
                pw_w = sd.get(pw_key)

            if pw_w is not None and bn_w is not None:
                # Fuse BN into pointwise/conv
                w_fused, b_fused = fuse_bn(pw_w, bn_w, bn_b, bn_m, bn_v)
                gguf_name_w = f"enc.{blk_idx}.sub.{sub}.pw.weight"
                gguf_name_b = f"enc.{blk_idx}.sub.{sub}.pw.bias"
                writer.add_tensor(gguf_name_w, w_fused.numpy().astype(np.float32),
                                  raw_dtype=GGMLQuantizationType.F32)
                writer.add_tensor(gguf_name_b, b_fused.numpy().astype(np.float32),
                                  raw_dtype=GGMLQuantizationType.F32)
                mapped += 2

        # Residual connection (1x1 conv + BN)
        if residual:
            res_w = sd.get(f"{prefix}.res.0.0.conv.weight")
            res_bn_w = sd.get(f"{prefix}.res.0.1.weight")
            res_bn_b = sd.get(f"{prefix}.res.0.1.bias")
            res_bn_m = sd.get(f"{prefix}.res.0.1.running_mean")
            res_bn_v = sd.get(f"{prefix}.res.0.1.running_var")
            if res_w is not None and res_bn_w is not None:
                w_fused, b_fused = fuse_bn(res_w, res_bn_w, res_bn_b, res_bn_m, res_bn_v)
                writer.add_tensor(f"enc.{blk_idx}.res.weight", w_fused.numpy().astype(np.float32),
                                  raw_dtype=GGMLQuantizationType.F32)
                writer.add_tensor(f"enc.{blk_idx}.res.bias", b_fused.numpy().astype(np.float32),
                                  raw_dtype=GGMLQuantizationType.F32)
                mapped += 2

    # Decoder: Linear(128→2)
    dec_w = sd.get("decoder.layer0.weight")
    dec_b = sd.get("decoder.layer0.bias")
    if dec_w is not None:
        writer.add_tensor("decoder.weight", dec_w.numpy().astype(np.float32),
                          raw_dtype=GGMLQuantizationType.F32)
        writer.add_tensor("decoder.bias", dec_b.numpy().astype(np.float32),
                          raw_dtype=GGMLQuantizationType.F32)
        mapped += 2

    print(f"\nWriting {mapped} tensors to {outfile}...")
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    size_kb = outfile.stat().st_size / 1024
    print(f"Done! {outfile} ({size_kb:.0f} KB, {mapped} tensors)")


if __name__ == "__main__":
    main()
