#!/usr/bin/env python3
"""
Convert third-intelligence/Haas-ja-v1.0 (a NeMo .nemo checkpoint with KDA predictor) → GGUF.

Haas couples:
  - 24-layer FastConformer encoder (identical to Parakeet-TDT 0.6B)
  - 2-layer Kimi Delta Attention (KDA) prediction network
  - TDT joint network (durations {0, 1, 2, 3, 4})
  - 3072-token SentencePiece Japanese vocabulary
"""

from __future__ import annotations

import argparse
import io
import sys
import tarfile
from pathlib import Path

import numpy as np

try:
    import gguf
except ImportError:
    sys.exit("pip install gguf")
try:
    import torch
except ImportError:
    sys.exit("pip install torch")
try:
    import sentencepiece as spm
except ImportError:
    sys.exit("pip install sentencepiece")
import yaml


def load_nemo(nemo_path: Path) -> dict:
    weights, config_str, spm_bytes = None, None, None
    print(f"Reading {nemo_path}...")
    with tarfile.open(nemo_path, "r") as tar:
        for m in tar.getmembers():
            if m.name.endswith("model_weights.ckpt"):
                f = tar.extractfile(m)
                weights = torch.load(f, map_location="cpu", weights_only=False)
            elif m.name.endswith("model_config.yaml"):
                f = tar.extractfile(m)
                config_str = f.read().decode("utf-8")
            elif m.name.endswith("tokenizer.model"):
                f = tar.extractfile(m)
                spm_bytes = f.read()

    if "state_dict" in weights:
        weights = weights["state_dict"]
    return {"weights": weights, "config_str": config_str, "spm_bytes": spm_bytes}


def load_extracted_dir(dir_path: Path) -> dict:
    print(f"Loading from extracted directory {dir_path}...")
    ckpt_path = dir_path / "model_weights.ckpt"
    weights = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    if "state_dict" in weights:
        weights = weights["state_dict"]
    config_str = (dir_path / "model_config.yaml").read_text("utf-8")
    spm_file = next(dir_path.glob("*tokenizer.model"))
    spm_bytes = spm_file.read_bytes()
    return {"weights": weights, "config_str": config_str, "spm_bytes": spm_bytes}


def remap_name(n: str) -> str | None:
    if n.endswith("num_batches_tracked"):
        return None

    # Preprocessor
    if n == "preprocessor.featurizer.fb":
        return "preprocessor.fb"
    if n == "preprocessor.featurizer.window":
        return "preprocessor.window"

    # Pre-encode
    if n.startswith("encoder.pre_encode."):
        return n.replace("encoder.pre_encode.", "encoder.pre.")

    # Conformer layers
    if n.startswith("encoder.layers."):
        rest = n[len("encoder.layers.") :]
        layer_id, sub = rest.split(".", 1)
        sub = (
            sub.replace("feed_forward1", "ff1")
            .replace("feed_forward2", "ff2")
            .replace("norm_feed_forward1", "norm_ff1")
            .replace("norm_feed_forward2", "norm_ff2")
            .replace("norm_self_att", "norm_attn")
            .replace("self_attn.linear_q", "attn.q")
            .replace("self_attn.linear_k", "attn.k")
            .replace("self_attn.linear_v", "attn.v")
            .replace("self_attn.linear_out", "attn.out")
            .replace("self_attn.linear_pos", "attn.pos")
            .replace("self_attn.pos_bias_u", "attn.pos_bias_u")
            .replace("self_attn.pos_bias_v", "attn.pos_bias_v")
            .replace("conv.pointwise_conv1", "conv.pw1")
            .replace("conv.depthwise_conv", "conv.dw")
            .replace("conv.pointwise_conv2", "conv.pw2")
            .replace("conv.batch_norm", "conv.bn")
        )
        return f"encoder.layers.{layer_id}.{sub}"

    # Decoder / Predictor (KDA)
    if n == "decoder.prediction.embed.weight":
        return "decoder.embed.weight"
    if n.startswith("decoder.prediction.dec_rnn.norms."):
        idx = n.split(".")[4]
        param = n.split(".")[5]
        return f"decoder.kda.norm.{idx}.{param}"
    if n.startswith("decoder.prediction.dec_rnn.final_norm."):
        param = n.split(".")[4]
        return f"decoder.kda.final_norm.{param}"
    if n.startswith("decoder.prediction.dec_rnn.layers."):
        parts = n.split(".")
        layer_idx = parts[4]
        sub = ".".join(parts[5:])
        if sub in ("q_proj.weight", "k_proj.weight", "v_proj.weight"):
            return f"decoder.kda.{layer_idx}.{sub}"
        if sub in ("q_conv1d.weight", "k_conv1d.weight", "v_conv1d.weight"):
            conv_name = sub.split("_")[0] + "_conv.weight"
            return f"decoder.kda.{layer_idx}.{conv_name}"
        if sub in ("f_proj.0.weight", "f_proj.1.weight"):
            return f"decoder.kda.{layer_idx}.{sub}"
        if sub == "b_proj.weight":
            return f"decoder.kda.{layer_idx}.b_proj.weight"
        if sub == "A_log":
            return f"decoder.kda.{layer_idx}.a_log"
        if sub == "dt_bias":
            return f"decoder.kda.{layer_idx}.dt_bias"
        if sub in ("g_proj.0.weight", "g_proj.1.weight", "g_proj.1.bias"):
            return f"decoder.kda.{layer_idx}.{sub}"
        if sub == "o_norm.weight":
            return f"decoder.kda.{layer_idx}.o_norm.weight"
        if sub == "o_proj.weight":
            return f"decoder.kda.{layer_idx}.o_proj.weight"

    # Joint
    if n == "joint.enc.weight":
        return "joint.enc.weight"
    if n == "joint.enc.bias":
        return "joint.enc.bias"
    if n == "joint.pred.weight":
        return "joint.pred.weight"
    if n == "joint.pred.bias":
        return "joint.pred.bias"
    if n == "joint.joint_net.2.weight":
        return "joint.out.weight"
    if n == "joint.joint_net.2.bias":
        return "joint.out.bias"

    print(f"  [warn] unmapped tensor: {n}", file=sys.stderr)
    return None


def is_f32_tensor(name: str, shape: tuple[int, ...]) -> bool:
    if name in ("preprocessor.fb", "preprocessor.window"):
        return True
    if any(s in name for s in ("pos_bias_u", "pos_bias_v", "conv.bn", "a_log", "dt_bias")):
        return True
    if len(shape) <= 1:
        return True
    return False


def convert(data: dict, out_path: Path, quant: str | None = None) -> None:
    sd = data["weights"]
    cfg = yaml.safe_load(data["config_str"])

    sp = spm.SentencePieceProcessor()
    sp.LoadFromSerializedProto(data["spm_bytes"])
    vocab = [sp.id_to_piece(i) for i in range(sp.get_piece_size())]
    print(f"  Loaded {len(vocab)} SentencePiece tokens")

    prep = cfg.get("preprocessor", {})
    enc_cfg = cfg.get("encoder", {})
    dec_cfg = cfg.get("decoder", {})
    pred_cfg = dec_cfg.get("prednet", {})
    joint_cfg = cfg.get("joint", {})
    joint_net = joint_cfg.get("jointnet", {})

    feat_in = enc_cfg.get("feat_in", prep.get("features", 128))
    sr = prep.get("sample_rate", 16000)
    n_fft = prep.get("n_fft", 512)
    ws = prep.get("window_size", 0.025)
    wst = prep.get("window_stride", 0.01)
    d_model = enc_cfg.get("d_model", 1024)
    n_layers = enc_cfg.get("n_layers", 24)
    n_heads = enc_cfg.get("n_heads", 8)
    head_dim = d_model // n_heads
    ff_dim = enc_cfg.get("ff_expansion_factor", 4) * d_model
    subsampling_factor = enc_cfg.get("subsampling_factor", 8)
    subsampling_channels = enc_cfg.get("subsampling_conv_channels", 256)
    conv_kernel = enc_cfg.get("conv_kernel_size", 9)
    xscaling = bool(enc_cfg.get("xscaling", True))
    pred_hidden = pred_cfg.get("pred_hidden", 640)
    pred_layers = pred_cfg.get("pred_rnn_layers", 2)
    joint_hidden = joint_net.get("joint_hidden", 640)

    decoding_cfg = cfg.get("decoding", {})
    tdt_durations = decoding_cfg.get("durations", [0, 1, 2, 3, 4])
    n_tdt_durations = len(tdt_durations)

    print(f"Writing GGUF to {out_path}...")
    writer = gguf.GGUFWriter(str(out_path), arch="haas")

    writer.add_name("Haas-ja-v1.0")
    writer.add_string("parakeet.language", "ja")
    writer.add_string("general.source.huggingface.repository", "third-intelligence/Haas-ja-v1.0")
    writer.add_string("general.source.url", "https://huggingface.co/third-intelligence/Haas-ja-v1.0")
    writer.add_string("general.license", "apache-2.0")

    writer.add_uint32("parakeet.sample_rate", sr)
    writer.add_uint32("parakeet.n_mels", feat_in)
    writer.add_uint32("parakeet.n_fft", n_fft)
    writer.add_uint32("parakeet.win_length", int(ws * sr))
    writer.add_uint32("parakeet.hop_length", int(wst * sr))
    writer.add_uint32("parakeet.d_model", d_model)
    writer.add_uint32("parakeet.n_layers", n_layers)
    writer.add_uint32("parakeet.n_heads", n_heads)
    writer.add_uint32("parakeet.head_dim", head_dim)
    writer.add_uint32("parakeet.ff_dim", ff_dim)
    writer.add_uint32("parakeet.subsampling_factor", subsampling_factor)
    writer.add_uint32("parakeet.subsampling_channels", subsampling_channels)
    writer.add_uint32("parakeet.conv_kernel", conv_kernel)
    writer.add_bool("parakeet.xscaling", xscaling)
    writer.add_uint32("parakeet.pred_hidden", pred_hidden)
    writer.add_uint32("parakeet.pred_layers", pred_layers)
    writer.add_uint32("parakeet.joint_hidden", joint_hidden)
    writer.add_uint32("parakeet.vocab_size", len(vocab))
    writer.add_uint32("parakeet.blank_id", len(vocab))
    writer.add_uint32("parakeet.n_tdt_durations", n_tdt_durations)
    writer.add_array("parakeet.tdt_durations", tdt_durations)
    writer.add_uint32("parakeet.frame_dur_cs", int(round(wst * subsampling_factor * 100)))

    # KDA hyperparameters
    writer.add_uint32("haas.kda_num_heads", pred_cfg.get("kda_num_heads", 5))
    writer.add_uint32("haas.kda_head_dim", pred_cfg.get("kda_head_dim", 64))
    writer.add_uint32("haas.kda_conv_size", pred_cfg.get("kda_conv_size", 4))
    writer.add_bool("haas.kda_final_norm", pred_cfg.get("kda_final_norm", True))

    writer.add_array("tokenizer.ggml.tokens", vocab)

    quant_type = None
    if quant:
        m = {"q4_k": gguf.GGMLQuantizationType.Q4_K, "q8_0": gguf.GGMLQuantizationType.Q8_0}
        quant_type = m.get(quant.lower())

    n_written = 0
    layers_seen = set()
    layers_with_dw_bias = set()

    for name in sorted(sd.keys()):
        gguf_name = remap_name(name)
        if gguf_name is None:
            continue
        t = sd[name].cpu().float().numpy()

        # If 3D conv weight in KDA, squeeze out channel dim 1: (320, 1, 4) -> (320, 4)
        if "_conv.weight" in gguf_name and t.ndim == 3 and t.shape[1] == 1:
            t = t.squeeze(1)

        raw_dtype = None
        if is_f32_tensor(gguf_name, t.shape):
            t = t.astype(np.float32)
        elif quant_type is not None and len(t.shape) >= 2 and t.shape[-1] % 32 == 0:
            try:
                t = gguf.quantize(t, quant_type)
                raw_dtype = quant_type
            except Exception:
                t = t.astype(np.float16)
        else:
            t = t.astype(np.float16)

        writer.add_tensor(gguf_name, t, raw_dtype=raw_dtype)
        n_written += 1

        if gguf_name.startswith("encoder.layers.") and ".conv.dw.weight" in gguf_name:
            li = int(gguf_name.split(".")[2])
            layers_seen.add(li)
        if gguf_name.startswith("encoder.layers.") and ".conv.dw.bias" in gguf_name:
            li = int(gguf_name.split(".")[2])
            layers_with_dw_bias.add(li)

    for li in sorted(layers_seen - layers_with_dw_bias):
        zero_bias = np.zeros(d_model, dtype=np.float32)
        writer.add_tensor(f"encoder.layers.{li}.conv.dw.bias", zero_bias)
        n_written += 1

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    sz_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"Successfully converted {n_written} tensors to {out_path} ({sz_mb:.1f} MB).")


def main():
    sys.stdout.reconfigure(line_buffering=True)
    parser = argparse.ArgumentParser(description="Convert Haas-ja-v1.0 to GGUF")
    parser.add_argument("--nemo", type=Path, help="Path to Haas-ja-v1.0.nemo archive")
    parser.add_argument("--dir", type=Path, help="Path to extracted directory")
    parser.add_argument("--output", type=Path, required=True, help="Output GGUF path")
    parser.add_argument("--quant", type=str, default=None, choices=["f16", "q8_0", "q4_k"])
    args = parser.parse_args()

    if args.dir:
        data = load_extracted_dir(args.dir)
    elif args.nemo:
        data = load_nemo(args.nemo)
    else:
        sys.exit("Must supply --nemo or --dir")

    quant = None if args.quant == "f16" else args.quant
    convert(data, args.output, quant=quant)


if __name__ == "__main__":
    main()
