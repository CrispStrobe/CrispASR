#!/usr/bin/env python3
"""
Convert any Wav2Vec2ForCTC model (HuggingFace) → GGUF F16.

Linear/attention/FFN weights are saved as F16.  CNN weights, biases, and
LayerNorm parameters are saved as F32.

    pip install gguf transformers torch
    python convert_wav2vec2_to_gguf.py \\
        --model-dir /path/to/wav2vec2-model \\
        --output    model-f16.gguf

    # Optional: quantize transformer weights (Python gguf library):
    python quantize_gguf.py model-f16.gguf model-q5_0.gguf Q5_0

Tensor naming convention (mirrors the C++ loader):
  cnn.{i}.conv.{weight|bias}      F32   CNN feature extractor
  cnn.{i}.norm.{weight|bias}      F32   GroupNorm (InstanceNorm) per layer
  feat_proj.ln.{weight|bias}      F32   Feature-projection LayerNorm
  feat_proj.{weight|bias}         F16   Feature-projection linear
  pos_conv.{weight|bias}          F32   Positional conv embedding (weight_norm removed)
  enc.ln.{weight|bias}            F32   Encoder global LayerNorm
  enc.{i}.ln1.{weight|bias}       F32   Pre-attention LayerNorm
  enc.{i}.attn.{q|k|v|out}.weight F16   Attention projections
  enc.{i}.attn.{q|k|v|out}.bias   F32
  enc.{i}.ln2.{weight|bias}       F32   Pre-FFN LayerNorm
  enc.{i}.ffn.fc{1|2}.weight      F16   FFN dense layers
  enc.{i}.ffn.fc{1|2}.bias        F32
  lm_head.{weight|bias}           F32   CTC output head
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

try:
    import gguf
except ImportError:
    sys.exit(
        "gguf package not found.  Install with:\n"
        "  pip install gguf\n"
        "or clone llama.cpp and run:\n"
        "  pip install -e /path/to/llama.cpp/gguf-py"
    )

try:
    from transformers import AutoModelForCTC, AutoConfig
except ImportError:
    sys.exit("transformers package not found.  Install with: pip install transformers")


ARCH = "wav2vec2"


def normalize_state_dict(sd: dict) -> dict:
    """Remap Data2Vec/HuBERT state dict keys to wav2vec2.* prefix."""
    out = {}
    for k, v in sd.items():
        nk = k
        for prefix in ("data2vec_audio.", "hubert."):
            if k.startswith(prefix):
                nk = "wav2vec2." + k[len(prefix):]
                break
        out[nk] = v
    return out


# ---------------------------------------------------------------------------
# Weight-norm helpers
# ---------------------------------------------------------------------------


def get_backbone(model):
    """Get the audio backbone (wav2vec2/data2vec_audio/hubert) from a ForCTC model."""
    for attr in ("wav2vec2", "data2vec_audio", "hubert"):
        if hasattr(model, attr):
            return getattr(model, attr)
    raise ValueError(f"Cannot find backbone in {type(model).__name__}")


def remove_weight_norm(model) -> None:
    """Remove weight_norm from pos_conv_embed so state_dict has plain .weight."""
    backbone = get_backbone(model)
    pce = backbone.encoder.pos_conv_embed
    # Wav2Vec2 has pce.conv; Data2Vec/HuBERT has pce.layers[i].conv
    convs = []
    if hasattr(pce, "conv"):
        convs = [pce.conv]
    elif hasattr(pce, "layers"):
        for layer in pce.layers:
            if hasattr(layer, "conv"):
                convs.append(layer.conv)
    for conv in convs:
        try:
            torch.nn.utils.remove_weight_norm(conv)
            print(f"  Removed weight_norm from {conv}")
        except ValueError:
            pass
    if not convs:
        print("  pos_conv_embed: no conv found, skipping weight_norm removal")


# ---------------------------------------------------------------------------
# Dtype helpers
# ---------------------------------------------------------------------------


def f32(t: torch.Tensor) -> np.ndarray:
    return t.detach().float().cpu().numpy().astype(np.float32)


def f16(t: torch.Tensor) -> np.ndarray:
    return t.detach().float().cpu().numpy().astype(np.float16)


# ---------------------------------------------------------------------------
# Layer writers
# ---------------------------------------------------------------------------


def write_cnn_layers(w: gguf.GGUFWriter, sd: dict, n_layers: int) -> None:
    # In HF wav2vec2 with feat_extract_norm="group":
    #   layer 0  → Wav2Vec2GroupNormConvLayer  (has layer_norm)
    #   layers 1+→ Wav2Vec2NoLayerNormConvLayer (no norm at all)
    for i in range(n_layers):
        p = f"wav2vec2.feature_extractor.conv_layers.{i}"
        w.add_tensor(f"cnn.{i}.conv.weight", f32(sd[f"{p}.conv.weight"]))
        if f"{p}.conv.bias" in sd:
            w.add_tensor(f"cnn.{i}.conv.bias", f32(sd[f"{p}.conv.bias"]))
        has_norm = f"{p}.layer_norm.weight" in sd
        # Store a flag (1 = has norm, 0 = no norm) in metadata
        w.add_uint32(f"wav2vec2.cnn_has_norm_{i}", int(has_norm))
        if has_norm:
            w.add_tensor(f"cnn.{i}.norm.weight", f32(sd[f"{p}.layer_norm.weight"]))
            w.add_tensor(f"cnn.{i}.norm.bias", f32(sd[f"{p}.layer_norm.bias"]))
        cout = sd[f"{p}.conv.weight"].shape[0]
        kern = sd[f"{p}.conv.weight"].shape[2]
        print(f"  cnn.{i}: out={cout} kernel={kern} norm={'yes' if has_norm else 'no'}")


def write_feature_projection(w: gguf.GGUFWriter, sd: dict) -> None:
    pfx = "wav2vec2.feature_projection"
    w.add_tensor("feat_proj.ln.weight", f32(sd[f"{pfx}.layer_norm.weight"]))
    w.add_tensor("feat_proj.ln.bias", f32(sd[f"{pfx}.layer_norm.bias"]))
    w.add_tensor("feat_proj.weight", f16(sd[f"{pfx}.projection.weight"]))
    w.add_tensor("feat_proj.bias", f32(sd[f"{pfx}.projection.bias"]))
    print(f"  feat_proj: {sd[f'{pfx}.projection.weight'].shape}")


def write_pos_conv(w: gguf.GGUFWriter, sd: dict, model) -> None:
    # weight_norm (old or new parametrize API) stores raw g/v in state_dict;
    # access the module attribute directly to get the reconstructed weight.
    backbone = get_backbone(model)
    pce = backbone.encoder.pos_conv_embed
    if hasattr(pce, "conv"):
        conv = pce.conv
        pos_w = conv.weight.detach().float()
        pos_b = conv.bias.detach().float()
        w.add_tensor("pos_conv.weight", pos_w.numpy().astype(np.float32))
        w.add_tensor("pos_conv.bias", pos_b.numpy().astype(np.float32))
        print(f"  pos_conv: {tuple(pos_w.shape)}")
    elif hasattr(pce, "layers"):
        # Data2Vec/HuBERT: pos_conv_embed.layers[i].conv — store ALL layers
        n_pos_layers = len(pce.layers)
        w.add_uint32(f"{ARCH}.num_pos_conv_layers", n_pos_layers)
        for li, layer in enumerate(pce.layers):
            conv = layer.conv
            pos_w = conv.weight.detach().float()
            pos_b = conv.bias.detach().float()
            w.add_tensor(f"pos_conv.{li}.weight", pos_w.numpy().astype(np.float32))
            w.add_tensor(f"pos_conv.{li}.bias", pos_b.numpy().astype(np.float32))
            if li == 0:
                # Also store as pos_conv.weight for backward compat with single-layer models
                w.add_tensor("pos_conv.weight", pos_w.numpy().astype(np.float32))
                w.add_tensor("pos_conv.bias", pos_b.numpy().astype(np.float32))
            print(f"  pos_conv.{li}: {tuple(pos_w.shape)} groups={conv.groups}")


def write_encoder_global_ln(w: gguf.GGUFWriter, sd: dict) -> None:
    pfx = "wav2vec2.encoder.layer_norm"
    w.add_tensor("enc.ln.weight", f32(sd[f"{pfx}.weight"]))
    w.add_tensor("enc.ln.bias", f32(sd[f"{pfx}.bias"]))


def write_encoder_layers(w: gguf.GGUFWriter, sd: dict, n_layers: int) -> None:
    for i in range(n_layers):
        p = f"wav2vec2.encoder.layers.{i}"

        # Pre-attention LayerNorm
        w.add_tensor(f"enc.{i}.ln1.weight", f32(sd[f"{p}.layer_norm.weight"]))
        w.add_tensor(f"enc.{i}.ln1.bias", f32(sd[f"{p}.layer_norm.bias"]))

        # Attention projections (F16 or F32 depending on --dtype)
        for proj, short in [
            ("q_proj", "q"),
            ("k_proj", "k"),
            ("v_proj", "v"),
            ("out_proj", "out"),
        ]:
            w.add_tensor(
                f"enc.{i}.attn.{short}.weight", wt(sd[f"{p}.attention.{proj}.weight"])
            )
            w.add_tensor(
                f"enc.{i}.attn.{short}.bias", f32(sd[f"{p}.attention.{proj}.bias"])
            )

        # Pre-FFN LayerNorm
        w.add_tensor(f"enc.{i}.ln2.weight", f32(sd[f"{p}.final_layer_norm.weight"]))
        w.add_tensor(f"enc.{i}.ln2.bias", f32(sd[f"{p}.final_layer_norm.bias"]))

        # FFN (F16 or F32 depending on --dtype)
        w.add_tensor(
            f"enc.{i}.ffn.fc1.weight",
            wt(sd[f"{p}.feed_forward.intermediate_dense.weight"]),
        )
        w.add_tensor(
            f"enc.{i}.ffn.fc1.bias",
            f32(sd[f"{p}.feed_forward.intermediate_dense.bias"]),
        )
        w.add_tensor(
            f"enc.{i}.ffn.fc2.weight", wt(sd[f"{p}.feed_forward.output_dense.weight"])
        )
        w.add_tensor(
            f"enc.{i}.ffn.fc2.bias", f32(sd[f"{p}.feed_forward.output_dense.bias"])
        )

        print(f"  enc.{i}: ok")


def write_lm_head(w: gguf.GGUFWriter, sd: dict) -> None:
    w.add_tensor("lm_head.weight", f32(sd["lm_head.weight"]))
    w.add_tensor("lm_head.bias", f32(sd["lm_head.bias"]))
    print(f"  lm_head: {sd['lm_head.weight'].shape}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert Wav2Vec2ForCTC HuggingFace model to GGUF"
    )
    p.add_argument(
        "--model-dir",
        type=Path,
        required=True,
        help="Path to HuggingFace Wav2Vec2ForCTC model directory (or HF model ID)",
    )
    p.add_argument("--output", type=Path, required=True, help="Output .gguf file path")
    p.add_argument(
        "--dtype",
        choices=["f16", "f32"],
        default="f16",
        help="Weight dtype for linear/attention/FFN tensors (default: f16)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    model_dir = args.model_dir
    out_path = args.output.resolve()

    # Set the weight dtype function globally
    global wt
    wt = f32 if args.dtype == "f32" else f16
    print(f"Weight dtype: {args.dtype}")

    # Accept both local paths and HF model IDs

    print(f"Loading model from: {model_dir}")
    model = AutoModelForCTC.from_pretrained(str(model_dir), trust_remote_code=True)
    model.eval()
    print(f"  Model class: {type(model).__name__}")

    print("Preprocessing weights:")
    remove_weight_norm(model)

    config = model.config
    sd = normalize_state_dict(model.state_dict())

    # Read vocab (id → token) — try local path first, then HF cache
    vocab_path = model_dir / "vocab.json"
    if vocab_path.exists():
        with open(vocab_path, encoding="utf-8") as f:
            vocab_dict = json.load(f)
    else:
        # Download vocab.json from HF
        from huggingface_hub import hf_hub_download

        vp = hf_hub_download(repo_id=str(model_dir), filename="vocab.json")
        with open(vp, encoding="utf-8") as f:
            vocab_dict = json.load(f)
    id_to_token = {v: k for k, v in vocab_dict.items()}
    vocab_list = [id_to_token.get(i, f"<{i}>") for i in range(config.vocab_size)]

    # --- GGUF writer ---
    print(f"\nWriting GGUF: {out_path}")
    writer = gguf.GGUFWriter(str(out_path), arch=ARCH)

    # Scalar metadata
    writer.add_uint32(f"{ARCH}.vocab_size", int(config.vocab_size))
    writer.add_uint32(f"{ARCH}.hidden_size", int(config.hidden_size))
    writer.add_uint32(f"{ARCH}.num_hidden_layers", int(config.num_hidden_layers))
    writer.add_uint32(f"{ARCH}.num_attention_heads", int(config.num_attention_heads))
    writer.add_uint32(f"{ARCH}.intermediate_size", int(config.intermediate_size))
    writer.add_uint32(
        f"{ARCH}.num_feat_extract_layers", int(config.num_feat_extract_layers)
    )
    writer.add_uint32(
        f"{ARCH}.num_conv_pos_embeddings", int(config.num_conv_pos_embeddings)
    )
    writer.add_uint32(
        f"{ARCH}.num_conv_pos_embedding_groups",
        int(config.num_conv_pos_embedding_groups),
    )
    writer.add_float32(f"{ARCH}.layer_norm_eps", float(config.layer_norm_eps))
    # CTC blank token is always index 0 in HF wav2vec2 (the <pad> token).
    # config.pad_token_id may differ (e.g. 1 for base models), but for CTC
    # greedy decoding we need the actual blank index.
    ctc_blank_id = 0
    writer.add_uint32(f"{ARCH}.pad_token_id", ctc_blank_id)
    # 0 = group norm, 1 = layer norm in CNN
    model_type = getattr(config, "model_type", "wav2vec2")
    is_data2vec = model_type == "data2vec-audio"
    is_hubert = model_type == "hubert"
    feat_norm_type = (
        1 if is_data2vec or is_hubert or getattr(config, "feat_extract_norm", "group") == "layer" else 0
    )
    writer.add_uint32(f"{ARCH}.feat_extract_norm_type", feat_norm_type)
    # 0 = post-norm, 1 = pre-norm. Data2Vec is POST-norm (despite LayerNorm CNN).
    # HuBERT is pre-norm (do_stable_layer_norm=True in config).
    stable_ln = 1 if is_hubert or getattr(config, "do_stable_layer_norm", False) else 0
    writer.add_uint32(f"{ARCH}.do_stable_layer_norm", stable_ln)
    # Data2Vec applies global encoder LN BEFORE transformer layers (unique to data2vec)
    global_ln_before = 1 if is_data2vec else 0
    writer.add_uint32(f"{ARCH}.global_ln_before_encoder", global_ln_before)
    if is_data2vec:
        print(f"  Detected data2vec: LayerNorm CNN, POST-norm encoder, global LN before layers")
    elif is_hubert:
        print(f"  Detected hubert: LayerNorm CNN, PRE-norm encoder")

    # CNN shape arrays (stored as individual keys for portability)
    for i, (dim, kern, stride) in enumerate(
        zip(config.conv_dim, config.conv_kernel, config.conv_stride)
    ):
        writer.add_uint32(f"{ARCH}.conv_dim_{i}", int(dim))
        writer.add_uint32(f"{ARCH}.conv_kernel_{i}", int(kern))
        writer.add_uint32(f"{ARCH}.conv_stride_{i}", int(stride))

    # Vocabulary
    writer.add_array("tokenizer.ggml.tokens", vocab_list)
    writer.add_uint32("tokenizer.ggml.padding_token_id", int(config.pad_token_id))

    # Tensors
    print("\nWriting tensors:")
    write_cnn_layers(writer, sd, config.num_feat_extract_layers)
    write_feature_projection(writer, sd)
    write_pos_conv(writer, sd, model)
    write_encoder_global_ln(writer, sd)
    write_encoder_layers(writer, sd, config.num_hidden_layers)
    write_lm_head(writer, sd)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    size_mb = out_path.stat().st_size / (1024 * 1024)
    stem = out_path.stem.replace("-f16", "")
    print(f"\nWrote {out_path}  ({size_mb:.1f} MB)")
    print("\nTo quantize transformer weights (Python, no llama.cpp needed):")
    print(f"  python quantize_gguf.py {out_path} {stem}-q5_0.gguf Q5_0")


if __name__ == "__main__":
    main()
