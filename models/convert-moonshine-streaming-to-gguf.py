#!/usr/bin/env python3
"""Convert HuggingFace Moonshine Streaming model weights to GGUF format.

Supports all three sizes: tiny (34M), small (123M), medium (245M).
The streaming architecture differs from regular moonshine in:
  - Audio frontend: raw waveform → frames → CMVN → asinh → Linear → CausalConv1d × 2
  - Encoder: sliding-window attention, unit-offset LayerNorm (gamma+1), no positional embeddings
  - Decoder: SiLU-gated MLP, learned positional embedding for cross-attention
  - Encoder/decoder may have different hidden sizes (small/medium)

Usage:
    python models/convert-moonshine-streaming-to-gguf.py --input UsefulSensors/moonshine-streaming-tiny --output moonshine-streaming-tiny.gguf
    python models/convert-moonshine-streaming-to-gguf.py --input /path/to/local/dir --output model.gguf --outtype f16
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

try:
    from gguf import GGUFWriter, GGMLQuantizationType
except ImportError:
    print("Error: gguf package not found. Install with: pip install gguf")
    sys.exit(1)

try:
    from safetensors import safe_open
except ImportError:
    print("Error: safetensors package not found. Install with: pip install safetensors")
    sys.exit(1)

try:
    from huggingface_hub import snapshot_download
except ImportError:
    print("Error: huggingface_hub package not found. Install with: pip install huggingface_hub")
    sys.exit(1)


# ── Static tensor name mapping ──────────────────────────────────────────────

TENSOR_NAME_MAP = {
    # Audio frontend (embedder)
    "model.encoder.embedder.comp.log_k":      "encoder.embedder.log_k",
    "model.encoder.embedder.linear.weight":    "encoder.embedder.linear.weight",
    "model.encoder.embedder.conv1.weight":     "encoder.embedder.conv1.weight",
    "model.encoder.embedder.conv1.bias":       "encoder.embedder.conv1.bias",
    "model.encoder.embedder.conv2.weight":     "encoder.embedder.conv2.weight",
    "model.encoder.embedder.conv2.bias":       "encoder.embedder.conv2.bias",
    # Encoder final norm (unit-offset: gamma+1 applied at convert time)
    "model.encoder.final_norm.gamma":          "encoder.output_norm.weight",
    # Decoder
    "model.decoder.embed_tokens.weight":       "decoder.embed_tokens.weight",
    "model.decoder.pos_emb.weight":            "decoder.pos_emb.weight",
    "model.decoder.norm.weight":               "decoder.output_norm.weight",
    "proj_out.weight":                         "decoder.output.weight",
    # Optional: encoder→decoder projection (small/medium only)
    "model.decoder.proj.weight":               "decoder.enc_proj.weight",
}

# Encoder layer tensor patterns (use .format(i=layer_idx))
ENCODER_LAYER_MAP = {
    "model.encoder.layers.{i}.input_layernorm.gamma":          "encoder.layers.{i}.attn_norm.weight",
    "model.encoder.layers.{i}.self_attn.q_proj.weight":        "encoder.layers.{i}.attn.q.weight",
    "model.encoder.layers.{i}.self_attn.k_proj.weight":        "encoder.layers.{i}.attn.k.weight",
    "model.encoder.layers.{i}.self_attn.v_proj.weight":        "encoder.layers.{i}.attn.v.weight",
    "model.encoder.layers.{i}.self_attn.o_proj.weight":        "encoder.layers.{i}.attn.o.weight",
    "model.encoder.layers.{i}.post_attention_layernorm.gamma": "encoder.layers.{i}.ffn_norm.weight",
    "model.encoder.layers.{i}.mlp.fc1.weight":                 "encoder.layers.{i}.ffn.fc1.weight",
    "model.encoder.layers.{i}.mlp.fc1.bias":                   "encoder.layers.{i}.ffn.fc1.bias",
    "model.encoder.layers.{i}.mlp.fc2.weight":                 "encoder.layers.{i}.ffn.fc2.weight",
    "model.encoder.layers.{i}.mlp.fc2.bias":                   "encoder.layers.{i}.ffn.fc2.bias",
}

# Decoder layer tensor patterns
DECODER_LAYER_MAP = {
    "model.decoder.layers.{i}.input_layernorm.weight":         "decoder.layers.{i}.attn_norm.weight",
    "model.decoder.layers.{i}.self_attn.q_proj.weight":        "decoder.layers.{i}.attn.q.weight",
    "model.decoder.layers.{i}.self_attn.k_proj.weight":        "decoder.layers.{i}.attn.k.weight",
    "model.decoder.layers.{i}.self_attn.v_proj.weight":        "decoder.layers.{i}.attn.v.weight",
    "model.decoder.layers.{i}.self_attn.o_proj.weight":        "decoder.layers.{i}.attn.o.weight",
    "model.decoder.layers.{i}.post_attention_layernorm.weight":"decoder.layers.{i}.cross_attn_norm.weight",
    "model.decoder.layers.{i}.encoder_attn.q_proj.weight":     "decoder.layers.{i}.cross_attn.q.weight",
    "model.decoder.layers.{i}.encoder_attn.k_proj.weight":     "decoder.layers.{i}.cross_attn.k.weight",
    "model.decoder.layers.{i}.encoder_attn.v_proj.weight":     "decoder.layers.{i}.cross_attn.v.weight",
    "model.decoder.layers.{i}.encoder_attn.o_proj.weight":     "decoder.layers.{i}.cross_attn.o.weight",
    "model.decoder.layers.{i}.final_layernorm.weight":         "decoder.layers.{i}.ffn_norm.weight",
    "model.decoder.layers.{i}.mlp.fc1.weight":                 "decoder.layers.{i}.ffn.fc1.weight",
    "model.decoder.layers.{i}.mlp.fc1.bias":                   "decoder.layers.{i}.ffn.fc1.bias",
    "model.decoder.layers.{i}.mlp.fc2.weight":                 "decoder.layers.{i}.ffn.fc2.weight",
    "model.decoder.layers.{i}.mlp.fc2.bias":                   "decoder.layers.{i}.ffn.fc2.bias",
}

# Conv1d tensors (PyTorch [OC, IC, K] → ggml ne[0]=K, ne[1]=IC, ne[2]=OC — no transpose needed)
CONV1D_TENSORS = {
    "model.encoder.embedder.conv1.weight",
    "model.encoder.embedder.conv2.weight",
}

# Unit-offset LayerNorm tensors: stored as gamma, we add +1.0 at convert time
# so the C++ runtime can use standard gamma*norm(x) instead of (gamma+1)*norm(x)
UNIT_OFFSET_GAMMA_TENSORS = {
    "model.encoder.final_norm.gamma",
}
# Per-layer unit-offset gammas (patterns, not full names)
UNIT_OFFSET_GAMMA_PATTERNS = [
    "input_layernorm.gamma",
    "post_attention_layernorm.gamma",
]


def build_full_tensor_map(enc_layers: int, dec_layers: int) -> dict[str, str]:
    """Build complete tensor name mapping including all layers."""
    mapping = dict(TENSOR_NAME_MAP)

    for i in range(enc_layers):
        for hf_pat, gguf_pat in ENCODER_LAYER_MAP.items():
            mapping[hf_pat.format(i=i)] = gguf_pat.format(i=i)

    for i in range(dec_layers):
        for hf_pat, gguf_pat in DECODER_LAYER_MAP.items():
            mapping[hf_pat.format(i=i)] = gguf_pat.format(i=i)

    return mapping


def is_unit_offset_gamma(hf_name: str) -> bool:
    """Check if a tensor is a unit-offset gamma that needs +1.0."""
    if hf_name in UNIT_OFFSET_GAMMA_TENSORS:
        return True
    return any(p in hf_name for p in UNIT_OFFSET_GAMMA_PATTERNS)


def load_model_dir(model_id: str) -> Path:
    """Download or locate model directory."""
    model_path = Path(model_id)
    if model_path.is_dir():
        print(f"Using local model directory: {model_path}")
        return model_path

    print(f"Downloading model from HuggingFace: {model_id}")
    path = snapshot_download(
        model_id,
        allow_patterns=["*.safetensors", "config.json", "tokenizer.json",
                        "tokenizer_config.json", "tokenizer.bin"],
    )
    print(f"Downloaded to: {path}")
    return Path(path)


def main():
    parser = argparse.ArgumentParser(description="Convert Moonshine Streaming to GGUF")
    parser.add_argument("--input", required=True, help="HF model ID or local directory")
    parser.add_argument("--output", required=True, help="Output GGUF path")
    parser.add_argument("--outtype", default="f32", choices=["f32", "f16"],
                        help="Output data type for 2D+ tensors (default: f32)")
    args = parser.parse_args()

    model_dir = load_model_dir(args.input)

    # Load config
    config_path = model_dir / "config.json"
    with open(config_path, encoding="utf-8") as f:
        config = json.load(f)

    enc_config = config.get("encoder_config", {})

    # Extract hyperparameters
    enc_hidden = enc_config.get("hidden_size", config.get("encoder_hidden_size", config["hidden_size"]))
    dec_hidden = config["hidden_size"]
    enc_layers = enc_config.get("num_hidden_layers", config["num_hidden_layers"])
    dec_layers = config["num_hidden_layers"]
    enc_heads = enc_config.get("num_attention_heads", config["num_attention_heads"])
    dec_heads = config["num_attention_heads"]
    enc_kv_heads = enc_config.get("num_key_value_heads", enc_heads)
    dec_kv_heads = config.get("num_key_value_heads", dec_heads)
    enc_intermediate = enc_config.get("intermediate_size", config["intermediate_size"])
    dec_intermediate = config["intermediate_size"]
    vocab_size = config["vocab_size"]
    sliding_windows = enc_config.get("sliding_windows", [[16, 4]] * enc_layers)
    rope_params = config.get("rope_parameters", {})
    partial_rotary_factor = rope_params.get("partial_rotary_factor", 0.8)
    rope_theta = rope_params.get("rope_theta", 10000.0)
    max_positions = config.get("max_position_embeddings", 4096)
    sample_rate = enc_config.get("sample_rate", 16000)
    frame_ms = enc_config.get("frame_ms", 5.0)
    # head_dim is explicit in config, NOT derived from hidden/heads
    enc_head_dim = enc_config.get("head_dim", enc_hidden // enc_heads)
    dec_head_dim = config.get("head_dim", dec_hidden // dec_heads)

    print(f"\nModel: moonshine-streaming")
    print(f"  Encoder: hidden={enc_hidden}, layers={enc_layers}, heads={enc_heads}, "
          f"kv_heads={enc_kv_heads}, intermediate={enc_intermediate}")
    print(f"  Decoder: hidden={dec_hidden}, layers={dec_layers}, heads={dec_heads}, "
          f"kv_heads={dec_kv_heads}, intermediate={dec_intermediate}")
    print(f"  Vocab: {vocab_size}, Max pos: {max_positions}")
    print(f"  Sliding windows: {sliding_windows}")
    print(f"  RoPE: theta={rope_theta}, partial_factor={partial_rotary_factor}")
    if enc_hidden != dec_hidden:
        print(f"  NOTE: encoder/decoder hidden sizes differ ({enc_hidden} vs {dec_hidden}) — projection layer expected")

    # Open safetensors
    st_files = sorted(model_dir.glob("*.safetensors"))
    if not st_files:
        print(f"Error: No .safetensors files found in {model_dir}")
        sys.exit(1)
    handles = [safe_open(str(f), framework="numpy") for f in st_files]
    tensor_names = {}
    for idx, h in enumerate(handles):
        for name in h.keys():
            tensor_names[name] = idx

    print(f"  Safetensors: {len(tensor_names)} tensors in {len(st_files)} file(s)")

    # Build tensor name mapping
    tensor_map = build_full_tensor_map(enc_layers, dec_layers)

    # Determine output dtype
    if args.outtype == "f16":
        out_dtype = np.float16
        ggml_type = GGMLQuantizationType.F16
    else:
        out_dtype = np.float32
        ggml_type = GGMLQuantizationType.F32

    # Create GGUF writer
    outfile = Path(args.output)
    writer = GGUFWriter(str(outfile), "moonshine_streaming")

    # Write metadata
    model_name = config.get("_name_or_path", args.input.split("/")[-1])
    writer.add_name(model_name)

    writer.add_uint32("moonshine_streaming.encoder.embedding_length", enc_hidden)
    writer.add_uint32("moonshine_streaming.encoder.block_count", enc_layers)
    writer.add_uint32("moonshine_streaming.encoder.attention.head_count", enc_heads)
    writer.add_uint32("moonshine_streaming.encoder.attention.head_count_kv", enc_kv_heads)
    writer.add_uint32("moonshine_streaming.encoder.feed_forward_length", enc_intermediate)
    writer.add_uint32("moonshine_streaming.encoder.attention.head_dim", enc_head_dim)

    writer.add_uint32("moonshine_streaming.decoder.embedding_length", dec_hidden)
    writer.add_uint32("moonshine_streaming.decoder.block_count", dec_layers)
    writer.add_uint32("moonshine_streaming.decoder.attention.head_count", dec_heads)
    writer.add_uint32("moonshine_streaming.decoder.attention.head_count_kv", dec_kv_heads)
    writer.add_uint32("moonshine_streaming.decoder.feed_forward_length", dec_intermediate)
    writer.add_uint32("moonshine_streaming.decoder.attention.head_dim", dec_head_dim)

    writer.add_uint32("moonshine_streaming.vocab_size", vocab_size)
    writer.add_uint32("moonshine_streaming.bos_token_id", config.get("bos_token_id", 1))
    writer.add_uint32("moonshine_streaming.eos_token_id", config.get("eos_token_id", 2))
    writer.add_uint32("moonshine_streaming.max_position_embeddings", max_positions)

    writer.add_float32("moonshine_streaming.rope.freq_base", rope_theta)
    writer.add_float32("moonshine_streaming.decoder.partial_rotary_factor", partial_rotary_factor)

    # Sliding window config: store as individual per-layer keys
    for li, (left, right) in enumerate(sliding_windows):
        writer.add_uint32(f"moonshine_streaming.encoder.layers.{li}.window_left", left)
        writer.add_uint32(f"moonshine_streaming.encoder.layers.{li}.window_right", right)

    # Audio frontend params
    writer.add_uint32("moonshine_streaming.audio.sample_rate", sample_rate)
    writer.add_float32("moonshine_streaming.audio.frame_ms", frame_ms)

    # Map and write tensors
    mapped_count = 0
    unmapped = []

    for hf_name in sorted(tensor_names.keys()):
        gguf_name = tensor_map.get(hf_name)
        if gguf_name is None:
            unmapped.append(hf_name)
            continue

        data = handles[tensor_names[hf_name]].get_tensor(hf_name)

        # Unit-offset gamma: add +1.0 so runtime uses standard LN multiply
        if is_unit_offset_gamma(hf_name):
            data = data.astype(np.float32) + 1.0
            print(f"  Unit-offset gamma: {hf_name} → {gguf_name} (added +1.0)")

        # Conv1d weights: PyTorch [OC, IC, K] → ggml ne[0]=K, ne[1]=IC, ne[2]=OC — no transpose
        if hf_name in CONV1D_TENSORS:
            print(f"  Conv1d weight: {hf_name} {data.shape}")

        # Scalar tensors: expand to 1-element array
        if data.ndim == 0:
            data = np.array([data.item()], dtype=np.float32)
            writer.add_tensor(gguf_name, data, raw_dtype=GGMLQuantizationType.F32)
        # 1D tensors (norms, biases): must stay F32 for ggml binary ops
        elif data.ndim <= 1:
            data = np.ascontiguousarray(data.astype(np.float32))
            writer.add_tensor(gguf_name, data, raw_dtype=GGMLQuantizationType.F32)
        else:
            data = np.ascontiguousarray(data.astype(out_dtype))
            writer.add_tensor(gguf_name, data, raw_dtype=ggml_type)
        mapped_count += 1

    if unmapped:
        print(f"\nWarning: {len(unmapped)} unmapped tensors:")
        for name in unmapped:
            print(f"  {name}")

    print(f"\nWriting {mapped_count} tensors to {outfile}")
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    size_mb = outfile.stat().st_size / 1024 / 1024
    print(f"Done! {outfile} ({size_mb:.1f} MB, {mapped_count} tensors)")


if __name__ == "__main__":
    main()
