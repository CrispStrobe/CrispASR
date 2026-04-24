#!/usr/bin/env python3
"""Convert HuggingFace Moonshine model weights to GGUF format."""

import argparse
import json
import struct
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


# Tensor name mapping: HuggingFace name → GGUF name
TENSOR_NAME_MAP = {
    # Encoder conv stem
    "model.encoder.conv1.weight": "encoder.conv1.weight",
    "model.encoder.groupnorm.weight": "encoder.groupnorm.weight",
    "model.encoder.groupnorm.bias": "encoder.groupnorm.bias",
    "model.encoder.conv2.weight": "encoder.conv2.weight",
    "model.encoder.conv2.bias": "encoder.conv2.bias",
    "model.encoder.conv3.weight": "encoder.conv3.weight",
    "model.encoder.conv3.bias": "encoder.conv3.bias",
    # Decoder
    # Encoder final norm
    "model.encoder.layer_norm.weight": "encoder.output_norm.weight",
    # Decoder
    "model.decoder.embed_tokens.weight": "decoder.embed_tokens.weight",
    "model.decoder.norm.weight": "decoder.output_norm.weight",
    "proj_out.weight": "decoder.output.weight",
}

# Encoder layer tensor patterns (use .format(i=layer_idx))
ENCODER_LAYER_MAP = {
    "model.encoder.layers.{i}.input_layernorm.weight": "encoder.layers.{i}.attn_norm.weight",
    "model.encoder.layers.{i}.self_attn.q_proj.weight": "encoder.layers.{i}.attn.q.weight",
    "model.encoder.layers.{i}.self_attn.k_proj.weight": "encoder.layers.{i}.attn.k.weight",
    "model.encoder.layers.{i}.self_attn.v_proj.weight": "encoder.layers.{i}.attn.v.weight",
    "model.encoder.layers.{i}.self_attn.o_proj.weight": "encoder.layers.{i}.attn.o.weight",
    "model.encoder.layers.{i}.post_attention_layernorm.weight": "encoder.layers.{i}.ffn_norm.weight",
    "model.encoder.layers.{i}.mlp.fc1.weight": "encoder.layers.{i}.ffn.fc1.weight",
    "model.encoder.layers.{i}.mlp.fc1.bias": "encoder.layers.{i}.ffn.fc1.bias",
    "model.encoder.layers.{i}.mlp.fc2.weight": "encoder.layers.{i}.ffn.fc2.weight",
    "model.encoder.layers.{i}.mlp.fc2.bias": "encoder.layers.{i}.ffn.fc2.bias",
}

# Decoder layer tensor patterns
DECODER_LAYER_MAP = {
    "model.decoder.layers.{i}.input_layernorm.weight": "decoder.layers.{i}.attn_norm.weight",
    "model.decoder.layers.{i}.self_attn.q_proj.weight": "decoder.layers.{i}.attn.q.weight",
    "model.decoder.layers.{i}.self_attn.k_proj.weight": "decoder.layers.{i}.attn.k.weight",
    "model.decoder.layers.{i}.self_attn.v_proj.weight": "decoder.layers.{i}.attn.v.weight",
    "model.decoder.layers.{i}.self_attn.o_proj.weight": "decoder.layers.{i}.attn.o.weight",
    "model.decoder.layers.{i}.post_attention_layernorm.weight": "decoder.layers.{i}.cross_attn_norm.weight",
    "model.decoder.layers.{i}.encoder_attn.q_proj.weight": "decoder.layers.{i}.cross_attn.q.weight",
    "model.decoder.layers.{i}.encoder_attn.k_proj.weight": "decoder.layers.{i}.cross_attn.k.weight",
    "model.decoder.layers.{i}.encoder_attn.v_proj.weight": "decoder.layers.{i}.cross_attn.v.weight",
    "model.decoder.layers.{i}.encoder_attn.o_proj.weight": "decoder.layers.{i}.cross_attn.o.weight",
    "model.decoder.layers.{i}.final_layernorm.weight": "decoder.layers.{i}.ffn_norm.weight",
    "model.decoder.layers.{i}.mlp.fc1.weight": "decoder.layers.{i}.ffn.fc1.weight",
    "model.decoder.layers.{i}.mlp.fc1.bias": "decoder.layers.{i}.ffn.fc1.bias",
    "model.decoder.layers.{i}.mlp.fc2.weight": "decoder.layers.{i}.ffn.fc2.weight",
    "model.decoder.layers.{i}.mlp.fc2.bias": "decoder.layers.{i}.ffn.fc2.bias",
}

# Conv1d tensor names that need transposition
CONV1D_TENSORS = {
    "model.encoder.conv1.weight",
    "model.encoder.conv2.weight",
    "model.encoder.conv3.weight",
}


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


def load_model_dir(model_id: str) -> Path:
    """Download or locate model directory."""
    model_path = Path(model_id)
    if model_path.is_dir():
        print(f"Using local model directory: {model_path}")
        return model_path

    print(f"Downloading model from HuggingFace: {model_id}")
    path = snapshot_download(
        model_id,
        allow_patterns=["*.safetensors", "config.json", "tokenizer.json", "tokenizer_config.json"],
    )
    print(f"Downloaded to: {path}")
    return Path(path)


def load_config(model_dir: Path) -> dict:
    """Load and return model config."""
    config_path = model_dir / "config.json"
    try:
        with open(config_path) as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: config.json not found in {model_dir}")
        sys.exit(1)


def get_config_value(config: dict, *keys):
    """Get a value from config trying multiple possible key names."""
    for key in keys:
        if key in config:
            return config[key]
    return None


def open_safetensors(model_dir: Path) -> list:
    """Open all safetensors files and return file handles (for lazy loading)."""
    st_files = sorted(model_dir.glob("*.safetensors"))
    if not st_files:
        print(f"Error: No .safetensors files found in {model_dir}")
        sys.exit(1)

    print(f"\nLoading safetensors from: {[f.name for f in st_files]}")
    return [safe_open(str(f), framework="numpy") for f in st_files]


def get_tensor_names(handles: list) -> dict[str, int]:
    """Get all tensor names across handles, mapping name → handle index."""
    names = {}
    for idx, handle in enumerate(handles):
        for name in handle.keys():
            names[name] = idx
    return names


def convert_tokenizer(model_dir: Path, out_dir: Path, config_vocab_size: int = 32768):
    """Convert tokenizer.json to tokenizer.bin format."""
    tokenizer_path = model_dir / "tokenizer.json"
    try:
        with open(tokenizer_path) as f:
            tokenizer_data = json.load(f)
    except FileNotFoundError:
        print(f"Warning: tokenizer.json not found in {model_dir}, skipping tokenizer conversion")
        return

    vocab = tokenizer_data.get("model", {}).get("vocab", {})
    if not vocab:
        print("Warning: No vocab found in tokenizer.json")
        return

    # Determine total vocab size: use config value to include added_tokens
    added_tokens = tokenizer_data.get("added_tokens", [])
    max_id = max(vocab.values())
    if added_tokens:
        max_added_id = max(entry.get("id", 0) for entry in added_tokens)
        max_id = max(max_id, max_added_id)
    vocab_size = max(max_id + 1, config_vocab_size)

    # Build token_id -> bytes mapping
    tokens = [b""] * vocab_size
    for token_str, token_id in vocab.items():
        if token_id < vocab_size:
            tokens[token_id] = token_str.encode("utf-8", errors="replace")

    # Merge added_tokens (includes special tokens at IDs >= 32000)
    for entry in added_tokens:
        tid = entry.get("id")
        content = entry.get("content", "")
        if tid is not None and tid < vocab_size:
            tokens[tid] = content.encode("utf-8", errors="replace")

    # Write binary format
    tokenizer_bin_path = out_dir / "tokenizer.bin"

    with open(tokenizer_bin_path, "wb") as f:
        for token_bytes in tokens:
            length = len(token_bytes)
            if length == 0:
                f.write(struct.pack("B", 0x00))
            elif length < 128:
                f.write(struct.pack("B", length))
                f.write(token_bytes)
            else:
                f.write(struct.pack("BB", (length % 128) + 128, length // 128))
                f.write(token_bytes)

    print(f"\nTokenizer saved to: {tokenizer_bin_path}")
    print(f"  Vocab size: {vocab_size}")
    print(f"  File size: {tokenizer_bin_path.stat().st_size:,} bytes")


def main():
    parser = argparse.ArgumentParser(description="Convert Moonshine HuggingFace model to GGUF")
    parser.add_argument("--model", required=True, help="HuggingFace model ID or local path")
    parser.add_argument("--outfile", required=True, help="Output GGUF file path")
    parser.add_argument("--outtype", default="f32", choices=["f32", "f16"], help="Output data type (default: f32)")
    args = parser.parse_args()

    outfile = Path(args.outfile)

    # Load model
    model_dir = load_model_dir(args.model)
    config = load_config(model_dir)

    # Extract config values (handle different naming conventions)
    hidden_size = get_config_value(config, "hidden_size", "d_model")
    intermediate_size = get_config_value(config, "intermediate_size", "encoder_ffn_dim")
    enc_layers = get_config_value(config, "encoder_num_hidden_layers", "num_encoder_layers", "encoder_layers")
    dec_layers = get_config_value(config, "decoder_num_hidden_layers", "num_decoder_layers", "decoder_layers")
    num_heads = get_config_value(config, "encoder_num_attention_heads", "decoder_num_attention_heads", "num_attention_heads")
    num_kv_heads = get_config_value(config, "encoder_num_key_value_heads", "decoder_num_key_value_heads", "num_key_value_heads")
    vocab_size = config.get("vocab_size", 32768)
    partial_rotary_factor = config.get("partial_rotary_factor", 0.9)
    rope_theta = config.get("rope_theta", 10000.0)
    layer_norm_eps = config.get("layer_norm_eps", config.get("layer_norm_epsilon", 1e-5))

    if hidden_size is None or enc_layers is None or dec_layers is None:
        print("Error: Could not determine model dimensions from config.json")
        print(f"  Config keys: {list(config.keys())}")
        sys.exit(1)

    # Determine model variant name
    if hidden_size <= 300:
        model_name = "moonshine-tiny"
    else:
        model_name = "moonshine-base"

    print(f"\nModel config:")
    print(f"  variant: {model_name}")
    print(f"  hidden_size={hidden_size}, intermediate_size={intermediate_size}")
    print(f"  enc_layers={enc_layers}, dec_layers={dec_layers}")
    print(f"  heads={num_heads}, kv_heads={num_kv_heads}")
    print(f"  partial_rotary_factor={partial_rotary_factor}")
    print(f"  vocab_size={vocab_size}")

    # Open safetensors lazily
    handles = open_safetensors(model_dir)
    tensor_names = get_tensor_names(handles)

    # Print all tensor names for debugging
    print(f"\nFound {len(tensor_names)} tensors:")
    for name in sorted(tensor_names.keys()):
        t = handles[tensor_names[name]].get_tensor(name)
        print(f"  {name}: {t.shape} {t.dtype}")
        del t

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
    writer = GGUFWriter(str(outfile), "moonshine")

    # Write metadata
    writer.add_name(model_name)

    writer.add_uint32("moonshine.encoder.embedding_length", hidden_size)
    writer.add_uint32("moonshine.encoder.block_count", enc_layers)
    writer.add_uint32("moonshine.encoder.attention.head_count", num_heads)
    writer.add_uint32("moonshine.encoder.attention.head_count_kv", num_kv_heads)
    writer.add_uint32("moonshine.encoder.feed_forward_length", intermediate_size)

    writer.add_uint32("moonshine.decoder.embedding_length", hidden_size)
    writer.add_uint32("moonshine.decoder.block_count", dec_layers)
    writer.add_uint32("moonshine.decoder.attention.head_count", num_heads)
    writer.add_uint32("moonshine.decoder.attention.head_count_kv", num_kv_heads)
    writer.add_uint32("moonshine.decoder.feed_forward_length", intermediate_size)

    writer.add_uint32("moonshine.vocab_size", vocab_size)
    writer.add_uint32("moonshine.bos_token_id", 1)
    writer.add_uint32("moonshine.eos_token_id", 2)

    writer.add_float32("moonshine.attention.layer_norm_epsilon", layer_norm_eps)
    writer.add_float32("moonshine.rope.freq_base", rope_theta)
    writer.add_float32("moonshine.encoder.partial_rotary_factor", partial_rotary_factor)
    writer.add_float32("moonshine.decoder.partial_rotary_factor", partial_rotary_factor)

    writer.add_uint32("moonshine.encoder.conv1.kernel_size", 127)
    writer.add_uint32("moonshine.encoder.conv1.stride", 64)
    writer.add_uint32("moonshine.encoder.conv2.kernel_size", 7)
    writer.add_uint32("moonshine.encoder.conv2.stride", 3)
    writer.add_uint32("moonshine.encoder.conv3.kernel_size", 3)
    writer.add_uint32("moonshine.encoder.conv3.stride", 2)

    # Map and write tensors (streamed one at a time)
    mapped_count = 0
    unmapped = []

    for hf_name in sorted(tensor_names.keys()):
        gguf_name = tensor_map.get(hf_name)
        if gguf_name is None:
            unmapped.append(hf_name)
            continue

        data = handles[tensor_names[hf_name]].get_tensor(hf_name)

        # Conv1d weights: PyTorch stores as [OC, IC, K].
        # ggml_conv_1d expects kernel ne[0]=K, ne[1]=IC, ne[2]=OC.
        # Since numpy (row-major) → ggml (column-major) reverses axes,
        # PyTorch [OC, IC, K] → ggml ne[0]=K, ne[1]=IC, ne[2]=OC — no transpose needed.
        if hf_name in CONV1D_TENSORS:
            print(f"  Conv1d weight (no transpose needed): {hf_name} {data.shape}")

        # Convert dtype (returns same array if already correct dtype and contiguous)
        data = np.ascontiguousarray(data.astype(out_dtype))

        writer.add_tensor(gguf_name, data, raw_dtype=ggml_type)
        mapped_count += 1

    if unmapped:
        print(f"\nWarning: {len(unmapped)} unmapped tensors:")
        for name in unmapped:
            print(f"  {name}")

    # Write GGUF file
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    file_size = outfile.stat().st_size
    print(f"\n{'='*60}")
    print(f"GGUF file written: {outfile}")
    print(f"  Model: {model_name}")
    print(f"  Tensors: {mapped_count}")
    print(f"  Type: {args.outtype}")
    print(f"  File size: {file_size:,} bytes ({file_size / 1024 / 1024:.1f} MB)")
    print(f"{'='*60}")

    # Convert tokenizer
    convert_tokenizer(model_dir, outfile.parent, vocab_size)


if __name__ == "__main__":
    main()
