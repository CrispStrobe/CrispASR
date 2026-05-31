#!/usr/bin/env python3
"""
Convert nari-labs/Dia-1.6B HuggingFace safetensors -> GGUF F16 for the
CrispASR `dia` backend.

Dia 1.6B is a dialogue-style TTS model with:
  - Text encoder: 12-layer Llama-style transformer (1024-d, 16 heads)
  - AR decoder: 18-layer transformer with cross-attention (2048-d, GQA 16q/4kv)
  - DAC codec: Descript Audio Codec (9 codebooks, 44.1 kHz)

The text encoder uses byte-level encoding with [S1]/[S2] speaker tags
mapped to bytes 0x01/0x02. The decoder generates 9 interleaved codebook
channels with a delay pattern [0, 8, 9, 10, 11, 12, 13, 14, 15].
Classifier-Free Guidance (CFG) is used during inference.

Architecture (from config.json):

  Encoder:
    hidden_size          = 1024
    n_layer              = 12
    n_head               = 16
    head_dim             = 128  (so hidden = head * head_dim isn't exact
                                 -- encoder Q/K/V project to decoder_hidden_size=2048)
    intermediate_size    = 4096
    vocab_size           = 256 (byte-level)
    max_position_embeddings = 1024

  Decoder:
    hidden_size          = 2048
    n_layer              = 18
    n_head (query)       = 16
    kv_heads             = 4    (GQA)
    head_dim             = 128
    intermediate_size    = 8192
    vocab_size           = 1028 (1024 DAC + BOS + EOS + PAD + 1 spare)
    max_position_embeddings = 3072
    channels             = 9

  Special tokens:
    EOS = 1024, PAD = 1025, BOS = 1026

  Delay pattern: [0, 8, 9, 10, 11, 12, 13, 14, 15]

Tensor naming convention (compatible with TTS.cpp):

  Encoder:
    dia.encoder.embedding              -> (256, 1024) embedding
    dia.encoder.norm                   -> (1024,) final RMSNorm
    dia.encoder.layers.N.pre_sa_norm   -> (1024,) self-attn pre-norm
    dia.encoder.layers.N.q_proj        -> (2048, 1024) Q projection
    dia.encoder.layers.N.k_proj        -> (2048, 1024) K projection
    dia.encoder.layers.N.v_proj        -> (2048, 1024) V projection
    dia.encoder.layers.N.o_proj        -> (1024, 2048) O projection
    dia.encoder.layers.N.post_sa_norm  -> (1024,) MLP pre-norm
    dia.encoder.layers.N.gate          -> (4096, 1024) MLP gate (from wi_fused split)
    dia.encoder.layers.N.up            -> (4096, 1024) MLP up   (from wi_fused split)
    dia.encoder.layers.N.wo            -> (1024, 4096) MLP down

  Decoder:
    dia.decoder.norm                   -> (2048,) final RMSNorm
    dia.decoder.embeddings.K           -> (1028, 2048) per-codebook embedding (K=0..8)
    dia.decoder.heads.K                -> (1028, 2048) per-codebook output head (K=0..8)
    dia.decoder.layers.N.pre_sa_norm   -> (2048,) self-attn pre-norm
    dia.decoder.layers.N.self_q_proj   -> (2048, 2048) self-attn Q
    dia.decoder.layers.N.self_k_proj   -> (512, 2048)  self-attn K (4 kv heads)
    dia.decoder.layers.N.self_v_proj   -> (512, 2048)  self-attn V (4 kv heads)
    dia.decoder.layers.N.self_o_proj   -> (2048, 2048) self-attn O
    dia.decoder.layers.N.pre_ca_norm   -> (2048,) cross-attn pre-norm
    dia.decoder.layers.N.cross_q_proj  -> (2048, 2048) cross-attn Q
    dia.decoder.layers.N.cross_k_proj  -> (2048, 1024) cross-attn K
    dia.decoder.layers.N.cross_v_proj  -> (2048, 1024) cross-attn V
    dia.decoder.layers.N.cross_o_proj  -> (2048, 2048) cross-attn O
    dia.decoder.layers.N.pre_mlp_norm  -> (2048,) MLP pre-norm
    dia.decoder.layers.N.gate          -> (8192, 2048) MLP gate
    dia.decoder.layers.N.up            -> (8192, 2048) MLP up
    dia.decoder.layers.N.wo            -> (2048, 8192) MLP down

  DAC (audio decoder):
    audio_encoder.*                    -> DAC weights (same naming as Zonos)

Usage:

    python models/convert-dia-to-gguf.py \\
        --input nari-labs/Dia-1.6B \\
        --output dia-1.6b-f16.gguf

    # Include DAC in the same GGUF:
    python models/convert-dia-to-gguf.py \\
        --input nari-labs/Dia-1.6B \\
        --output dia-1.6b-f16.gguf \\
        --include-dac
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
    from safetensors import safe_open
except ImportError:
    sys.exit("pip install safetensors")

try:
    import torch
except ImportError:
    sys.exit("pip install torch")

try:
    from huggingface_hub import snapshot_download
except ImportError:
    sys.exit("pip install huggingface_hub")


# ---------------------------------------------------------------------------
# Model directory resolution
# ---------------------------------------------------------------------------

def load_model_dir(model_id: str) -> Path:
    p = Path(model_id)
    if p.is_dir():
        return p
    print(f"Downloading {model_id} from HuggingFace...")
    return Path(snapshot_download(model_id))


# ---------------------------------------------------------------------------
# Dia-specific PyTorch key -> GGUF name mapping
# ---------------------------------------------------------------------------

def map_encoder_tensor(py_name: str) -> tuple[str, bool]:
    """Map encoder PyTorch key to GGUF name.

    Returns (gguf_name, needs_transpose).
    The fused wi_fused tensor is split by the caller.
    """
    parts = py_name.split(".")
    # encoder.embedding.kernel -> dia.encoder.embedding
    if parts[0] == "embedding":
        return "dia.encoder.embedding", False
    if parts[0] == "norm":
        return "dia.encoder.norm", False
    if parts[0] == "layers":
        layer_idx = parts[1]
        prefix = f"dia.encoder.layers.{layer_idx}"
        rest = ".".join(parts[2:])
        if rest == "self_attention.q_proj.kernel":
            return f"{prefix}.q_proj", True
        if rest == "self_attention.k_proj.kernel":
            return f"{prefix}.k_proj", True
        if rest == "self_attention.v_proj.kernel":
            return f"{prefix}.v_proj", True
        if rest == "self_attention.o_proj.kernel":
            return f"{prefix}.o_proj", True
        if rest == "pre_sa_norm.scale":
            return f"{prefix}.pre_sa_norm", False
        if rest == "post_sa_norm.scale":
            return f"{prefix}.post_sa_norm", False
        if rest == "mlp.wi_fused.kernel":
            # Fused gate+up -- caller splits dim 1
            return f"{prefix}.wi_fused", True
        if rest == "mlp.wo.kernel":
            return f"{prefix}.wo", True
    return None, False


def map_decoder_tensor(py_name: str) -> tuple[str, bool]:
    """Map decoder PyTorch key to GGUF name.

    Returns (gguf_name, needs_transpose).
    """
    parts = py_name.split(".")
    if parts[0] == "embeddings":
        idx = parts[1]
        return f"dia.decoder.embeddings.{idx}", False
    if parts[0] == "norm":
        return "dia.decoder.norm", False
    if parts[0] == "logits_dense":
        # logits_dense.kernel shape: (hidden_size, n_heads, vocab_size)
        # Split by caller into per-head tensors
        return "dia.decoder.logits_dense", True
    if parts[0] == "layers":
        layer_idx = parts[1]
        prefix = f"dia.decoder.layers.{layer_idx}"
        rest = ".".join(parts[2:])
        # Self-attention
        if rest == "self_attention.q_proj.kernel":
            return f"{prefix}.self_q_proj", True
        if rest == "self_attention.k_proj.kernel":
            return f"{prefix}.self_k_proj", True
        if rest == "self_attention.v_proj.kernel":
            return f"{prefix}.self_v_proj", True
        if rest == "self_attention.o_proj.kernel":
            return f"{prefix}.self_o_proj", True
        # Cross-attention
        if rest == "cross_attention.q_proj.kernel":
            return f"{prefix}.cross_q_proj", True
        if rest == "cross_attention.k_proj.kernel":
            return f"{prefix}.cross_k_proj", True
        if rest == "cross_attention.v_proj.kernel":
            return f"{prefix}.cross_v_proj", True
        if rest == "cross_attention.o_proj.kernel":
            return f"{prefix}.cross_o_proj", True
        # Norms
        if rest == "pre_sa_norm.scale":
            return f"{prefix}.pre_sa_norm", False
        if rest == "pre_ca_norm.scale":
            return f"{prefix}.pre_ca_norm", False
        if rest == "pre_mlp_norm.scale":
            return f"{prefix}.pre_mlp_norm", False
        # MLP
        if rest == "mlp.wi_fused.kernel":
            return f"{prefix}.wi_fused", True
        if rest == "mlp.wo.kernel":
            return f"{prefix}.wo", True
    return None, False


# ---------------------------------------------------------------------------
# DAC codec tensor mapping
# ---------------------------------------------------------------------------

def map_dac_tensor(py_name: str) -> tuple[str, bool]:
    """Map DAC model PyTorch key to GGUF name with audio_encoder prefix."""
    # The DAC model in Dia uses the descript-audio-codec package.
    # We map to the same naming convention as the Zonos DAC converter.
    # Prefix with "audio_encoder." for compatibility with TTS.cpp.

    # Quantizer
    if py_name.startswith("quantizer."):
        parts = py_name.split(".")
        if "codebook" in py_name:
            # quantizer.quantizers.N.codebook -> audio_encoder.quant.N.codebook
            idx = parts[2]
            return f"audio_encoder.quant.{idx}.codebook", False
        if "out_proj" in py_name:
            idx = parts[2]
            if "weight" in py_name:
                return f"audio_encoder.quant_proj.{idx}.weight", True
            if "bias" in py_name:
                return f"audio_encoder.quant_proj.{idx}.bias", False

    # Decoder
    if py_name.startswith("decoder."):
        # Complex mapping -- simplified for the most common patterns
        return f"audio_encoder.{py_name}", False

    return None, False


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------

def convert(model_dir: Path, output_path: str, include_dac: bool = False):
    # Load safetensors
    st_files = sorted(model_dir.glob("*.safetensors"))
    if not st_files:
        sys.exit(f"No safetensors files found in {model_dir}")

    print(f"Found {len(st_files)} safetensors file(s)")

    writer = GGUFWriter(output_path, "dia")

    # ---- Metadata ----
    writer.add_uint32("dia.attn_head_size", 128)
    writer.add_uint32("dia.eos_token_id", 1024)
    writer.add_uint32("dia.bos_token_id", 1026)
    writer.add_uint32("dia.pad_token_id", 1025)
    writer.add_uint32("dia.max_delay", 15)

    # Encoder config
    writer.add_uint32("dia.encoder.max_context_length", 1024)
    writer.add_uint32("dia.encoder.attn_heads", 16)
    writer.add_uint32("dia.encoder.layers", 12)
    writer.add_uint32("dia.encoder.hidden_size", 1024)
    writer.add_uint32("dia.encoder.intermediate_size", 4096)

    # Decoder config
    writer.add_uint32("dia.decoder.hidden_size", 2048)
    writer.add_uint32("dia.decoder.layers", 18)
    writer.add_uint32("dia.decoder.output_heads", 9)
    writer.add_uint32("dia.decoder.attn_heads", 16)
    writer.add_uint32("dia.decoder.query_heads", 4)  # kv_heads
    writer.add_uint32("dia.decoder.output_vocab_size", 1028)
    writer.add_uint32("dia.decoder.audio_vocab_size", 1024)
    writer.add_uint32("dia.decoder.max_generation_size", 3072)
    writer.add_uint32("dia.decoder.intermediate_size", 8192)

    writer.add_float32("dia.rope_theta", 10000.0)
    writer.add_float32("dia.rms_norm_eps", 1e-5)

    n_tensors = 0
    ftype = GGMLQuantizationType.F16

    for st_file in st_files:
        print(f"Processing {st_file.name}...")
        with safe_open(str(st_file), framework="pt") as f:
            for key in f.keys():
                tensor = f.get_tensor(key)

                # Determine if encoder or decoder tensor
                gguf_name = None
                needs_transpose = False

                if key.startswith("encoder."):
                    py_name = key[len("encoder."):]
                    gguf_name, needs_transpose = map_encoder_tensor(py_name)
                elif key.startswith("decoder."):
                    py_name = key[len("decoder."):]
                    gguf_name, needs_transpose = map_decoder_tensor(py_name)
                elif key.startswith("dac_model.") and include_dac:
                    py_name = key[len("dac_model."):]
                    gguf_name, needs_transpose = map_dac_tensor(py_name)

                if gguf_name is None:
                    if not key.startswith("dac_model."):
                        print(f"  SKIP (unmapped): {key} {list(tensor.shape)}")
                    continue

                data = tensor.float().numpy()

                # Handle fused wi_fused (gate + up concatenated)
                if gguf_name.endswith(".wi_fused"):
                    base_name = gguf_name[:-len(".wi_fused")]
                    # wi_fused shape: (hidden_size, 2, intermediate_size) or
                    # similar fused layout. Split into gate and up.
                    if data.ndim == 3:
                        gate = data[:, 0, :]
                        up = data[:, 1, :]
                    else:
                        # Handle potential 2D fused: (2*intermediate, hidden)
                        half = data.shape[0] // 2
                        gate = data[:half, :]
                        up = data[half:, :]

                    # Transpose for GGUF (row-major weights)
                    gate_t = gate.T.copy()
                    up_t = up.T.copy()

                    gate_f16 = gate_t.astype(np.float16)
                    up_f16 = up_t.astype(np.float16)

                    writer.add_tensor(f"{base_name}.gate", gate_f16)
                    writer.add_tensor(f"{base_name}.up", up_f16)
                    n_tensors += 2
                    print(f"  {base_name}.gate {list(gate_f16.shape)}")
                    print(f"  {base_name}.up {list(up_f16.shape)}")
                    continue

                # Handle logits_dense (per-head split)
                if gguf_name == "dia.decoder.logits_dense":
                    # Shape: (hidden_size, n_heads, vocab_size) with DenseGeneral
                    # Or after reshape: we need per-head output heads
                    n_heads = 9
                    if data.ndim == 3:
                        for i in range(n_heads):
                            head = data[:, i, :]  # (hidden_size, vocab_size)
                            head_t = head.T.copy()  # (vocab_size, hidden_size)
                            head_f16 = head_t.astype(np.float16)
                            writer.add_tensor(f"dia.decoder.heads.{i}", head_f16)
                            n_tensors += 1
                            print(f"  dia.decoder.heads.{i} {list(head_f16.shape)}")
                    else:
                        print(f"  WARNING: unexpected logits_dense shape {list(data.shape)}")
                    continue

                # Standard tensor handling
                if needs_transpose and data.ndim >= 2:
                    # DenseGeneral uses tensordot convention; reshape + transpose
                    # Q/K/V: kernel shape (in_dim, n_heads, head_dim)
                    #   -> GGUF: (n_heads * head_dim, in_dim) row-major
                    # O: kernel shape (n_heads, head_dim, out_dim)
                    #   -> GGUF: (out_dim, n_heads * head_dim) row-major
                    if data.ndim == 3:
                        # (in_dim, n_heads, head_dim) -> (n_heads*head_dim, in_dim)
                        data = data.reshape(data.shape[0], -1).T.copy()
                    else:
                        data = data.T.copy()

                # Choose precision
                if data.ndim == 1:
                    # Norms, biases: keep as F32
                    out = data.astype(np.float32)
                else:
                    out = data.astype(np.float16)

                writer.add_tensor(gguf_name, out)
                n_tensors += 1
                print(f"  {gguf_name} {list(out.shape)}")

    print(f"\nTotal tensors: {n_tensors}")
    print(f"Writing {output_path}...")
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print("Done.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Convert Dia 1.6B to GGUF")
    parser.add_argument("--input", required=True,
                        help="HuggingFace model ID or local directory")
    parser.add_argument("--output", required=True,
                        help="Output GGUF file path")
    parser.add_argument("--include-dac", action="store_true",
                        help="Include DAC codec weights in the same GGUF")
    args = parser.parse_args()

    model_dir = load_model_dir(args.input)
    convert(model_dir, args.output, args.include_dac)


if __name__ == "__main__":
    main()
