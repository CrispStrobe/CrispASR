#!/usr/bin/env python3
"""Convert GLM-ASR-Nano (zai-org/GLM-ASR-Nano-2512) to GGUF.

Architecture:
  - Audio encoder: Whisper-style (1280 hidden, 32 layers, partial RoPE)
  - Projector: 4-frame stack → linear(5120→4096,GELU) → linear(4096→2048)
  - LLM decoder: Llama-style (2048 hidden, 28 layers, GQA 16/4, SwiGLU)

Usage:
  python models/convert-glm-asr-to-gguf.py \
      --input zai-org/GLM-ASR-Nano-2512 \
      --output glm-asr-nano.gguf
"""

import argparse
import json
import os
import struct
import sys

import numpy as np

try:
    import gguf
except ImportError:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ggml", "python"))
    import gguf


def bf16_to_f16(data: np.ndarray) -> np.ndarray:
    """Convert BF16 (as uint16) to FP16."""
    # BF16 → F32 → F16
    f32 = data.view(np.uint16).astype(np.uint32) << 16
    return np.frombuffer(f32.tobytes(), dtype=np.float32).astype(np.float16)


def main():
    parser = argparse.ArgumentParser(description="Convert GLM-ASR-Nano to GGUF")
    parser.add_argument("--input", required=True, help="HF model ID or local path")
    parser.add_argument("--output", required=True, help="Output GGUF path")
    args = parser.parse_args()

    print(f"Loading: {args.input}")

    # Load model
    from transformers import AutoConfig, AutoTokenizer
    config = AutoConfig.from_pretrained(args.input, trust_remote_code=True)

    # Load weights
    from safetensors import safe_open
    from huggingface_hub import hf_hub_download

    if os.path.isdir(args.input):
        st_path = os.path.join(args.input, "model.safetensors")
    else:
        os.environ.setdefault("HF_HOME", "/mnt/akademie_storage/huggingface")
        st_path = hf_hub_download(args.input, "model.safetensors",
                                   cache_dir=os.environ.get("HF_HOME", None) + "/hub"
                                   if os.environ.get("HF_HOME") else None)

    import torch
    sd = {}
    with safe_open(st_path, framework="pt") as f:
        for key in f.keys():
            t = f.get_tensor(key)
            # Convert BF16 to F32 numpy (numpy doesn't support BF16)
            if t.dtype == torch.bfloat16:
                sd[key] = t.float().numpy()
            else:
                sd[key] = t.numpy()
    print(f"  Loaded {len(sd)} tensors")

    # Extract config
    audio_cfg = config.audio_config
    text_cfg = config.text_config

    # Extract mel filterbank + window from processor
    from transformers import AutoProcessor
    proc = AutoProcessor.from_pretrained(args.input, trust_remote_code=True)
    fe = proc.feature_extractor
    mel_filters = np.asarray(fe.mel_filters, dtype=np.float32)  # (n_freqs, n_mels)
    win = np.hanning(fe.n_fft + 1)[:-1].astype(np.float32)  # Hann window
    print(f"  mel_filters shape: {mel_filters.shape}")
    print(f"  mel_window shape: {win.shape}")

    # Create GGUF writer
    writer = gguf.GGUFWriter(args.output, "glmasr")
    writer.add_name("GLM-ASR-Nano")

    # Audio encoder hparams
    writer.add_uint32("glmasr.audio.hidden_size", audio_cfg.hidden_size)  # 1280
    writer.add_uint32("glmasr.audio.num_layers", audio_cfg.num_hidden_layers)  # 32
    writer.add_uint32("glmasr.audio.num_heads", audio_cfg.num_attention_heads)  # 20
    writer.add_uint32("glmasr.audio.num_kv_heads", audio_cfg.num_key_value_heads)  # 20
    writer.add_uint32("glmasr.audio.intermediate_size", audio_cfg.intermediate_size)  # 5120
    writer.add_uint32("glmasr.audio.num_mel_bins", audio_cfg.num_mel_bins)  # 128
    writer.add_uint32("glmasr.audio.max_position_embeddings", audio_cfg.max_position_embeddings)  # 1500
    writer.add_float32("glmasr.audio.partial_rotary_factor", audio_cfg.partial_rotary_factor)  # 0.5

    # LLM hparams
    writer.add_uint32("glmasr.llm.hidden_size", text_cfg.hidden_size)  # 2048
    writer.add_uint32("glmasr.llm.num_layers", text_cfg.num_hidden_layers)  # 28
    writer.add_uint32("glmasr.llm.num_heads", text_cfg.num_attention_heads)  # 16
    writer.add_uint32("glmasr.llm.num_kv_heads", text_cfg.num_key_value_heads)  # 4
    writer.add_uint32("glmasr.llm.intermediate_size", text_cfg.intermediate_size)  # 6144
    writer.add_uint32("glmasr.llm.vocab_size", text_cfg.vocab_size)  # 59264
    writer.add_uint32("glmasr.llm.max_position_embeddings", text_cfg.max_position_embeddings)  # 8192
    writer.add_float32("glmasr.llm.rms_norm_eps", text_cfg.rms_norm_eps)  # 1e-5

    # Special tokens
    writer.add_uint32("glmasr.audio_token_id", config.audio_token_id)  # 59260
    eos_ids = text_cfg.eos_token_id if isinstance(text_cfg.eos_token_id, list) else [text_cfg.eos_token_id]
    for i, eid in enumerate(eos_ids):
        writer.add_uint32(f"glmasr.eos_token_id_{i}", eid)
    writer.add_uint32("glmasr.num_eos_tokens", len(eos_ids))
    writer.add_uint32("glmasr.bos_token_id", text_cfg.bos_token_id)  # 1

    # Tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.input, trust_remote_code=True)
        vocab = tokenizer.get_vocab()
        vocab_size = max(vocab.values()) + 1
        vocab_list = [""] * vocab_size
        for token, idx in vocab.items():
            if idx < vocab_size:
                vocab_list[idx] = token
        # Store as GGUF string array
        writer.add_array("tokenizer.ggml.tokens", vocab_list)
        print(f"  Tokenizer: {vocab_size} tokens")
    except Exception as e:
        print(f"  Warning: tokenizer load failed: {e}")

    # Write tensors
    print(f"\nWriting: {args.output}")

    def f16(t):
        """Convert to F16."""
        if t.dtype == np.float32:
            return t.astype(np.float16)
        elif t.dtype == np.uint16:  # BF16
            return bf16_to_f16(t)
        elif t.dtype == np.float16:
            return t
        else:
            return t.astype(np.float32).astype(np.float16)

    def f32(t):
        """Convert to F32."""
        if t.dtype == np.uint16:  # BF16
            return (t.astype(np.uint32) << 16).view(np.float32)
        return t.astype(np.float32)

    def is_f32_tensor(name, shape):
        """Tensors that should stay F32."""
        if "norm" in name:
            return True
        if name.endswith(".bias"):
            return True
        if len(shape) <= 1:
            return True
        return False

    # Bake mel filterbank + window into the GGUF
    writer.add_tensor("audio.mel_filters", mel_filters)
    writer.add_tensor("audio.mel_window", win)
    tensor_count = 2
    print(f"  audio.mel_filters                                          {str(mel_filters.shape):20s} F32")
    print(f"  audio.mel_window                                           {str(win.shape):20s} F32")

    for name, tensor in sorted(sd.items()):
        # Map HF names to GGUF names
        gguf_name = name
        gguf_name = gguf_name.replace("audio_tower.", "audio.")
        gguf_name = gguf_name.replace("language_model.model.", "llm.")
        gguf_name = gguf_name.replace("language_model.lm_head", "lm_head")
        gguf_name = gguf_name.replace("multi_modal_projector.", "proj.")
        gguf_name = gguf_name.replace("self_attn.", "attn.")
        gguf_name = gguf_name.replace("input_layernorm", "attn_norm")
        gguf_name = gguf_name.replace("post_attention_layernorm", "ffn_norm")
        gguf_name = gguf_name.replace("mlp.fc1", "ffn.up")
        gguf_name = gguf_name.replace("mlp.fc2", "ffn.down")
        gguf_name = gguf_name.replace("mlp.gate_proj", "ffn.gate")
        gguf_name = gguf_name.replace("mlp.up_proj", "ffn.up")
        gguf_name = gguf_name.replace("mlp.down_proj", "ffn.down")
        gguf_name = gguf_name.replace("embed_tokens", "token_embd")
        gguf_name = gguf_name.replace("layers.", "blk.")
        gguf_name = gguf_name.replace("attn.q_proj", "attn_q")
        gguf_name = gguf_name.replace("attn.k_proj", "attn_k")
        gguf_name = gguf_name.replace("attn.v_proj", "attn_v")
        gguf_name = gguf_name.replace("attn.o_proj", "attn_out")

        if is_f32_tensor(name, tensor.shape):
            data = f32(tensor)
            print(f"  {gguf_name:60s} {str(tensor.shape):20s} F32")
        else:
            data = f16(tensor)
            print(f"  {gguf_name:60s} {str(tensor.shape):20s} F16")

        writer.add_tensor(gguf_name, data)
        tensor_count += 1

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    file_size = os.path.getsize(args.output)
    print(f"\nDone: {args.output} ({file_size / 1e9:.2f} GB, {tensor_count} tensors)")


if __name__ == "__main__":
    main()
