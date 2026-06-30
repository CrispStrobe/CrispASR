#!/usr/bin/env python3
"""Convert OpenMOSS-Team/MOSS-Transcribe-preview-2B to GGUF format for CrispASR.

Architecture (config.json + stock transformers Qwen3OmniMoeAudioEncoder):
  Audio encoder (qwen3_omni_moe_audio_encoder): 128-mel → 3× Conv2d(stride 2,
    480ch) → conv_out Linear(7680→1280, NO bias) → +sinusoidal pos → 32
    Whisper-style pre-LN layers (1280d, 20 heads, FFN 5120, windowed attn) →
    ln_post → proj1(1280→1280)+gelu → proj2(1280→2048).  output_dim=2048.
  Audio adapter: MossGatedMLP(2048 → 8192 → 2048, SiLU gate, no bias).
  LM: Qwen3-1.7B (28L, 2048d, 16Q/8KV heads, head_dim 128, SwiGLU 6144,
    QK-norm, RoPE θ=1e6, vocab 151936, TIED embeddings → no lm_head tensor).

Differs from MOSS-Audio-4B: no DeepStack, conv_out(no bias)+proj1/proj2/ln_post
encoder head, smaller LM. Mel front-end is identical (128-bin Whisper, n_fft=400,
hop=160 — per the README's MelConfig, NOT the file's 80/640 defaults).

Streams tensors one-at-a-time via safe_open (BF16→F16). ~10 GB peak RAM.

Usage:
    python models/convert-moss-transcribe-to-gguf.py \\
        --input OpenMOSS-Team/MOSS-Transcribe-preview-2B \\
        --output moss-transcribe-preview-2b-f16.gguf

    # Then quantize:
    crispasr-quantize moss-transcribe-preview-2b-f16.gguf \\
                      moss-transcribe-preview-2b-q4_k.gguf q4_k
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

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


def load_model_dir(model_id: str) -> Path:
    model_path = Path(model_id)
    if model_path.is_dir():
        return model_path
    print(f"Downloading model from HuggingFace: {model_id}")
    path = snapshot_download(model_id,
                             allow_patterns=["*.safetensors", "*.json",
                                             "merges.txt", "vocab.json"])
    return Path(path)


def map_tensor_name(hf_name: str):
    """Map HF tensor name → GGUF name for MOSS-Transcribe.

    Returns None for tensors that should be skipped (e.g. a separate lm_head,
    which is tied to embed_tokens and absent in this checkpoint anyway).
    """
    name = hf_name

    # Strip the top-level MossModel prefix.
    if name.startswith("model."):
        name = name[len("model."):]

    # ---- Audio encoder (stock Qwen3OmniMoeAudioEncoder) ----
    if name.startswith("audio_model."):
        name = name[len("audio_model."):]
        # Conv front-end: conv2d1/2/3 → enc.conv1/2/3
        name = name.replace("conv2d1.", "enc.conv1.")
        name = name.replace("conv2d2.", "enc.conv2.")
        name = name.replace("conv2d3.", "enc.conv3.")
        name = name.replace("conv_out.", "enc.conv_out.")
        # Encoder output head
        name = name.replace("ln_post.", "enc.ln_post.")
        name = name.replace("proj1.", "enc.proj1.")
        name = name.replace("proj2.", "enc.proj2.")
        # Transformer layers
        name = name.replace("layers.", "enc.blk.")
        name = name.replace(".self_attn_layer_norm.", ".attn_norm.")
        name = name.replace(".final_layer_norm.", ".ffn_norm.")
        name = name.replace(".self_attn.q_proj.", ".attn.q.")
        name = name.replace(".self_attn.k_proj.", ".attn.k.")
        name = name.replace(".self_attn.v_proj.", ".attn.v.")
        name = name.replace(".self_attn.out_proj.", ".attn.o.")
        name = name.replace(".fc1.", ".ffn.fc1.")
        name = name.replace(".fc2.", ".ffn.fc2.")
        return name

    # ---- Audio adapter (MossGatedMLP, final encoder → LM space) ----
    if name.startswith("audio_adapter."):
        name = name.replace("audio_adapter.gate_proj.", "adapter.gate.")
        name = name.replace("audio_adapter.up_proj.", "adapter.up.")
        name = name.replace("audio_adapter.down_proj.", "adapter.down.")
        return name

    # ---- Language model (Qwen3) ----
    if name.startswith("language_model."):
        name = name.replace("language_model.embed_tokens.", "llm.embed.")
        name = name.replace("language_model.norm.", "llm.final_norm.")
        name = name.replace("language_model.layers.", "llm.blk.")
        name = name.replace(".mlp.gate_proj.", ".ffn.gate.")
        name = name.replace(".mlp.up_proj.", ".ffn.up.")
        name = name.replace(".mlp.down_proj.", ".ffn.down.")
        name = name.replace(".self_attn.q_proj.", ".attn.q.")
        name = name.replace(".self_attn.k_proj.", ".attn.k.")
        name = name.replace(".self_attn.v_proj.", ".attn.v.")
        name = name.replace(".self_attn.o_proj.", ".attn.o.")
        name = name.replace(".self_attn.q_norm.", ".attn.q_norm.")
        name = name.replace(".self_attn.k_norm.", ".attn.k_norm.")
        name = name.replace(".input_layernorm.", ".attn_norm.")
        name = name.replace(".post_attention_layernorm.", ".ffn_norm.")
        return name

    # Separate lm_head (tied → ignore if present)
    if name.startswith("lm_head."):
        return None

    return name


def main():
    parser = argparse.ArgumentParser(description="Convert MOSS-Transcribe-preview-2B to GGUF")
    parser.add_argument("--input", required=True, help="HF model ID or local directory")
    parser.add_argument("--output", required=True, help="Output GGUF path")
    parser.add_argument("--outtype", default="f16", choices=["f32", "f16"],
                        help="Output data type for 2D+ tensors (default: f16)")
    args = parser.parse_args()

    model_dir = load_model_dir(args.input)

    with open(model_dir / "config.json", encoding="utf-8") as f:
        config = json.load(f)

    ac = config.get("audio_config", {})
    lc = config.get("language_config", {})

    # Mel front-end: README's canonical MelConfig is 128-bin / n_fft 400 / hop 160.
    # (processing_Moss.py defaults of 80 / 640 are NOT used at inference.)
    n_mels = 128
    mel_n_fft = 400
    mel_hop = 160

    print(f"\nMOSS-Transcribe-preview-2B")
    print(f"  Audio encoder: {ac.get('encoder_layers', 32)}L, d={ac.get('d_model', 1280)}, "
          f"heads={ac.get('encoder_attention_heads', 20)}, FFN={ac.get('encoder_ffn_dim', 5120)}, "
          f"output_dim={ac.get('output_dim', 2048)}")
    print(f"  Adapter hidden: {config.get('adapter_hidden_size', 8192)}")
    print(f"  LM: {lc.get('num_hidden_layers', 28)}L, hidden={lc.get('hidden_size', 2048)}, "
          f"heads={lc.get('num_attention_heads', 16)}, kv_heads={lc.get('num_key_value_heads', 8)}, "
          f"head_dim={lc.get('head_dim', 128)}, ffn={lc.get('intermediate_size', 6144)}")
    print(f"  Vocab: {lc.get('vocab_size', 151936)} (tied embeddings)")

    if args.outtype == "f16":
        out_dtype = np.float16
        ggml_type = GGMLQuantizationType.F16
    else:
        out_dtype = np.float32
        ggml_type = GGMLQuantizationType.F32

    st_files = sorted(model_dir.glob("*.safetensors"))
    if not st_files:
        st_files = sorted(model_dir.glob("model-*.safetensors"))
    handles = [safe_open(str(f), framework="pt") for f in st_files]
    tensor_names = {}
    for idx, h in enumerate(handles):
        for name in h.keys():
            tensor_names[name] = idx
    print(f"  Safetensors: {len(tensor_names)} tensors in {len(st_files)} file(s)")

    outfile = Path(args.output)
    writer = GGUFWriter(str(outfile), "moss_transcribe", use_temp_file=True)
    writer.add_name("MOSS-Transcribe-preview-2B")

    # Audio encoder params
    writer.add_uint32("moss_transcribe.enc.num_mel_bins", n_mels)
    writer.add_uint32("moss_transcribe.enc.encoder_layers", ac.get("encoder_layers", 32))
    writer.add_uint32("moss_transcribe.enc.d_model", ac.get("d_model", 1280))
    writer.add_uint32("moss_transcribe.enc.encoder_attention_heads",
                      ac.get("encoder_attention_heads", 20))
    writer.add_uint32("moss_transcribe.enc.encoder_ffn_dim", ac.get("encoder_ffn_dim", 5120))
    writer.add_uint32("moss_transcribe.enc.downsample_hidden_size",
                      ac.get("downsample_hidden_size", 480))
    writer.add_uint32("moss_transcribe.enc.max_source_positions",
                      ac.get("max_source_positions", 1500))
    writer.add_uint32("moss_transcribe.enc.n_window", ac.get("n_window", 50))
    writer.add_uint32("moss_transcribe.enc.n_window_infer", ac.get("n_window_infer", 800))
    writer.add_uint32("moss_transcribe.enc.output_dim", ac.get("output_dim", 2048))
    writer.add_float32("moss_transcribe.enc.layer_norm_eps", ac.get("layer_norm_eps", 1e-5))

    # Adapter params
    writer.add_uint32("moss_transcribe.adapter.hidden_size",
                      config.get("adapter_hidden_size", 8192))

    # LM params (Qwen3-1.7B)
    writer.add_uint32("moss_transcribe.llm.hidden_size", lc.get("hidden_size", 2048))
    writer.add_uint32("moss_transcribe.llm.num_layers", lc.get("num_hidden_layers", 28))
    writer.add_uint32("moss_transcribe.llm.num_heads", lc.get("num_attention_heads", 16))
    writer.add_uint32("moss_transcribe.llm.num_kv_heads", lc.get("num_key_value_heads", 8))
    writer.add_uint32("moss_transcribe.llm.head_dim", lc.get("head_dim", 128))
    writer.add_uint32("moss_transcribe.llm.intermediate_size", lc.get("intermediate_size", 6144))
    writer.add_uint32("moss_transcribe.llm.vocab_size", lc.get("vocab_size", 151936))
    writer.add_uint32("moss_transcribe.llm.max_position_embeddings",
                      lc.get("max_position_embeddings", 40960))
    writer.add_float32("moss_transcribe.llm.rope_theta", lc.get("rope_theta", 1000000.0))
    writer.add_float32("moss_transcribe.llm.rms_norm_eps", lc.get("rms_norm_eps", 1e-6))
    writer.add_bool("moss_transcribe.llm.tied_embeddings", lc.get("tie_word_embeddings", True))

    # Special tokens (Qwen3 tokenizer; processing_Moss.py legacy layout)
    writer.add_uint32("moss_transcribe.bos_token_id",
                      lc.get("bos_token_id", 151643))
    writer.add_uint32("moss_transcribe.eos_token_id",
                      lc.get("eos_token_id", 151645))
    writer.add_uint32("moss_transcribe.start_token_id", 151644)   # <|im_start|>
    writer.add_uint32("moss_transcribe.audio_start_id", 151669)   # <|audio_bos|>
    writer.add_uint32("moss_transcribe.audio_end_id", 151670)     # <|audio_eos|>

    # Bake mel filterbank + Hann window from WhisperFeatureExtractor so the C++
    # runtime uses the exact same filters as the Python reference.
    try:
        from transformers import WhisperFeatureExtractor
        fe = WhisperFeatureExtractor(
            feature_size=n_mels, sampling_rate=16000,
            hop_length=mel_hop, n_fft=mel_n_fft)
        mel_filters = np.ascontiguousarray(np.asarray(fe.mel_filters, dtype=np.float32))
        writer.add_tensor("audio.mel_filters", mel_filters)
        print(f"  mel_filters shape: {mel_filters.shape}")
        win = np.asarray(
            [0.5 * (1.0 - np.cos(2.0 * np.pi * i / mel_n_fft)) for i in range(mel_n_fft)],
            dtype=np.float32)
        writer.add_tensor("audio.mel_window", win)
        print(f"  mel_window shape: {win.shape}")
    except ImportError:
        print("  WARNING: transformers not available, skipping mel filter bake")

    # Tokenizer: GPT-2 byte-level BPE vocab + merges (Qwen3)
    tok_path = model_dir / "tokenizer.json"
    vocab_size = lc.get("vocab_size", 151936)
    if tok_path.exists():
        with open(tok_path, encoding="utf-8") as f:
            tok_data = json.load(f)
        vocab = tok_data.get("model", {}).get("vocab", {})
        added = tok_data.get("added_tokens", [])
        tokens = [f"[PAD{i}]" for i in range(vocab_size)]
        for token, idx in vocab.items():
            if 0 <= idx < vocab_size:
                tokens[idx] = token
        for entry in added:
            tid = entry.get("id")
            content = entry.get("content")
            if content and tid is not None and 0 <= tid < vocab_size:
                tokens[tid] = content
        writer.add_tokenizer_model("gpt2")
        writer.add_token_list(tokens)
        print(f"  Tokenizer: {len(tokens)} tokens ({len(vocab)} BPE + {len(added)} added)")
        raw_merges = tok_data.get("model", {}).get("merges", [])
        merges = [" ".join(m) if isinstance(m, list) else m for m in raw_merges]
    else:
        merges = []

    merges_path = model_dir / "merges.txt"
    if merges_path.exists():
        merges = []
        with open(merges_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if line and not line.startswith("#"):
                    merges.append(line)
    if merges:
        writer.add_token_merges(merges)
        print(f"  Merges: {len(merges)}")

    # Map and write tensors — stream one-at-a-time to minimize RAM.
    mapped = 0
    skipped = []
    for hf_name in sorted(tensor_names.keys()):
        gguf_name = map_tensor_name(hf_name)
        if gguf_name is None:
            skipped.append(hf_name)
            continue
        h = handles[tensor_names[hf_name]]
        tensor = h.get_tensor(hf_name)
        if tensor.dtype == torch.bfloat16:
            arr = tensor.to(torch.float32).numpy()
        else:
            arr = tensor.numpy()
        if arr.ndim >= 2:
            arr = arr.astype(out_dtype)
            dtype = ggml_type
        else:
            arr = arr.astype(np.float32)
            dtype = GGMLQuantizationType.F32
        writer.add_tensor(gguf_name, arr, raw_dtype=dtype)
        mapped += 1
        del tensor, arr

    print(f"\n  Mapped {mapped} tensors, skipped {len(skipped)}")
    for s in skipped:
        print(f"    SKIP: {s}")

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"\n  Written: {outfile} ({outfile.stat().st_size / 1024**3:.2f} GB)")


if __name__ == "__main__":
    main()
