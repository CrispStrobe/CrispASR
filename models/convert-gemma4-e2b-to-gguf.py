#!/usr/bin/env python3
"""Convert an audio-capable Gemma 4 model to GGUF format for CrispASR.

Supported inputs — the `gemma4` architecture (USM Conformer audio, 1024-dim):
    google/gemma-4-E2B-it   model_type=gemma4   35L×1536  audio 1024  (shipped)
    google/gemma-4-E4B-it   model_type=gemma4   42L×2560  audio 1024  (larger)

NOT yet supported — `gemma4_unified` (a *different* architecture):
    google/gemma-4-12B-it   model_type=gemma4_unified  48L×3840  audio 640
    The 12B uses `Gemma4UnifiedForConditionalGeneration` with a different
    audio encoder (640-dim, `audio_embed_dim` / `audio_samples_per_token`
    keys) and a unified decoder — its tensors do not match this converter's
    USM-Conformer name map, so the emitted GGUF crashes at the first audio
    slice (issue #196). Supporting it needs a separate converter + backend
    audio path. E2B/E4B all feature native audio and work today.

NOTE: the GGUF `general.architecture` string written here is `gemma4e2b`
(CrispASR's internal name for the `gemma4` E2B/E4B family).

Conversion streams tensors one at a time (use_temp_file), so RAM stays
bounded (~16 GB is plenty); the binding constraint is disk for the
download + output.

Usage:
    python models/convert-gemma4-e2b-to-gguf.py --input google/gemma-4-E2B-it --output gemma4-e2b-it.gguf
    python models/convert-gemma4-e2b-to-gguf.py --input google/gemma-4-E4B-it --output gemma4-e4b-it.gguf --outtype f16
    python models/convert-gemma4-e2b-to-gguf.py --input /path/to/local/dir   --output model.gguf

Designed to run on Kaggle (≥16 GB RAM) or better.
"""

import argparse
import json
import struct
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
        print(f"Using local model directory: {model_path}")
        return model_path
    print(f"Downloading model from HuggingFace: {model_id}")
    path = snapshot_download(
        model_id,
        allow_patterns=["*.safetensors", "config.json", "tokenizer.json",
                        "tokenizer_config.json", "generation_config.json"],
    )
    print(f"Downloaded to: {path}")
    return Path(path)


# ── Tensor name mapping ─────────────────────────────────────────────────────
# We strip the clipped-linear min/max scalars (quantization-aware training artifacts)
# and map the rest to a clean GGUF naming scheme.

SKIP_PATTERNS = [
    "model.vision_tower.",   # skip vision encoder (not needed for ASR)
    "model.embed_vision.",   # skip vision embedding
]
# QAT clipping scalars on Gemma4ClippableLinear. We KEEP these for the
# audio tower (HF applies them at inference per `use_clipped_linears=True`)
# but skip for the vision tower (already filtered above).


def should_skip(name: str) -> bool:
    return any(p in name for p in SKIP_PATTERNS)


def map_tensor_name(hf_name: str) -> str | None:
    """Map HuggingFace tensor name to GGUF name. Returns None to skip."""
    if should_skip(hf_name):
        return None

    name = hf_name

    # Audio tower
    name = name.replace("model.audio_tower.layers.", "audio.layers.")
    name = name.replace("model.audio_tower.output_proj.", "audio.output_proj.")
    name = name.replace("model.audio_tower.subsample_conv_projection.", "audio.subsample.")

    # Clipped linear: .linear.weight → .weight (strip the .linear. indirection)
    name = name.replace(".linear.weight", ".weight")

    # Audio conformer sub-modules
    name = name.replace(".feed_forward1.", ".ffn1.")
    name = name.replace(".feed_forward2.", ".ffn2.")
    name = name.replace(".ffw_layer_1.", ".up.")
    name = name.replace(".ffw_layer_2.", ".down.")
    name = name.replace(".pre_layer_norm.", ".pre_ln.")
    name = name.replace(".post_layer_norm.", ".post_ln.")
    name = name.replace(".lconv1d.", ".conv.")
    name = name.replace(".depthwise_conv1d.", ".dw_conv.")
    name = name.replace(".linear_start.", ".gate_proj.")
    name = name.replace(".linear_end.", ".out_proj.")
    name = name.replace(".conv_norm.", ".conv_ln.")
    name = name.replace(".norm_pre_attn.", ".attn_pre_ln.")
    name = name.replace(".norm_post_attn.", ".attn_post_ln.")
    name = name.replace(".norm_out.", ".out_ln.")
    name = name.replace(".self_attn.", ".attn.")
    name = name.replace(".q_proj.", ".q.")
    name = name.replace(".k_proj.", ".k.")
    name = name.replace(".v_proj.", ".v.")
    name = name.replace(".post.", ".o.")  # output projection in audio attn
    name = name.replace(".relative_k_proj.", ".rel_k.")

    # Audio subsampling
    name = name.replace(".input_proj_linear.", ".input_proj.")
    name = name.replace(".layer0.conv.", ".conv0.")
    name = name.replace(".layer0.norm.", ".norm0.")
    name = name.replace(".layer1.conv.", ".conv1.")
    name = name.replace(".layer1.norm.", ".norm1.")

    # Audio embedding projection
    name = name.replace("model.embed_audio.embedding_projection.", "audio.embed_proj.")

    # Language model
    name = name.replace("model.language_model.", "llm.")
    name = name.replace(".self_attn.", ".attn.")
    name = name.replace(".input_layernorm.", ".attn_norm.")
    name = name.replace(".post_attention_layernorm.", ".post_attn_norm.")
    name = name.replace(".pre_feedforward_layernorm.", ".pre_ffn_norm.")
    name = name.replace(".post_feedforward_layernorm.", ".post_ffn_norm.")
    name = name.replace(".post_per_layer_input_norm.", ".post_ple_norm.")
    name = name.replace(".per_layer_input_gate.", ".ple_gate.")
    name = name.replace(".per_layer_projection.", ".ple_proj.")
    name = name.replace(".mlp.gate_proj.", ".ffn.gate.")
    name = name.replace(".mlp.up_proj.", ".ffn.up.")
    name = name.replace(".mlp.down_proj.", ".ffn.down.")

    # Gemma4 double-wide MLP norms (used when use_double_wide_mlp=true,
    # which the E2B-it config sets). Each layer has TWO MLP halves with
    # their own pre/post norms — different from a vanilla SwiGLU.
    name = name.replace(".pre_feedforward_layernorm_2.", ".pre_ffn_norm_2.")
    name = name.replace(".post_feedforward_layernorm_1.", ".post_ffn_norm_1.")
    name = name.replace(".post_feedforward_layernorm_2.", ".post_ffn_norm_2.")

    # Per-attention norm tensors (q_norm/k_norm/v_norm). Gemma4 also has
    # an optional v_norm that vanilla GQA/MQA models lack.
    name = name.replace(".attn.v_norm.", ".attn.v_norm.")

    return name


def main():
    parser = argparse.ArgumentParser(description="Convert Gemma-4-E2B to GGUF")
    parser.add_argument("--input", required=True, help="HF model ID or local directory")
    parser.add_argument("--output", required=True, help="Output GGUF path")
    parser.add_argument("--outtype", default="f16", choices=["f32", "f16"],
                        help="Output data type for 2D+ tensors (default: f16)")
    args = parser.parse_args()

    model_dir = load_model_dir(args.input)

    # Load config
    with open(model_dir / "config.json", encoding="utf-8") as f:
        config = json.load(f)

    ac = config.get("audio_config", {})
    tc = config.get("text_config", {})

    # Guard: this converter + the C++ gemma4 backend implement the `gemma4`
    # architecture (E2B / E4B): a USM Conformer audio encoder (1024-dim) plus
    # the standard gemma4 decoder. The 12B checkpoint is a DIFFERENT
    # architecture — `gemma4_unified` / Gemma4UnifiedForConditionalGeneration
    # — with a 640-dim audio encoder (audio_embed_dim / audio_samples_per_token)
    # and a unified decoder, whose tensor names do not match the map below.
    # Converting it anyway produces a GGUF that loads but crashes at the first
    # audio slice (issue #196: Windows 0xC0000005 / return code 3221225477).
    # Detect it up front and fail loudly rather than emit a broken GGUF.
    model_type = config.get("model_type", "")
    architectures = config.get("architectures", []) or []
    if model_type == "gemma4_unified" or any("Unified" in a for a in architectures):
        print(
            "\nError: this is the `gemma4_unified` architecture "
            f"(model_type={model_type!r}, {architectures}) — the Gemma-4 12B "
            "model.\n"
            "  It uses a different 640-dim audio encoder + unified decoder "
            "that this converter and the C++ gemma4 backend do not support "
            "yet (issue #196).\n"
            "  Supported today (model_type=gemma4, USM Conformer 1024-dim):\n"
            "    google/gemma-4-E2B-it   (already shipped)\n"
            "    google/gemma-4-E4B-it   (larger; same architecture)",
            file=sys.stderr,
        )
        sys.exit(1)
    if not ac:
        print(
            "\nError: this checkpoint has no `audio_config` — it is not an "
            "audio-capable Gemma 4 model and cannot be used for ASR.\n"
            "  Use google/gemma-4-E2B-it or google/gemma-4-E4B-it.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"\nGemma-4-E2B")
    print(f"  Audio: {ac.get('num_hidden_layers')}L, hidden={ac.get('hidden_size')}, "
          f"heads={ac.get('num_attention_heads')}, conv_k={ac.get('conv_kernel_size')}")
    print(f"  LLM:   {tc.get('num_hidden_layers')}L, hidden={tc.get('hidden_size')}, "
          f"heads={tc.get('num_attention_heads')}, kv_heads={tc.get('num_key_value_heads')}, "
          f"head_dim={tc.get('head_dim')}")
    print(f"  Vocab: {tc.get('vocab_size')}")

    # Output dtype
    if args.outtype == "f16":
        out_dtype = np.float16
        ggml_type = GGMLQuantizationType.F16
    else:
        out_dtype = np.float32
        ggml_type = GGMLQuantizationType.F32

    # Open safetensors
    st_files = sorted(model_dir.glob("*.safetensors"))
    if not st_files:
        print(f"Error: No .safetensors files found in {model_dir}")
        sys.exit(1)
    handles = [safe_open(str(f), framework="pt") for f in st_files]
    tensor_names = {}
    for idx, h in enumerate(handles):
        for name in h.keys():
            tensor_names[name] = idx
    print(f"  Safetensors: {len(tensor_names)} tensors in {len(st_files)} file(s)")

    # Belt-and-suspenders: even if a config carried an `audio_config`, refuse
    # to emit a GGUF when the actual audio-tower weights are absent (this is
    # what produces the silent-then-crash GGUF in issue #196).
    if not any("audio_tower" in n for n in tensor_names):
        print(
            "\nError: no `model.audio_tower.*` tensors found in the "
            "checkpoint — cannot build an audio-LLM GGUF.\n"
            "  Use an audio-capable Gemma 3n model (E2B / E4B); plain "
            "Gemma 3 (12B/27B) has no audio tower.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Create GGUF writer
    outfile = Path(args.output)
    writer = GGUFWriter(str(outfile), "gemma4e2b", use_temp_file=True)

    # Write metadata
    writer.add_name(config.get("_name_or_path", "gemma-4-E2B-it"))

    # Audio encoder params
    writer.add_uint32("gemma4e2b.audio.hidden_size", ac.get("hidden_size", 1024))
    writer.add_uint32("gemma4e2b.audio.num_layers", ac.get("num_hidden_layers", 12))
    writer.add_uint32("gemma4e2b.audio.num_heads", ac.get("num_attention_heads", 8))
    writer.add_uint32("gemma4e2b.audio.conv_kernel_size", ac.get("conv_kernel_size", 5))
    writer.add_uint32("gemma4e2b.audio.chunk_size", ac.get("attention_chunk_size", 12))
    writer.add_uint32("gemma4e2b.audio.context_left", ac.get("attention_context_left", 13))
    writer.add_uint32("gemma4e2b.audio.output_proj_dims", ac.get("output_proj_dims", 1536))
    writer.add_float32("gemma4e2b.audio.residual_weight", ac.get("residual_weight", 0.5))
    writer.add_float32("gemma4e2b.audio.attention_logit_cap", ac.get("attention_logit_cap", 50.0))

    # LLM params
    writer.add_uint32("gemma4e2b.llm.hidden_size", tc.get("hidden_size", 1536))
    writer.add_uint32("gemma4e2b.llm.num_layers", tc.get("num_hidden_layers", 35))
    writer.add_uint32("gemma4e2b.llm.num_heads", tc.get("num_attention_heads", 8))
    writer.add_uint32("gemma4e2b.llm.num_kv_heads", tc.get("num_key_value_heads", 1))
    writer.add_uint32("gemma4e2b.llm.head_dim", tc.get("head_dim", 256))
    writer.add_uint32("gemma4e2b.llm.intermediate_size", tc.get("intermediate_size", 6144))
    writer.add_uint32("gemma4e2b.llm.vocab_size", tc.get("vocab_size", 262144))
    writer.add_uint32("gemma4e2b.llm.max_position_embeddings", tc.get("max_position_embeddings", 131072))
    writer.add_uint32("gemma4e2b.llm.sliding_window", tc.get("sliding_window", 512))
    writer.add_float32("gemma4e2b.llm.rope_theta", tc.get("rope_theta", 10000.0))
    writer.add_float32("gemma4e2b.llm.final_logit_softcapping", tc.get("final_logit_softcapping", 30.0))
    writer.add_float32("gemma4e2b.llm.rms_norm_eps", tc.get("rms_norm_eps", 1e-6))

    # Gemma4-specific architecture flags. These were missed by the
    # original converter and are required for KV sharing, the
    # sliding/full attention split, and the double-wide MLP.
    writer.add_uint32("gemma4e2b.llm.hidden_size_per_layer_input",
                      tc.get("hidden_size_per_layer_input", 256))
    writer.add_uint32("gemma4e2b.llm.num_kv_shared_layers",
                      tc.get("num_kv_shared_layers", 0))
    writer.add_uint32("gemma4e2b.llm.global_head_dim",
                      tc.get("global_head_dim", tc.get("head_dim", 256)))
    writer.add_bool("gemma4e2b.llm.use_double_wide_mlp",
                    tc.get("use_double_wide_mlp", False))
    writer.add_bool("gemma4e2b.llm.attention_k_eq_v",
                    tc.get("attention_k_eq_v", False))
    # layer_types is a list of strings ("sliding_attention" / "full_attention").
    # Persist as a 1-byte-per-layer mask: 1 = full, 0 = sliding.
    layer_types = tc.get("layer_types", [])
    if layer_types:
        full_mask = [1 if t == "full_attention" else 0 for t in layer_types]
        writer.add_array("gemma4e2b.llm.layer_full_mask", full_mask)
    # rope_theta for the FULL-attention layers (the partial_rotary one).
    rope_params = tc.get("rope_parameters", {}) or {}
    full_rope = rope_params.get("full_attention", {}) or {}
    writer.add_float32("gemma4e2b.llm.rope_theta_full",
                       full_rope.get("rope_theta", 1000000.0))
    writer.add_float32("gemma4e2b.llm.partial_rotary_factor",
                       full_rope.get("partial_rotary_factor", 1.0))

    # Tokenizer: store BPE tokens from tokenizer.json
    tok_path = model_dir / "tokenizer.json"
    if tok_path.exists():
        with open(tok_path, encoding="utf-8") as f:
            tok_data = json.load(f)
        vocab = tok_data.get("model", {}).get("vocab", {})
        if vocab:
            tokens = [""] * len(vocab)
            for token, idx in vocab.items():
                if idx < len(tokens):
                    tokens[idx] = token
            writer.add_token_list(tokens)
            # NOTE: merges skipped — add_token_merges produces GGUF type 9
            # which our C reader rejects ("invalid GGUF type 9"). The BPE
            # decoder in core/bpe.h works from vocab alone. 514K merges
            # would also bloat the header by ~30 MB.
            print(f"  Tokenizer: {len(tokens)} tokens (merges skipped)")

    # Generate and store mel filterbank + Hann window (same as qwen3/voxtral)
    # Gemma4 uses WhisperFeatureExtractor: n_fft=400, hop=160, n_mels=128, sr=16000
    n_fft = 400
    n_mels = 128
    sr = 16000
    n_freqs = n_fft // 2 + 1

    # Hann window [n_fft]
    hann = np.hanning(n_fft + 1)[:n_fft].astype(np.float32)
    writer.add_tensor("audio.mel_window", hann, raw_dtype=GGMLQuantizationType.F32)

    # Mel filterbank [n_freqs, n_mels] (HTK scale, slaney normalization)
    mel_lo = 2595.0 * np.log10(1.0 + 0.0 / 700.0)
    mel_hi = 2595.0 * np.log10(1.0 + (sr / 2.0) / 700.0)
    mel_pts = 700.0 * (10.0 ** (np.linspace(mel_lo, mel_hi, n_mels + 2) / 2595.0) - 1.0)
    fft_freqs = np.arange(n_freqs) * sr / n_fft
    fb = np.zeros((n_freqs, n_mels), dtype=np.float64)
    for m in range(n_mels):
        lo, ctr, hi = mel_pts[m], mel_pts[m + 1], mel_pts[m + 2]
        if hi > lo:
            enorm = 2.0 / (hi - lo)
            for f in range(n_freqs):
                if lo <= fft_freqs[f] <= ctr and ctr > lo:
                    fb[f, m] = (fft_freqs[f] - lo) / (ctr - lo) * enorm
                elif ctr < fft_freqs[f] <= hi and hi > ctr:
                    fb[f, m] = (hi - fft_freqs[f]) / (hi - ctr) * enorm
    fb = np.ascontiguousarray(fb.astype(np.float32))
    writer.add_tensor("audio.mel_filters", fb, raw_dtype=GGMLQuantizationType.F32)
    print(f"  Mel: {n_mels}-bin filterbank ({n_freqs}x{n_mels}), Hann window ({n_fft})")

    # Map and write tensors (streaming, one at a time)
    mapped = 0
    skipped = 0
    for hf_name in sorted(tensor_names.keys()):
        gguf_name = map_tensor_name(hf_name)
        if gguf_name is None:
            skipped += 1
            continue

        data = handles[tensor_names[hf_name]].get_tensor(hf_name).to(torch.float32).numpy()

        # Scalar tensors → 1-element F32
        if data.ndim == 0:
            data = np.array([data.item()], dtype=np.float32)
            writer.add_tensor(gguf_name, data, raw_dtype=GGMLQuantizationType.F32)
        # 1D tensors (norms, biases, scales) → F32
        elif data.ndim <= 1:
            data = np.ascontiguousarray(data.astype(np.float32))
            writer.add_tensor(gguf_name, data, raw_dtype=GGMLQuantizationType.F32)
        else:
            data = np.ascontiguousarray(data.astype(out_dtype))
            writer.add_tensor(gguf_name, data, raw_dtype=ggml_type)

        mapped += 1
        if mapped % 100 == 0:
            print(f"  [{mapped}] {gguf_name}")

    print(f"\nMapped: {mapped}, Skipped: {skipped} (clipping scalars + vision)")
    print(f"Writing to {outfile}...")
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    size_mb = outfile.stat().st_size / 1024 / 1024
    print(f"Done! {outfile} ({size_mb:.1f} MB, {mapped} tensors)")


if __name__ == "__main__":
    main()
