#!/usr/bin/env python3
"""Convert TADA aligner (wav2vec2-large + Llama CTC head) → GGUF.

The TADA aligner is a wav2vec2-large model fine-tuned with a CTC head that
maps to the Llama-3.2 tokenizer vocabulary (128256 classes). It takes 16 kHz
audio and produces per-frame log-probabilities over the Llama token space,
which are then used by a DP alignment algorithm to find the optimal
text-to-audio alignment.

The output GGUF uses the same tensor naming as convert-wav2vec2-to-gguf.py
so that the existing wav2vec2_load() + wav2vec2_compute_logits() C++ runtime
works unchanged. The Llama tokenizer tokens+merges are embedded for the
DP alignment to tokenize the transcript.

Usage:
    python models/convert-tada-aligner-to-gguf.py \\
        --codec-repo HumeAI/tada-codec \\
        --output tada-aligner-en.gguf

    # Language-specific:
    python models/convert-tada-aligner-to-gguf.py \\
        --codec-repo HumeAI/tada-codec \\
        --language fr \\
        --output tada-aligner-fr.gguf

Supported languages: ar, ch, de, es, fr, it, ja, pl, pt (omit for English)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

try:
    from gguf import GGUFWriter, GGMLQuantizationType
except ImportError:
    sys.exit("gguf not found: pip install gguf")

try:
    import torch
except ImportError:
    sys.exit("torch not found: pip install torch")

try:
    from safetensors import safe_open
except ImportError:
    sys.exit("safetensors not found: pip install safetensors")

try:
    from huggingface_hub import hf_hub_download
except ImportError:
    sys.exit("huggingface_hub not found: pip install huggingface_hub")


ARCH = "wav2vec2"
SUPPORTED_LANGUAGES = {"ar", "ch", "de", "es", "fr", "it", "ja", "pl", "pt"}

# wav2vec2-large architecture constants
W2V_HIDDEN = 1024
W2V_LAYERS = 24
W2V_HEADS = 16
W2V_INTERMEDIATE = 4096
W2V_NUM_CNN_LAYERS = 7
W2V_CONV_DIM = [512, 512, 512, 512, 512, 512, 512]
W2V_CONV_KERNEL = [10, 3, 3, 3, 3, 2, 2]
W2V_CONV_STRIDE = [5, 2, 2, 2, 2, 2, 2]
W2V_POS_CONV_K = 128
W2V_POS_CONV_GROUPS = 16


def f32(t: torch.Tensor) -> np.ndarray:
    return t.detach().float().cpu().numpy().astype(np.float32)


def f16(t: torch.Tensor) -> np.ndarray:
    return t.detach().float().cpu().numpy().astype(np.float16)


def get_tensor(handles, name_to_idx, name):
    """Load a single tensor from the safetensors file(s)."""
    idx = name_to_idx[name]
    return handles[idx].get_tensor(name)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--codec-repo", default="HumeAI/tada-codec",
                    help="HuggingFace repo containing aligner weights")
    ap.add_argument("--language", default=None,
                    help=f"Language code. Supported: {', '.join(sorted(SUPPORTED_LANGUAGES))}")
    ap.add_argument("--output", required=True, help="Output .gguf path")
    ap.add_argument("--outtype", default="f16", choices=["f32", "f16"],
                    help="Weight dtype for linear/attention layers (default: f16)")
    ap.add_argument("--tokenizer", default="unsloth/Llama-3.2-1B",
                    help="Tokenizer to embed (default: unsloth/Llama-3.2-1B)")
    args = ap.parse_args()

    lang = args.language
    if lang and lang not in SUPPORTED_LANGUAGES:
        sys.exit(f"Unknown language '{lang}'. Supported: {', '.join(sorted(SUPPORTED_LANGUAGES))}")

    subfolder = f"aligner-{lang}" if lang else "aligner"
    lang_label = lang or "en"

    # ── Download aligner weights ──
    print(f"Downloading TADA aligner ({lang_label}) from {args.codec_repo}/{subfolder}...")
    st_path = hf_hub_download(args.codec_repo, f"{subfolder}/model.safetensors")
    cfg_path = hf_hub_download(args.codec_repo, f"{subfolder}/config.json")

    with open(cfg_path, encoding="utf-8") as f:
        cfg = json.load(f)
    print(f"  Config: {json.dumps(cfg, indent=2)}")

    # ── Load safetensors ──
    handles = [safe_open(st_path, framework="pt")]
    name_to_idx = {}
    for i, h in enumerate(handles):
        for k in h.keys():
            name_to_idx[k] = i

    all_keys = sorted(name_to_idx.keys())
    print(f"  {len(all_keys)} tensors in safetensors")

    # Print first few keys to understand naming
    for k in all_keys[:5]:
        t = handles[name_to_idx[k]].get_tensor(k)
        print(f"    {k}: {list(t.shape)} {t.dtype}")
    if len(all_keys) > 5:
        print(f"    ... ({len(all_keys)} total)")
        # Print lm_head
        for k in all_keys:
            if "lm_head" in k:
                t = handles[name_to_idx[k]].get_tensor(k)
                print(f"    {k}: {list(t.shape)} {t.dtype}")

    # ── Load tokenizer ──
    print(f"\nLoading tokenizer: {args.tokenizer}")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    vocab_size = len(tokenizer)
    print(f"  Vocab size: {vocab_size}")

    # ── Remap tensor names ──
    # The Aligner stores weights as:
    #   encoder.wav2vec2.feature_extractor.conv_layers.{i}.conv.weight
    #   encoder.wav2vec2.feature_extractor.conv_layers.{i}.layer_norm.weight
    #   encoder.wav2vec2.feature_projection.layer_norm.weight
    #   encoder.wav2vec2.feature_projection.projection.weight
    #   encoder.wav2vec2.encoder.pos_conv_embed.conv.parametrizations.weight.original{0,1}
    #   encoder.wav2vec2.encoder.layer_norm.weight
    #   encoder.wav2vec2.encoder.layers.{i}.attention.{q,k,v,out}_proj.weight
    #   encoder.wav2vec2.encoder.layers.{i}.layer_norm.weight
    #   encoder.wav2vec2.encoder.layers.{i}.feed_forward.intermediate_dense.weight
    #   encoder.wav2vec2.encoder.layers.{i}.feed_forward.output_dense.weight
    #   encoder.wav2vec2.encoder.layers.{i}.final_layer_norm.weight
    #   encoder.lm_head.weight
    #   encoder.lm_head.bias

    # We need to strip "encoder." and then normalize "wav2vec2.*" same as
    # convert-wav2vec2-to-gguf.py does.

    out_dtype = np.float16 if args.outtype == "f16" else np.float32
    out_qt = GGMLQuantizationType.F16 if args.outtype == "f16" else GGMLQuantizationType.F32

    # Detect weight-norm in pos_conv
    wn_g_key = None
    wn_v_key = None
    for k in all_keys:
        if "pos_conv_embed.conv.parametrizations.weight.original0" in k:
            wn_g_key = k
        elif "pos_conv_embed.conv.parametrizations.weight.original1" in k:
            wn_v_key = k

    # Materialize weight-norm for pos_conv
    # weight_norm dim=0 default: g has shape [Cout, 1, 1], v has shape [Cout, Cin/G, K]
    # But TADA stores g as [1, 1, K] (dim=2?). Handle both cases.
    pos_conv_weight = None
    if wn_g_key and wn_v_key:
        g = get_tensor(handles, name_to_idx, wn_g_key).to(torch.float32)
        v = get_tensor(handles, name_to_idx, wn_v_key).to(torch.float32)
        # Determine which dim was normed: g has size 1 on all dims except the normed one
        # For standard weight_norm with dim=0: norm over dims (1, 2, ...)
        # Detect: if g.shape[0] == 1 and g.shape[-1] > 1, it's dim=-1 or dim=2
        if g.shape[0] == 1 and v.shape[0] > 1:
            # Weight norm was applied on a non-zero dim. The g tensor broadcasts.
            # Compute v_norm for each "group" along the g-expanded dim
            # g shape [1, 1, K], v shape [Cout, Cin/G, K]
            # norm over dims (0, 1) keeping dim 2
            norm_dims = tuple(i for i in range(v.ndim) if g.shape[i] == 1 and v.shape[i] > 1)
            if not norm_dims:
                norm_dims = tuple(range(v.ndim - 1))
        else:
            norm_dims = tuple(range(1, v.ndim))
        v_norm = torch.linalg.vector_norm(v, dim=norm_dims, keepdim=True)
        pos_conv_weight = v * (g / (v_norm + 1e-12))
        print(f"  WN materialized: pos_conv g={list(g.shape)} v={list(v.shape)} "
              f"norm_dims={norm_dims} → w={list(pos_conv_weight.shape)}")

    # ── Write GGUF ──
    out_path = Path(args.output)
    print(f"\nWriting GGUF: {out_path}")
    w = GGUFWriter(str(out_path), arch=ARCH, use_temp_file=False)
    w.add_name(f"tada-aligner-{lang_label}")

    # Detect CNN norm type from tensors: if layer 0 has layer_norm, it's
    # "group" variant (InstanceNorm on layer 0 only); if all layers have
    # layer_norm it's "layer" variant (LayerNorm on all CNN layers).
    cnn_layer0_has_norm = f"encoder.wav2vec2.feature_extractor.conv_layers.0.layer_norm.weight" in name_to_idx
    cnn_layer1_has_norm = f"encoder.wav2vec2.feature_extractor.conv_layers.1.layer_norm.weight" in name_to_idx
    # group: InstanceNorm on layer 0 only (feat_extract_norm_type=0)
    # layer: LayerNorm on all layers (feat_extract_norm_type=1)
    feat_norm_type = 1 if cnn_layer1_has_norm else 0
    print(f"  CNN norm type: {'layer' if feat_norm_type else 'group'} "
          f"(layer0_norm={cnn_layer0_has_norm}, layer1_norm={cnn_layer1_has_norm})")

    # Metadata (same keys as convert-wav2vec2-to-gguf.py)
    w.add_uint32(f"{ARCH}.vocab_size", vocab_size)
    w.add_uint32(f"{ARCH}.hidden_size", W2V_HIDDEN)
    w.add_uint32(f"{ARCH}.num_hidden_layers", W2V_LAYERS)
    w.add_uint32(f"{ARCH}.num_attention_heads", W2V_HEADS)
    w.add_uint32(f"{ARCH}.intermediate_size", W2V_INTERMEDIATE)
    w.add_uint32(f"{ARCH}.num_feat_extract_layers", W2V_NUM_CNN_LAYERS)
    w.add_uint32(f"{ARCH}.num_conv_pos_embeddings", W2V_POS_CONV_K)
    w.add_uint32(f"{ARCH}.num_conv_pos_embedding_groups", W2V_POS_CONV_GROUPS)
    w.add_float32(f"{ARCH}.layer_norm_eps", 1e-5)
    w.add_uint32(f"{ARCH}.pad_token_id", 0)
    w.add_uint32(f"{ARCH}.feat_extract_norm_type", feat_norm_type)
    w.add_uint32(f"{ARCH}.do_stable_layer_norm", 1)    # pre-norm (wav2vec2-large)
    w.add_uint32(f"{ARCH}.global_ln_before_encoder", 0)

    for i in range(W2V_NUM_CNN_LAYERS):
        w.add_uint32(f"{ARCH}.conv_dim_{i}", W2V_CONV_DIM[i])
        w.add_uint32(f"{ARCH}.conv_kernel_{i}", W2V_CONV_KERNEL[i])
        w.add_uint32(f"{ARCH}.conv_stride_{i}", W2V_CONV_STRIDE[i])

    # TADA-specific metadata
    if lang:
        w.add_string("tada.aligner.language", lang)

    # Tokenizer — embed Llama tokens for DP alignment
    vocab_list = [tokenizer.convert_ids_to_tokens(i) or f"<{i}>" for i in range(vocab_size)]
    w.add_array("tokenizer.ggml.tokens", vocab_list)
    w.add_uint32("tokenizer.ggml.padding_token_id", 0)

    # Embed BPE merges (REQUIRED for C++ tokenization). Without them the C++
    # core_bpe::tokenize_simple falls back to byte-level tokens (e.g. a 5-word
    # transcript becomes ~26 byte-tokens), which mismatches the runtime's BPE
    # prompt tokenization and breaks voice cloning (empty/garbled synth).
    # tokenizers' get_merges() is version-dependent, so parse tokenizer.json
    # directly. Its "model.merges" is either ["a b", ...] (older) or
    # [["a","b"], ...] (tokenizers >= 0.20); core_bpe wants "a b" strings in
    # rank order.
    merges = []
    try:
        import json as _json
        tj = _json.loads(tokenizer.backend_tokenizer.to_str())
        raw = tj.get("model", {}).get("merges", [])
        for m in raw:
            merges.append(m if isinstance(m, str) else (m[0] + " " + m[1]))
    except Exception as e:
        print(f"  WARNING: could not extract BPE merges ({e}); "
              "C++ tokenization will byte-fall-back and voice refs will be broken")
    if merges:
        w.add_array("tokenizer.ggml.merges", merges)
        print(f"  Embedded {len(merges)} BPE merges")

    # ── Write tensors ──
    print("\nWriting tensors:")

    # Helper to select F32 or F16
    def wt(t):
        """Convert tensor to output dtype for weight tensors."""
        arr = t.detach().float().cpu().numpy()
        return np.ascontiguousarray(arr.astype(out_dtype))

    def write_t(gguf_name, hf_name, force_f32=False):
        """Load tensor from safetensors and write to GGUF."""
        t = get_tensor(handles, name_to_idx, hf_name).to(torch.float32)
        if force_f32 or t.ndim <= 1:
            arr = np.ascontiguousarray(f32(t))
            w.add_tensor(gguf_name, arr, raw_dtype=GGMLQuantizationType.F32)
        else:
            arr = np.ascontiguousarray(wt(t))
            w.add_tensor(gguf_name, arr, raw_dtype=out_qt)
        return t

    n_written = 0

    # CNN feature extractor
    # Note: TADA aligner uses "group" norm variant — only layer 0 has
    # InstanceNorm (stored as layer_norm in HF), layers 1-6 have no norm
    # and no bias.
    for i in range(W2V_NUM_CNN_LAYERS):
        pfx = f"encoder.wav2vec2.feature_extractor.conv_layers.{i}"
        write_t(f"cnn.{i}.conv.weight", f"{pfx}.conv.weight", force_f32=True)
        bias_key = f"{pfx}.conv.bias"
        if bias_key in name_to_idx:
            write_t(f"cnn.{i}.conv.bias", bias_key, force_f32=True)
        ln_key = f"{pfx}.layer_norm.weight"
        has_norm = ln_key in name_to_idx
        w.add_uint32(f"{ARCH}.cnn_has_norm_{i}", int(has_norm))
        if has_norm:
            write_t(f"cnn.{i}.norm.weight", f"{pfx}.layer_norm.weight", force_f32=True)
            write_t(f"cnn.{i}.norm.bias", f"{pfx}.layer_norm.bias", force_f32=True)
        n_written += 1
        print(f"  cnn.{i}: norm={'yes' if has_norm else 'no'} bias={'yes' if bias_key in name_to_idx else 'no'}")

    # Feature projection
    fp_pfx = "encoder.wav2vec2.feature_projection"
    write_t("feat_proj.ln.weight", f"{fp_pfx}.layer_norm.weight", force_f32=True)
    write_t("feat_proj.ln.bias", f"{fp_pfx}.layer_norm.bias", force_f32=True)
    write_t("feat_proj.weight", f"{fp_pfx}.projection.weight")
    write_t("feat_proj.bias", f"{fp_pfx}.projection.bias", force_f32=True)
    n_written += 1
    print(f"  feat_proj: ok")

    # Positional conv (weight-norm materialized)
    if pos_conv_weight is not None:
        arr = np.ascontiguousarray(f32(pos_conv_weight))
        w.add_tensor("pos_conv.weight", arr, raw_dtype=GGMLQuantizationType.F32)
    else:
        # Fallback: try direct weight key
        pc_key = "encoder.wav2vec2.encoder.pos_conv_embed.conv.weight"
        if pc_key in name_to_idx:
            write_t("pos_conv.weight", pc_key, force_f32=True)
    # Bias
    pc_bias_key = "encoder.wav2vec2.encoder.pos_conv_embed.conv.bias"
    if pc_bias_key in name_to_idx:
        write_t("pos_conv.bias", pc_bias_key, force_f32=True)
    n_written += 1
    print(f"  pos_conv: ok")

    # Encoder global LayerNorm
    enc_pfx = "encoder.wav2vec2.encoder"
    write_t("enc.ln.weight", f"{enc_pfx}.layer_norm.weight", force_f32=True)
    write_t("enc.ln.bias", f"{enc_pfx}.layer_norm.bias", force_f32=True)
    n_written += 1

    # Transformer encoder layers
    for i in range(W2V_LAYERS):
        lp = f"{enc_pfx}.layers.{i}"

        # Pre-attention LayerNorm
        write_t(f"enc.{i}.ln1.weight", f"{lp}.layer_norm.weight", force_f32=True)
        write_t(f"enc.{i}.ln1.bias", f"{lp}.layer_norm.bias", force_f32=True)

        # Attention projections
        for proj, short in [("q_proj", "q"), ("k_proj", "k"), ("v_proj", "v"), ("out_proj", "out")]:
            write_t(f"enc.{i}.attn.{short}.weight", f"{lp}.attention.{proj}.weight")
            write_t(f"enc.{i}.attn.{short}.bias", f"{lp}.attention.{proj}.bias", force_f32=True)

        # Pre-FFN LayerNorm
        write_t(f"enc.{i}.ln2.weight", f"{lp}.final_layer_norm.weight", force_f32=True)
        write_t(f"enc.{i}.ln2.bias", f"{lp}.final_layer_norm.bias", force_f32=True)

        # FFN
        write_t(f"enc.{i}.ffn.fc1.weight", f"{lp}.feed_forward.intermediate_dense.weight")
        write_t(f"enc.{i}.ffn.fc1.bias", f"{lp}.feed_forward.intermediate_dense.bias", force_f32=True)
        write_t(f"enc.{i}.ffn.fc2.weight", f"{lp}.feed_forward.output_dense.weight")
        write_t(f"enc.{i}.ffn.fc2.bias", f"{lp}.feed_forward.output_dense.bias", force_f32=True)

        n_written += 1
        if i == 0 or i == W2V_LAYERS - 1 or (i + 1) % 8 == 0:
            print(f"  enc.{i}: ok")

    # LM head (CTC output → 128256 classes)
    lm_w_key = "encoder.lm_head.weight"
    lm_b_key = "encoder.lm_head.bias"
    # LM head is huge (128256 × 1024, ~525 MB F32). It is the CTC projection that
    # drives the alignment argmax/DP, so it is stored F32 by default (safest).
    # TADA_ALIGNER_LMHEAD_F16=1 stores it at --outtype precision (F16 halves it to
    # ~262 MB) — only enable if validated (forced alignment is robust to logit
    # rounding, but #192 measured q4_k on the ENCODER already costs ~220 ms drift).
    lm_head_force_f32 = os.environ.get("TADA_ALIGNER_LMHEAD_F16", "") not in ("1", "true", "yes")
    write_t("lm_head.weight", lm_w_key, force_f32=lm_head_force_f32)
    if lm_b_key in name_to_idx:
        write_t("lm_head.bias", lm_b_key, force_f32=True)
    else:
        # Some models may not have bias
        print("  lm_head: no bias found, creating zeros")
        w.add_tensor("lm_head.bias", np.zeros(vocab_size, dtype=np.float32),
                      raw_dtype=GGMLQuantizationType.F32)
    lm_t = get_tensor(handles, name_to_idx, lm_w_key)
    print(f"  lm_head: {list(lm_t.shape)} → {vocab_size} classes")
    n_written += 1

    print(f"\n  Total tensor groups written: {n_written}")

    # Finalize
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()

    sz = out_path.stat().st_size / (1024 * 1024)
    print(f"\nDone: {out_path} ({sz:.1f} MB)")
    print(f"\nThis GGUF is compatible with wav2vec2_load() — use it as the aligner")
    print(f"component of the --make-ref pipeline.")


if __name__ == "__main__":
    main()
