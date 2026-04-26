#!/usr/bin/env python3
"""Convert Facebook OmniASR-CTC to GGUF.

Architecture: wav2vec2-style CNN frontend + Transformer encoder + CTC head.
Raw 16kHz PCM input → 9812-token SentencePiece output.

Sizes: 300M (24L, d=1024), 1B (48L, d=1280), 3B, 7B

Usage:
  python models/convert-omniasr-ctc-to-gguf.py \
      --input facebook/omniASR-CTC-300M \
      --output omniasr-ctc-300m.gguf
"""

import argparse
import os
import sys

import numpy as np

try:
    import gguf
except ImportError:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ggml", "python"))
    import gguf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="HF model ID (e.g. facebook/omniASR-CTC-300M)")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    import torch
    from huggingface_hub import hf_hub_download, list_repo_files

    model_name = args.input.split("/")[-1]

    # Detect format: v1 (fairseq2 .pt) vs v2 (HF transformers safetensors)
    repo_files = list_repo_files(args.input)
    is_v2 = "model.safetensors" in repo_files and "config.json" in repo_files
    is_v1 = any(f.endswith(".pt") for f in repo_files)

    if is_v2:
        print(f"Detected HF transformers format (v2): {args.input}")
        from safetensors.torch import load_file
        import json

        sf_path = hf_hub_download(args.input, "model.safetensors")
        cfg_path = hf_hub_download(args.input, "config.json")
        vocab_path = hf_hub_download(args.input, "vocab.json")

        sd_raw = load_file(sf_path)
        with open(cfg_path) as f:
            cfg = json.load(f)
        with open(vocab_path) as f:
            vocab_json = json.load(f)

        # Map v2 tensor names to v1 names for unified processing
        sd = {}
        for v2_name, tensor in sd_raw.items():
            v1 = v2_name
            # Top-level replacements (order matters — longest prefix first)
            v1 = v1.replace("wav2vec2.feature_extractor.conv_layers.", "encoder_frontend.feature_extractor.layers.")
            v1 = v1.replace("wav2vec2.feature_projection.projection.", "encoder_frontend.model_dim_proj.")
            v1 = v1.replace("wav2vec2.feature_projection.layer_norm.", "encoder_frontend.post_extract_layer_norm.")
            v1 = v1.replace("wav2vec2.encoder.pos_conv_embed.conv.", "encoder_frontend.pos_encoder.conv.")
            v1 = v1.replace("wav2vec2.encoder.layer_norm.", "encoder.layer_norm.")
            v1 = v1.replace("wav2vec2.encoder.layers.", "encoder.layers.")
            v1 = v1.replace("lm_head.", "final_proj.")
            # Sub-module replacements
            v1 = v1.replace(".attention.", ".self_attn.")
            v1 = v1.replace(".out_proj.", ".output_proj.")
            v1 = v1.replace(".feed_forward.intermediate_dense.", ".ffn.inner_proj.")
            v1 = v1.replace(".feed_forward.output_dense.", ".ffn.output_proj.")
            v1 = v1.replace(".final_layer_norm.", ".ffn_layer_norm.")
            # v2 has .layer_norm per encoder layer = self_attn_layer_norm in v1
            if "encoder.layers." in v1 and ".layer_norm." in v1 and "ffn" not in v1:
                v1 = v1.replace(".layer_norm.", ".self_attn_layer_norm.")
            sd[v1] = tensor

        # Build vocab from vocab.json
        vocab = [""] * len(vocab_json)
        for token, idx in vocab_json.items():
            if idx < len(vocab):
                vocab[idx] = token

        print(f"Loaded {len(sd)} tensors from {model_name} (v2 format)")
        print(f"Vocab: {len(vocab)} tokens from vocab.json")

    elif is_v1:
        import sentencepiece as spm

        pt_file = next(f for f in repo_files if f.endswith(".pt"))
        pt_path = hf_hub_download(args.input, pt_file)
        tok_path = hf_hub_download(args.input, "omniASR_tokenizer.model")

        ckpt = torch.load(pt_path, map_location="cpu", weights_only=False)
        sd = ckpt["model"]

        sp = spm.SentencePieceProcessor()
        sp.Load(tok_path)
        vocab = [sp.IdToPiece(i) for i in range(sp.GetPieceSize())]

        print(f"Loaded {len(sd)} tensors from {model_name} (v1 format)")
    else:
        print(f"ERROR: unrecognized model format in {args.input}")
        sys.exit(1)

    # Infer architecture (works for both v1 and v2 after name mapping)
    n_enc = max(int(k.split('.')[2]) for k in sd if k.startswith("encoder.layers.")) + 1
    d_model = sd["encoder.layers.0.self_attn.q_proj.weight"].shape[0]
    d_ffn = sd["encoder.layers.0.ffn.inner_proj.weight"].shape[0]
    # Infer n_heads from config if available (v2), else from known architecture
    if is_v2 and "num_attention_heads" in cfg:
        n_heads = cfg["num_attention_heads"]
    else:
        # v1 fairseq2 models: 300M has n_heads=16 (hd=64), 1B has n_heads=16 (hd=80),
        # 3B has n_heads=20 (hd=64), 7B has n_heads=20 (hd=64)
        # Use head_dim=64 as default for v1 models
        n_heads = d_model // 64
    vocab_size = sd["final_proj.weight"].shape[0]
    n_cnn = max(int(k.split('.')[3]) for k in sd
                if k.startswith("encoder_frontend.feature_extractor.layers.")) + 1

    print(f"  n_enc={n_enc}, d_model={d_model}, d_ffn={d_ffn}, n_heads={n_heads}")
    print(f"  vocab={vocab_size}, cnn_layers={n_cnn}")

    # CNN kernel sizes and strides (infer from weight shapes)
    cnn_info = []
    for i in range(n_cnn):
        w = sd[f"encoder_frontend.feature_extractor.layers.{i}.conv.weight"]
        cnn_info.append((w.shape[0], w.shape[1], w.shape[2]))  # (out_ch, in_ch, kernel)
    strides = [5] + [2] * (n_cnn - 1)  # layer 0: stride 5, rest: stride 2
    print(f"  CNN: {[(f'{oc}x{ic}xk{k}s{s}') for (oc,ic,k), s in zip(cnn_info, strides)]}")

    # Tokenizer: already loaded as `vocab` list for both v1 and v2

    # For v2: token IDs from config.json
    if is_v2:
        bos_id = cfg.get("bos_token_id", 0)
        eos_id = cfg.get("eos_token_id", 2)
        pad_id = cfg.get("pad_token_id", 0)
        unk_id = 3  # standard
    else:
        import sentencepiece as spm
        sp_obj = spm.SentencePieceProcessor()
        sp_obj.Load(tok_path)
        bos_id = sp_obj.bos_id()
        eos_id = sp_obj.eos_id()
        pad_id = sp_obj.pad_id()
        unk_id = sp_obj.unk_id()

    # Create GGUF
    writer = gguf.GGUFWriter(args.output, "omniasr-ctc")
    writer.add_name(f"OmniASR-CTC-{model_name.split('-')[-1]}")
    writer.add_uint32("omniasr.d_model", d_model)
    writer.add_uint32("omniasr.d_ffn", d_ffn)
    head_dim = d_model // n_heads
    writer.add_uint32("omniasr.n_heads", n_heads)
    writer.add_uint32("omniasr.head_dim", head_dim)
    writer.add_uint32("omniasr.n_enc_layers", n_enc)
    writer.add_uint32("omniasr.n_cnn_layers", n_cnn)
    writer.add_uint32("omniasr.vocab_size", vocab_size)
    writer.add_uint32("omniasr.bos_id", bos_id)
    writer.add_uint32("omniasr.eos_id", eos_id)
    writer.add_uint32("omniasr.pad_id", pad_id)
    writer.add_uint32("omniasr.unk_id", unk_id)
    writer.add_uint32("omniasr.model_type", 0)  # 0=CTC
    # Store CNN strides for runtime
    writer.add_array("omniasr.cnn_strides", strides)

    writer.add_array("tokenizer.ggml.tokens", vocab)

    def f16(t):
        return t.astype(np.float16) if t.dtype == np.float32 else t

    def f32(t):
        return t.astype(np.float32)

    # Shorten tensor names to fit 64-char limit
    def shorten(name):
        name = name.replace("encoder_frontend.feature_extractor.layers.", "cnn.")
        name = name.replace("encoder_frontend.model_dim_proj.", "proj.")
        name = name.replace("encoder.layers.", "enc.")
        name = name.replace("encoder.layer_norm.", "enc_ln.")
        name = name.replace("self_attn.", "attn.")
        name = name.replace("self_attn_layer_norm.", "attn_ln.")
        name = name.replace("ffn_layer_norm.", "ffn_ln.")
        name = name.replace("ffn.inner_proj.", "ffn.up.")
        name = name.replace("ffn.output_proj.", "ffn.down.")
        name = name.replace("layer_norm.", "ln.")
        name = name.replace("final_proj.", "ctc.")
        name = name.replace("output_proj.", "out.")
        return name

    # Pre-compute weight normalization for pos_encoder conv.
    # v2 (HF): use torch._weight_norm to materialize the combined weight
    #   correctly (the parametrize dim may differ from the fairseq2 convention).
    # v1 (fairseq2): weight_g and weight_v with per-output-channel norm.
    pg_key = "encoder_frontend.pos_encoder.conv.parametrizations.weight.original0"
    pv_key = "encoder_frontend.pos_encoder.conv.parametrizations.weight.original1"
    if pg_key in sd and pv_key in sd:
        wg = sd[pg_key]
        wv = sd[pv_key]
        # Use torch._weight_norm with the correct dim (read from the model)
        # Safest: load the model and read conv.weight directly
        try:
            from transformers import AutoModel
            hf_model = AutoModel.from_pretrained(args.input)
            w_combined = hf_model.encoder.pos_conv_embed.conv.weight.detach()
            b_combined = hf_model.encoder.pos_conv_embed.conv.bias.detach()
            print(f"  pos_conv weight from HF model: {w_combined.shape} (materialized)")
        except Exception as e:
            print(f"  WARNING: could not load HF model for pos_conv weight ({e})")
            print(f"  Falling back to manual weight norm (may be inaccurate)")
            v_norm = wv.reshape(wv.shape[0], -1).norm(dim=1).reshape(-1, 1, 1)
            w_combined = (wg / (v_norm + 1e-12)) * wv
            b_combined = sd["encoder_frontend.pos_encoder.conv.bias"]
        sd["pos_conv.weight"] = w_combined
        sd["pos_conv.bias"] = b_combined
        del sd[pg_key]
        del sd[pv_key]
        if "encoder_frontend.pos_encoder.conv.bias" in sd:
            del sd["encoder_frontend.pos_encoder.conv.bias"]
    elif "encoder_frontend.pos_encoder.conv.weight_v" in sd:
        wv = sd["encoder_frontend.pos_encoder.conv.weight_v"]  # [OC, IC/G, K]
        wg = sd["encoder_frontend.pos_encoder.conv.weight_g"]  # [1, 1, K]
        # Weight normalization: w = g * v / ||v||
        # fairseq2 uses weight_norm dim=2: norm over all dims except K
        # wg: [1, 1, K=128], wv: [OC, IC/G, K=128]
        # norm = ||v|| over dims [0,1] → [1, 1, K]
        v_norm = wv.norm(dim=[0, 1], keepdim=True)  # [1, 1, K]
        w_combined = wg * (wv / (v_norm + 1e-12))
        sd["pos_conv.weight"] = w_combined
        sd["pos_conv.bias"] = sd["encoder_frontend.pos_encoder.conv.bias"]
        print(f"  Pre-computed pos_conv weight: {w_combined.shape}")
        # Remove raw weight_g/weight_v to avoid storing them
        del sd["encoder_frontend.pos_encoder.conv.weight_g"]
        del sd["encoder_frontend.pos_encoder.conv.weight_v"]
        del sd["encoder_frontend.pos_encoder.conv.bias"]

    tensor_count = 0
    for name in sorted(sd.keys()):
        t = sd[name].float().numpy()
        gguf_name = shorten(name)

        if len(gguf_name) >= 64:
            print(f"  WARNING: name too long ({len(gguf_name)}): {gguf_name}")
            continue

        # Store norms/biases as F32, weights as F16
        if "norm" in name or name.endswith(".bias") or len(t.shape) <= 1:
            data = f32(t)
        else:
            data = f16(t)

        writer.add_tensor(gguf_name, data)
        tensor_count += 1
        if tensor_count <= 5 or tensor_count % 50 == 0:
            print(f"  [{tensor_count}] {gguf_name:50s} {str(data.shape):20s}")

    print(f"  total: {tensor_count} tensors")

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    sz = os.path.getsize(args.output)
    print(f"\nDone: {args.output} ({sz / 1e9:.2f} GB, {tensor_count} tensors)")


if __name__ == "__main__":
    main()
