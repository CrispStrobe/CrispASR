#!/usr/bin/env python3
"""Convert Facebook OmniASR-LLM to GGUF.

Architecture: wav2vec2 encoder (24L, d=1024) +
              Linear(1024→4096) + 12-layer LLaMA decoder (d=4096, 8 heads, SwiGLU, RMSNorm, RoPE) +
              LM head (4096→9812).

Usage:
  python models/convert-omniasr-llm-to-gguf.py \
      --input facebook/omniASR-LLM-300M \
      --output omniasr-llm-300m.gguf
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

from gguf import GGMLQuantizationType


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="HF model ID (e.g. facebook/omniASR-LLM-300M)")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    import torch
    import sentencepiece as spm
    from huggingface_hub import hf_hub_download

    model_name = args.input.split("/")[-1]
    pt_file = next(f for f in [f"{model_name}.pt", "model.pt"]
                   if True)  # just use the first
    pt_path = hf_hub_download(args.input, f"{model_name}.pt")
    tok_path = hf_hub_download(args.input, "omniASR_tokenizer.model")

    ckpt = torch.load(pt_path, map_location="cpu", weights_only=False)
    sd = ckpt["model"]
    print(f"Loaded {len(sd)} tensors from {model_name}")

    # Infer encoder architecture
    n_enc = max(int(k.split('.')[2]) for k in sd if k.startswith("encoder.layers.")) + 1
    d_enc = sd["encoder.layers.0.self_attn.q_proj.weight"].shape[0]
    d_ffn_enc = sd["encoder.layers.0.ffn.inner_proj.weight"].shape[0]
    n_heads_enc = 16  # OmniASR uses 16 heads for all sizes
    n_cnn = max(int(k.split('.')[3]) for k in sd
                if k.startswith("encoder_frontend.feature_extractor.layers.")) + 1

    # Infer decoder architecture
    n_dec = max(int(k.split('.')[2]) for k in sd if k.startswith("llama_decoder.layers.")) + 1
    d_dec = sd["llama_decoder.layers.0.ffn.gate_proj.weight"].shape[1]
    d_ffn_dec = sd["llama_decoder.layers.0.ffn.gate_proj.weight"].shape[0]
    n_heads_dec = d_dec // 512  # head_dim=512 for LLaMA decoder

    vocab_size = sd["final_proj.weight"].shape[0]
    n_langs = sd["lang_embeddings.weight"].shape[0] if "lang_embeddings.weight" in sd else 0

    print(f"  Encoder: n_enc={n_enc}, d={d_enc}, ffn={d_ffn_enc}, heads={n_heads_enc}, cnn={n_cnn}")
    print(f"  Decoder: n_dec={n_dec}, d={d_dec}, ffn={d_ffn_dec}, heads={n_heads_dec}")
    print(f"  vocab={vocab_size}, n_langs={n_langs}")

    # CNN strides
    cnn_info = []
    for i in range(n_cnn):
        w = sd[f"encoder_frontend.feature_extractor.layers.{i}.conv.weight"]
        cnn_info.append((w.shape[0], w.shape[1], w.shape[2]))
    strides = [5] + [2] * (n_cnn - 1)

    # Tokenizer
    sp = spm.SentencePieceProcessor()
    sp.Load(tok_path)
    vocab = [sp.IdToPiece(i) for i in range(sp.GetPieceSize())]

    # Language codes
    lang_codes = []
    if "lang_embeddings.weight" in sd:
        # fairseq2 stores language codes in the tokenizer's control symbols
        # For now, just use numeric indices
        lang_codes = [f"lang_{i}" for i in range(n_langs)]

    # Pre-compute pos_conv weight normalization
    # fairseq2 weight_norm dim=2: norm over dims [0,1], gain per K position
    if "encoder_frontend.pos_encoder.conv.weight_v" in sd:
        wv = sd["encoder_frontend.pos_encoder.conv.weight_v"]
        wg = sd["encoder_frontend.pos_encoder.conv.weight_g"]
        v_norm = wv.norm(dim=[0, 1], keepdim=True)
        w_combined = wg * (wv / (v_norm + 1e-12))
        sd["pos_conv.weight"] = w_combined
        sd["pos_conv.bias"] = sd["encoder_frontend.pos_encoder.conv.bias"]
        print(f"  Pre-computed pos_conv weight: {w_combined.shape}")
        del sd["encoder_frontend.pos_encoder.conv.weight_g"]
        del sd["encoder_frontend.pos_encoder.conv.weight_v"]
        del sd["encoder_frontend.pos_encoder.conv.bias"]

    # Create GGUF
    writer = gguf.GGUFWriter(args.output, "omniasr-ctc")  # same arch string for runtime
    writer.add_name(f"OmniASR-LLM-{model_name.split('-')[-1]}")
    writer.add_uint32("omniasr.d_model", d_enc)
    writer.add_uint32("omniasr.d_ffn", d_ffn_enc)
    writer.add_uint32("omniasr.n_heads", n_heads_enc)
    writer.add_uint32("omniasr.head_dim", d_enc // n_heads_enc)
    writer.add_uint32("omniasr.n_enc_layers", n_enc)
    writer.add_uint32("omniasr.n_cnn_layers", n_cnn)
    writer.add_uint32("omniasr.vocab_size", vocab_size)
    writer.add_uint32("omniasr.bos_id", sp.bos_id())
    writer.add_uint32("omniasr.eos_id", sp.eos_id())
    writer.add_uint32("omniasr.pad_id", sp.pad_id())
    writer.add_uint32("omniasr.unk_id", sp.unk_id())
    writer.add_uint32("omniasr.model_type", 1)  # 1=LLM
    writer.add_uint32("omniasr.d_dec", d_dec)
    writer.add_uint32("omniasr.d_ffn_dec", d_ffn_dec)
    writer.add_uint32("omniasr.n_heads_dec", n_heads_dec)
    writer.add_uint32("omniasr.head_dim_dec", d_dec // n_heads_dec)
    writer.add_uint32("omniasr.n_dec_layers", n_dec)
    writer.add_uint32("omniasr.n_langs", n_langs)
    writer.add_array("omniasr.cnn_strides", strides)
    writer.add_array("tokenizer.ggml.tokens", vocab)

    if lang_codes:
        writer.add_array("omniasr.lang_codes", lang_codes)

    def f16(t):
        return t.astype(np.float16) if t.dtype == np.float32 else t

    def f32(t):
        return t.astype(np.float32)

    # Shorten tensor names
    def shorten(name):
        name = name.replace("encoder_frontend.feature_extractor.layers.", "cnn.")
        name = name.replace("encoder_frontend.model_dim_proj.", "proj.")
        name = name.replace("encoder_frontend.post_extract_layer_norm.", "post_extract_ln.")
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
        name = name.replace("llama_decoder.layers.", "dec.")
        name = name.replace("llama_decoder.layer_norm.", "dec_ln.")
        name = name.replace("text_frontend.", "text_emb.")
        name = name.replace("encoder_proj.", "enc_proj.")
        name = name.replace("lang_embeddings.", "lang_emb.")
        return name

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
            dtype = GGMLQuantizationType.F32
        else:
            data = f16(t)
            dtype = GGMLQuantizationType.F16

        writer.add_tensor(gguf_name, data, raw_dtype=dtype)
        tensor_count += 1
        if tensor_count <= 5 or tensor_count % 100 == 0:
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
