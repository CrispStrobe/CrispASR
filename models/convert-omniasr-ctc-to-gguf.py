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
    import sentencepiece as spm
    from huggingface_hub import hf_hub_download

    # Download model + tokenizer
    model_name = args.input.split("/")[-1]
    pt_path = hf_hub_download(args.input, f"{model_name}.pt")
    tok_path = hf_hub_download(args.input, "omniASR_tokenizer.model")

    ckpt = torch.load(pt_path, map_location="cpu", weights_only=False)
    sd = ckpt["model"]
    print(f"Loaded {len(sd)} tensors from {model_name}")

    # Infer architecture
    n_enc = max(int(k.split('.')[2]) for k in sd if k.startswith("encoder.layers.")) + 1
    d_model = sd["encoder.layers.0.self_attn.q_proj.weight"].shape[0]
    d_ffn = sd["encoder.layers.0.ffn.inner_proj.weight"].shape[0]
    n_heads = d_model // 64  # head_dim=64
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

    # Tokenizer
    sp = spm.SentencePieceProcessor()
    sp.Load(tok_path)
    vocab = [sp.IdToPiece(i) for i in range(sp.GetPieceSize())]

    # Create GGUF
    writer = gguf.GGUFWriter(args.output, "omniasr-ctc")
    writer.add_name(f"OmniASR-CTC-{model_name.split('-')[-1]}")
    writer.add_uint32("omniasr.d_model", d_model)
    writer.add_uint32("omniasr.d_ffn", d_ffn)
    writer.add_uint32("omniasr.n_heads", n_heads)
    writer.add_uint32("omniasr.n_enc_layers", n_enc)
    writer.add_uint32("omniasr.n_cnn_layers", n_cnn)
    writer.add_uint32("omniasr.vocab_size", vocab_size)
    writer.add_uint32("omniasr.bos_id", sp.bos_id())
    writer.add_uint32("omniasr.eos_id", sp.eos_id())
    writer.add_uint32("omniasr.pad_id", sp.pad_id())
    writer.add_uint32("omniasr.unk_id", sp.unk_id())
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
