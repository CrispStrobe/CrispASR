#!/usr/bin/env python3
"""Convert FireRedPunc (BERT token classifier) to GGUF.

Architecture: BERT-base (12L, d=768, 12 heads, d_ffn=3072)
              + Linear(768, 5) classifier head.
              5 classes: space(0), ，(1), 。(2), ？(3), ！(4)

Usage:
    python models/convert-fireredpunc-to-gguf.py \
        --input /path/to/FireRedPunc \
        --output fireredpunc.gguf
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
    parser.add_argument("--input", required=True, help="Path to FireRedPunc model dir")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    import torch

    # Load checkpoint
    ckpt_path = os.path.join(args.input, "model.pth.tar")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"No model.pth.tar in {args.input}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt["model_state_dict"]
    model_args = ckpt.get("args")
    print(f"Loaded {len(sd)} tensors")

    # Architecture params
    n_layers = max(int(k.split('.')[3]) for k in sd if k.startswith("bert.encoder.layer.")) + 1
    d_model = sd["bert.embeddings.LayerNorm.weight"].shape[0]
    d_ffn = sd["bert.encoder.layer.0.intermediate.dense.weight"].shape[0]
    n_heads = d_model // 64  # head_dim=64
    vocab_size = sd["bert.embeddings.word_embeddings.weight"].shape[0]
    max_pos = sd["bert.embeddings.position_embeddings.weight"].shape[0]
    n_classes = sd["classifier.weight"].shape[0]

    print(f"  Layers: {n_layers}, d_model: {d_model}, d_ffn: {d_ffn}, heads: {n_heads}")
    print(f"  Vocab: {vocab_size}, max_pos: {max_pos}, classes: {n_classes}")

    # Load vocabulary
    vocab_path = os.path.join(args.input, "chinese-bert-wwm-ext_vocab.txt")
    if not os.path.exists(vocab_path):
        # Fall back to chinese-lert-base vocab
        vocab_path = os.path.join(args.input, "chinese-lert-base", "vocab.txt")
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = [line.strip() for line in f]
    print(f"  Vocabulary: {len(vocab)} tokens from {vocab_path}")

    # Load output labels
    out_dict_path = os.path.join(args.input, "out_dict")
    labels = []
    with open(out_dict_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if parts:
                label = parts[0]
                if label == "<space>":
                    label = " "
                labels.append(label)
    print(f"  Labels: {labels}")

    # Create GGUF
    writer = gguf.GGUFWriter(args.output, "fireredpunc")
    writer.add_name("FireRedPunc")

    writer.add_uint32("fireredpunc.d_model", d_model)
    writer.add_uint32("fireredpunc.d_ffn", d_ffn)
    writer.add_uint32("fireredpunc.n_heads", n_heads)
    writer.add_uint32("fireredpunc.n_layers", n_layers)
    writer.add_uint32("fireredpunc.vocab_size", vocab_size)
    writer.add_uint32("fireredpunc.max_pos", max_pos)
    writer.add_uint32("fireredpunc.n_classes", n_classes)
    writer.add_uint32("fireredpunc.cls_id", getattr(model_args, 'cls_id', 101))
    writer.add_uint32("fireredpunc.pad_id", 0)

    writer.add_array("tokenizer.ggml.tokens", vocab)
    writer.add_array("fireredpunc.labels", labels)

    def f16(t):
        return t.astype(np.float16) if t.dtype == np.float32 else t

    def f32(t):
        return t.astype(np.float32)

    # Shorten tensor names
    def shorten(name):
        name = name.replace("bert.embeddings.", "emb.")
        name = name.replace("bert.encoder.layer.", "enc.")
        name = name.replace("attention.self.query.", "attn.q.")
        name = name.replace("attention.self.key.", "attn.k.")
        name = name.replace("attention.self.value.", "attn.v.")
        name = name.replace("attention.output.dense.", "attn.out.")
        name = name.replace("attention.output.LayerNorm.", "attn.ln.")
        name = name.replace("intermediate.dense.", "ffn.up.")
        name = name.replace("output.dense.", "ffn.down.")
        name = name.replace("output.LayerNorm.", "ffn.ln.")
        name = name.replace("word_embeddings.", "tok_emb.")
        name = name.replace("position_embeddings.", "pos_emb.")
        name = name.replace("token_type_embeddings.", "type_emb.")
        name = name.replace("LayerNorm.", "ln.")
        name = name.replace("classifier.", "cls.")
        return name

    tensor_count = 0
    for name in sorted(sd.keys()):
        t = sd[name].float().numpy()
        gguf_name = shorten(name)

        # Norms/biases as F32, weights as F16
        if "ln." in gguf_name or "LayerNorm" in name or name.endswith(".bias") or len(t.shape) <= 1:
            data = f32(t)
        else:
            data = f16(t)

        writer.add_tensor(gguf_name, data)
        tensor_count += 1
        if tensor_count <= 5 or tensor_count % 50 == 0:
            print(f"  [{tensor_count}] {gguf_name:50s} {str(data.shape):20s}")

    print(f"  Total: {tensor_count} tensors")

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    sz = os.path.getsize(args.output)
    print(f"\nDone: {args.output} ({sz / 1e6:.1f} MB, {tensor_count} tensors)")


if __name__ == "__main__":
    main()
