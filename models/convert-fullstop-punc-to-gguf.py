#!/usr/bin/env python3
"""Convert oliverguhr/fullstop-punctuation-multilang-large to GGUF.

Architecture: XLM-RoBERTa-large (24L, d=1024, 16 heads, d_ffn=4096)
              + Linear(1024, 6) classifier head.
              6 classes: 0(none), .(period), ,(comma), ?(question), -(dash), :(colon)

Usage:
    python models/convert-fullstop-punc-to-gguf.py \
        --input oliverguhr/fullstop-punctuation-multilang-large \
        --output fullstop-punc.gguf
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
    parser.add_argument("--input", required=True, help="HF model ID or local path")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForTokenClassification, AutoTokenizer, AutoConfig

    print(f"Loading model from {args.input}...")
    config = AutoConfig.from_pretrained(args.input)
    model = AutoModelForTokenClassification.from_pretrained(args.input, dtype=torch.float32)
    sd = model.state_dict()
    print(f"Loaded {len(sd)} tensors")

    # Architecture params
    n_layers = config.num_hidden_layers
    d_model = config.hidden_size
    d_ffn = config.intermediate_size
    n_heads = config.num_attention_heads
    vocab_size = config.vocab_size
    max_pos = config.max_position_embeddings
    n_classes = config.num_labels

    print(f"  Layers: {n_layers}, d_model: {d_model}, d_ffn: {d_ffn}, heads: {n_heads}")
    print(f"  Vocab: {vocab_size}, max_pos: {max_pos}, classes: {n_classes}")
    print(f"  Labels: {config.id2label}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.input)

    # Get vocab as list of strings
    vocab = [tokenizer.convert_ids_to_tokens(i) for i in range(vocab_size)]

    # Get labels
    labels = [config.id2label[i] for i in range(n_classes)]
    # Map '0' to space
    labels = [' ' if l == '0' else l for l in labels]
    print(f"  Labels (mapped): {labels}")

    # Create GGUF
    writer = gguf.GGUFWriter(args.output, "fireredpunc")
    writer.add_name("fullstop-punctuation-multilang-large")

    writer.add_uint32("fireredpunc.d_model", d_model)
    writer.add_uint32("fireredpunc.d_ffn", d_ffn)
    writer.add_uint32("fireredpunc.n_heads", n_heads)
    writer.add_uint32("fireredpunc.n_layers", n_layers)
    writer.add_uint32("fireredpunc.vocab_size", vocab_size)
    writer.add_uint32("fireredpunc.max_pos", max_pos)
    writer.add_uint32("fireredpunc.n_classes", n_classes)
    writer.add_uint32("fireredpunc.cls_id", 0)   # <s> for RoBERTa
    writer.add_uint32("fireredpunc.pad_id", 1)   # <pad>
    writer.add_string("fireredpunc.tokenizer_type", "sentencepiece")

    writer.add_array("tokenizer.ggml.tokens", vocab)
    writer.add_array("fireredpunc.labels", labels)

    def f16(t):
        return t.astype(np.float16) if t.dtype == np.float32 else t

    def f32(t):
        return t.astype(np.float32)

    # Shorten tensor names — same as FireRedPunc but with roberta. prefix
    def shorten(name):
        name = name.replace("roberta.embeddings.", "emb.")
        name = name.replace("roberta.encoder.layer.", "enc.")
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

        # Norms/biases as F32, weights as F16.
        # Embedding table stays F32 — F16 causes precision loss through 24 layers.
        if "ln." in gguf_name or "LayerNorm" in name or name.endswith(".bias") or len(t.shape) <= 1:
            data = f32(t)
        elif "tok_emb" in gguf_name or "pos_emb" in gguf_name or "type_emb" in gguf_name:
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
