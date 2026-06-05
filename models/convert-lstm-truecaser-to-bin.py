#!/usr/bin/env python3
"""Convert mayhewsw/pytorch-truecaser model to compact binary format.

Architecture: Embedding(vocab, 50) → BiLSTM(50→150, 2 layers) → Linear(300→2)
Labels: L (lowercase), U (uppercase) — per character

Usage:
    # Download model first:
    wget https://github.com/mayhewsw/pytorch-truecaser/releases/download/v1.0/wmt-truecaser-model-de.tar.gz
    tar xzf wmt-truecaser-model-de.tar.gz

    python models/convert-lstm-truecaser-to-bin.py \
        --input wmt-truecaser-de/ --output truecaser-lstm-de.bin
"""

import argparse
import os
import struct
import sys

import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Directory with weights.th + vocabulary/")
    parser.add_argument("--output", required=True, help="Output binary file")
    args = parser.parse_args()

    # Load weights
    sd = torch.load(os.path.join(args.input, "weights.th"), map_location="cpu", weights_only=False)
    print(f"Loaded {len(sd)} tensors")

    # Load vocabulary
    with open(os.path.join(args.input, "vocabulary", "tokens.txt")) as f:
        tokens = [line.rstrip("\n") for line in f]

    # AllenNLP vocab: index 0 = @@PADDING@@, then tokens.txt at indices 1..N
    vocab = ["@@PADDING@@"] + tokens
    vocab_size = len(vocab)
    print(f"Vocab: {vocab_size} tokens")

    # Architecture params
    embed_dim = 50
    hidden_size = 150
    n_layers = 2
    n_labels = 2  # L, U

    # Extract and convert to numpy
    def np_f32(key):
        return sd[key].float().numpy()

    with open(args.output, "wb") as f:
        # Magic + header
        f.write(b"LSTM")
        f.write(struct.pack("<IIII", vocab_size, embed_dim, hidden_size, n_layers))

        # Vocabulary: each token as uint16_len + utf8 bytes
        for tok in vocab:
            tb = tok.encode("utf-8")
            f.write(struct.pack("<H", len(tb)))
            f.write(tb)

        # Embedding weights: [vocab_size, embed_dim]
        emb = np_f32("text_field_embedder.token_embedder_tokens.weight")
        f.write(emb.tobytes())
        print(f"  embedding: {emb.shape}")

        # BiLSTM weights: for each layer, for each direction (fwd, rev):
        #   weight_ih: [4*hidden, input_size]
        #   weight_hh: [4*hidden, hidden_size]
        #   bias_ih: [4*hidden]
        #   bias_hh: [4*hidden]
        for layer in range(n_layers):
            for direction in ["", "_reverse"]:
                suffix = f"_l{layer}{direction}"
                wih = np_f32(f"encoder._module.weight_ih{suffix}")
                whh = np_f32(f"encoder._module.weight_hh{suffix}")
                bih = np_f32(f"encoder._module.bias_ih{suffix}")
                bhh = np_f32(f"encoder._module.bias_hh{suffix}")
                f.write(wih.tobytes())
                f.write(whh.tobytes())
                f.write(bih.tobytes())
                f.write(bhh.tobytes())
                print(f"  lstm{suffix}: wih={wih.shape} whh={whh.shape}")

        # Projection: Linear(300, 2)
        proj_w = np_f32("tag_projection_layer._module.weight")
        proj_b = np_f32("tag_projection_layer._module.bias")
        f.write(proj_w.tobytes())
        f.write(proj_b.tobytes())
        print(f"  projection: w={proj_w.shape} b={proj_b.shape}")

    sz = os.path.getsize(args.output)
    print(f"\nDone: {args.output} ({sz / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
