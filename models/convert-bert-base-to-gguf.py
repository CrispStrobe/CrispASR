#!/usr/bin/env python
"""
Convert bert-base-uncased to GGUF for MeloTTS BERT conditioning.

Only extracts layers 0-9 (hidden_states[-3] = layer 9 output).
Layers 10-11 and the MLM head are not needed.

Usage:
    python models/convert-bert-base-to-gguf.py --output bert-base-uncased.gguf
"""

import argparse
import sys
import numpy as np

try:
    from gguf import GGUFWriter
except ImportError:
    sys.exit("pip install gguf")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    ap.add_argument("--model-id", default="bert-base-uncased")
    ap.add_argument("--n-layers", type=int, default=10,
                    help="Number of transformer layers to keep (default 10 = hidden_states[-3])")
    args = ap.parse_args()

    import os
    os.environ.setdefault("HF_HOME", "/mnt/storage/huggingface")

    import torch
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    print(f"Loading {args.model_id}...")
    model = AutoModelForMaskedLM.from_pretrained(args.model_id)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)

    sd = model.state_dict()

    writer = GGUFWriter(args.output, arch="bert")

    # Metadata
    writer.add_uint32("bert.n_layers", args.n_layers)
    writer.add_uint32("bert.hidden_size", model.config.hidden_size)
    writer.add_uint32("bert.n_heads", model.config.num_attention_heads)
    writer.add_uint32("bert.intermediate_size", model.config.intermediate_size)
    writer.add_uint32("bert.vocab_size", model.config.vocab_size)
    writer.add_uint32("bert.max_position_embeddings", model.config.max_position_embeddings)
    writer.add_uint32("bert.type_vocab_size", model.config.type_vocab_size)
    writer.add_float32("bert.layer_norm_eps", model.config.layer_norm_eps)

    # Tokenizer vocab (for text → token IDs)
    vocab = tokenizer.get_vocab()
    # Store as JSON string
    import json
    writer.add_string("bert.vocab_json", json.dumps(vocab))

    # Write tensors
    n_written = 0
    for name, param in sorted(sd.items()):
        # Skip layers we don't need
        skip = False
        for i in range(args.n_layers, 12):
            if f"layer.{i}." in name:
                skip = True
                break
        if "cls." in name or "pooler." in name:
            skip = True
        if skip:
            continue

        # Remove "bert." prefix for cleaner names
        clean_name = name
        if clean_name.startswith("bert."):
            clean_name = clean_name[5:]

        arr = param.cpu().numpy()

        # Embeddings + 1D → F32; linear weights → F16
        is_emb = "embedding" in clean_name
        if arr.ndim <= 1 or is_emb:
            writer.add_tensor(clean_name, arr.astype(np.float32))
        else:
            writer.add_tensor(clean_name, arr.astype(np.float16))
        n_written += 1

    print(f"Written {n_written} tensors")

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    import os
    size = os.path.getsize(args.output)
    print(f"Output: {args.output} ({size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
