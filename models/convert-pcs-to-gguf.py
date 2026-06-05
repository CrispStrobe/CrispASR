#!/usr/bin/env python3
"""Convert 1-800-BAD-CODE/xlm-roberta_punctuation_fullstop_truecase ONNX to GGUF.

Architecture: XLM-RoBERTa-base (12L, d=768, 12 heads, d_ffn=3072, GELU)
              + 4 classification heads:
                - post_punc: Linear(768,256) + Linear(256,17)  — 17 punctuation classes
                - pre_punc:  Linear(768,256) + Linear(256,2)   — 2 pre-punc classes
                - sbd:       Linear(772,128) + Linear(128,2)   — sentence boundary
                - truecase:  Linear(769,128) + Linear(128,16)  — per-char upper/lower

Usage:
    python models/convert-pcs-to-gguf.py \\
        --input 1-800-BAD-CODE/xlm-roberta_punctuation_fullstop_truecase \\
        --output pcs-xlmr-base.gguf
"""

import argparse
import json
import os
import sys

import numpy as np

try:
    import gguf
except ImportError:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ggml", "python"))
    import gguf

try:
    import onnx
    from onnx import numpy_helper
except ImportError:
    print("Error: onnx not found. pip install onnx")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="HF model ID or local path")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    # Download files
    from huggingface_hub import hf_hub_download
    import yaml

    onnx_path = hf_hub_download(args.input, "model.onnx")
    config_path = hf_hub_download(args.input, "config.yaml")
    sp_path = hf_hub_download(args.input, "sp.model")

    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    print(f"Loading ONNX model from {onnx_path}...")
    model = onnx.load(onnx_path)

    # Build weight dict from ONNX initializers
    weights = {}
    for init in model.graph.initializer:
        weights[init.name] = numpy_helper.to_array(init)

    print(f"Loaded {len(weights)} tensors")

    # Identify architecture params from weight shapes
    vocab_size = weights["bert_model.embeddings.word_embeddings.weight"].shape[0]
    d_model = weights["bert_model.embeddings.word_embeddings.weight"].shape[1]
    max_pos = weights["bert_model.embeddings.position_embeddings.weight"].shape[0]

    # Count layers
    n_layers = 0
    while f"bert_model.encoder.layer.{n_layers}.attention.self.query.bias" in weights:
        n_layers += 1

    # FFN size from first layer
    ffn_key = None
    for k in weights:
        if "layer.0" in k and "intermediate" in k and "weight" in k:
            ffn_key = k
            break
    if ffn_key is None:
        # Try onnx::MatMul pattern
        for k, v in weights.items():
            if k.startswith("onnx::MatMul") and v.shape == (d_model, 3072):
                d_ffn = 3072
                break
        else:
            d_ffn = 3072
    else:
        d_ffn = weights[ffn_key].shape[0]

    n_heads = 12  # XLM-R base

    post_labels = config.get("post_labels", [])
    pre_labels = config.get("pre_labels", [])
    n_post = len(post_labels)
    n_pre = len(pre_labels)

    print(f"Architecture: {n_layers}L, d={d_model}, ffn={d_ffn}, heads={n_heads}")
    print(f"Vocab: {vocab_size}, max_pos: {max_pos}")
    print(f"Post-punc labels ({n_post}): {post_labels}")
    print(f"Pre-punc labels ({n_pre}): {pre_labels}")

    # Load SentencePiece vocab
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor()
    sp.Load(sp_path)
    sp_vocab = [sp.IdToPiece(i) for i in range(sp.GetPieceSize())]
    print(f"SP vocab: {len(sp_vocab)} pieces")

    # Create GGUF
    writer = gguf.GGUFWriter(args.output, "pcs")
    writer.add_name("xlm-roberta-pcs-47lang")

    writer.add_uint32("pcs.d_model", d_model)
    writer.add_uint32("pcs.d_ffn", d_ffn)
    writer.add_uint32("pcs.n_heads", n_heads)
    writer.add_uint32("pcs.n_layers", n_layers)
    writer.add_uint32("pcs.vocab_size", vocab_size)
    writer.add_uint32("pcs.max_pos", max_pos)
    writer.add_uint32("pcs.n_post_labels", n_post)
    writer.add_uint32("pcs.n_pre_labels", n_pre)
    writer.add_uint32("pcs.pad_id", 1)
    writer.add_uint32("pcs.cls_id", 0)  # <s>

    writer.add_array("tokenizer.ggml.tokens", sp_vocab)
    writer.add_array("pcs.post_labels", post_labels)
    writer.add_array("pcs.pre_labels", pre_labels)

    # Map ONNX tensor names to GGUF names
    tensor_map = {}

    # Embeddings
    tensor_map["bert_model.embeddings.word_embeddings.weight"] = "emb.tok_emb.weight"
    tensor_map["bert_model.embeddings.position_embeddings.weight"] = "emb.pos_emb.weight"
    tensor_map["bert_model.embeddings.token_type_embeddings.weight"] = "emb.type_emb.weight"
    tensor_map["bert_model.embeddings.LayerNorm.mod.weight"] = "emb.ln.weight"
    tensor_map["bert_model.embeddings.LayerNorm.mod.bias"] = "emb.ln.bias"

    # Encoder layers
    for i in range(n_layers):
        prefix = f"bert_model.encoder.layer.{i}"
        out_prefix = f"enc.{i}"

        # The ONNX model stores Q/K/V weights as MatMul nodes (transposed)
        # and biases as named initializers
        tensor_map[f"{prefix}.attention.self.query.bias"] = f"{out_prefix}.attn.q.bias"
        tensor_map[f"{prefix}.attention.self.key.bias"] = f"{out_prefix}.attn.k.bias"
        tensor_map[f"{prefix}.attention.self.value.bias"] = f"{out_prefix}.attn.v.bias"
        tensor_map[f"{prefix}.attention.output.dense.bias"] = f"{out_prefix}.attn.out.bias"
        tensor_map[f"{prefix}.attention.output.LayerNorm.mod.weight"] = f"{out_prefix}.attn.ln.weight"
        tensor_map[f"{prefix}.attention.output.LayerNorm.mod.bias"] = f"{out_prefix}.attn.ln.bias"
        tensor_map[f"{prefix}.intermediate.dense.bias"] = f"{out_prefix}.ffn.up.bias"
        tensor_map[f"{prefix}.output.dense.bias"] = f"{out_prefix}.ffn.down.bias"
        tensor_map[f"{prefix}.output.LayerNorm.mod.weight"] = f"{out_prefix}.ffn.ln.weight"
        tensor_map[f"{prefix}.output.LayerNorm.mod.bias"] = f"{out_prefix}.ffn.ln.bias"

    # Find unnamed MatMul weights (Q/K/V/output/FFN weights stored as onnx::MatMul_*)
    # These are in order: for each layer, QKV weights, output, intermediate, output
    # We need to map them carefully.

    # Collect all onnx::MatMul weights
    matmul_keys = sorted([k for k in weights if k.startswith("onnx::MatMul_")],
                        key=lambda k: int(k.split("_")[1]))
    print(f"\n{len(matmul_keys)} onnx::MatMul tensors")

    # The first n_layers * 6 MatMul weights are the encoder layer weights:
    # Per layer: Q_w, K_w, V_w, Out_w, FFN_up_w, FFN_down_w
    # After that: head weights
    layer_matmuls = n_layers * 6
    for i in range(n_layers):
        base = i * 6
        out_prefix = f"enc.{i}"
        tensor_map[matmul_keys[base + 0]] = f"{out_prefix}.attn.q.weight"
        tensor_map[matmul_keys[base + 1]] = f"{out_prefix}.attn.k.weight"
        tensor_map[matmul_keys[base + 2]] = f"{out_prefix}.attn.v.weight"
        tensor_map[matmul_keys[base + 3]] = f"{out_prefix}.attn.out.weight"
        tensor_map[matmul_keys[base + 4]] = f"{out_prefix}.ffn.up.weight"
        tensor_map[matmul_keys[base + 5]] = f"{out_prefix}.ffn.down.weight"

    # Remaining MatMul weights are the 4 heads
    head_keys = matmul_keys[layer_matmuls:]
    print(f"Head MatMul weights: {len(head_keys)}")
    for k in head_keys:
        shape = weights[k].shape
        print(f"  {k}: {shape}")

    # Map head weights. From the shapes:
    # [768, 256] → post_punc head layer 1
    # [256, 17]  → post_punc head layer 2
    # [768, 256] → pre_punc head layer 1
    # [256, 2]   → pre_punc head layer 2
    # [772, 128] → sbd head layer 1 (768 + 4 post_punc embedding)
    # [128, 2]   → sbd head layer 2
    # Then there may be embedding weights for post_punc → sbd
    # [769, 128] → truecase head layer 1 (768 + 1 sbd)
    # [128, 16]  → truecase head layer 2

    # Let's identify by shape
    head_idx = 0
    head_names = []
    for k in head_keys:
        shape = weights[k].shape
        if shape == (768, 256) and head_idx == 0:
            tensor_map[k] = "head.post.fc1.weight"
            head_idx += 1
        elif shape == (256, n_post):
            tensor_map[k] = "head.post.fc2.weight"
        elif shape == (768, 256) and head_idx >= 1:
            tensor_map[k] = "head.pre.fc1.weight"
            head_idx += 1
        elif shape == (256, n_pre):
            tensor_map[k] = "head.pre.fc2.weight"
        elif len(shape) == 2 and shape[0] > 768 and shape[0] <= 784 and shape[1] == 128:
            if shape[0] == 772:  # 768 + 4 (post_punc embedding dim)
                tensor_map[k] = "head.sbd.fc1.weight"
            elif shape[0] == 769:  # 768 + 1 (sbd embedding)
                tensor_map[k] = "head.tc.fc1.weight"
            else:
                tensor_map[k] = f"head.unknown_{shape}.weight"
        elif shape == (128, 2):
            # Could be sbd or tc final layer
            if "head.sbd.fc1.weight" in tensor_map.values() and "head.sbd.fc2.weight" not in tensor_map.values():
                tensor_map[k] = "head.sbd.fc2.weight"
            else:
                tensor_map[k] = f"head.sbd_or_tc.fc2.weight"
        elif shape == (128, 16):
            tensor_map[k] = "head.tc.fc2.weight"
        else:
            # Could be embedding tables for inter-head conditioning
            tensor_map[k] = f"head.emb_{shape[0]}x{shape[1]}"
            print(f"  -> unmapped: {k} {shape}")

    # Map known non-MatMul head weights
    decoder_map = {
        "_decoder._punct_emb.weight": "head.post_emb.weight",
        "_decoder._punct_head_post._linears.0.bias": "head.post.fc1.bias",
        "_decoder._punct_head_post._linears.1.bias": "head.post.fc2.bias",
        "_decoder._punct_head_pre._linears.0.bias": "head.pre.fc1.bias",
        "_decoder._punct_head_pre._linears.1.bias": "head.pre.fc2.bias",
        "_decoder._seg_head._linears.0.bias": "head.sbd.fc1.bias",
        "_decoder._seg_head._linears.1.bias": "head.sbd.fc2.bias",
        "_decoder._cap_head._linears.0.bias": "head.tc.fc1.bias",
        "_decoder._cap_head._linears.1.bias": "head.tc.fc2.bias",
    }
    for onnx_k, gguf_k in decoder_map.items():
        if onnx_k in weights:
            tensor_map[onnx_k] = gguf_k

    # Check for any remaining unmapped weights
    for k in weights:
        if k not in tensor_map:
            print(f"  Unmapped: {k} {weights[k].shape}")

    def f16(t):
        return t.astype(np.float16) if t.dtype == np.float32 else t

    def f32(t):
        return t.astype(np.float32)

    # Write tensors
    count = 0
    for onnx_name, gguf_name in sorted(tensor_map.items(), key=lambda x: x[1]):
        if onnx_name not in weights:
            print(f"  WARNING: {onnx_name} not found in weights!")
            continue
        t = weights[onnx_name]

        # ONNX MatMul weights may need transposing: ONNX stores as [in, out]
        # but our runtime expects [out, in] for ggml_mul_mat.
        # Actually, check: ggml_mul_mat(W, x) computes x @ W^T, so if W is
        # [out, in] in row-major, that's correct. ONNX MatMul stores [in, out],
        # so we need to transpose.
        if ".weight" in gguf_name and len(t.shape) == 2 and "emb" not in gguf_name:
            t = t.T.copy()

        # Norms/biases as F32, weights as F16.
        # Head weights (sbd/tc) are read directly by CPU code — must be F32.
        if "ln." in gguf_name or gguf_name.endswith(".bias") or len(t.shape) <= 1:
            data = f32(t)
        elif "emb" in gguf_name:
            data = f32(t)  # embeddings stay F32
        elif "head.sbd." in gguf_name or "head.tc." in gguf_name:
            data = f32(t)  # CPU-evaluated heads must be F32
        else:
            data = f16(t)

        writer.add_tensor(gguf_name, data)
        count += 1
        if count <= 5 or count % 50 == 0:
            print(f"  [{count}] {gguf_name:50s} {str(data.shape):20s} {'F32' if data.dtype == np.float32 else 'F16'}")

    print(f"  Total: {count} tensors")

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    sz = os.path.getsize(args.output)
    print(f"\nDone: {args.output} ({sz / 1e6:.1f} MB, {count} tensors)")


if __name__ == "__main__":
    main()
