#!/usr/bin/env python3
"""Convert omniASR-CTC (fairseq2) checkpoints to wav2vec2 GGUF.

Source: https://huggingface.co/Steveeeeeeen/omniASR-CTC-{300M,1B,3B,7B}

Architecture: standard wav2vec2ForCTC — 7-layer strided CNN feature extractor,
feature projection (LayerNorm + Linear), positional conv embedding, N transformer
encoder layers (pre-norm), and a linear CTC head. This is the same architecture
that our src/wav2vec2-ggml.cpp runtime already implements.

The only fairseq2-specific wrinkle is weight-normalised positional convolution:
fairseq2 stores `weight_g` (gain, [1,1,K]) and `weight_v` (direction, [Cout,Cin/g,K])
separately, while the HF wav2vec2 convention stores a single merged weight. We
merge them at conversion time: `w = weight_g * (weight_v / norm(weight_v))`.

Usage:
  python models/convert-omniasr-ctc-to-gguf.py \\
      --input  /path/to/omniasr-ctc-300m \\
      --output models/omniasr-ctc-300m.gguf
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np

try:
    import gguf
except ImportError:
    sys.exit("pip install gguf")
try:
    import torch
except ImportError:
    sys.exit("pip install torch")
try:
    import sentencepiece as spm
except ImportError:
    spm = None


# ---------------------------------------------------------------------------
# Tensor name mapping: fairseq2 → wav2vec2-ggml GGUF convention
# ---------------------------------------------------------------------------

def remap(name: str) -> str | None:
    n = name

    # wav2vec2-ggml.cpp uses short tensor names. Full mapping:

    # Encoder global layer norm (must come BEFORE the generic .layer_norm.→.norm.
    # replacement so `encoder.layer_norm.weight` doesn't get caught by the CNN rule)
    n = n.replace("encoder.layer_norm.weight", "enc.ln.weight")
    n = n.replace("encoder.layer_norm.bias",   "enc.ln.bias")

    # CNN feature extractor: cnn.N.conv.{weight,bias}, cnn.N.norm.{weight,bias}
    n = n.replace("encoder_frontend.feature_extractor.layers.", "cnn.")
    n = n.replace(".layer_norm.", ".norm.")

    # Post-extract LayerNorm → feat_proj.ln.{weight,bias}
    n = n.replace("encoder_frontend.post_extract_layer_norm.weight", "feat_proj.ln.weight")
    n = n.replace("encoder_frontend.post_extract_layer_norm.bias",   "feat_proj.ln.bias")

    # Feature projection → feat_proj.{weight,bias}
    n = n.replace("encoder_frontend.model_dim_proj.weight", "feat_proj.weight")
    n = n.replace("encoder_frontend.model_dim_proj.bias",   "feat_proj.bias")

    # Positional conv — weight_g/weight_v merge done separately
    if "pos_encoder.conv.weight_g" in name or "pos_encoder.conv.weight_v" in name:
        return None
    n = n.replace("encoder_frontend.pos_encoder.conv.bias", "pos_conv.bias")

    # Encoder layers: enc.N.attn.{q,k,v,out}.{weight,bias} etc
    if n.startswith("encoder.layers."):
        rest = n[len("encoder.layers."):]
        layer_id, sub = rest.split(".", 1)
        sub = (sub
            .replace("self_attn.q_proj",       "attn.q")
            .replace("self_attn.k_proj",       "attn.k")
            .replace("self_attn.v_proj",       "attn.v")
            .replace("self_attn.output_proj",  "attn.out")
            .replace("self_attn_layer_norm",   "ln1")
            .replace("ffn.inner_proj",         "ffn.fc1")
            .replace("ffn.output_proj",        "ffn.fc2")
            .replace("ffn_layer_norm",         "ln2")
        )
        n = f"enc.{layer_id}.{sub}"

    # CTC head
    n = n.replace("final_proj.", "lm_head.")

    return n


def is_f32(name: str) -> bool:
    return name.endswith(".bias") or "norm" in name or len(name.split(".")) <= 2


def convert(input_dir: Path, out_path: Path) -> None:
    ckpt_files = sorted(input_dir.glob("*.pt")) + sorted(input_dir.glob("*.bin"))
    if not ckpt_files:
        sys.exit(f"No .pt/.bin checkpoint found in {input_dir}")

    print(f"Loading: {ckpt_files[0]}")
    raw = torch.load(str(ckpt_files[0]), map_location="cpu", weights_only=False)
    sd = raw.get("model", raw.get("state_dict", raw))

    # Extract hparams from tensor shapes
    q_w = sd["encoder.layers.0.self_attn.q_proj.weight"]
    hidden_size = q_w.shape[0]
    ff_w = sd["encoder.layers.0.ffn.inner_proj.weight"]
    intermediate_size = ff_w.shape[0]
    ctc_w = sd["final_proj.weight"]
    vocab_size = ctc_w.shape[0]
    n_layers = max(int(k.split(".")[2]) for k in sd if k.startswith("encoder.layers.")) + 1

    # CNN hparams
    conv_dims = []
    conv_kernels = []
    conv_strides = []  # not stored in weights — use wav2vec2 defaults
    for i in range(7):
        cw = sd.get(f"encoder_frontend.feature_extractor.layers.{i}.conv.weight")
        if cw is not None:
            conv_dims.append(cw.shape[0])
            conv_kernels.append(cw.shape[2])
    # Standard wav2vec2 strides
    conv_strides = [5, 2, 2, 2, 2, 2, 2]

    # Positional conv
    pos_conv_k = sd["encoder_frontend.pos_encoder.conv.weight_v"].shape[2]
    pos_conv_g = sd["encoder_frontend.pos_encoder.conv.weight_v"].shape[1]  # Cin/groups

    # n_heads: not directly stored. Infer from the fact that wav2vec2-base uses
    # head_dim * n_heads = hidden_size. Common configs: hidden=768→12 heads (64),
    # hidden=1024→16 heads (64), hidden=1280→16 heads (80). We'll store it in
    # the GGUF and let the runtime read it.
    n_heads = hidden_size // 64  # assume head_dim=64 for omniASR (fairseq2 default)

    # Check if there's layer_norm in the CNN (stable LN variant)
    has_layer_norm = f"encoder_frontend.feature_extractor.layers.0.layer_norm.weight" in sd
    feat_extract_norm_type = 1 if has_layer_norm else 0

    print(f"  hidden={hidden_size} layers={n_layers} heads={n_heads} ff={intermediate_size}")
    print(f"  vocab={vocab_size} CNN dims={conv_dims} kernels={conv_kernels}")
    print(f"  pos_conv K={pos_conv_k} groups={hidden_size // pos_conv_g}")
    print(f"  feat_extract_norm_type={'layer_norm' if has_layer_norm else 'group_norm'}")

    # Vocab
    vocab = None
    spm_path = list(input_dir.glob("*tokenizer*"))
    if spm_path and spm:
        sp = spm.SentencePieceProcessor(model_file=str(spm_path[0]))
        vocab = [sp.id_to_piece(i) for i in range(sp.get_piece_size())]
        print(f"  spm: {spm_path[0].name} ({len(vocab)} pieces)")

    # ---- Write GGUF ----
    print(f"\nWriting: {out_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = gguf.GGUFWriter(str(out_path), arch="wav2vec2", use_temp_file=True)

    writer.add_uint32("wav2vec2.vocab_size",                    vocab_size)
    writer.add_uint32("wav2vec2.hidden_size",                   hidden_size)
    writer.add_uint32("wav2vec2.num_hidden_layers",             n_layers)
    writer.add_uint32("wav2vec2.num_attention_heads",           n_heads)
    writer.add_uint32("wav2vec2.intermediate_size",             intermediate_size)
    writer.add_uint32("wav2vec2.num_feat_extract_layers",       7)
    writer.add_uint32("wav2vec2.num_conv_pos_embeddings",       pos_conv_k)
    writer.add_uint32("wav2vec2.num_conv_pos_embedding_groups", hidden_size // pos_conv_g)
    writer.add_uint32("wav2vec2.pad_token_id",                  vocab_size - 1)  # CTC blank
    writer.add_uint32("wav2vec2.feat_extract_norm_type",        feat_extract_norm_type)
    for i in range(7):
        writer.add_uint32(f"wav2vec2.conv_dim_{i}",    conv_dims[i] if i < len(conv_dims) else 512)
        writer.add_uint32(f"wav2vec2.conv_kernel_{i}",  conv_kernels[i] if i < len(conv_kernels) else 3)
        writer.add_uint32(f"wav2vec2.conv_stride_{i}",  conv_strides[i])

    if vocab:
        writer.add_array("tokenizer.ggml.tokens", vocab)

    # Merge pos conv weight_g + weight_v
    wg = sd["encoder_frontend.pos_encoder.conv.weight_g"]
    wv = sd["encoder_frontend.pos_encoder.conv.weight_v"]
    # weight_g: (1, 1, K)  weight_v: (Cout, Cin/groups, K)
    # merged weight = weight_g * weight_v / norm(weight_v, dim=...) where norm is over
    # the (Cin/groups, K) dimensions for each output channel.
    wv_norm = wv.norm(dim=(1, 2), keepdim=True).clamp(min=1e-12)
    pos_conv_w = (wg * wv / wv_norm).float().numpy()
    writer.add_tensor("pos_conv.weight", pos_conv_w.astype(np.float32))

    # Write all other tensors
    n_written = 1  # pos_conv_w already written
    n_f16 = 0
    n_f32 = 1
    for name in sorted(sd.keys()):
        gguf_name = remap(name)
        if gguf_name is None:
            continue
        t = sd[name].cpu().float().numpy()
        if is_f32(gguf_name):
            t = t.astype(np.float32)
            n_f32 += 1
        else:
            t = t.astype(np.float16)
            n_f16 += 1
        writer.add_tensor(gguf_name, t)
        n_written += 1
        if n_written <= 20 or n_written % 100 == 0:
            print(f"  {gguf_name:60s}  {str(t.shape):20s}  {t.dtype}")

    print(f"\n  total: {n_written} tensors (F16: {n_f16}, F32: {n_f32})")
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"Done: {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    convert(args.input, args.output)
