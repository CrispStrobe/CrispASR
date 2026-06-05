#!/usr/bin/env python3
"""Convert FunASR Paraformer (paraformer-zh / paraformer-en) to GGUF.

Architecture:
  50-block SANMEncoder (1 entry block 560→512 + 49 main blocks 512→512)
  → CifPredictorV2 (Conv1d + Linear + sigmoid → CIF accumulation)
  → 16-block ParaformerSANMDecoder (FSMN self-attn + cross-attn + FFN)
  → output_layer (Linear 512→vocab)

The encoder is structurally identical to the FunASR-Nano SANM encoder and
reuses the same core_sanm::build_block() helper in the C++ runtime.

Usage:
  python models/convert-paraformer-to-gguf.py \\
      --input funasr/paraformer-zh \\
      --output paraformer-zh.gguf
"""

import argparse
import json
import os
import re
import struct
import sys

import numpy as np


def to_np(t):
    import torch
    t = t.detach()
    if t.dtype == torch.bfloat16:
        return t.float().numpy()
    return t.numpy()


def add_tensor(writer, name: str, t: np.ndarray, *, force_f32: bool = False):
    assert len(name) < 64, f"GGUF name too long ({len(name)}): {name}"
    if not force_f32 and t.ndim >= 2:
        data = np.ascontiguousarray(t.astype(np.float16))
        dtype = "F16"
    else:
        data = np.ascontiguousarray(t.astype(np.float32))
        dtype = "F32"
    writer.add_tensor(name, data)
    return dtype


def parse_kaldi_cmvn(path):
    """Parse am.mvn (Kaldi AddShift + Rescale) into shift/scale arrays."""
    with open(path, encoding="utf-8") as f:
        text = f.read()
    # Extract the two vectors between [ ... ]
    vectors = re.findall(r'\[\s*([-\d\s.eE+]+)\s*\]', text)
    if len(vectors) < 2:
        raise ValueError(f"Expected 2 vectors in {path}, found {len(vectors)}")
    shift = np.fromstring(vectors[1], sep=' ', dtype=np.float32)  # AddShift
    scale = np.fromstring(vectors[2], sep=' ', dtype=np.float32)  # Rescale
    return shift, scale


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True,
                    help="HF model ID or local snapshot dir")
    ap.add_argument("--output", required=True, help="Output GGUF path")
    args = ap.parse_args()

    import torch
    import gguf
    from huggingface_hub import snapshot_download

    if os.path.isdir(args.input):
        base = args.input
    else:
        print(f"Downloading {args.input}")
        base = snapshot_download(
            repo_id=args.input,
            allow_patterns=["model.pt", "config.yaml", "tokens.json", "am.mvn",
                            "configuration.json", "seg_dict"],
        )
    print(f"Loading from {base}")

    model_pt = os.path.join(base, "model.pt")
    print("  Loading state dict ...")
    sd = torch.load(model_pt, map_location="cpu", weights_only=False)
    print(f"  {len(sd)} tensors")

    # ---- Hyperparameters (hardcoded for paraformer-zh / paraformer-en) ----
    hp = dict(
        n_mels=80, lfr_m=7, lfr_n=6, sample_rate=16000,
        frame_length_ms=25, frame_shift_ms=10,
        d_model=512, n_heads=4, ffn_dim=2048,
        n_enc_blocks0=1, n_enc_blocks=49, sanm_kernel=11,
        n_dec_blocks=16, dec_ffn_dim=2048,
        cif_conv_kernel=3,
    )

    # Read vocab size from tokens.json
    tok_path = os.path.join(base, "tokens.json")
    with open(tok_path, encoding="utf-8") as f:
        tokens = json.load(f)
    vocab_size = len(tokens)
    hp["vocab_size"] = vocab_size
    print(f"  Vocab: {vocab_size} tokens")

    # ---- Open GGUF writer ----
    writer = gguf.GGUFWriter(args.output, "paraformer")
    writer.add_name("Paraformer-zh")
    for k, v in hp.items():
        writer.add_uint32(f"paraformer.{k}", int(v))

    # ---- Write CMVN ----
    cmvn_path = os.path.join(base, "am.mvn")
    if os.path.isfile(cmvn_path):
        shift, scale = parse_kaldi_cmvn(cmvn_path)
        add_tensor(writer, "paraformer.cmvn_shift", shift, force_f32=True)
        add_tensor(writer, "paraformer.cmvn_scale", scale, force_f32=True)
        print(f"  CMVN: shift {shift.shape}, scale {scale.shape}")

    # ---- Write tokenizer ----
    tok_data = "\n".join(tokens).encode("utf-8")
    writer.add_array("tokenizer.tokens", [t.encode("utf-8") for t in tokens])

    # ---- Tensor write helpers ----
    n_written = 0
    n_f16 = 0
    n_f32 = 0

    def write(name, tensor, *, force_f32=False):
        nonlocal n_written, n_f16, n_f32
        t = to_np(tensor)
        dtype = add_tensor(writer, name, t, force_f32=force_f32)
        if dtype == "F16":
            n_f16 += 1
        else:
            n_f32 += 1
        n_written += 1

    # ---- Encoder: encoders0 (1 entry block, 560→512) ----
    for L in range(hp["n_enc_blocks0"]):
        p = f"encoder.encoders0.{L}"
        o = f"paraformer.enc0.blk.{L}"
        write(f"{o}.norm1.w", sd[f"{p}.norm1.weight"], force_f32=True)
        write(f"{o}.norm1.b", sd[f"{p}.norm1.bias"], force_f32=True)
        write(f"{o}.norm2.w", sd[f"{p}.norm2.weight"], force_f32=True)
        write(f"{o}.norm2.b", sd[f"{p}.norm2.bias"], force_f32=True)
        write(f"{o}.attn.qkv.w", sd[f"{p}.self_attn.linear_q_k_v.weight"])
        write(f"{o}.attn.qkv.b", sd[f"{p}.self_attn.linear_q_k_v.bias"], force_f32=True)
        write(f"{o}.attn.out.w", sd[f"{p}.self_attn.linear_out.weight"])
        write(f"{o}.attn.out.b", sd[f"{p}.self_attn.linear_out.bias"], force_f32=True)
        fsmn = to_np(sd[f"{p}.self_attn.fsmn_block.weight"])
        if fsmn.ndim == 3 and fsmn.shape[1] == 1:
            fsmn = fsmn.squeeze(1)
        add_tensor(writer, f"{o}.attn.fsmn.w", fsmn)
        n_written += 1; n_f16 += 1
        write(f"{o}.ffn.l1.w", sd[f"{p}.feed_forward.w_1.weight"])
        write(f"{o}.ffn.l1.b", sd[f"{p}.feed_forward.w_1.bias"], force_f32=True)
        write(f"{o}.ffn.l2.w", sd[f"{p}.feed_forward.w_2.weight"])
        write(f"{o}.ffn.l2.b", sd[f"{p}.feed_forward.w_2.bias"], force_f32=True)

    # ---- Encoder: main blocks (49 blocks, 512→512) ----
    for L in range(hp["n_enc_blocks"]):
        p = f"encoder.encoders.{L}"
        o = f"paraformer.enc.blk.{L}"
        write(f"{o}.norm1.w", sd[f"{p}.norm1.weight"], force_f32=True)
        write(f"{o}.norm1.b", sd[f"{p}.norm1.bias"], force_f32=True)
        write(f"{o}.norm2.w", sd[f"{p}.norm2.weight"], force_f32=True)
        write(f"{o}.norm2.b", sd[f"{p}.norm2.bias"], force_f32=True)
        write(f"{o}.attn.qkv.w", sd[f"{p}.self_attn.linear_q_k_v.weight"])
        write(f"{o}.attn.qkv.b", sd[f"{p}.self_attn.linear_q_k_v.bias"], force_f32=True)
        write(f"{o}.attn.out.w", sd[f"{p}.self_attn.linear_out.weight"])
        write(f"{o}.attn.out.b", sd[f"{p}.self_attn.linear_out.bias"], force_f32=True)
        fsmn = to_np(sd[f"{p}.self_attn.fsmn_block.weight"])
        if fsmn.ndim == 3 and fsmn.shape[1] == 1:
            fsmn = fsmn.squeeze(1)
        add_tensor(writer, f"{o}.attn.fsmn.w", fsmn)
        n_written += 1; n_f16 += 1
        write(f"{o}.ffn.l1.w", sd[f"{p}.feed_forward.w_1.weight"])
        write(f"{o}.ffn.l1.b", sd[f"{p}.feed_forward.w_1.bias"], force_f32=True)
        write(f"{o}.ffn.l2.w", sd[f"{p}.feed_forward.w_2.weight"])
        write(f"{o}.ffn.l2.b", sd[f"{p}.feed_forward.w_2.bias"], force_f32=True)

    # ---- Encoder: after_norm ----
    write("paraformer.enc.after_norm.w", sd["encoder.after_norm.weight"], force_f32=True)
    write("paraformer.enc.after_norm.b", sd["encoder.after_norm.bias"], force_f32=True)

    # ---- CIF Predictor ----
    # Conv1d(512, 512, 3, padding=1) → ReLU → Linear(512, 1) → sigmoid
    write("paraformer.cif.conv.w", sd["predictor.cif_conv1d.weight"])
    write("paraformer.cif.conv.b", sd["predictor.cif_conv1d.bias"], force_f32=True)
    write("paraformer.cif.out.w", sd["predictor.cif_output.weight"])
    write("paraformer.cif.out.b", sd["predictor.cif_output.bias"], force_f32=True)

    # ---- Decoder: 16 SANM+CrossAttn blocks ----
    for L in range(hp["n_dec_blocks"]):
        p = f"decoder.decoders.{L}"
        o = f"paraformer.dec.blk.{L}"
        # norm1 (before FSMN self-attn)
        write(f"{o}.norm1.w", sd[f"{p}.norm1.weight"], force_f32=True)
        write(f"{o}.norm1.b", sd[f"{p}.norm1.bias"], force_f32=True)
        # FSMN self-attention (depthwise conv only, no QKV)
        fsmn = to_np(sd[f"{p}.self_attn.fsmn_block.weight"])
        if fsmn.ndim == 3 and fsmn.shape[1] == 1:
            fsmn = fsmn.squeeze(1)
        add_tensor(writer, f"{o}.fsmn.w", fsmn)
        n_written += 1; n_f16 += 1
        # norm2 (before cross-attention)
        write(f"{o}.norm2.w", sd[f"{p}.norm2.weight"], force_f32=True)
        write(f"{o}.norm2.b", sd[f"{p}.norm2.bias"], force_f32=True)
        # Cross-attention: Q from decoder, fused K+V from encoder
        write(f"{o}.cross.q.w", sd[f"{p}.src_attn.linear_q.weight"])
        write(f"{o}.cross.q.b", sd[f"{p}.src_attn.linear_q.bias"], force_f32=True)
        write(f"{o}.cross.kv.w", sd[f"{p}.src_attn.linear_k_v.weight"])
        write(f"{o}.cross.kv.b", sd[f"{p}.src_attn.linear_k_v.bias"], force_f32=True)
        write(f"{o}.cross.out.w", sd[f"{p}.src_attn.linear_out.weight"])
        write(f"{o}.cross.out.b", sd[f"{p}.src_attn.linear_out.bias"], force_f32=True)
        # norm3 (before FFN)
        write(f"{o}.norm3.w", sd[f"{p}.norm3.weight"], force_f32=True)
        write(f"{o}.norm3.b", sd[f"{p}.norm3.bias"], force_f32=True)
        # FFN: w_1 → norm → w_2 (note: decoder FFN has internal LayerNorm!)
        write(f"{o}.ffn.l1.w", sd[f"{p}.feed_forward.w_1.weight"])
        write(f"{o}.ffn.l1.b", sd[f"{p}.feed_forward.w_1.bias"], force_f32=True)
        write(f"{o}.ffn.norm.w", sd[f"{p}.feed_forward.norm.weight"], force_f32=True)
        write(f"{o}.ffn.norm.b", sd[f"{p}.feed_forward.norm.bias"], force_f32=True)
        write(f"{o}.ffn.l2.w", sd[f"{p}.feed_forward.w_2.weight"])
        # Note: decoder FFN w_2 has NO bias in the upstream model

    # ---- Decoder: decoders3 (post-processing block) ----
    if "decoder.decoders3.0.norm1.weight" in sd:
        p = "decoder.decoders3.0"
        o = "paraformer.dec.post"
        write(f"{o}.norm1.w", sd[f"{p}.norm1.weight"], force_f32=True)
        write(f"{o}.norm1.b", sd[f"{p}.norm1.bias"], force_f32=True)
        write(f"{o}.ffn.l1.w", sd[f"{p}.feed_forward.w_1.weight"])
        write(f"{o}.ffn.l1.b", sd[f"{p}.feed_forward.w_1.bias"], force_f32=True)
        write(f"{o}.ffn.norm.w", sd[f"{p}.feed_forward.norm.weight"], force_f32=True)
        write(f"{o}.ffn.norm.b", sd[f"{p}.feed_forward.norm.bias"], force_f32=True)
        write(f"{o}.ffn.l2.w", sd[f"{p}.feed_forward.w_2.weight"])

    # ---- Decoder: after_norm + output_layer + embed ----
    write("paraformer.dec.after_norm.w", sd["decoder.after_norm.weight"], force_f32=True)
    write("paraformer.dec.after_norm.b", sd["decoder.after_norm.bias"], force_f32=True)
    write("paraformer.dec.output.w", sd["decoder.output_layer.weight"])
    write("paraformer.dec.output.b", sd["decoder.output_layer.bias"], force_f32=True)
    write("paraformer.dec.embed.w", sd["decoder.embed.0.weight"])

    # ---- Finalize ----
    print(f"\n  Total {n_written} tensors  (F16: {n_f16}, F32: {n_f32})")
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    sz = os.path.getsize(args.output) / (1024 * 1024)
    print(f"\nDone: {args.output} ({sz:.2f} MB)")


if __name__ == "__main__":
    main()
