#!/usr/bin/env python3
"""Convert FireRedLID to GGUF.

The LID model reuses the FireRedASR2-AED encoder but has a separate
6-layer Transformer decoder for language identification (120 languages).

This converter produces a SINGLE GGUF with both encoder + LID decoder.

Usage:
  python models/convert-firered-lid-to-gguf.py \
      --input FireRedTeam/FireRedLID \
      --output firered-lid.gguf
"""

import argparse
import os
import struct
import sys

import numpy as np

try:
    import gguf
except ImportError:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ggml", "python"))
    import gguf


def main():
    parser = argparse.ArgumentParser(description="Convert FireRedLID to GGUF")
    parser.add_argument("--input", required=True, help="HF model ID or local path")
    parser.add_argument("--output", required=True, help="Output GGUF path")
    args = parser.parse_args()

    import torch
    from huggingface_hub import hf_hub_download

    # Download model files
    if os.path.isdir(args.input):
        base = args.input
    else:
        for f in ["model.pth.tar", "dict.txt", "cmvn.ark"]:
            hf_hub_download(args.input, f)
        p = hf_hub_download(args.input, "model.pth.tar")
        base = os.path.dirname(p)

    ckpt = torch.load(os.path.join(base, "model.pth.tar"), map_location="cpu", weights_only=False)
    model_args = ckpt["args"]
    sd = ckpt["model_state_dict"]

    print(f"FireRedLID: {len(sd)} tensors")
    print(f"  lid_odim={model_args.lid_odim}, n_layers_lid_dec={model_args.n_layers_lid_dec}")
    print(f"  d_model={model_args.d_model}, n_head={model_args.n_head}")

    # Load vocab
    vocab = []
    dict_path = os.path.join(base, "dict.txt")
    with open(dict_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if parts:
                vocab.append(parts[0])
    print(f"  Vocabulary: {len(vocab)} tokens")

    # Parse CMVN
    cmvn_path = os.path.join(base, "cmvn.ark")
    cmvn_mean = cmvn_std = None
    with open(cmvn_path, "rb") as f:
        data = f.read()
    for i in range(len(data) - 3):
        if data[i:i+3] in (b"BDM", b"BFM"):
            idx = i + 3
            if data[idx] == 0x20: idx += 1
            assert data[idx] == 4; idx += 1
            rows = struct.unpack("<i", data[idx:idx+4])[0]; idx += 4
            assert data[idx] == 4; idx += 1
            cols = struct.unpack("<i", data[idx:idx+4])[0]; idx += 4
            elem_size = 8 if data[i:i+3] == b"BDM" else 4
            dtype = "<f8" if data[i:i+3] == b"BDM" else "<f4"
            vals = np.frombuffer(data[idx:idx+rows*cols*elem_size], dtype=dtype).reshape(rows, cols)
            count = vals[0, -1]
            cmvn_mean = (vals[0, :-1] / count).astype(np.float32)
            cmvn_var = vals[1, :-1] / count - cmvn_mean.astype(np.float64)**2
            cmvn_std = np.sqrt(np.maximum(cmvn_var, 1e-10)).astype(np.float32)
            break

    # Create GGUF
    writer = gguf.GGUFWriter(args.output, "firered-lid")
    writer.add_name("FireRedLID")

    # Hyperparameters (encoder same as ASR, decoder is LID-specific)
    writer.add_uint32("firered.d_model", model_args.d_model)
    writer.add_uint32("firered.n_head", model_args.n_head)
    writer.add_uint32("firered.n_head_dec", model_args.layer_n_head)  # LID decoder uses 8 heads
    writer.add_uint32("firered.d_inner", model_args.d_inner)
    writer.add_uint32("firered.n_layers_enc", model_args.n_layers_enc)
    writer.add_uint32("firered.n_layers_dec", model_args.n_layers_lid_dec)  # LID decoder layers
    writer.add_uint32("firered.idim", model_args.idim)
    writer.add_uint32("firered.odim", model_args.lid_odim)  # 120 languages
    writer.add_uint32("firered.subsample", model_args.subsample)
    writer.add_uint32("firered.kernel_size", model_args.kernel_size)
    writer.add_uint32("firered.pe_maxlen", model_args.pe_maxlen)
    writer.add_uint32("firered.sos_id", model_args.sos_id)
    writer.add_uint32("firered.eos_id", model_args.eos_id)
    writer.add_uint32("firered.blank_id", model_args.blank_id)
    writer.add_uint32("firered.pad_id", model_args.pad_id)
    writer.add_uint32("firered.context", 7)

    # Tokenizer
    writer.add_array("tokenizer.ggml.tokens", vocab)

    # CMVN
    if cmvn_mean is not None:
        writer.add_tensor("cmvn.mean", cmvn_mean)
        writer.add_tensor("cmvn.std", cmvn_std)

    # Tensor name shortening (same as ASR converter)
    def shorten_name(name):
        name = name.replace("encoder.layer_stack.", "enc.")
        name = name.replace("encoder.input_preprocessor.", "enc.preproc.")
        name = name.replace("encoder.positional_encoding.", "enc.pe.")
        name = name.replace("lid_decoder.layer_stack.", "dec.")
        name = name.replace("lid_decoder.positional_encoding.", "dec.pe.")
        name = name.replace("lid_decoder.layer_norm_out.", "dec.norm_out.")
        name = name.replace("lid_decoder.tgt_word_emb.", "dec.emb.")
        name = name.replace("lid_decoder.tgt_word_prj.", "dec.prj.")
        name = name.replace("cross_attn_norm.", "xattn_norm.")
        name = name.replace("cross_attn.", "xattn.")
        name = name.replace("self_attn_norm.", "sattn_norm.")
        name = name.replace("self_attn.", "sattn.")
        name = name.replace("layer_norm_q.", "ln_q.")
        name = name.replace("layer_norm_k.", "ln_k.")
        name = name.replace("layer_norm_v.", "ln_v.")
        name = name.replace("layer_norm.", "ln.")
        name = name.replace("batch_norm.", "bn.")
        name = name.replace("depthwise_conv.", "dw.")
        name = name.replace("pointwise_conv1.", "pw1.")
        name = name.replace("pointwise_conv2.", "pw2.")
        name = name.replace("pre_layer_norm.", "pre_ln.")
        name = name.replace("linear_pos.", "lin_pos.")
        name = name.replace("depthwise_conv.", "dw.")
        name = name.replace("pointwise_conv1.", "pw1.")
        name = name.replace("pointwise_conv2.", "pw2.")
        return name

    def f16(t):
        return t.astype(np.float16) if t.dtype == np.float32 else t

    def f32(t):
        return t.astype(np.float32)

    tensor_count = 0
    for name, tensor in sorted(sd.items()):
        t = tensor.float().numpy() if tensor.dtype == torch.bfloat16 else tensor.numpy()
        gguf_name = shorten_name(name)
        if len(gguf_name) >= 64:
            print(f"  WARNING: name too long ({len(gguf_name)}): {gguf_name}")
            continue

        # Squeeze Conv1d kernel=1 weights from 3D [out, in, 1] to 2D [out, in]
        # so quantizer can process them as regular matrices.
        if len(t.shape) == 3 and 1 in t.shape and ("pointwise_conv" in name):
            t = t.squeeze()

        if "norm" in name or name.endswith(".bias") or "pe" in name or len(t.shape) <= 1:
            data = f32(t)
        else:
            data = f16(t)
        writer.add_tensor(gguf_name, data)
        tensor_count += 1
        if tensor_count <= 5 or tensor_count % 100 == 0:
            print(f"  [{tensor_count}] {gguf_name:55s} {str(t.shape):25s}")

    print(f"  ... total: {tensor_count} tensors")

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    file_size = os.path.getsize(args.output)
    print(f"\nDone: {args.output} ({file_size / 1e9:.2f} GB, {tensor_count} tensors)")


if __name__ == "__main__":
    main()
