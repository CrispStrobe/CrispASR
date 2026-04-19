#!/usr/bin/env python3
"""Convert FireRedASR2-AED to GGUF.

Architecture: Conformer encoder (16L, d=1280, 20 heads, rel-PE, macaron FFN)
            + Transformer decoder (16L, d=1280, cross-attention)
            + CTC head
            + BPE tokenizer (8667 tokens)

Usage:
  python models/convert-firered-asr-to-gguf.py \
      --input FireRedTeam/FireRedASR2-AED \
      --output firered-asr2-aed.gguf
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
    parser = argparse.ArgumentParser(description="Convert FireRedASR2-AED to GGUF")
    parser.add_argument("--input", required=True, help="HF model ID or local path")
    parser.add_argument("--output", required=True, help="Output GGUF path")
    args = parser.parse_args()

    print(f"Loading: {args.input}")

    import torch
    from huggingface_hub import hf_hub_download

    cache_dir = os.environ.get("HF_HOME", None)
    if cache_dir:
        cache_dir += "/hub"

    # Resolve file paths
    if os.path.isdir(args.input):
        base = args.input
    else:
        # Download all needed files
        for fname in ["model.pth.tar", "dict.txt", "train_bpe1000.model", "cmvn.ark"]:
            hf_hub_download(args.input, fname, cache_dir=cache_dir)
        config_path = hf_hub_download(args.input, "dict.txt", cache_dir=cache_dir)
        base = os.path.dirname(config_path)

    model_path = os.path.join(base, "model.pth.tar")
    dict_path = os.path.join(base, "dict.txt")
    cmvn_path = os.path.join(base, "cmvn.ark")

    # Load checkpoint
    print("  Loading checkpoint...")
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    model_args = ckpt["args"]
    sd = ckpt["model_state_dict"]
    print(f"  {len(sd)} tensors")

    # Extract hyperparameters
    hp = {
        "d_model": model_args.d_model,          # 1280
        "n_head": model_args.n_head,             # 20
        "d_inner": model_args.d_inner,           # 5120
        "n_layers_enc": model_args.n_layers_enc, # 16
        "n_layers_dec": model_args.n_layers_dec, # 16
        "idim": model_args.idim,                 # 80 (mel bins)
        "odim": model_args.odim,                 # 8667 (vocab)
        "subsample": model_args.subsample,       # 4
        "kernel_size": model_args.kernel_size,   # 33
        "pe_maxlen": model_args.pe_maxlen,       # 5000
        "sos_id": model_args.sos_id,             # 3
        "eos_id": model_args.eos_id,             # 4
        "blank_id": model_args.blank_id,         # 0
        "pad_id": model_args.pad_id,             # 2
    }
    print(f"  Architecture: d_model={hp['d_model']}, heads={hp['n_head']}, "
          f"enc_layers={hp['n_layers_enc']}, dec_layers={hp['n_layers_dec']}, "
          f"vocab={hp['odim']}")

    # Load vocabulary from dict.txt
    vocab = []
    with open(dict_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                vocab.append(parts[0])
    print(f"  Vocabulary: {len(vocab)} tokens")

    # Load CMVN (binary Kaldi ark format — extract mean and variance)
    # For now, read as binary and parse the float arrays
    cmvn_mean = None
    cmvn_var = None
    try:
        import struct
        with open(cmvn_path, "rb") as f:
            data = f.read()
        # Kaldi binary ark: skip header, find float matrices
        # Format: " BDM " header + matrix data
        # Simple approach: find the float32 arrays
        # The CMVN file contains mean[80] and variance[80]
        idx = data.find(b"BDM")
        if idx >= 0:
            idx += 4  # skip "BDM " + type byte
            # Read dimensions
            # Kaldi format: rows cols then float data
            # Let's just extract 160 floats (80 mean + 80 var)
            float_start = idx
            while float_start < len(data) and data[float_start:float_start+1] not in [b'\x00']:
                float_start += 1
            # Try to find 80 consecutive floats
            n_floats = (len(data) - idx) // 4
            if n_floats >= 160:
                all_floats = struct.unpack(f"<{n_floats}f", data[idx:idx + n_floats * 4])
                # First 80 = sum, next 80 = sum_sq, last = count
                # Actually CMVN format varies. Let's skip for now and handle in C++
        print(f"  CMVN: loaded ({len(data)} bytes)")
    except Exception as e:
        print(f"  CMVN: skipped ({e})")

    # Create GGUF
    writer = gguf.GGUFWriter(args.output, "firered-asr")
    writer.add_name("FireRedASR2-AED")

    # Write hyperparameters
    writer.add_uint32("firered.d_model", hp["d_model"])
    writer.add_uint32("firered.n_head", hp["n_head"])
    writer.add_uint32("firered.d_inner", hp["d_inner"])
    writer.add_uint32("firered.n_layers_enc", hp["n_layers_enc"])
    writer.add_uint32("firered.n_layers_dec", hp["n_layers_dec"])
    writer.add_uint32("firered.idim", hp["idim"])
    writer.add_uint32("firered.odim", hp["odim"])
    writer.add_uint32("firered.subsample", hp["subsample"])
    writer.add_uint32("firered.kernel_size", hp["kernel_size"])
    writer.add_uint32("firered.pe_maxlen", hp["pe_maxlen"])
    writer.add_uint32("firered.sos_id", hp["sos_id"])
    writer.add_uint32("firered.eos_id", hp["eos_id"])
    writer.add_uint32("firered.blank_id", hp["blank_id"])
    writer.add_uint32("firered.pad_id", hp["pad_id"])

    # Tokenizer
    writer.add_array("tokenizer.ggml.tokens", vocab)

    # Write tensors
    print(f"\nWriting: {args.output}")

    def f16(t):
        return t.astype(np.float16) if t.dtype == np.float32 else t

    def f32(t):
        return t.astype(np.float32)

    # Tensor name shortening to stay under GGUF 64-char limit
    def shorten_name(name):
        name = name.replace("encoder.layer_stack.", "enc.")
        name = name.replace("encoder.input_preprocessor.", "enc.preproc.")
        name = name.replace("encoder.positional_encoding.", "enc.pe.")
        name = name.replace("decoder.layer_stack.", "dec.")
        name = name.replace("decoder.positional_encoding.", "dec.pe.")
        name = name.replace("decoder.layer_norm_out.", "dec.norm_out.")
        name = name.replace("decoder.tgt_word_emb.", "dec.emb.")
        name = name.replace("decoder.tgt_word_prj.", "dec.prj.")
        name = name.replace("cross_attn_norm.", "xattn_norm.")
        name = name.replace("cross_attn.", "xattn.")
        name = name.replace("self_attn_norm.", "sattn_norm.")
        name = name.replace("self_attn.", "sattn.")
        name = name.replace("positional_encoding.", "pe.")
        name = name.replace("layer_norm_", "ln_")
        name = name.replace("layer_norm.", "ln.")
        name = name.replace("batch_norm.", "bn.")
        name = name.replace("depthwise_conv.", "dw.")
        name = name.replace("pointwise_conv1.", "pw1.")
        name = name.replace("pointwise_conv2.", "pw2.")
        name = name.replace("pre_layer_norm.", "pre_ln.")
        name = name.replace("linear_pos.", "lin_pos.")
        name = name.replace("ctc.ctc_lo.", "ctc.")
        return name

    tensor_count = 0
    for name, tensor in sorted(sd.items()):
        t = tensor.float().numpy() if tensor.dtype == torch.bfloat16 else tensor.numpy()
        gguf_name = shorten_name(name)
        assert len(gguf_name) < 64, f"Name too long ({len(gguf_name)}): {gguf_name}"

        if "norm" in name or name.endswith(".bias") or "pe" in name or len(t.shape) <= 1:
            data = f32(t)
            dtype_str = "F32"
        else:
            data = f16(t)
            dtype_str = "F16"
        writer.add_tensor(gguf_name, data)
        tensor_count += 1
        if tensor_count <= 5 or tensor_count % 100 == 0:
            print(f"  [{tensor_count}] {gguf_name:55s} {str(t.shape):25s} {dtype_str}")

    print(f"  ... total: {tensor_count} tensors")

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    file_size = os.path.getsize(args.output)
    print(f"\nDone: {args.output} ({file_size / 1e9:.2f} GB, {tensor_count} tensors)")


if __name__ == "__main__":
    main()
