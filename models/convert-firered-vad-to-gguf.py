#!/usr/bin/env python3
"""Convert FireRedVAD (DFSMN) to GGUF.

Architecture: 8-block DFSMN (Feedforward Sequential Memory Network)
  - Input: 80-dim fbank → fc1(256) → fc2(128) → 8× FSMN blocks → dnn(256) → out(1)
  - Each FSMN block: lookback_conv(k=20) + lookahead_conv(k=20) + skip connection
  - Total params: ~588K (2.2 MB)

Three variants: VAD (non-streaming), Stream-VAD (streaming), AED (multi-label)

Usage:
  python models/convert-firered-vad-to-gguf.py \
      --input FireRedTeam/FireRedVAD \
      --variant VAD \
      --output firered-vad.gguf
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
    parser = argparse.ArgumentParser(description="Convert FireRedVAD to GGUF")
    parser.add_argument("--input", required=True, help="HF model ID or local path")
    parser.add_argument("--variant", default="VAD", choices=["VAD", "Stream-VAD", "AED"])
    parser.add_argument("--output", required=True, help="Output GGUF path")
    args = parser.parse_args()

    import torch
    from huggingface_hub import hf_hub_download

    # Download model
    if os.path.isdir(args.input):
        model_path = os.path.join(args.input, args.variant, "model.pth.tar")
        cmvn_path = os.path.join(args.input, args.variant, "cmvn.ark")
    else:
        model_path = hf_hub_download(args.input, f"{args.variant}/model.pth.tar")
        cmvn_path = hf_hub_download(args.input, f"{args.variant}/cmvn.ark")

    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    model_args = ckpt["args"]
    sd = ckpt["model_state_dict"]

    print(f"FireRedVAD ({args.variant}): {len(sd)} tensors, "
          f"{sum(v.numel() for v in sd.values()):,} params")
    print(f"  R={model_args.R}, H={model_args.H}, P={model_args.P}, "
          f"N1={model_args.N1}, N2={model_args.N2}, idim={model_args.idim}")

    # Parse CMVN
    with open(cmvn_path, "rb") as f:
        data = f.read()
    cmvn_mean = cmvn_std = None
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
    writer = gguf.GGUFWriter(args.output, "firered-vad")
    writer.add_name(f"FireRedVAD-{args.variant}")

    writer.add_uint32("firered_vad.R", model_args.R)  # number of FSMN blocks
    writer.add_uint32("firered_vad.H", model_args.H)  # hidden size
    writer.add_uint32("firered_vad.P", model_args.P)  # projection size
    writer.add_uint32("firered_vad.N1", model_args.N1)  # lookback context
    writer.add_uint32("firered_vad.N2", model_args.N2)  # lookahead context
    writer.add_uint32("firered_vad.S1", model_args.S1)  # lookback stride
    writer.add_uint32("firered_vad.S2", model_args.S2)  # lookahead stride
    writer.add_uint32("firered_vad.idim", model_args.idim)
    writer.add_uint32("firered_vad.odim", model_args.odim)

    # CMVN
    if cmvn_mean is not None:
        writer.add_tensor("cmvn.mean", cmvn_mean)
        writer.add_tensor("cmvn.std", cmvn_std)

    # Write tensors (model is tiny — keep as F32)
    tensor_count = 0
    for name, tensor in sorted(sd.items()):
        t = tensor.numpy()
        writer.add_tensor(name, t.astype(np.float32))
        tensor_count += 1

    print(f"\nWriting: {args.output}")
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    file_size = os.path.getsize(args.output)
    print(f"Done: {args.output} ({file_size / 1e6:.1f} MB, {tensor_count} tensors)")


if __name__ == "__main__":
    main()
