#!/usr/bin/env python3
"""Convert SpeechBrain ECAPA-TDNN LID to GGUF.

Model: speechbrain/lang-id-voxlingua107-ecapa (Apache-2.0)
Architecture: ECAPA-TDNN with SE-Res2Net blocks + attentive statistical pooling
  - 60-dim mel fbank input
  - channels=[1024,1024,1024,1024,3072], kernels=[5,3,3,3,1], dilations=[1,2,3,4,1]
  - 107 language classes
  - 21M params, ~84 MB F32

Usage:
  python models/convert-ecapa-tdnn-lid-to-gguf.py \
      --input speechbrain/lang-id-voxlingua107-ecapa \
      --output ecapa-lid-107.gguf
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
    parser = argparse.ArgumentParser(description="Convert ECAPA-TDNN LID to GGUF")
    parser.add_argument("--input", required=True, help="HF model ID or local path")
    parser.add_argument("--output", required=True, help="Output GGUF path")
    args = parser.parse_args()

    import torch
    from huggingface_hub import hf_hub_download

    # Download model files
    if os.path.isdir(args.input):
        base = args.input
    else:
        for f in ["embedding_model.ckpt", "classifier.ckpt", "label_encoder.txt"]:
            hf_hub_download(args.input, f)
        base = os.path.dirname(hf_hub_download(args.input, "embedding_model.ckpt"))

    emb_ckpt = torch.load(os.path.join(base, "embedding_model.ckpt"), map_location="cpu", weights_only=False)
    cls_ckpt = torch.load(os.path.join(base, "classifier.ckpt"), map_location="cpu", weights_only=False)

    # Read language labels (skip metadata lines like '======' and 'starting_index')
    labels = []
    label_path = os.path.join(base, "label_encoder.txt")
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if "=>" not in line or line.startswith("="):
                continue
            # Format: 'en: English' => 20
            label_str = line.split("=>")[0].strip().strip("'")
            if ":" not in label_str:
                continue  # skip 'starting_index' and similar metadata
            code = label_str.split(":")[0].strip()
            labels.append(code)

    print(f"ECAPA-TDNN LID: {len(emb_ckpt)} emb tensors, {len(cls_ckpt)} cls tensors, {len(labels)} labels")

    # Create GGUF
    writer = gguf.GGUFWriter(args.output, "ecapa-tdnn-lid")
    writer.add_name("ECAPA-TDNN-LID-VoxLingua107")

    # Hyperparameters
    writer.add_uint32("ecapa.n_mels", 60)
    writer.add_uint32("ecapa.n_classes", len(labels))
    writer.add_uint32("ecapa.channels_0", 1024)
    writer.add_uint32("ecapa.channels_mfa", 3072)
    writer.add_uint32("ecapa.attention_channels", 128)
    writer.add_uint32("ecapa.lin_neurons", 256)
    writer.add_uint32("ecapa.n_se_res2net_blocks", 3)  # blocks 1,2,3
    writer.add_uint32("ecapa.res2net_scale", 8)  # 1024/128 = 8 sub-bands

    # Tokenizer (language labels)
    writer.add_array("tokenizer.ggml.tokens", labels)

    def f32(t):
        return t.float().numpy()

    def f16(t):
        return t.float().numpy().astype(np.float16)

    # Write embedding model tensors
    tensor_count = 0
    for name, tensor in sorted(emb_ckpt.items()):
        if "num_batches_tracked" in name:
            continue  # Skip BN tracking counters
        t = tensor.float().numpy()

        # Shorten names
        gguf_name = "emb." + name
        gguf_name = gguf_name.replace(".conv.conv.", ".conv.")
        gguf_name = gguf_name.replace(".norm.norm.", ".bn.")

        if len(gguf_name) >= 64:
            print(f"  WARNING: name too long ({len(gguf_name)}): {gguf_name}")
            # Further shorten
            gguf_name = gguf_name.replace("res2net_block.blocks.", "r2n.")
            gguf_name = gguf_name.replace("se_block.", "se.")

        # Squeeze kernel=1 conv weights
        if len(t.shape) == 3 and t.shape[-1] == 1:
            t = t.squeeze(-1)

        # Store norms/biases as F32, weights as F16
        if "bn." in gguf_name or "bias" in name or len(t.shape) <= 1:
            data = f32(tensor)
        else:
            data = f16(tensor)

        writer.add_tensor(gguf_name, data)
        tensor_count += 1
        if tensor_count <= 5 or tensor_count % 20 == 0:
            print(f"  [{tensor_count}] {gguf_name:55s} {str(data.shape):20s}")

    # Write classifier tensors
    for name, tensor in sorted(cls_ckpt.items()):
        if "num_batches_tracked" in name:
            continue
        t = tensor.float().numpy()
        gguf_name = "cls." + name
        gguf_name = gguf_name.replace(".norm.norm.", ".bn.")
        gguf_name = gguf_name.replace(".linear.w.", ".linear.")

        if "bn." in gguf_name or "bias" in name or len(t.shape) <= 1:
            data = f32(tensor)
        else:
            data = f16(tensor)

        writer.add_tensor(gguf_name, data)
        tensor_count += 1
        print(f"  [{tensor_count}] {gguf_name:55s} {str(data.shape):20s}")

    print(f"  ... total: {tensor_count} tensors")

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    file_size = os.path.getsize(args.output)
    print(f"\nDone: {args.output} ({file_size / 1e6:.1f} MB, {tensor_count} tensors)")


if __name__ == "__main__":
    main()
