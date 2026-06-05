#!/usr/bin/env python3
"""Convert NVIDIA TitaNet-Large speaker verification model to GGUF.

Model: nvidia/speakerverification_en_titanet_large (CC-BY-4.0)
Architecture: TitaNet-Large — depthwise separable Conv1D encoder + ASP decoder
  - 80-dim mel spectrogram input (Hann window, 512-point FFT)
  - Jasper-style encoder: 5 blocks (prolog + 3 mega + epilog)
  - Prolog:  DW-Conv1d(80→80, k=3) + PW-Conv1d(80→1024) + BN + SE
  - Mega 1:  3× DW-Sep-Conv(1024, k=7)  + SE + residual
  - Mega 2:  3× DW-Sep-Conv(1024, k=11) + SE + residual
  - Mega 3:  3× DW-Sep-Conv(1024, k=15) + SE + residual
  - Epilog:  DW-Conv1d(1024→1024, k=1) + PW-Conv1d(1024→3072) + BN + SE
  - Decoder: ASP(3072 → 6144) + Linear(6144 → 192) → L2-normalize
  - 23M params, ~46 MB GGUF (F16 weights)
  - 192-d speaker embedding output

Usage:
  python models/convert-titanet-to-gguf.py \
      --input nvidia/speakerverification_en_titanet_large \
      --output titanet-large.gguf
"""

import argparse
import io
import os
import sys
import tarfile

import numpy as np

try:
    import gguf
except ImportError:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ggml", "python"))
    import gguf


def main():
    parser = argparse.ArgumentParser(description="Convert TitaNet speaker verification to GGUF")
    parser.add_argument("--input", required=True, help="HF model ID or local .nemo path")
    parser.add_argument("--output", required=True, help="Output GGUF path")
    args = parser.parse_args()

    import torch
    import yaml

    # Locate the .nemo file
    if os.path.isfile(args.input) and args.input.endswith(".nemo"):
        nemo_path = args.input
    else:
        from huggingface_hub import hf_hub_download
        # Find the .nemo file in the repo
        from huggingface_hub import list_repo_files
        files = list_repo_files(args.input)
        nemo_files = [f for f in files if f.endswith(".nemo")]
        if not nemo_files:
            print(f"ERROR: No .nemo files found in {args.input}")
            sys.exit(1)
        nemo_path = hf_hub_download(args.input, nemo_files[0])

    print(f"Loading: {nemo_path}")

    # Extract config and weights from .nemo tarball
    with tarfile.open(nemo_path) as tar:
        cfg = yaml.safe_load(tar.extractfile("model_config.yaml"))
        wf = tar.extractfile("model_weights.ckpt")
        ckpt = torch.load(io.BytesIO(wf.read()), map_location="cpu", weights_only=False)

    # Parse architecture from config
    jasper = cfg["encoder"]["jasper"]
    n_blocks = len(jasper)
    n_mels = cfg["preprocessor"]["features"]
    n_fft = cfg["preprocessor"]["n_fft"]
    emb_dim = cfg["decoder"]["emb_sizes"]
    channels = jasper[0]["filters"]
    epilog_channels = jasper[-1]["filters"]
    block_kernels = [b["kernel"][0] for b in jasper]
    block_repeats = [b["repeat"] for b in jasper]

    print(f"TitaNet: {n_blocks} blocks, {n_mels} mels, n_fft={n_fft}, emb_dim={emb_dim}")
    print(f"  channels={channels}, epilog_channels={epilog_channels}")
    print(f"  kernels={block_kernels}, repeats={block_repeats}")
    print(f"  {len(ckpt)} weight tensors")

    # Build NeMo → GGUF name mapping
    # NeMo structure per mega block (repeat=R):
    #   mconv.{0+5*s}: depthwise conv   (s = sub-block 0..R-1)
    #   mconv.{1+5*s}: pointwise conv
    #   mconv.{2+5*s}: batch norm
    #   mconv.{3+5*s}: ReLU (no weights)
    #   mconv.{4+5*s}: Dropout (no weights)
    #   mconv.{5*R}:   SE block (only for repeat=1 blocks where no skip from 3,4)
    #     For repeat=1: mconv indices are 0,1,2 + SE at 3
    #     For repeat=3: mconv indices are 0-2, 5-7, 10-12 + SE at 13
    #   res.0.0: residual conv (only for blocks with residual=True)
    #   res.0.1: residual BN

    def map_name(nemo_name):
        """Map NeMo weight name to short GGUF name."""
        parts = nemo_name.split(".")

        if parts[0] == "preprocessor":
            if "fb" in nemo_name:
                return "mel_filterbank"
            if "window" in nemo_name:
                return "hann_window"
            return None

        if parts[0] == "encoder":
            bi = int(parts[2])  # block index
            prefix = f"enc.b{bi}"

            if parts[3] == "mconv":
                layer_idx = int(parts[4])
                R = block_repeats[bi]
                se_idx = 5 * R - 2  # SE index: repeat=1 → 3, repeat=3 → 13

                if layer_idx == se_idx:
                    # SE block
                    fc_idx = parts[6]  # '0' or '2'
                    fc_name = "fc1" if fc_idx == "0" else "fc2"
                    return f"{prefix}.se.{fc_name}.w"

                # Which sub-block?
                sub = layer_idx // 5
                pos = layer_idx % 5  # 0=dw, 1=pw, 2=bn

                if pos == 0:
                    return f"{prefix}.s{sub}.dw.w"
                elif pos == 1:
                    return f"{prefix}.s{sub}.pw.w"
                elif pos == 2:
                    suffix = parts[5]  # weight/bias/running_mean/running_var
                    bn_map = {"weight": "w", "bias": "b", "running_mean": "m", "running_var": "v"}
                    if suffix in bn_map:
                        return f"{prefix}.s{sub}.bn.{bn_map[suffix]}"
                    return None  # num_batches_tracked

            elif parts[3] == "res":
                # res.0.0.conv.weight → residual conv
                # res.0.1.{weight,bias,running_mean,running_var} → residual BN
                sub_idx = int(parts[5])  # 0=conv module, 1=BN module
                if sub_idx == 0:
                    return f"{prefix}.res.conv.w"
                else:
                    suffix = parts[6]
                    bn_map = {"weight": "w", "bias": "b", "running_mean": "m", "running_var": "v"}
                    if suffix in bn_map:
                        return f"{prefix}.res.bn.{bn_map[suffix]}"
                    return None

        if parts[0] == "decoder":
            if "_pooling" in nemo_name:
                if "attention_layer.0.conv_layer" in nemo_name:
                    if "weight" in parts[-1]:
                        return "dec.asp.tdnn.w"
                    return "dec.asp.tdnn.b"
                if "attention_layer.0.bn" in nemo_name:
                    bn_map = {"weight": "w", "bias": "b", "running_mean": "m", "running_var": "v"}
                    if parts[-1] in bn_map:
                        return f"dec.asp.bn.{bn_map[parts[-1]]}"
                    return None
                if "attention_layer.2" in nemo_name:
                    if "weight" in parts[-1]:
                        return "dec.asp.conv.w"
                    return "dec.asp.conv.b"

            if "emb_layers.0.0" in nemo_name:
                bn_map = {"weight": "w", "bias": "b", "running_mean": "m", "running_var": "v"}
                if parts[-1] in bn_map:
                    return f"dec.pool_bn.{bn_map[parts[-1]]}"
                return None

            if "emb_layers.0.1" in nemo_name:
                if "weight" in parts[-1]:
                    return "dec.fc.w"
                return "dec.fc.b"

            if parts[1] == "final":
                return None  # Skip classifier head (not needed for embeddings)

        return None

    # Create GGUF
    writer = gguf.GGUFWriter(args.output, "titanet")
    writer.add_name("TitaNet-Large")

    # Hyperparameters
    writer.add_uint32("titanet.n_mels", n_mels)
    writer.add_uint32("titanet.n_fft", n_fft)
    writer.add_uint32("titanet.channels", channels)
    writer.add_uint32("titanet.epilog_channels", epilog_channels)
    writer.add_uint32("titanet.emb_dim", emb_dim)
    writer.add_uint32("titanet.n_blocks", n_blocks)
    writer.add_array("titanet.block_repeats", block_repeats)
    writer.add_array("titanet.block_kernels", block_kernels)

    # SE reduction channels (detect from first SE layer)
    se_key = [k for k in ckpt if "fc.0.weight" in k and "encoder" in k]
    se_channels = ckpt[se_key[0]].shape[0] if se_key else 128
    writer.add_uint32("titanet.se_channels", se_channels)

    # NeMo's encoder BN uses eps=0.001 (not the PyTorch default 1e-5)
    writer.add_float32("titanet.bn_eps_encoder", 0.001)
    writer.add_float32("titanet.bn_eps_decoder", 1e-5)

    # Write tensors
    tensor_count = 0
    skipped = []
    for name in sorted(ckpt.keys()):
        if "num_batches_tracked" in name:
            continue

        gguf_name = map_name(name)
        if gguf_name is None:
            skipped.append(name)
            continue

        t = ckpt[name].float().numpy()

        # Squeeze kernel=1 conv weights: [C_out, C_in, 1] → [C_out, C_in]
        if len(t.shape) == 3 and t.shape[-1] == 1:
            t = t.squeeze(-1)

        # Reshape mel filterbank: [1, 80, 257] → [80, 257]
        if gguf_name == "mel_filterbank" and len(t.shape) == 3:
            t = t.squeeze(0)

        # Store BN params and biases as F32, conv weights as F16
        if ".bn." in gguf_name or ".b" == gguf_name[-2:] or gguf_name == "mel_filterbank" or gguf_name == "hann_window" or len(t.shape) <= 1:
            data = t.astype(np.float32)
        else:
            data = t.astype(np.float16)

        writer.add_tensor(gguf_name, data)
        tensor_count += 1
        if tensor_count <= 10 or tensor_count % 20 == 0:
            print(f"  [{tensor_count:3d}] {gguf_name:30s} {str(data.shape):20s} {data.dtype}")

    print(f"  ... total: {tensor_count} tensors")
    if skipped:
        print(f"  skipped {len(skipped)}: {', '.join(skipped[:5])}{'...' if len(skipped) > 5 else ''}")

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    file_size = os.path.getsize(args.output)
    print(f"\nDone: {args.output} ({file_size / 1e6:.1f} MB, {tensor_count} tensors)")


if __name__ == "__main__":
    main()
