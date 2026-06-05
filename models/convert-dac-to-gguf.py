#!/usr/bin/env python3
"""
Convert descript/dac_44khz HuggingFace model -> GGUF F16/F32 for the
CrispASR `dac_decoder` core component.

Descript Audio Codec (DAC) is a neural audio codec with 9 residual VQ
codebooks at 44.1 kHz output. Used by Zonos TTS (PLAN #130) and
potentially by Dia (PLAN #136) and other DAC-based models.

DAC decoder architecture (descript/dac_44khz):
  - Quantizer: 9 codebooks, each 1024 entries x 8-dim, with out_proj to
    hidden_size=1024. Strides all 1 (flat RVQ, no multi-rate).
  - Decoder:
    [0] Conv1d(1024, 1536, k=7, p=3) - input conv
    Decoder blocks (4 stages, upsampling_ratios=[8,8,4,2]):
    [1] DecoderBlock(1536 -> 768, stride=8) - 8x upsample
    [2] DecoderBlock(768 -> 384, stride=8) - 8x upsample
    [3] DecoderBlock(384 -> 192, stride=4) - 4x upsample
    [4] DecoderBlock(192 -> 96, stride=2) - 2x upsample
    [5] Snake1d(96)
    [6] Conv1d(96, 1, k=7, p=3) - output conv
    [7] Tanh

  DecoderBlock(in_ch -> out_ch, stride s):
    [0] Snake1d(in_ch)
    [1] ConvTranspose1d(in_ch, out_ch, k=2s, s, p=s/2)
    [2] ResidualUnit(out_ch, dilation=1)
    [3] ResidualUnit(out_ch, dilation=3)
    [4] ResidualUnit(out_ch, dilation=9)

  ResidualUnit(dim, dilation=d):
    y = Snake1d(dim) -> Conv1d(dim, dim, k=7, p=3*d, d=d) ->
        Snake1d(dim) -> Conv1d(dim, dim, k=1) -> y
    return x + y

  Snake1d: y = x + (1/alpha) * sin^2(alpha * x)

Total upsampling: 8*8*4*2 = 512. So ~86 codes/s -> 44100 Hz.

Usage:
    python models/convert-dac-to-gguf.py \\
        --input descript/dac_44khz \\
        --output /mnt/storage/zonos-tts/dac-44khz-f16.gguf
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

try:
    import gguf
    from gguf import GGUFWriter, GGMLQuantizationType
except ImportError:
    sys.exit("pip install gguf")

try:
    import torch
except ImportError:
    sys.exit("pip install torch")

try:
    from huggingface_hub import snapshot_download
except ImportError:
    sys.exit("pip install huggingface_hub")


def load_model_dir(model_id: str) -> Path:
    p = Path(model_id)
    if p.is_dir():
        return p
    print(f"Downloading {model_id}...", file=sys.stderr)
    return Path(snapshot_download(model_id, allow_patterns=[
        "*.safetensors", "*.json", "*.bin", "*.pt",
    ]))


# ---------------------------------------------------------------------------
# Weight name mapping
# ---------------------------------------------------------------------------

def map_dac_tensor_name(hf_name: str) -> str | None:
    """Map HuggingFace DacModel weight names to GGUF names."""
    n = hf_name

    # Skip encoder weights (we only need the decoder + quantizer)
    if n.startswith("encoder.") or n.startswith("model.encoder."):
        return None

    # Remove "model." prefix if present
    if n.startswith("model."):
        n = n[6:]

    # Quantizer codebooks: quantizer.codebooks.N.* -> dac.quant.N.*
    n = n.replace("quantizer.codebooks.", "dac.quant.")

    # Quantizer out_proj: quantizer.out_proj.N.* -> dac.quant_proj.N.*
    # (linear projection from codebook_dim to hidden_size)
    n = n.replace("quantizer.out_proj.", "dac.quant_proj.")

    # Decoder input conv
    n = n.replace("decoder.model.0.", "dac.dec.in_conv.")

    # Decoder blocks
    # decoder.model.{1,2,3,4} -> dac.dec.blk.{0,1,2,3}
    for blk_in, blk_out in [(1, 0), (2, 1), (3, 2), (4, 3)]:
        prefix = f"decoder.model.{blk_in}."
        if n.startswith(prefix):
            rest = n[len(prefix):]
            # Block internal structure:
            # .0 = Snake1d (alpha)
            # .1 = ConvTranspose1d (weight, bias)
            # .2, .3, .4 = ResidualUnit (dilation 1, 3, 9)
            #   ResidualUnit internal:
            #   .block.0 = Snake1d
            #   .block.1 = Conv1d (depthwise or regular)
            #   .block.2 = Snake1d
            #   .block.3 = Conv1d (pointwise)
            n = f"dac.dec.blk.{blk_out}.{rest}"
            break

    # Decoder output: Snake + Conv1d + Tanh
    n = n.replace("decoder.model.5.", "dac.dec.out_snake.")
    n = n.replace("decoder.model.6.", "dac.dec.out_conv.")

    # Clean up common suffixes
    n = n.replace(".weight_g", ".weight_g")  # weight_norm decomposition
    n = n.replace(".weight_v", ".weight_v")

    return n


def main():
    ap = argparse.ArgumentParser(description="Convert DAC 44kHz to GGUF")
    ap.add_argument("--input", default="descript/dac_44khz",
                    help="HF model ID or local path")
    ap.add_argument("--output", required=True,
                    help="Output GGUF file path")
    ap.add_argument("--quant", default="f16",
                    choices=["f32", "f16"],
                    help="Quantization (default: f16). DAC is small; q8 not needed.")
    args = ap.parse_args()

    model_dir = load_model_dir(args.input)

    # Load config
    config_path = model_dir / "config.json"
    if not config_path.exists():
        sys.exit(f"No config.json found in {model_dir}")
    with open(config_path, encoding="utf-8") as f:
        config = json.load(f)

    n_codebooks = config.get("n_codebooks", 9)
    codebook_size = config.get("codebook_size", 1024)
    codebook_dim = config.get("codebook_dim", 8)
    hidden_size = config.get("hidden_size", 1024)
    decoder_hidden = config.get("decoder_hidden_size", 1536)
    sampling_rate = config.get("sampling_rate", 44100)
    hop_length = config.get("hop_length", 512)
    upsampling_ratios = config.get("upsampling_ratios", [8, 8, 4, 2])

    print(f"DAC config:", file=sys.stderr)
    print(f"  n_codebooks={n_codebooks}, codebook_size={codebook_size}, "
          f"codebook_dim={codebook_dim}", file=sys.stderr)
    print(f"  hidden_size={hidden_size}, decoder_hidden={decoder_hidden}",
          file=sys.stderr)
    print(f"  sampling_rate={sampling_rate}, hop_length={hop_length}",
          file=sys.stderr)
    print(f"  upsampling_ratios={upsampling_ratios}", file=sys.stderr)

    # Create GGUF writer
    writer = GGUFWriter(args.output, "dac-44khz")

    # Write metadata
    writer.add_uint32("dac.n_codebooks", n_codebooks)
    writer.add_uint32("dac.codebook_size", codebook_size)
    writer.add_uint32("dac.codebook_dim", codebook_dim)
    writer.add_uint32("dac.hidden_size", hidden_size)
    writer.add_uint32("dac.decoder_hidden_size", decoder_hidden)
    writer.add_uint32("dac.sampling_rate", sampling_rate)
    writer.add_uint32("dac.hop_length", hop_length)
    writer.add_uint32("dac.n_decoder_blocks", len(upsampling_ratios))
    for i, r in enumerate(upsampling_ratios):
        writer.add_uint32(f"dac.upsampling_ratio.{i}", r)

    # Load model weights
    # DAC uses weight_norm, so we need to handle weight_g and weight_v
    # and reconstruct the actual weight: w = weight_g * weight_v / ||weight_v||
    try:
        from transformers import DacModel
        print("Loading DAC model via transformers...", file=sys.stderr)
        model = DacModel.from_pretrained(args.input)
        model.eval()
        state_dict = model.state_dict()
    except ImportError:
        # Fallback: load safetensors directly
        from safetensors import safe_open
        st_files = list(model_dir.glob("*.safetensors"))
        if not st_files:
            sys.exit("No safetensors files found and transformers not available")
        state_dict = {}
        for st_path in st_files:
            with safe_open(st_path, framework="pt") as f:
                for k in f.keys():
                    state_dict[k] = f.get_tensor(k)

    # Write tensors (decoder + quantizer only)
    n_written = 0
    skipped = 0
    for hf_name in sorted(state_dict.keys()):
        gguf_name = map_dac_tensor_name(hf_name)
        if gguf_name is None:
            skipped += 1
            continue

        tensor = state_dict[hf_name]
        if tensor.dtype == torch.bfloat16:
            tensor = tensor.float()
        shape = list(tensor.shape)

        # Choose dtype: 1D -> F32, 2D -> F16 or F32
        if len(shape) == 1:
            qtype = GGMLQuantizationType.F32
            data = np.ascontiguousarray(tensor.float().numpy())
        elif args.quant == "f16":
            qtype = GGMLQuantizationType.F16
            data = np.ascontiguousarray(tensor.float().numpy().astype(np.float16))
        else:
            qtype = GGMLQuantizationType.F32
            data = np.ascontiguousarray(tensor.float().numpy())

        writer.add_tensor(gguf_name, data, raw_dtype=qtype)
        n_written += 1

        if n_written <= 5 or n_written % 50 == 0:
            print(f"  [{n_written}] {hf_name} -> {gguf_name} "
                  f"{shape} {qtype.name}", file=sys.stderr)

    print(f"\nTensors written: {n_written}, skipped (encoder): {skipped}",
          file=sys.stderr)

    print(f"Writing GGUF to {args.output}...", file=sys.stderr)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    import os
    size_mb = os.path.getsize(args.output) / (1024 * 1024)
    print(f"Done. Output: {args.output} ({size_mb:.1f} MB)", file=sys.stderr)


if __name__ == "__main__":
    main()
