#!/usr/bin/env python3
# convert-miocodec-to-gguf.py — MioCodec v2 (44.1kHz wave decoder) to GGUF
#
# Converts the MioCodec-25Hz-44.1kHz-v2 audio codec model to GGUF format.
# This is the DECODER-ONLY GGUF — the WavLM encoder is separate (too large
# to bundle, and frozen/shared across models).
#
# Usage:
#   OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1 \
#   TMPDIR=/mnt/volume1/tmp-overflow \
#   python models/convert-miocodec-to-gguf.py \
#       --model Aratako/MioCodec-25Hz-44.1kHz-v2 \
#       --output miocodec-v2-44k-f16.gguf --dtype f16

import argparse
import json
import os
import struct
import sys

import numpy as np

# Lazy per-tensor loading via safetensors
from safetensors import safe_open

# gguf writer
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ggml", "scripts"))
try:
    from gguf import GGUFWriter, GGMLQuantizationType
except ImportError:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from ggml.scripts.gguf import GGUFWriter, GGMLQuantizationType


def resolve_model_path(model_id: str) -> tuple:
    """Resolve HF repo or local path to safetensors + config paths."""
    if os.path.isdir(model_id):
        st = os.path.join(model_id, "model.safetensors")
        cfg = os.path.join(model_id, "config.yaml")
        return st, cfg
    # Download from HF
    from huggingface_hub import hf_hub_download
    st = hf_hub_download(model_id, "model.safetensors")
    cfg = hf_hub_download(model_id, "config.yaml")
    return st, cfg


def parse_config(config_path: str) -> dict:
    """Parse the config.yaml and extract model hyperparameters."""
    import yaml
    with open(config_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    init = raw["model"]["init_args"]
    cfg = init["config"]

    # Extract transformer configs
    local_enc = init["local_encoder"]["init_args"]
    wave_pre = init["wave_prenet"]["init_args"]
    wave_dec = init["wave_decoder"]["init_args"]
    fsq = init["local_quantizer"]["init_args"]
    glob = init["global_encoder"]["init_args"]

    return {
        # Audio
        "sample_rate": cfg["sample_rate"],
        "n_fft": cfg["n_fft"],
        "hop_length": cfg["hop_length"],
        # FSQ
        "fsq_levels": fsq["levels"],
        "fsq_input_dim": fsq["input_dim"],
        "fsq_output_dim": fsq["output_dim"],
        # Local encoder
        "local_enc_dim": local_enc["dim"],
        "local_enc_n_layers": local_enc["n_layers"],
        "local_enc_n_heads": local_enc["n_heads"],
        "local_enc_window_size": local_enc.get("window_size", 0) or 0,
        "local_enc_rope_theta": local_enc.get("rope_theta", 10000.0),
        # Wave prenet
        "wave_prenet_dim": wave_pre["dim"],
        "wave_prenet_output_dim": wave_pre.get("output_dim", wave_pre["dim"]),
        "wave_prenet_n_layers": wave_pre["n_layers"],
        "wave_prenet_n_heads": wave_pre["n_heads"],
        "wave_prenet_window_size": wave_pre.get("window_size", 0) or 0,
        "wave_prenet_rope_theta": wave_pre.get("rope_theta", 10000.0),
        # Wave decoder (AdaLN-Zero)
        "wave_dec_dim": wave_dec["dim"],
        "wave_dec_n_layers": wave_dec["n_layers"],
        "wave_dec_n_heads": wave_dec["n_heads"],
        "wave_dec_window_size": wave_dec.get("window_size", 0) or 0,
        "wave_dec_rope_theta": wave_dec.get("rope_theta", 10000.0),
        "wave_dec_adaln_cond_dim": wave_dec.get("adanorm_condition_dim", 128),
        # Wave decoder misc
        "wave_decoder_dim": cfg["wave_decoder_dim"],
        "wave_upsample_factor": cfg["wave_upsample_factor"],
        "wave_resnet_num_blocks": cfg["wave_resnet_num_blocks"],
        "wave_resnet_kernel_size": cfg["wave_resnet_kernel_size"],
        "wave_resnet_num_groups": cfg["wave_resnet_num_groups"],
        "wave_upsampler_factors": cfg.get("wave_upsampler_factors", []),
        "wave_upsampler_kernel_sizes": cfg.get("wave_upsampler_kernel_sizes", []),
        "downsample_factor": cfg["downsample_factor"],
        # Global encoder
        "global_enc_input_channels": glob["input_channels"],
        "global_enc_output_channels": glob["output_channels"],
        "global_enc_num_layers": glob["num_layers"],
        "global_enc_dim": glob["dim"],
        "global_enc_intermediate_dim": glob["intermediate_dim"],
        # SSL
        "local_ssl_layers": cfg["local_ssl_layers"],
        "global_ssl_layers": cfg["global_ssl_layers"],
    }


def main():
    parser = argparse.ArgumentParser(description="Convert MioCodec to GGUF")
    parser.add_argument("--model", required=True, help="HF repo ID or local path")
    parser.add_argument("--output", required=True, help="Output GGUF path")
    parser.add_argument("--dtype", default="f16", choices=["f16", "f32"], help="Weight dtype")
    args = parser.parse_args()

    st_path, cfg_path = resolve_model_path(args.model)
    hparams = parse_config(cfg_path)

    print(f"Model: {args.model}")
    print(f"Sample rate: {hparams['sample_rate']}")
    print(f"FSQ levels: {hparams['fsq_levels']} (codebook size: {np.prod(hparams['fsq_levels'])})")
    print(f"Local encoder: {hparams['local_enc_n_layers']}L, {hparams['local_enc_dim']}d")
    print(f"Wave prenet: {hparams['wave_prenet_n_layers']}L, {hparams['wave_prenet_dim']}d→{hparams['wave_prenet_output_dim']}d")
    print(f"Wave decoder: {hparams['wave_dec_n_layers']}L, {hparams['wave_dec_dim']}d (AdaLN cond={hparams['wave_dec_adaln_cond_dim']})")
    print(f"Output: {args.output} ({args.dtype})")

    gguf_dtype = GGMLQuantizationType.F16 if args.dtype == "f16" else GGMLQuantizationType.F32

    writer = GGUFWriter(args.output, "miocodec", use_temp_file=True)

    # Write hyperparameters
    writer.add_uint32("miocodec.sample_rate", hparams["sample_rate"])
    writer.add_uint32("miocodec.n_fft", hparams["n_fft"])
    writer.add_uint32("miocodec.hop_length", hparams["hop_length"])
    writer.add_uint32("miocodec.downsample_factor", hparams["downsample_factor"])

    # FSQ
    writer.add_array("miocodec.fsq.levels", hparams["fsq_levels"])
    writer.add_uint32("miocodec.fsq.input_dim", hparams["fsq_input_dim"])
    writer.add_uint32("miocodec.fsq.output_dim", hparams["fsq_output_dim"])

    # Local encoder
    writer.add_uint32("miocodec.local_enc.n_layers", hparams["local_enc_n_layers"])
    writer.add_uint32("miocodec.local_enc.dim", hparams["local_enc_dim"])
    writer.add_uint32("miocodec.local_enc.n_heads", hparams["local_enc_n_heads"])
    writer.add_uint32("miocodec.local_enc.window_size", hparams["local_enc_window_size"])
    writer.add_float32("miocodec.local_enc.rope_theta", hparams["local_enc_rope_theta"])

    # Wave prenet
    writer.add_uint32("miocodec.wave_prenet.n_layers", hparams["wave_prenet_n_layers"])
    writer.add_uint32("miocodec.wave_prenet.dim", hparams["wave_prenet_dim"])
    writer.add_uint32("miocodec.wave_prenet.output_dim", hparams["wave_prenet_output_dim"])
    writer.add_uint32("miocodec.wave_prenet.n_heads", hparams["wave_prenet_n_heads"])
    writer.add_uint32("miocodec.wave_prenet.window_size", hparams["wave_prenet_window_size"])
    writer.add_float32("miocodec.wave_prenet.rope_theta", hparams["wave_prenet_rope_theta"])

    # Wave decoder
    writer.add_uint32("miocodec.wave_dec.n_layers", hparams["wave_dec_n_layers"])
    writer.add_uint32("miocodec.wave_dec.dim", hparams["wave_dec_dim"])
    writer.add_uint32("miocodec.wave_dec.n_heads", hparams["wave_dec_n_heads"])
    writer.add_uint32("miocodec.wave_dec.window_size", hparams["wave_dec_window_size"])
    writer.add_float32("miocodec.wave_dec.rope_theta", hparams["wave_dec_rope_theta"])
    writer.add_uint32("miocodec.wave_dec.adaln_cond_dim", hparams["wave_dec_adaln_cond_dim"])

    # Wave upsampler
    writer.add_uint32("miocodec.wave_upsample_factor", hparams["wave_upsample_factor"])
    writer.add_uint32("miocodec.wave_resnet_num_blocks", hparams["wave_resnet_num_blocks"])
    if hparams["wave_upsampler_factors"]:
        writer.add_array("miocodec.wave_upsampler.factors", hparams["wave_upsampler_factors"])
        writer.add_array("miocodec.wave_upsampler.kernel_sizes", hparams["wave_upsampler_kernel_sizes"])

    # Global encoder
    writer.add_uint32("miocodec.global_enc.output_channels", hparams["global_enc_output_channels"])
    writer.add_uint32("miocodec.global_enc.num_layers", hparams["global_enc_num_layers"])
    writer.add_uint32("miocodec.global_enc.dim", hparams["global_enc_dim"])
    writer.add_uint32("miocodec.global_enc.intermediate_dim", hparams["global_enc_intermediate_dim"])

    # SSL layer indices
    writer.add_array("miocodec.local_ssl_layers", hparams["local_ssl_layers"])
    writer.add_array("miocodec.global_ssl_layers", hparams["global_ssl_layers"])

    # Write tensors
    print(f"\nConverting tensors...")
    sf = safe_open(st_path, framework="pt")
    n_tensors = 0
    total_bytes = 0

    # Norms and biases always F32; everything else uses requested dtype
    def should_keep_f32(name: str) -> bool:
        if name.endswith(".bias"):
            return True
        if "norm" in name and "weight" in name:
            return True
        if "gamma" in name:
            return True
        if "alpha" in name or "beta" in name:  # SnakeBeta params
            return True
        return False

    for name in sorted(sf.keys()):
        tensor = sf.get_tensor(name).float().numpy()

        if should_keep_f32(name) or args.dtype == "f32":
            data = tensor.astype(np.float32)
        else:
            data = tensor.astype(np.float16)

        writer.add_tensor(name, data)
        n_tensors += 1
        total_bytes += tensor.nbytes
        if n_tensors % 50 == 0:
            print(f"  {n_tensors} tensors written ({total_bytes / 1024 / 1024:.1f} MB F32)...")

    print(f"\nTotal: {n_tensors} tensors, {total_bytes / 1024 / 1024:.1f} MB (F32)")
    print(f"Writing GGUF...")
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    out_size = os.path.getsize(args.output)
    print(f"Done: {args.output} ({out_size / 1024 / 1024:.1f} MB)")


if __name__ == "__main__":
    main()
