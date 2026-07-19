#!/usr/bin/env python3
"""Convert SaruLab Sidon v0.1 predictor + continuous DAC decoder to GGUF.

The converter consumes the locally downloaded Hugging Face repositories:

  * facebook/w2v-bert-2.0 (base model.safetensors + config)
  * sarulab-speech/sidon_raw_weight (LoRA adapter + decoder_state_dict.pt)

Only the first eight w2v-BERT layers used by Sidon are written.  The LoRA
deltas on ffn{1,2}.output_dense are merged into the base weights.  Decoder
weight-normalisation is removed so the native runtime only loads dense
convolution kernels.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

try:
    import gguf
except ImportError:
    sys.exit("missing Python package 'gguf'; install the converter dependencies and retry")

from transformers import SeamlessM4TFeatureExtractor


N_LAYERS = 8
RATES = [8, 5, 4, 3, 2]


def f32(t: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(t, torch.Tensor):
        t = t.detach().float().cpu().numpy()
    return np.ascontiguousarray(t, dtype=np.float32)


def f16(t: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(t, torch.Tensor):
        t = t.detach().float().cpu().numpy()
    return np.ascontiguousarray(t, dtype=np.float16)


def merged_weight_norm(g: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Materialise torch.nn.utils.weight_norm(..., dim=0)."""
    vf = v.float()
    reduce_dims = tuple(range(1, vf.ndim))
    norm = torch.linalg.vector_norm(vf, dim=reduce_dims, keepdim=True)
    return vf * (g.float() / norm.clamp_min(1e-12))


def output_name(base_name: str) -> str:
    if base_name.startswith("feature_projection."):
        return "predictor." + base_name
    if base_name.startswith("encoder.layers."):
        return "predictor." + base_name
    raise ValueError(f"unexpected predictor tensor: {base_name}")


def decoder_name(name: str) -> str:
    return "decoder." + name


def main() -> None:
    ap = argparse.ArgumentParser(description="Convert Sidon v0.1 to GGUF")
    ap.add_argument("--base", type=Path, required=True, help="local facebook/w2v-bert-2.0 directory")
    ap.add_argument("--sidon", type=Path, required=True, help="local sarulab-speech/sidon_raw_weight directory")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--dtype", choices=["f16", "f32"], default="f16")
    args = ap.parse_args()

    base_weights = args.base / "model.safetensors"
    adapter_weights = args.sidon / "adapter_model.safetensors"
    adapter_config = args.sidon / "adapter_config.json"
    decoder_weights = args.sidon / "decoder_state_dict.pt"
    for p in (
        base_weights,
        adapter_weights,
        adapter_config,
        decoder_weights,
        args.base / "config.json",
        args.base / "preprocessor_config.json",
    ):
        if not p.is_file():
            sys.exit(f"missing required file: {p}")

    with open(args.base / "config.json", encoding="utf-8") as f:
        cfg = json.load(f)
    with open(adapter_config, encoding="utf-8") as f:
        adapter_cfg = json.load(f)
    if (
        adapter_cfg.get("peft_type") != "LORA"
        or adapter_cfg.get("fan_in_fan_out", False)
        or adapter_cfg.get("use_dora", False)
        or adapter_cfg.get("rank_pattern")
        or adapter_cfg.get("alpha_pattern")
    ):
        raise ValueError("unsupported Sidon LoRA configuration")
    lora_rank = int(adapter_cfg["r"])
    lora_alpha = float(adapter_cfg["lora_alpha"])
    if lora_rank <= 0 or not np.isfinite(lora_alpha):
        raise ValueError("invalid Sidon LoRA rank or alpha")
    lora_scale = lora_alpha / (np.sqrt(lora_rank) if adapter_cfg.get("use_rslora", False) else lora_rank)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = gguf.GGUFWriter(str(args.output), "sidon")
    writer.add_string("general.name", "Sidon v0.1")
    writer.add_file_type(
        gguf.GGMLQuantizationType.F16 if args.dtype == "f16" else gguf.GGMLQuantizationType.F32
    )
    writer.add_uint32("sidon.predictor.layers", N_LAYERS)
    writer.add_uint32("sidon.hidden_size", int(cfg["hidden_size"]))
    writer.add_uint32("sidon.intermediate_size", int(cfg["intermediate_size"]))
    writer.add_uint32("sidon.attention_heads", int(cfg["num_attention_heads"]))
    writer.add_uint32("sidon.conv_kernel", int(cfg["conv_depthwise_kernel_size"]))
    writer.add_uint32("sidon.relative_left", int(cfg["left_max_position_embeddings"]))
    writer.add_uint32("sidon.relative_right", int(cfg["right_max_position_embeddings"]))
    writer.add_float32("sidon.layer_norm_eps", float(cfg["layer_norm_eps"]))
    writer.add_uint32("sidon.input_sample_rate", 16000)
    writer.add_uint32("sidon.output_sample_rate", 48000)
    writer.add_uint32("sidon.feature_dim", 160)
    writer.add_uint32("sidon.mel_bins", 80)
    writer.add_uint32("sidon.decoder_blocks", len(RATES))
    writer.add_uint32("sidon.decoder_hop", int(np.prod(RATES)))
    for i, rate in enumerate(RATES):
        writer.add_uint32(f"sidon.decoder_rate.{i}", rate)

    # Store the exact Transformers frontend constants.  This removes a major
    # source of cross-runtime drift (window/mel-scale convention).
    frontend = SeamlessM4TFeatureExtractor.from_pretrained(str(args.base), local_files_only=True)
    writer.add_tensor("frontend.window", f32(frontend.window))
    writer.add_tensor("frontend.mel_filters", f32(frontend.mel_filters))

    matrix_cast = f16 if args.dtype == "f16" else f32
    count = 2

    with safe_open(str(adapter_weights), framework="pt", device="cpu") as af:
        adapter = {k: af.get_tensor(k) for k in af.keys()}

    merged_adapter_keys: set[str] = set()
    with safe_open(str(base_weights), framework="pt", device="cpu") as bf:
        for name in bf.keys():
            if name.startswith("encoder.layers."):
                layer = int(name.split(".")[2])
                if layer >= N_LAYERS:
                    continue
            elif not name.startswith("feature_projection."):
                continue

            tensor = bf.get_tensor(name)
            if name.endswith(".output_dense.weight"):
                apfx = "base_model.model." + name[: -len(".weight")]
                a_name = apfx + ".lora_A.weight"
                b_name = apfx + ".lora_B.weight"
                if (a_name in adapter) != (b_name in adapter):
                    raise KeyError(f"incomplete LoRA pair for {name}")
                if a_name in adapter:
                    tensor = tensor.float() + lora_scale * (adapter[b_name].float() @ adapter[a_name].float())
                    merged_adapter_keys.update((a_name, b_name))

            data = matrix_cast(tensor) if tensor.ndim >= 2 else f32(tensor)
            writer.add_tensor(output_name(name), data)
            count += 1

    unused_adapter_keys = set(adapter) - merged_adapter_keys
    if unused_adapter_keys or len(merged_adapter_keys) != 4 * N_LAYERS:
        raise ValueError(
            f"unexpected Sidon LoRA layout: merged {len(merged_adapter_keys)} tensors, "
            f"unused={sorted(unused_adapter_keys)}"
        )
    del adapter

    decoder = torch.load(decoder_weights, map_location="cpu", weights_only=True)
    for name in decoder:
        if name.endswith(".weight_g") and name[: -len("weight_g")] + "weight_v" not in decoder:
            raise KeyError(f"missing weight_v pair for {name}")
    consumed: set[str] = set()
    for name, tensor in decoder.items():
        if name in consumed or name.endswith(".weight_g"):
            continue
        if name.endswith(".weight_v"):
            stem = name[: -len("weight_v")]
            g_name = stem + "weight_g"
            if g_name not in decoder:
                raise KeyError(f"missing {g_name} for {name}")
            tensor = merged_weight_norm(decoder[g_name], tensor)
            name = stem + "weight"
            consumed.add(g_name)
        data = matrix_cast(tensor) if tensor.ndim >= 2 and not name.endswith(".alpha") else f32(tensor)
        writer.add_tensor(decoder_name(name), data)
        count += 1

    print(f"writing {count} tensors to {args.output}", file=sys.stderr)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"done: {args.output} ({args.output.stat().st_size / 2**20:.1f} MiB)", file=sys.stderr)


if __name__ == "__main__":
    main()
