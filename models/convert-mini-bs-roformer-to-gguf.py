#!/usr/bin/env python3
"""Convert HiDolen/Mini-BS-RoFormer-V2.5-46.2M to a CrispASR GGUF.

The converter reads safetensors directly; it does not execute remote model
code. The pinned upstream MIT blueprint is used separately by the reference
dumper. Matrix weights use F16 (or diagnostic F32); DeltaRMSNorm parameters,
learned probes/register tokens and exact periodic Hann windows stay F32.
Use crispasr-quantize for Q8_0 and Q4_K after conversion.

    python models/convert-mini-bs-roformer-to-gguf.py \
        --model HiDolen/Mini-BS-RoFormer-V2.5-46.2M \
        --output mini-bs-roformer-f16.gguf

All checkpoint names and matrix layouts are retained verbatim. In particular,
``*.scale`` contains the original DeltaRMSNorm delta: the runtime must use
``gamma = 1 + scale``, not treat it as an ordinary RMSNorm weight.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path

MODEL_ID = "HiDolen/Mini-BS-RoFormer-V2.5-46.2M"
MODEL_REVISION = "38dafc78834607aec57d8c0cb3568523f9ea5563"
ARCH = "mini-bs-roformer"
PREFIX = "mini_bs_roformer"
SOURCES = ("bass", "drums", "other", "vocals")
INT_FIELDS = (
    "hidden_size", "band_proj_size", "intermediate_size", "num_hidden_layers",
    "num_attention_heads", "num_key_value_heads", "head_dim",
    "num_input_channels", "num_stems", "register_token_num", "time_conv_length",
    "time_transformer_depth", "freq_transformer_depth", "stft_n_fft",
    "stft_n_fft_out", "stft_hop_length", "wave_chunk_size", "wave_sample_rate",
)
SUPPORTED_INTS = dict(zip(INT_FIELDS, (
    512, 256, 1024, 6, 8, 4, 64, 2, 4, 4, 4, 1, 1, 4096, 2048, 512, 352800, 44100,
)))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_model(source: str, revision: str) -> tuple[Path, str]:
    local = Path(source).expanduser()
    if local.is_dir():
        marker = local / "source_revision.txt"
        if marker.is_file():
            local_revision = marker.read_text(encoding="utf-8").strip()
        elif local.parent.name == "snapshots" and re.fullmatch(r"[0-9a-f]{40}", local.name):
            local_revision = local.name
        else:
            local_revision = "local-unverified"
        return local, local_revision
    if source != MODEL_ID:
        raise ValueError(f"expected a local snapshot directory or {MODEL_ID}, got {source!r}")
    from huggingface_hub import snapshot_download

    path = Path(snapshot_download(
        repo_id=source,
        revision=revision,
        allow_patterns=["config.json", "model.safetensors", "*.py", "README.md"],
    ))
    return path, path.name


def validate_config(config: dict) -> None:
    if config.get("model_type") != "bs_roformer":
        raise ValueError("expected model_type=bs_roformer")
    for key in INT_FIELDS:
        value = config.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if config["num_input_channels"] != 2 or config["num_stems"] != len(SOURCES):
        raise ValueError("this architecture requires the four-stem stereo V2.5 model")
    if config["hidden_size"] != config["num_attention_heads"] * config["head_dim"]:
        raise ValueError("hidden_size must equal num_attention_heads * head_dim")
    if config["num_attention_heads"] % config["num_key_value_heads"]:
        raise ValueError("query heads must be divisible by key/value heads")
    if config["head_dim"] % 2:
        raise ValueError("split-half RoPE requires an even head_dim")
    for key in ("rms_norm_eps", "rope_base"):
        value = config.get(key)
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{key} must be finite and positive")
    for key, fft_key in (("freqs_per_bands", "stft_n_fft"),
                         ("freqs_per_bands_out", "stft_n_fft_out")):
        bands = config.get(key)
        if not isinstance(bands, list) or not bands or any(
            isinstance(n, bool) or not isinstance(n, int) or n <= 0 for n in bands
        ):
            raise ValueError(f"{key} must be a nonempty positive integer array")
        if sum(bands) != config[fft_key] // 2 + 1:
            raise ValueError(f"{key} does not cover the complete real FFT")
    if len(config["freqs_per_bands"]) != len(config["freqs_per_bands_out"]):
        raise ValueError("encoder and decoder must use the same number of bands")
    for key, value in SUPPORTED_INTS.items():
        if config[key] != value:
            raise ValueError(f"unsupported V2.5 configuration: {key} must be {value}")
    if len(config["freqs_per_bands"]) != 80:
        raise ValueError("the supported V2.5 model requires 80 bands")
    if config["rms_norm_eps"] != 1e-6 or config["rope_base"] != 10000.0:
        raise ValueError("the supported V2.5 model requires rms_norm_eps=1e-6 and rope_base=10000")


def expected_shapes(config: dict) -> dict[str, tuple[int, ...]]:
    """Reject incomplete/wrong-version checkpoints before producing a GGUF."""
    d = config["hidden_size"]
    p = config["band_proj_size"]
    ff = config["intermediate_size"]
    c = config["num_input_channels"]
    tc = config["time_conv_length"]
    q = config["num_attention_heads"] * config["head_dim"]
    kv = config["num_key_value_heads"] * config["head_dim"]
    rn = config["register_token_num"]
    shapes: dict[str, tuple[int, ...]] = {}
    root = "freq_domain_model."
    mask = root + "mask_model."

    def mlp(name: str, din: int, mid: int, dout: int) -> None:
        shapes[name + "gate_proj.weight"] = (mid, din)
        shapes[name + "up_proj.weight"] = (mid, din)
        shapes[name + "down_proj.weight"] = (dout, mid)

    def attention(name: str) -> None:
        shapes[name + "q_proj.weight"] = (q, d)
        shapes[name + "k_proj.weight"] = (kv, d)
        shapes[name + "v_proj.weight"] = (kv, d)
        shapes[name + "o_proj.weight"] = (d, q)

    for i, (nin, nout) in enumerate(zip(config["freqs_per_bands"], config["freqs_per_bands_out"])):
        shapes[f"{mask}band_split.to_features.{i}.0.scale"] = (2 * c * nin,)
        shapes[f"{mask}band_split.to_features.{i}.1.weight"] = (p, 2 * c * nin)
        shapes[f"{mask}mask_estimator.to_freq_mlps.{i}.weight"] = (2 * c * nout, p)
        shapes[f"{mask}mask_estimator.to_gate_mlps.{i}.weight"] = (c * nout, p)
    shapes[mask + "time_conv_norm.scale"] = (p,)
    shapes[mask + "time_deconv_norm.scale"] = (p,)
    mlp(mask + "time_conv.", p * tc, d * tc, d)
    mlp(mask + "time_deconv.", d, d * tc, p * tc)
    shapes[mask + "encoder.register_tokens"] = (rn, rn, d)
    shapes[mask + "encoder.final_norm.scale"] = (d,)
    for i in range(config["num_hidden_layers"]):
        for axis, depth_key in enumerate(("time_transformer_depth", "freq_transformer_depth")):
            for j in range(config[depth_key]):
                name = f"{mask}encoder.layers.{i}.{axis}.layers.{j}."
                shapes[name + "input_layernorm.scale"] = (d,)
                shapes[name + "post_attention_layernorm.scale"] = (d,)
                attention(name + "self_attn.")
                mlp(name + "mlp.", d, ff, d)
    for name, probes in (("freq_pooling", 1), ("time_pooling", config["num_stems"])):
        base = root + name + "."
        shapes[base + "probe"] = (probes, d)
        shapes[base + "mlp_layernorm.scale"] = (d,)
        attention(base + "attn.")
        mlp(base + "mlp.", d, ff, d)
    return shapes


def keep_f32(name: str) -> bool:
    return name.endswith((".scale", ".probe", ".register_tokens"))


def convert(args: argparse.Namespace) -> None:
    import numpy as np
    import torch
    from gguf import GGMLQuantizationType, GGUFWriter
    from safetensors import safe_open

    model_dir, revision = resolve_model(args.model, args.revision)
    config_path = model_dir / "config.json"
    weights_path = model_dir / "model.safetensors"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    validate_config(config)
    expected = expected_shapes(config)
    with safe_open(str(weights_path), framework="pt", device="cpu") as checkpoint:
        actual = set(checkpoint.keys())
        if actual != set(expected):
            raise ValueError(f"checkpoint keys differ: missing={sorted(set(expected) - actual)}, "
                             f"unexpected={sorted(actual - set(expected))}")
        for name, shape in expected.items():
            if tuple(checkpoint.get_slice(name).get_shape()) != shape:
                raise ValueError(f"{name}: expected shape {shape}, "
                                 f"got {checkpoint.get_slice(name).get_shape()}")

        args.output.parent.mkdir(parents=True, exist_ok=True)
        writer = GGUFWriter(str(args.output), ARCH, use_temp_file=True)
        writer.add_name("Mini-BS-RoFormer-V2.5-46.2M")
        writer.add_description("HiDolen four-stem stereo band-split RoFormer V2.5")
        writer.add_string("general.license", "mit")
        writer.add_string("general.source.huggingface.repository", MODEL_ID)
        writer.add_string("general.source.revision", revision)
        writer.add_string("general.source.model_sha256", sha256(weights_path))
        writer.add_string("general.source.config_sha256", sha256(config_path))
        for source in ("modeling_bs_roformer.py", "configuration_bs_roformer.py"):
            path = model_dir / source
            if path.is_file():
                writer.add_string(f"general.source.{path.stem}_sha256", sha256(path))
        writer.add_file_type(GGMLQuantizationType.F16 if args.dtype == "f16" else GGMLQuantizationType.F32)
        for key in INT_FIELDS:
            writer.add_uint32(f"{PREFIX}.{key}", config[key])
        writer.add_uint32(f"{PREFIX}.num_bands", len(config["freqs_per_bands"]))
        for key in ("rms_norm_eps", "rope_base"):
            writer.add_float32(f"{PREFIX}.{key}", config[key])
        for key in ("freqs_per_bands", "freqs_per_bands_out"):
            writer.add_array(f"{PREFIX}.{key}", config[key])
        writer.add_string(f"{PREFIX}.sources_json", json.dumps(SOURCES))
        writer.add_string(f"{PREFIX}.norm_type", "delta_rms_norm")
        writer.add_string(f"{PREFIX}.transformers_version", config.get("transformers_version", ""))

        for name in sorted(expected):
            array = checkpoint.get_tensor(name).float().numpy()
            if not np.isfinite(array).all():
                raise ValueError(f"nonfinite checkpoint tensor: {name}")
            dtype = np.float32 if args.dtype == "f32" or keep_f32(name) else np.float16
            if dtype == np.float16 and np.max(np.abs(array)) > np.finfo(np.float16).max:
                raise ValueError(f"F16 overflow in {name}; use --dtype f32")
            writer.add_tensor(name, np.ascontiguousarray(array, dtype=dtype))
        # The upstream creates these non-persistent buffers with torch, so save
        # the same coefficients rather than recomputing an almost-equal window.
        writer.add_tensor("aux.stft_window", torch.hann_window(config["stft_n_fft"]).numpy())
        writer.add_tensor("aux.stft_out_window", torch.hann_window(config["stft_n_fft_out"]).numpy())
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()
    count = sum(int(np.prod(shape)) for shape in expected.values())
    print(f"Wrote {args.output}: {len(expected)} learned tensors, {count:,} parameters, "
          f"{args.dtype} matrices and F32 norms/probes/registers/windows")
    print(f"Source: {MODEL_ID}@{revision}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=MODEL_ID, help="local snapshot directory or pinned model ID")
    parser.add_argument("--revision", default=MODEL_REVISION, help="revision for remote downloads")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dtype", choices=("f16", "f32"), default="f16")
    args = parser.parse_args()
    try:
        convert(args)
    except (ValueError, FileNotFoundError) as error:
        parser.exit(1, f"error: {error}\n")


if __name__ == "__main__":
    main()
