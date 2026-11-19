#!/usr/bin/env python3
"""Convert Inworld Voice Isolator v1 (Apache-2.0) to a CrispASR GGUF.

Upstream: https://huggingface.co/inworld/voice-isolator (weights) and
https://github.com/inworld-ai/voice-isolator (inference code). Two checkpoints
share one band-split dual-path architecture:

  denoiser.safetensors           voice-isolator-v1           offline, two-sided
  denoiser_realtime.safetensors  voice-isolator-v1-realtime  causal, look-ahead

The architecture is read from the safetensors ``config`` metadata exactly as
the upstream ``read_checkpoint`` does; nothing is guessed. The converter reads
safetensors directly and never executes upstream code.

    python models/convert-voice-isolator-to-gguf.py \
        --input denoiser.safetensors --output voice-isolator-f16.gguf

Tensor names are kept verbatim. Matrix weights become F16 (``--dtype f32``
keeps everything F32 for parity work); norms, biases, Mamba ``A_log``/``D``,
depthwise convolutions and embeddings stay F32. Conv1d weights
``[inner, 1, k]`` are stored as ``[inner, k]`` (ggml ``ssm_conv`` layout).
Band edges are computed with upstream ``band_edges`` and stored, so the
runtime never re-derives the rounding.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path

import numpy as np

ARCH = "voice-isolator"
PREFIX = "voice_isolator"
UPSTREAM_REPO = "inworld/voice-isolator"
UPSTREAM_REVISION = "9b0285b3fe8403aa8e2f1dbb2165a80a4294c970"
UPSTREAM_CODE = "inworld-ai/voice-isolator@c73b3fab81a870410cbac50d2f05f25edc4d10aa"
KNOWN_SHA256 = {
    "647335a6aeb4dc706b895532c209d51e0e84a4849d17b339f00d4392a050ae53": "voice-isolator-v1",
    "76598f05523c0669e76562a8b2f31c9d395fca316e80c907ee11775e7497ae0f": "voice-isolator-v1-realtime",
}
# Fields this runtime implements; any other non-default field is refused, the
# same policy as upstream VoiceIsolatorConfig.from_dict.
SUPPORTED = {
    "n_fft": 2048, "hop_length": 480, "sample_rate": 48000, "num_heads": 8,
    "mamba_state": 16, "mamba_conv": 4, "mamba_expand": 2,
    "exclusive_self_attention": True, "exclusive_axes": "time",
    "dual_mode": False, "mode_film": False, "correction_bound": 0.5,
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_safetensors(path: Path):
    with path.open("rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
        base = 8 + n
    meta = header.pop("__metadata__", {})
    data = np.memmap(path, dtype=np.uint8, mode="r")
    tensors = {}
    for name, info in header.items():
        if info["dtype"] != "F32":
            raise SystemExit(f"{name}: unexpected dtype {info['dtype']}")
        a, b = info["data_offsets"]
        tensors[name] = np.frombuffer(data[base + a:base + b], dtype=np.float32).reshape(info["shape"])
    return meta, tensors


def band_edges(cfg: dict) -> list[tuple[int, int]]:
    """Upstream model.band_edges, verbatim."""
    n_bins = cfg["n_fft"] // 2 + 1
    hz_per_bin = cfg["sample_rate"] / cfg["n_fft"]
    edges, start = [], 0
    for width_hz, upto_hz in cfg["band_plan"]:
        width = max(1, round(width_hz / hz_per_bin))
        upto = min(n_bins, round(upto_hz / hz_per_bin))
        while start < upto:
            end = min(start + width, upto)
            edges.append((start, end))
            start = end
    if start < n_bins:
        edges.append((start, n_bins))
    return edges


def keep_f32(name: str, array: np.ndarray) -> bool:
    if array.ndim != 2 or name.endswith(".bias"):
        return True
    return any(s in name for s in ("A_log", "conv1d", "band_embedding", "lookahead_embedding", "condition", "norm"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, required=True, help="denoiser.safetensors or denoiser_realtime.safetensors")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--dtype", choices=("f16", "f32"), default="f16")
    args = ap.parse_args()

    digest = sha256(args.input)
    variant = KNOWN_SHA256.get(digest)
    if variant is None:
        print(f"warning: {args.input} sha256 {digest} is not a pinned release checkpoint")
        variant = "unknown"
    meta, tensors = read_safetensors(args.input)
    if "config" not in meta:
        raise SystemExit("safetensors metadata lacks the 'config' entry")
    cfg = json.loads(meta["config"])
    for key, value in SUPPORTED.items():
        if cfg.get(key, value) != value:
            raise SystemExit(f"unsupported {key}={cfg.get(key)!r} (runtime implements {value!r})")
    if any(t not in ("transformer", "mamba") for t in cfg["block_types"]):
        raise SystemExit("unsupported block type")
    if cfg.get("offline_passes", 1) != 1:
        raise SystemExit("offline_passes != 1 is not implemented")
    edges = band_edges(cfg)
    if len(edges) != tensors["band_embedding"].shape[0]:
        raise SystemExit("band plan does not match band_embedding")
    for i, (s, e) in enumerate(edges):
        if tensors[f"encoders.{i}.1.weight"].shape != (cfg["width"], 2 * (e - s)):
            raise SystemExit(f"encoder {i} shape mismatch")
    causal = bool(cfg.get("causal", False))
    choices = list(cfg.get("lookahead_frames_choices", []))

    # imported after validation so the refusal checks need no gguf install
    import gguf  # noqa: PLC0415

    w = gguf.GGUFWriter(str(args.output), ARCH)
    w.add_name("Inworld Voice Isolator v1" + (" realtime" if causal else ""))
    w.add_description("48 kHz band-split dual-path (attention + Mamba) voice isolation")
    w.add_string("general.license", "apache-2.0")
    w.add_string("general.source.huggingface.repository", UPSTREAM_REPO)
    w.add_string("general.source.revision", UPSTREAM_REVISION)
    w.add_string("general.source.code", UPSTREAM_CODE)
    w.add_string("general.source.model_sha256", digest)
    w.add_string(f"{PREFIX}.variant", variant)
    w.add_string(f"{PREFIX}.config_json", meta["config"])
    w.add_file_type(gguf.GGMLQuantizationType.F16 if args.dtype == "f16" else gguf.GGMLQuantizationType.F32)
    for key in ("width", "num_heads", "n_fft", "hop_length", "sample_rate", "mamba_state", "mamba_conv",
                "mamba_expand"):
        w.add_uint32(f"{PREFIX}.{key}", int(cfg[key]))
    w.add_uint32(f"{PREFIX}.n_blocks", len(cfg["block_types"]))
    w.add_array(f"{PREFIX}.block_is_mamba", [1 if t == "mamba" else 0 for t in cfg["block_types"]])
    w.add_uint32(f"{PREFIX}.n_bands", len(edges))
    w.add_array(f"{PREFIX}.band_edges", [x for pair in edges for x in pair])
    w.add_float32(f"{PREFIX}.correction_bound", float(cfg["correction_bound"]))
    w.add_float32(f"{PREFIX}.chunk_seconds", float(cfg["chunk_seconds"]))
    w.add_float32(f"{PREFIX}.overlap_seconds", float(cfg["overlap_seconds"]))
    w.add_bool(f"{PREFIX}.causal", causal)
    w.add_uint32(f"{PREFIX}.lookahead_frames", int(cfg.get("lookahead_frames", 0)))
    w.add_array(f"{PREFIX}.lookahead_choices", choices if choices else [0])
    w.add_uint32(f"{PREFIX}.n_lookahead_choices", len(choices))
    w.add_uint32(f"{PREFIX}.attention_window_frames", int(cfg.get("attention_window_frames", 0)))
    w.add_string(f"{PREFIX}.sources_json", json.dumps(["voice", "background"]))

    for name in sorted(tensors):
        a = tensors[name]
        if "conv1d.weight" in name:
            a = a.reshape(a.shape[0], a.shape[-1])
        if args.dtype == "f16" and not keep_f32(name, a):
            a = a.astype(np.float16)
        else:
            a = a.astype(np.float32)
        w.add_tensor(name, np.ascontiguousarray(a))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"wrote {args.output} ({variant}, {len(tensors)} tensors, {len(edges)} bands, causal={causal})")


if __name__ == "__main__":
    main()
