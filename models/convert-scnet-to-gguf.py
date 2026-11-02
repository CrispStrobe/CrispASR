#!/usr/bin/env python3
"""Convert an author's SCNet or SCNet-large best_state checkpoint to GGUF.

The source checkout must match the pinned official implementation. Download
the exact author-linked checkpoint separately and record its SHA-256 first;
this tool never downloads an unverified substitute or executes pickle code.

  python models/convert-scnet-to-gguf.py --source-dir ../SCNet-upstream \
      --checkpoint checkpoint.th --checkpoint-sha256 <64-hex-digest> \
      --variant scnet --output scnet-f16.gguf

Matrices use F16 by default; biases and normalization vectors remain F32.
Use crispasr-quantize for Q8_0/Q4_K after conversion. Tensor layout version 1:
Conv/Conv1d [out,in,kF,kT] -> [out,in*kF*kT]; ConvTranspose2d
[in,out,kF,1] -> transpose(1,2,3,0).reshape(out*kF,in). LSTM and dense
matrices retain their original [out,in] layout and gate ordering.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from reference_backends.scnet import (  # noqa: E402
    CHECKPOINT_URLS, LICENSE_URL, SOURCE_HASHES, SOURCE_REVISION, SOURCE_URL,
    expected_shapes, load_checkpoint, model_config, sha256, verify_source_files,
)


def convert_tensor(name, value, dtype="f16"):
    """Convert layout only, retaining full bias/normalization precision."""
    import numpy as np

    if dtype not in ("f16", "f32"):
        raise ValueError("dtype must be f16 or f32")
    if hasattr(value, "detach"):
        value = value.detach().float().cpu().numpy()
    array = np.asarray(value, dtype=np.float32)
    if not np.isfinite(array).all():
        raise ValueError(f"nonfinite checkpoint tensor: {name}")
    if ".convtrs." in name and name.endswith(".weight"):
        if array.ndim != 4 or array.shape[-1] != 1:
            raise ValueError(f"unsupported ConvTranspose2d shape for {name}: {array.shape}")
        cin, cout, kernel, _ = array.shape
        array = array.transpose(1, 2, 3, 0).reshape(cout * kernel, cin)
    elif array.ndim > 2:
        array = array.reshape(array.shape[0], -1)
    target = np.float32 if dtype == "f32" or array.ndim == 1 else np.float16
    if target == np.float16 and np.max(np.abs(array)) > np.finfo(np.float16).max:
        raise ValueError(f"F16 overflow in {name}; use diagnostic --dtype f32")
    return np.ascontiguousarray(array, dtype=target)


def validate_config(config, variant):
    expected = model_config(variant)
    actual = config.get("model", config)
    if not isinstance(actual, dict) or set(actual) != set(expected):
        raise ValueError("model config must contain exactly the supported SCNet constructor fields")
    for key, value in expected.items():
        if actual[key] != value or isinstance(actual[key], bool) != isinstance(value, bool):
            raise ValueError(f"unsupported {variant} configuration: {key} must equal {value!r}")


def write_metadata(writer, variant, dtype, checkpoint_sha256, source_dir, config_sha256=None):
    cfg = model_config(variant)
    writer.add_name("SCNet-large" if variant == "scnet-large" else "SCNet")
    writer.add_description("Author's four-stem stereo Sparse Compression Network; native ggml layout v1")
    writer.add_file_type(1 if dtype == "f16" else 0)
    writer.add_string("general.license", "mit")
    writer.add_string("general.license.link", LICENSE_URL)
    writer.add_string("general.license.text", (source_dir / "LICENSE").read_text(encoding="utf-8"))
    writer.add_string("general.author", "Weinan Tong and SCNet authors")
    writer.add_string("general.source.url", SOURCE_URL)
    writer.add_string("general.source.revision", SOURCE_REVISION)
    writer.add_string("general.source.hashes_json", json.dumps(SOURCE_HASHES, sort_keys=True))
    writer.add_string("general.source.model_sha256", checkpoint_sha256)
    writer.add_string("scnet.checkpoint.sha256", checkpoint_sha256)
    writer.add_string("scnet.checkpoint.url", CHECKPOINT_URLS[variant])
    writer.add_string("scnet.checkpoint.state_key", "best_state")
    writer.add_string("scnet.variant", variant)
    writer.add_uint32("scnet.tensor_layout_version", 1)
    if config_sha256:
        writer.add_string("scnet.config.sha256", config_sha256)
    integers = {
        "n_fft": cfg["nfft"], "hop_length": cfg["hop_size"], "win_length": cfg["win_size"],
        "sample_rate": 44100, "audio_channels": 2, "num_sources": 4,
        "num_dplayer": cfg["num_dplayer"], "expand": cfg["expand"],
        "compress": cfg["compress"], "conv_kernel": cfg["conv_kernel"],
    }
    for name, value in integers.items():
        writer.add_uint32("scnet." + name, value)
    for name in ("dims", "band_stride", "band_kernel", "conv_depths"):
        writer.add_array("scnet." + name, cfg[name])
    writer.add_array("scnet.band_sr", cfg["band_SR"])
    writer.add_array("scnet.source_names", cfg["sources"])
    writer.add_bool("scnet.normalized", cfg["normalized"])


def convert(args):
    from gguf import GGUFWriter

    source_dir = verify_source_files(args.source_dir)
    config_hash = None
    if args.config:
        import yaml

        validate_config(yaml.safe_load(args.config.read_text(encoding="utf-8")), args.variant)
        config_hash = sha256(args.config)
    state, digest = load_checkpoint(args.checkpoint, args.variant, expected_sha256=args.checkpoint_sha256)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=f".{args.output.name}.", suffix=".partial", dir=args.output.parent)
    os.close(handle)
    writer = None
    try:
        writer = GGUFWriter(temporary, "scnet", use_temp_file=True)
        writer.temp_file = tempfile.SpooledTemporaryFile(max_size=8 * 1024 * 1024, dir=args.output.parent)
        write_metadata(writer, args.variant, args.dtype, digest, source_dir, config_hash)
        for name in sorted(state):
            writer.add_tensor(name, convert_tensor(name, state[name], args.dtype))
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()
        writer = None
        os.replace(temporary, args.output)
    finally:
        if writer is not None:
            writer.close()
            if writer.temp_file is not None:
                writer.temp_file.close()
        Path(temporary).unlink(missing_ok=True)
    provenance = {
        "architecture": "scnet", "variant": args.variant, "tensor_layout_version": 1,
        "source_revision": SOURCE_REVISION, "source_url": SOURCE_URL,
        "source_hashes": SOURCE_HASHES, "checkpoint_url": CHECKPOINT_URLS[args.variant],
        "checkpoint_sha256": digest, "state_key": "best_state", "model_config": model_config(args.variant),
        "gguf_sha256": sha256(args.output), "dtype": args.dtype, "tensor_count": len(state),
        "license": "MIT", "checkpoint_license_evidence": LICENSE_URL,
        "checkpoint_hash_origin": "caller-supplied expected digest; upstream publishes no checksum",
    }
    args.output.with_name(args.output.name + ".provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.output.with_name(args.output.name + ".LICENSE.txt").write_text(
        (source_dir / "LICENSE").read_text(encoding="utf-8"), encoding="utf-8")
    count = sum(math.prod(shape) for shape in expected_shapes(args.variant).values())
    print(f"Wrote {args.output}: {len(state)} tensors, {count:,} parameters, {args.dtype} matrices + F32 vectors")
    print(f"Original source: {SOURCE_URL}@{SOURCE_REVISION}; checkpoint SHA-256: {digest}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True, help="expected digest of the exact author checkpoint")
    parser.add_argument("--variant", choices=tuple(CHECKPOINT_URLS), default="scnet")
    parser.add_argument("--config", type=Path, help="optional author config; all model fields must match the supported variant")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dtype", choices=("f16", "f32"), default="f16")
    args = parser.parse_args()
    try:
        convert(args)
    except (ValueError, FileNotFoundError) as error:
        parser.exit(1, f"error: {error}\n")


if __name__ == "__main__":
    main()
