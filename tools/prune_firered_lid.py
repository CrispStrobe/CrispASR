#!/usr/bin/env python3
"""Prune FireRedLID encoder layers for smaller LID model.

Creates a new GGUF with fewer encoder layers. Two strategies:
  --strategy prune   : keep only selected layers (drop the rest)
  --strategy slerp   : merge adjacent pairs via SLERP interpolation

Usage:
  # Keep layers 0,1,14,15 (first 2 + last 2 of 16):
  python tools/prune_firered_lid.py --input firered-lid.gguf --output lid-4L.gguf --keep 0,1,14,15

  # SLERP merge pairs (0+1, 2+3, ..., 14+15) → 8 layers:
  python tools/prune_firered_lid.py --input firered-lid.gguf --output lid-8L-slerp.gguf --strategy slerp --pairs 0-1,2-3,4-5,6-7,8-9,10-11,12-13,14-15

  # Aggressive: keep 0,1 + slerp(7,8) + 14,15 → 4 layers:
  python tools/prune_firered_lid.py --input firered-lid.gguf --output lid-4L-slerp.gguf --strategy slerp --pairs 0,1,7-8,14,15
"""

import argparse
import math
import os
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ggml", "python"))
import gguf


def slerp(t, v0, v1, eps=1e-8):
    """Spherical linear interpolation between two vectors/tensors."""
    v0_flat = v0.flatten().astype(np.float64)
    v1_flat = v1.flatten().astype(np.float64)
    n0 = np.linalg.norm(v0_flat)
    n1 = np.linalg.norm(v1_flat)
    if n0 < eps or n1 < eps:
        return ((1 - t) * v0 + t * v1).astype(v0.dtype)
    v0n = v0_flat / n0
    v1n = v1_flat / n1
    dot = np.clip(np.dot(v0n, v1n), -1.0, 1.0)
    omega = np.arccos(dot)
    if abs(omega) < eps:
        return ((1 - t) * v0 + t * v1).astype(v0.dtype)
    so = np.sin(omega)
    result = (np.sin((1 - t) * omega) / so) * v0_flat + (np.sin(t * omega) / so) * v1_flat
    # Rescale to average magnitude
    result = result * ((n0 + n1) / 2.0) / (np.linalg.norm(result) + eps)
    return result.reshape(v0.shape).astype(v0.dtype)


def parse_layer_spec(spec_str, n_layers):
    """Parse layer specification.

    Formats:
      "0,1,14,15"          → keep these layers as-is
      "0,1,7-8,14,15"      → keep 0,1,14,15 as-is; slerp merge 7+8
      "0-1,2-3,...,14-15"   → slerp each pair
    """
    items = []
    for part in spec_str.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-", 1)
            items.append(("slerp", int(a), int(b)))
        else:
            items.append(("keep", int(part), int(part)))
    return items


def read_tensor_data(reader, tensor):
    """Read tensor data from GGUF reader."""
    # The tensor data offset is relative to the data section start
    offset = tensor.data_offset
    with open(reader.path if hasattr(reader, 'path') else reader._path, "rb") as f:
        f.seek(offset)
        nbytes = int(np.prod(tensor.shape)) * gguf.GGML_QUANT_SIZES.get(tensor.tensor_type, (1, 1))[1] // gguf.GGML_QUANT_SIZES.get(tensor.tensor_type, (1, 1))[0]
        # Actually just read the raw bytes
        data_size = 1
        for d in tensor.shape:
            data_size *= d
        if tensor.tensor_type == gguf.GGMLQuantizationType.F32:
            dtype = np.float32
        elif tensor.tensor_type == gguf.GGMLQuantizationType.F16:
            dtype = np.float16
        else:
            dtype = np.float32  # will need special handling
        return np.frombuffer(f.read(data_size * dtype().itemsize), dtype=dtype).reshape(tensor.shape)


def main():
    parser = argparse.ArgumentParser(description="Prune/merge FireRedLID encoder layers")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--keep", type=str, default=None,
                        help="Layers to keep/merge, e.g. '0,1,7-8,14,15'")
    parser.add_argument("--slerp-t", type=float, default=0.5,
                        help="SLERP interpolation factor (0.5 = equal mix)")
    parser.add_argument("--dec-keep", type=str, default=None,
                        help="Decoder layers to keep (default: all). E.g. '0,5' for first+last")
    args = parser.parse_args()

    reader = gguf.GGUFReader(args.input)

    # Read hyperparams
    def get_field_val(name):
        f = reader.fields.get(name)
        if f is None:
            return None
        return f.parts[-1].tolist()[0] if len(f.parts) > 0 else None

    n_layers_enc = get_field_val("firered.n_layers_enc") or 16
    n_layers_dec = get_field_val("firered.n_layers_dec") or 6
    print(f"Input: {n_layers_enc} enc layers, {n_layers_dec} dec layers, {len(reader.tensors)} tensors")

    # Parse layer specs
    if args.keep is None:
        # Default: keep first 2, last 2
        enc_spec = parse_layer_spec("0,1,14,15", n_layers_enc)
    else:
        enc_spec = parse_layer_spec(args.keep, n_layers_enc)

    if args.dec_keep is not None:
        dec_layers_keep = [int(x.strip()) for x in args.dec_keep.split(",")]
    else:
        dec_layers_keep = list(range(n_layers_dec))

    # Compute output layer count
    n_out_enc = len(enc_spec)
    n_out_dec = len(dec_layers_keep)
    print(f"Output: {n_out_enc} enc layers, {n_out_dec} dec layers")

    # Build tensor name mapping
    # For each output layer, map old tensor names → new tensor names
    tensor_map = {}  # old_name → (new_name, transform)
    # transform = None (copy), or ("slerp", other_layer_idx, t)

    # Non-layer tensors: copy as-is
    for t in reader.tensors:
        name = t.name
        if not name.startswith("enc.") and not name.startswith("dec."):
            tensor_map[name] = (name, None)
        elif name.startswith("enc.pe.") or name.startswith("enc.preproc."):
            tensor_map[name] = (name, None)
        elif name.startswith("dec.pe.") or name.startswith("dec.emb.") or \
             name.startswith("dec.prj.") or name.startswith("dec.norm_out."):
            tensor_map[name] = (name, None)

    # Encoder layers
    for out_idx, spec in enumerate(enc_spec):
        mode, a, b = spec
        if mode == "keep":
            # Direct copy, renumber
            prefix_old = f"enc.{a}."
            prefix_new = f"enc.{out_idx}."
            for t in reader.tensors:
                if t.name.startswith(prefix_old):
                    new_name = prefix_new + t.name[len(prefix_old):]
                    tensor_map[t.name] = (new_name, None)
        elif mode == "slerp":
            prefix_a = f"enc.{a}."
            prefix_b = f"enc.{b}."
            prefix_new = f"enc.{out_idx}."
            for t in reader.tensors:
                if t.name.startswith(prefix_a):
                    suffix = t.name[len(prefix_a):]
                    other_name = prefix_b + suffix
                    new_name = prefix_new + suffix
                    tensor_map[t.name] = (new_name, ("slerp", other_name, args.slerp_t))

    # Decoder layers
    for out_idx, old_idx in enumerate(dec_layers_keep):
        prefix_old = f"dec.{old_idx}."
        prefix_new = f"dec.{out_idx}."
        for t in reader.tensors:
            if t.name.startswith(prefix_old):
                new_name = prefix_new + t.name[len(prefix_old):]
                tensor_map[t.name] = (new_name, None)

    # Build name→tensor lookup
    tensor_by_name = {t.name: t for t in reader.tensors}

    # Read all tensor data we need
    print("Reading tensor data...")
    # We need the raw file path
    input_path = args.input

    def read_tensor_np(t):
        """Read tensor as numpy array in its native ggml memory layout.

        IMPORTANT: The GGUF Python writer reverses numpy shapes for ggml
        column-major convention. To preserve the original layout, we store
        tensors as 1D flat arrays — the writer will create an equivalent
        tensor with shape [n_elements] which we fix by copying the shape
        metadata separately. Actually, the simplest correct approach:
        reshape using the REVERSED ggml shape (= row-major / numpy order).
        """
        offset = t.data_offset
        # t.shape is in ggml order (ne[0], ne[1], ...). For numpy, reverse it.
        np_shape = tuple(reversed(t.shape.tolist()))
        with open(input_path, "rb") as f:
            f.seek(offset)
            if t.tensor_type == gguf.GGMLQuantizationType.F32:
                data = np.frombuffer(f.read(int(np.prod(t.shape)) * 4), dtype=np.float32)
            elif t.tensor_type == gguf.GGMLQuantizationType.F16:
                data = np.frombuffer(f.read(int(np.prod(t.shape)) * 2), dtype=np.float16)
            else:
                raise ValueError(f"Cannot read type {t.tensor_type} for slerp")
        return data.reshape(np_shape)

    # Create output GGUF
    writer = gguf.GGUFWriter(args.output, "firered-lid")
    writer.add_name("FireRedLID-pruned")

    # Write firered KV pairs (hardcoded keys, read from input)
    uint32_keys = [
        "firered.d_model", "firered.n_head", "firered.n_head_dec",
        "firered.d_inner", "firered.idim", "firered.odim",
        "firered.subsample", "firered.kernel_size", "firered.pe_maxlen",
        "firered.sos_id", "firered.eos_id", "firered.blank_id",
        "firered.pad_id", "firered.context",
    ]
    for key in uint32_keys:
        val = get_field_val(key)
        if val is not None:
            writer.add_uint32(key, int(val))

    # Override layer counts
    writer.add_uint32("firered.n_layers_enc", n_out_enc)
    writer.add_uint32("firered.n_layers_dec", n_out_dec)

    # Tokenizer: read vocab strings using gguf C API via subprocess
    # (The Python GGUFReader's string array parsing is unreliable)
    import subprocess, json
    vocab_str = subprocess.check_output([
        "python3", "-c",
        f"""
import ctypes, struct
path = "{args.input}"
# Use gguf C bindings to read string array properly
import sys; sys.path.insert(0, "ggml/python")
import gguf
r = gguf.GGUFReader(path)
# Read token strings via the original GGUF file binary
import json
# Parse the raw binary to extract string array
# Actually, just re-read from HuggingFace dict.txt if available
import os
dict_path = os.path.join(os.path.dirname(path), "dict.txt")
if not os.path.exists(dict_path):
    from huggingface_hub import hf_hub_download
    dict_path = hf_hub_download("FireRedTeam/FireRedLID", "dict.txt")
vocab = []
with open(dict_path) as f:
    for line in f:
        parts = line.strip().split()
        if parts:
            vocab.append(parts[0])
print(json.dumps(vocab))
"""
    ], text=True).strip()
    vocab = json.loads(vocab_str)
    writer.add_array("tokenizer.ggml.tokens", vocab)

    # Write tensors
    print("Writing tensors...")
    tensor_count = 0
    for t in reader.tensors:
        if t.name not in tensor_map:
            continue
        new_name, transform = tensor_map[t.name]

        data = read_tensor_np(t)

        if transform is not None and transform[0] == "slerp":
            _, other_name, slerp_t = transform
            other_t = tensor_by_name.get(other_name)
            if other_t is None:
                print(f"  WARNING: {other_name} not found for slerp, copying {t.name}")
            else:
                other_data = read_tensor_np(other_t)
                # Convert to f32 for slerp, then back
                d0 = data.astype(np.float32)
                d1 = other_data.astype(np.float32)
                merged = slerp(slerp_t, d0, d1)
                data = merged.astype(data.dtype)

        writer.add_tensor(new_name, data)
        tensor_count += 1
        if tensor_count <= 3 or tensor_count % 50 == 0:
            print(f"  [{tensor_count}] {new_name:50s} {str(data.shape):20s}")

    print(f"  total: {tensor_count} tensors")

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    sz = os.path.getsize(args.output)
    print(f"\nDone: {args.output} ({sz / 1e6:.0f} MB, {tensor_count} tensors)")
    print(f"  enc: {n_layers_enc} → {n_out_enc} layers")
    print(f"  dec: {n_layers_dec} → {n_out_dec} layers")


if __name__ == "__main__":
    main()
