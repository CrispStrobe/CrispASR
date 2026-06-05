#!/usr/bin/env python3
"""Patch an existing mimo-asr GGUF to fuse per-LM-layer Q/K/V tensors.

PLAN #60d landed the runtime + converter changes for fused QKV.
Re-running the full BF16 → F16 / Q4_K conversion is bottlenecked on a
nearly-full external disk (~0.8 MB/min sustained), so this script
performs the equivalent transformation directly on a finished GGUF
by byte-concat'ing the per-layer Q/K/V weight bytes (valid for F16 /
F32 / Q4_K / Q8_0 because each row is independently quantised) and
F32 bias values. The output is bit-identical to what the updated
converter would produce.

Usage:
    python tools/patch_mimo_asr_fuse_qkv.py <input.gguf> <output.gguf>
"""

import argparse
import sys
from pathlib import Path

import numpy as np

try:
    import gguf
except ImportError:
    print("Error: gguf package not found. Install with: pip install gguf")
    sys.exit(1)


GGML_TYPE_F32 = gguf.GGMLQuantizationType.F32


def main():
    parser = argparse.ArgumentParser(description="Fuse mimo-asr LM Q/K/V tensors")
    parser.add_argument("input", type=Path, help="Input GGUF (existing mimo-asr)")
    parser.add_argument("output", type=Path, help="Output GGUF (fused QKV)")
    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: input {args.input} not found")
        sys.exit(1)

    print(f"Reading {args.input}...")
    reader = gguf.GGUFReader(str(args.input))
    print(f"  {len(reader.tensors)} tensors, {len(reader.fields)} KV fields")

    tensors_by_name = {t.name: t for t in reader.tensors}

    lm_layer_idxs = set()
    for t in reader.tensors:
        if t.name.startswith("model.layers.") and ".attn." in t.name:
            parts = t.name.split(".")
            if (len(parts) == 6 and parts[3] == "attn"
                    and parts[4] in ("q", "k", "v")
                    and parts[5] in ("weight", "bias")):
                lm_layer_idxs.add(int(parts[2]))
    print(f"  Detected {len(lm_layer_idxs)} LM layers with separate Q/K/V")
    if not lm_layer_idxs:
        print("Nothing to fuse. Input may already have fused QKV.")
        sys.exit(0)

    # ---- Set up writer ----
    arch_field = reader.fields["general.architecture"]
    arch = bytes(arch_field.parts[arch_field.data[0]]).decode("utf-8")
    writer = gguf.GGUFWriter(str(args.output), arch, use_temp_file=True)

    # ---- Copy KV metadata fields ----
    skip = {"GGUF.version", "GGUF.tensor_count", "GGUF.kv_count", "general.architecture"}
    n_fields_copied = 0
    for fname, field in reader.fields.items():
        if fname in skip:
            continue
        ft = field.types[0]
        if ft == gguf.GGUFValueType.STRING:
            value = bytes(field.parts[field.data[0]]).decode("utf-8")
            writer.add_string(fname, value)
        elif ft == gguf.GGUFValueType.ARRAY:
            elem_type = field.types[1]
            if elem_type == gguf.GGUFValueType.STRING:
                values = [bytes(field.parts[i]).decode("utf-8") for i in field.data]
            else:
                values = [field.parts[i].tolist()[0] for i in field.data]
            writer.add_array(fname, values)
        elif ft == gguf.GGUFValueType.UINT32:
            writer.add_uint32(fname, int(field.parts[field.data[0]][0]))
        elif ft == gguf.GGUFValueType.INT32:
            writer.add_int32(fname, int(field.parts[field.data[0]][0]))
        elif ft == gguf.GGUFValueType.UINT64:
            writer.add_uint64(fname, int(field.parts[field.data[0]][0]))
        elif ft == gguf.GGUFValueType.INT64:
            writer.add_int64(fname, int(field.parts[field.data[0]][0]))
        elif ft == gguf.GGUFValueType.UINT16:
            writer.add_uint16(fname, int(field.parts[field.data[0]][0]))
        elif ft == gguf.GGUFValueType.INT16:
            writer.add_int16(fname, int(field.parts[field.data[0]][0]))
        elif ft == gguf.GGUFValueType.UINT8:
            writer.add_uint8(fname, int(field.parts[field.data[0]][0]))
        elif ft == gguf.GGUFValueType.INT8:
            writer.add_int8(fname, int(field.parts[field.data[0]][0]))
        elif ft == gguf.GGUFValueType.FLOAT32:
            writer.add_float32(fname, float(field.parts[field.data[0]][0]))
        elif ft == gguf.GGUFValueType.FLOAT64:
            writer.add_float64(fname, float(field.parts[field.data[0]][0]))
        elif ft == gguf.GGUFValueType.BOOL:
            writer.add_bool(fname, bool(field.parts[field.data[0]][0]))
        else:
            print(f"  Warning: skipping field '{fname}' of unhandled type {ft}")
            continue
        n_fields_copied += 1
    print(f"  Copied {n_fields_copied} KV fields")

    # ---- Emit tensors ----
    handled = set()
    n_written = 0
    n_fused = 0

    def add_raw(name, t):
        # GGUFReader.data is a numpy memmap of either the element-typed
        # array (F32/F16) or the raw uint8 byte buffer (quant types).
        # gguf.GGUFWriter.add_tensor_info derives the logical shape:
        #   - numpy float dtype → shape is element shape (taken as-is)
        #   - numpy uint8 dtype → shape is byte shape, converted to
        #     logical shape via quant_shape_from_byte_shape
        # So we pass data.shape unmodified; do NOT supply raw_shape.
        data = np.ascontiguousarray(t.data)
        writer.add_tensor(name, data, raw_dtype=t.tensor_type)

    for t in reader.tensors:
        if t.name in handled:
            continue

        is_lm_qkv = False
        if t.name.startswith("model.layers.") and ".attn." in t.name:
            parts = t.name.split(".")
            if (len(parts) == 6 and parts[3] == "attn"
                    and parts[4] in ("q", "k", "v")
                    and parts[5] in ("weight", "bias")):
                is_lm_qkv = True

        if not is_lm_qkv:
            add_raw(t.name, t)
            handled.add(t.name)
            n_written += 1
            continue

        layer_idx = int(t.name.split(".")[2])
        suffix = t.name.split(".")[5]
        q_name = f"model.layers.{layer_idx}.attn.q.{suffix}"
        k_name = f"model.layers.{layer_idx}.attn.k.{suffix}"
        v_name = f"model.layers.{layer_idx}.attn.v.{suffix}"
        if q_name in handled and k_name in handled and v_name in handled:
            continue
        q_t, k_t, v_t = tensors_by_name[q_name], tensors_by_name[k_name], tensors_by_name[v_name]
        if q_t.tensor_type != k_t.tensor_type or q_t.tensor_type != v_t.tensor_type:
            print(f"Error: Q/K/V tensor types differ for layer {layer_idx} {suffix}")
            sys.exit(1)
        fused_dtype = q_t.tensor_type

        if suffix == "bias":
            fused_data = np.concatenate([q_t.data, k_t.data, v_t.data]).astype(np.float32, copy=False)
            writer.add_tensor(f"model.layers.{layer_idx}.attn.qkv.bias",
                              np.ascontiguousarray(fused_data), raw_dtype=GGML_TYPE_F32)
        else:
            # Byte/element concat along axis=0 (the per-row dim in
            # GGUFReader.data, which is q_out for weights). Each row is
            # independently quantised so concat is safe across F16/F32 and
            # all per-row quant types (Q4_K, Q8_0, ...). For quant types
            # axis=0 is the q_out row count (number of independent rows);
            # the inner dim is bytes-per-row and stays unchanged. For
            # F16/F32 axis=0 is also q_out (logical-shape row count).
            fused_data = np.concatenate([q_t.data, k_t.data, v_t.data], axis=0)
            writer.add_tensor(f"model.layers.{layer_idx}.attn.qkv.weight",
                              np.ascontiguousarray(fused_data), raw_dtype=fused_dtype)
        handled.add(q_name)
        handled.add(k_name)
        handled.add(v_name)
        n_fused += 1
        if n_fused % 24 == 0:
            print(f"  fused {n_fused} {suffix} tensors so far")

    print(f"  Total: {n_written} pass-through, {n_fused} fused triplets")
    print(f"Writing {args.output}...")
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    size_gb = args.output.stat().st_size / 1024**3
    print(f"Done! {args.output} ({size_gb:.2f} GB)")


if __name__ == "__main__":
    main()
