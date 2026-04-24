#!/usr/bin/env python3
"""Splice the Qwen2.5-7B vocabulary into an existing VibeVoice-ASR GGUF.

Older VibeVoice GGUFs (produced before the converter learned to fall back to
Qwen/Qwen2.5-7B) ship without `tokenizer.ggml.tokens`. The runtime then
generates correct token IDs but detokenizes to an empty string. This tool
rewrites such a GGUF adding the missing vocabulary — no tensor data changes,
so it's cheap and loss-free.

Design: pure metadata-level splice. Tensor blobs are streamed byte-for-byte
from input to output; only the KV section grows (~2.6 MB of UTF-8 strings).

Usage:
  python tools/vibevoice_add_tokenizer.py \
      --input vibevoice-asr-7b-q4_k.gguf \
      --output vibevoice-asr-7b-q4_k-fixed.gguf

Requires: transformers (for the Qwen tokenizer) + the vendored gguf module.
No torch required — this is a metadata rewrite.
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


def load_qwen_vocab(vocab_size):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B", trust_remote_code=True)
    vocab_map = tok.get_vocab()  # str -> id
    inv = {v: k for k, v in vocab_map.items()}
    max_id = max(inv.keys())
    # Match the converter's behaviour: size the array by max_id + 1,
    # filling unknowns with "<unk_<i>>". Runtime only needs IDs that the
    # model actually emits (all inside Qwen's defined range).
    vocab_list = [inv.get(i, f"<unk_{i}>") for i in range(max_id + 1)]
    print(f"  loaded {len(vocab_list)} tokens from Qwen/Qwen2.5-7B (model vocab_size={vocab_size})")
    return vocab_list


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="existing VibeVoice GGUF without tokenizer")
    ap.add_argument("--output", required=True, help="path for the rewritten GGUF")
    args = ap.parse_args()

    reader = gguf.GGUFReader(args.input)

    # Sanity: refuse to rewrite if tokens already present.
    if "tokenizer.ggml.tokens" in reader.fields:
        print("input already contains tokenizer.ggml.tokens — nothing to do", file=sys.stderr)
        return 0

    # Pull architecture name and vocab size
    arch_field = reader.fields.get("general.architecture")
    arch = bytes(arch_field.parts[arch_field.data[0]]).decode("utf-8") if arch_field else "vibevoice-asr"
    vs_field = reader.fields.get("vibevoice.vocab_size")
    vocab_size = int(vs_field.parts[vs_field.data[0]]) if vs_field else 152064

    vocab_list = load_qwen_vocab(vocab_size)

    writer = gguf.GGUFWriter(args.output, arch)

    # Copy every existing KV (except the flag we'll overwrite below).
    SKIP_KEYS = {
        "GGUF.version", "GGUF.tensor_count", "GGUF.kv_count",  # auto-managed
        "general.architecture",                                 # written by GGUFWriter ctor
        "vibevoice.has_tokenizer",                              # overwritten
    }
    for key, field in reader.fields.items():
        if key in SKIP_KEYS:
            continue
        # gguf.GGUFReader exposes the raw type; mirror it through GGUFWriter.
        t = field.types[0] if field.types else None
        if t == gguf.GGUFValueType.STRING:
            s = bytes(field.parts[field.data[0]]).decode("utf-8", "replace")
            writer.add_string(key, s)
        elif t == gguf.GGUFValueType.UINT32:
            writer.add_uint32(key, int(field.parts[field.data[0]]))
        elif t == gguf.GGUFValueType.INT32:
            writer.add_int32(key, int(field.parts[field.data[0]]))
        elif t == gguf.GGUFValueType.UINT64:
            writer.add_uint64(key, int(field.parts[field.data[0]]))
        elif t == gguf.GGUFValueType.FLOAT32:
            writer.add_float32(key, float(field.parts[field.data[0]]))
        elif t == gguf.GGUFValueType.BOOL:
            writer.add_bool(key, bool(field.parts[field.data[0]]))
        elif t == gguf.GGUFValueType.ARRAY:
            # Element type is field.types[1]
            elem_t = field.types[1]
            if elem_t == gguf.GGUFValueType.INT32:
                writer.add_array(key, [int(field.parts[off]) for off in field.data])
            elif elem_t == gguf.GGUFValueType.UINT32:
                writer.add_array(key, [int(field.parts[off]) for off in field.data])
            elif elem_t == gguf.GGUFValueType.STRING:
                writer.add_array(key, [bytes(field.parts[off]).decode("utf-8", "replace") for off in field.data])
            else:
                raise RuntimeError(f"unsupported array elem type for key {key}: {elem_t}")
        else:
            raise RuntimeError(f"unsupported KV type for key {key}: {t}")

    # Add the missing vocabulary + flip the flag.
    writer.add_array("tokenizer.ggml.tokens", vocab_list)
    writer.add_uint32("vibevoice.has_tokenizer", 1)

    # Copy every tensor by reference — no dtype changes, no reshapes.
    #
    # gguf-py has two gotchas that together silently corrupt the output:
    #
    #   1. GGUFReader exposes quantized tensors with dtype=uint8 and a
    #      numpy shape = (outer, bytes_per_row). When add_tensor sees
    #      dtype=uint8, it calls quant_shape_from_byte_shape to recover
    #      the logical shape — but it takes shape[-1] as "bytes per row"
    #      and derives the logical last-dim from that. A prior attempt
    #      (flatten to 1D uint8) bypassed truncation but stored a 1D
    #      header shape, so the runtime saw lm_head.weight as rank-1 and
    #      ggml_mul_mat failed. Another prior attempt (raw_shape=logical)
    #      crashed because the uint8 branch reinterpreted that logical
    #      shape as bytes-per-row.
    #
    #   2. ggml tensor shape is stored ne0-first (fastest-varying first);
    #      numpy's shape is reversed. So raw_shape must be
    #      reversed(reader_t.shape).
    #
    # Robust path: view the uint8 buffer as int8 — same bytes, but the
    # dtype check in GGUFWriter.add_tensor_info no longer fires, so it
    # trusts our raw_shape verbatim. Then the correct 2D logical shape
    # and full-length byte buffer both land in the output.
    for t in reader.tensors:
        ggml_shape = tuple(int(x) for x in reversed(t.shape))
        data = t.data.view(np.int8) if t.data.dtype == np.uint8 else t.data
        writer.add_tensor(t.name, data, raw_shape=ggml_shape, raw_dtype=t.tensor_type)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    sz = os.path.getsize(args.output)
    print(f"Done: {args.output} ({sz / 1e9:.2f} GB, {len(reader.tensors)} tensors, +{len(vocab_list)} vocab strings)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
