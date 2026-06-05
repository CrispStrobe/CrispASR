#!/usr/bin/env python
"""
Patch a chatterbox-t3 GGUF to add the missing `tokenizer.ggml.tokens` +
`tokenizer.ggml.merges` keys from the source HF tokenizer.json. Writes a
new GGUF to <out>.

Background: an older revision of convert-chatterbox-to-gguf.py wrote only
`chatterbox.t3.text_tokens` and never wrote merges. The C++ loader then
fell through to legacy char-level tokenize_text, which doesn't match
python's BPE. R9 #5 LEARNINGS documents this.
"""
import sys
import json
from pathlib import Path

try:
    import gguf
except ImportError:
    sys.exit("install: pip install gguf")


def main(argv):
    if len(argv) != 4:
        sys.exit(f"usage: {argv[0]} INPUT.gguf TOKENIZER.json OUTPUT.gguf")
    inp = Path(argv[1])
    tok = Path(argv[2])
    out = Path(argv[3])

    reader = gguf.GGUFReader(inp)
    # Load tokenizer.json
    tok_data = json.loads(tok.read_text())
    model = tok_data.get("model", {})
    vocab = model.get("vocab", {})
    if not vocab:
        sys.exit("tokenizer.json has no vocab")
    max_id = max(vocab.values())
    tokens = [""] * (max_id + 1)
    for t, idx in vocab.items():
        if idx < len(tokens):
            tokens[idx] = t
    merges_raw = model.get("merges", [])
    merges = []
    for m in merges_raw:
        if isinstance(m, list) and len(m) == 2:
            merges.append(f"{m[0]} {m[1]}")
        elif isinstance(m, str):
            merges.append(m)

    print(f"tokens: {len(tokens)}  merges: {len(merges)}")

    # Build a writer that copies the existing GGUF + adds the new keys
    arch = "chatterbox.t3"
    # Try detecting arch from the reader
    for kv in reader.fields.values():
        if kv.name == "general.architecture":
            arch_bytes = bytes(kv.parts[kv.data[0]])
            arch = arch_bytes.decode("utf-8", "replace")
            break
    writer = gguf.GGUFWriter(out, arch, use_temp_file=False)

    # Copy existing KV pairs
    seen_keys = set()
    AUTO_KEYS = {
        "GGUF.version",
        "GGUF.tensor_count",
        "GGUF.kv_count",
        "general.architecture",  # writer constructor sets this
    }
    for name, field in reader.fields.items():
        if name in ("tokenizer.ggml.tokens", "tokenizer.ggml.merges"):
            continue  # we'll write our own
        if name in AUTO_KEYS:
            continue
        seen_keys.add(name)
        # Re-emit by type
        try:
            val = field.contents()
        except Exception:
            val = None
        if val is None:
            continue
        # gguf-py types
        t = field.types[0] if field.types else None
        if t == gguf.GGUFValueType.UINT32:
            writer.add_uint32(name, int(val))
        elif t == gguf.GGUFValueType.INT32:
            writer.add_int32(name, int(val))
        elif t == gguf.GGUFValueType.FLOAT32:
            writer.add_float32(name, float(val))
        elif t == gguf.GGUFValueType.BOOL:
            writer.add_bool(name, bool(val))
        elif t == gguf.GGUFValueType.STRING:
            writer.add_string(name, str(val))
        elif t == gguf.GGUFValueType.UINT64:
            writer.add_uint64(name, int(val))
        elif t == gguf.GGUFValueType.INT64:
            writer.add_int64(name, int(val))
        elif t == gguf.GGUFValueType.FLOAT64:
            writer.add_float64(name, float(val))
        elif t == gguf.GGUFValueType.ARRAY:
            # array of something
            elem_t = field.types[1] if len(field.types) > 1 else None
            if elem_t == gguf.GGUFValueType.STRING:
                writer.add_array(name, [str(v) for v in val])
            else:
                # arrays of numbers
                writer.add_array(name, list(val))
        else:
            print(f"skip unknown type for {name}: {t}")

    writer.add_array("tokenizer.ggml.tokens", tokens)
    writer.add_array("tokenizer.ggml.merges", merges)

    # Copy tensors. For quantized tensors, t.data is the raw byte buffer
    # (the numpy array's shape IS the byte shape); just pass raw_dtype so
    # the writer doesn't try to re-quantize.
    for t in reader.tensors:
        writer.add_tensor(t.name, t.data, raw_dtype=t.tensor_type)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"wrote: {out}")


if __name__ == "__main__":
    main(sys.argv)
