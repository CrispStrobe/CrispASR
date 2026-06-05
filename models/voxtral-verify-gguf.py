#!/usr/bin/env python3
"""Verify a Voxtral GGUF has all expected tensors with the right shapes.

Run after `models/convert-voxtral-to-gguf.py` to catch missing or
mis-named tensors before wiring up the C++ loader.

Usage:
  python3 models/voxtral-verify-gguf.py path/to/voxtral-mini-3b-2507.gguf
"""

from __future__ import annotations
import sys
import gguf

# Expected tensor names + their shape *patterns* (in PyTorch / source order;
# the GGUF reader returns dims reversed from the source numpy shape).
EXPECTED = {
    # Audio encoder front-end
    "audio.conv.1.weight": ("F16", (3, 128, 1280)),
    "audio.conv.1.bias": ("F32", (1280,)),
    "audio.conv.2.weight": ("F16", (3, 1280, 1280)),
    "audio.conv.2.bias": ("F32", (1280,)),
    "audio.embed_positions": ("F32", (1280, 1500)),  # ggml ne reversal
    "audio.ln_post.weight": ("F32", (1280,)),
    "audio.ln_post.bias": ("F32", (1280,)),
    # Multi-modal projector
    "proj1.weight": ("F16", (5120, 3072)),  # nn.Linear(5120 -> 3072), ggml ne reversed
    "proj2.weight": ("F16", (3072, 3072)),
    # LLM top-level
    "token_embd.weight": ("F16", (3072, 131072)),
    "output_norm.weight": ("F32", (3072,)),
    "output.weight": ("F16", (3072, 131072)),
}

# Per-layer expectations
AUDIO_LAYER = {
    "attn_norm.weight": ("F32", (1280,)),
    "attn_norm.bias": ("F32", (1280,)),
    "attn_q.weight": ("F16", (1280, 1280)),
    "attn_q.bias": ("F32", (1280,)),
    "attn_k.weight": ("F16", (1280, 1280)),  # K-proj has NO bias
    "attn_v.weight": ("F16", (1280, 1280)),
    "attn_v.bias": ("F32", (1280,)),
    "attn_out.weight": ("F16", (1280, 1280)),
    "attn_out.bias": ("F32", (1280,)),
    "ffn_norm.weight": ("F32", (1280,)),
    "ffn_norm.bias": ("F32", (1280,)),
    "ffn_up.weight": ("F16", (1280, 5120)),  # ggml ne reversed
    "ffn_up.bias": ("F32", (5120,)),
    "ffn_down.weight": ("F16", (5120, 1280)),  # ggml ne reversed
    "ffn_down.bias": ("F32", (1280,)),
}

LLM_LAYER = {
    "attn_norm.weight": ("F32", (3072,)),
    "attn_q.weight": ("F16", (3072, 4096)),  # ggml ne reversed
    "attn_k.weight": ("F16", (3072, 1024)),
    "attn_v.weight": ("F16", (3072, 1024)),
    "attn_output.weight": ("F16", (4096, 3072)),
    "ffn_norm.weight": ("F32", (3072,)),
    "ffn_gate.weight": ("F16", (3072, 8192)),
    "ffn_up.weight": ("F16", (3072, 8192)),
    "ffn_down.weight": ("F16", (8192, 3072)),
}

EXPECTED_KV = {
    "voxtral.sample_rate": 16000,
    "voxtral.n_mels": 128,
    "voxtral.n_fft": 400,
    "voxtral.hop_length": 160,
    "voxtral.audio.n_layers": 32,
    "voxtral.audio.d_model": 1280,
    "voxtral.audio.n_heads": 20,
    "voxtral.audio.head_dim": 64,
    "voxtral.audio.ff_dim": 5120,
    "voxtral.audio.max_pos": 1500,
    "voxtral.proj.in_dim": 5120,
    "voxtral.proj.out_dim": 3072,
    "voxtral.proj.frame_stack": 4,
    "voxtral.llm.n_layers": 30,
    "voxtral.llm.d_model": 3072,
    "voxtral.llm.n_heads": 32,
    "voxtral.llm.n_kv_heads": 8,
    "voxtral.llm.head_dim": 128,
    "voxtral.llm.ff_dim": 8192,
    "voxtral.llm.vocab_size": 131072,
    "voxtral.audio_token_id": 24,
}


def main():
    if len(sys.argv) < 2:
        print("usage: voxtral-verify-gguf.py path/to/voxtral.gguf", file=sys.stderr)
        sys.exit(1)
    path = sys.argv[1]
    print(f"Verifying {path} ...")
    r = gguf.GGUFReader(path)

    # Build a name→tensor lookup
    by_name = {t.name: t for t in r.tensors}
    print(f"  total tensors in GGUF: {len(by_name)}")

    errors = []

    # Check top-level expected
    for name, (ftype, expected_shape) in EXPECTED.items():
        t = by_name.get(name)
        if t is None:
            errors.append(f"  MISSING: {name}")
            continue
        actual = tuple(int(x) for x in t.shape)
        if actual != expected_shape:
            errors.append(f"  SHAPE: {name}: got {actual}, want {expected_shape}")
        if t.tensor_type.name != ftype:
            errors.append(f"  TYPE: {name}: got {t.tensor_type.name}, want {ftype}")

    # Check audio layers
    for li in range(32):
        for suf, (ftype, sh) in AUDIO_LAYER.items():
            full = f"audio.blk.{li}.{suf}"
            t = by_name.get(full)
            if t is None:
                errors.append(f"  MISSING: {full}")
                continue
            actual = tuple(int(x) for x in t.shape)
            if actual != sh:
                errors.append(f"  SHAPE: {full}: got {actual}, want {sh}")
            if t.tensor_type.name != ftype:
                errors.append(f"  TYPE: {full}: got {t.tensor_type.name}, want {ftype}")

    # Check LLM layers
    for li in range(30):
        for suf, (ftype, sh) in LLM_LAYER.items():
            full = f"blk.{li}.{suf}"
            t = by_name.get(full)
            if t is None:
                errors.append(f"  MISSING: {full}")
                continue
            actual = tuple(int(x) for x in t.shape)
            if actual != sh:
                errors.append(f"  SHAPE: {full}: got {actual}, want {sh}")
            if t.tensor_type.name != ftype:
                errors.append(f"  TYPE: {full}: got {t.tensor_type.name}, want {ftype}")

    # Check KV
    fields = {f.name: f for f in r.fields.values()}
    for k, expected in EXPECTED_KV.items():
        f = fields.get(k)
        if f is None:
            errors.append(f"  MISSING KV: {k}")
            continue
        try:
            v = int(f.parts[f.data[0]])
            if v != expected:
                errors.append(f"  KV: {k}: got {v}, want {expected}")
        except Exception as e:
            errors.append(f"  KV: {k}: parse error {e}")

    # Tekken blobs
    for k in (
        "tokenizer.tekken.pattern",
        "tokenizer.tekken.specials",
        "tokenizer.tekken.vocab",
        "tokenizer.tekken.n_specials",
        "tokenizer.tekken.n_vocab",
    ):
        if k not in fields:
            errors.append(f"  MISSING tekken blob: {k}")

    if errors:
        print(f"\n  {len(errors)} ERRORS:")
        for e in errors[:30]:
            print(e)
        if len(errors) > 30:
            print(f"  ... and {len(errors)-30} more")
        sys.exit(1)
    else:
        print("\n  all expected tensors present and well-formed:")
        print(
            f"    audio:    32 layers × 15 tensors = {32*15}  + 7 top-level = {32*15+7}"
        )
        print("    proj:     2")
        print(
            f"    LLM:      30 layers × 9 tensors = {30*9}    + 3 top-level = {30*9+3}"
        )
        print(f"    expected total: {32*15+7 + 2 + 30*9+3}, got {len(by_name)}")
        print(f"    KV metadata: {len(EXPECTED_KV)} keys verified")
        print("    Tekken blobs: 5/5 present")
        print("\n  PASS")


if __name__ == "__main__":
    main()
