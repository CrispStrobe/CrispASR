#!/usr/bin/env python3
"""
Convert OuteAI/OuteTTS-0.3-1B (OlmoForCausalLM, CC BY 4.0) HuggingFace
safetensors -> GGUF F16/F32 for the CrispASR `outetts` backend.

OuteTTS is an LLM-based TTS: OLMo 1B generates interleaved text + audio
tokens autoregressively; audio tokens index into a WavTokenizer single
codebook (4096 entries, 75 tok/s, 24 kHz).  The codec decoder lives in
a separate GGUF (convert-wavtokenizer-to-gguf.py).

Architecture (from OuteAI/OuteTTS-0.3-1B/config.json):

    architectures        = ["OlmoForCausalLM"]
    hidden_size          = 2048
    num_hidden_layers    = 16
    num_attention_heads  = 16
    num_key_value_heads  = 16   (MHA, not GQA)
    head_dim             = 128
    intermediate_size    = 8192
    vocab_size           = 57344
    rope_theta           = 10000.0
    tie_word_embeddings  = true
    max_position_embeddings = 4096

OuteTTS V2 special tokens (discovered from tokenizer.json):
    <|im_start|>, <|im_end|>,
    <|text_start|>, <|text_end|>,
    <|audio_start|>, <|audio_end|>,
    <|space|>,
    <|t_X.XX|> (duration markers, 0.01-30.00 in 0.01 steps),
    <|0|> .. <|4099|> (audio code tokens)
    plus punctuation markers: <|period|>, <|comma|>, etc.

Usage:
    python models/convert-outetts-to-gguf.py \\
        --input OuteAI/OuteTTS-0.3-1B \\
        --output /mnt/storage/outetts/outetts-0.3-1b-f16.gguf
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

try:
    import gguf
    from gguf import GGUFWriter, GGMLQuantizationType
except ImportError:
    sys.exit("pip install gguf")

try:
    from safetensors import safe_open
except ImportError:
    sys.exit("pip install safetensors")

try:
    import torch
except ImportError:
    sys.exit("pip install torch")

try:
    from huggingface_hub import snapshot_download
except ImportError:
    sys.exit("pip install huggingface_hub")


def load_model_dir(model_id: str) -> Path:
    p = Path(model_id)
    if p.is_dir():
        return p
    print(f"Downloading {model_id}...", file=sys.stderr)
    return Path(snapshot_download(model_id, allow_patterns=[
        "*.safetensors", "*.json", "*.txt", "tokenizer*", "*.model",
    ]))


# ---------------------------------------------------------------------------
# Tensor name remapping -- HF OLMo -> GGUF talker.*
# ---------------------------------------------------------------------------

def map_tensor_name(hf_name: str) -> str | None:
    if hf_name.endswith(".rotary_emb.inv_freq"):
        return None

    n = hf_name

    # Top-level
    n = n.replace("model.embed_tokens.", "talker.token_embd.")
    n = n.replace("model.norm.", "talker.output_norm.")
    n = n.replace("lm_head.", "talker.output.")
    n = n.replace("model.layers.", "talker.blk.")

    # Per-layer renames
    n = n.replace(".self_attn.q_proj.", ".attn_q.")
    n = n.replace(".self_attn.k_proj.", ".attn_k.")
    n = n.replace(".self_attn.v_proj.", ".attn_v.")
    n = n.replace(".self_attn.o_proj.", ".attn_output.")
    n = n.replace(".input_layernorm.", ".attn_norm.")
    n = n.replace(".post_attention_layernorm.", ".ffn_norm.")
    n = n.replace(".mlp.gate_proj.", ".ffn_gate.")
    n = n.replace(".mlp.up_proj.", ".ffn_up.")
    n = n.replace(".mlp.down_proj.", ".ffn_down.")

    return n


# ---------------------------------------------------------------------------
# Tokenizer + OuteTTS special-token discovery
# ---------------------------------------------------------------------------

def discover_specials(model_dir: Path) -> dict:
    """Locate audio tokens and control tokens from tokenizer.json."""
    tj = model_dir / "tokenizer.json"
    if not tj.exists():
        sys.exit("no tokenizer.json -- cannot resolve OuteTTS special tokens")
    with open(tj, encoding="utf-8") as f:
        tjd = json.load(f)

    added = tjd.get("added_tokens", [])
    tok_map = {it["content"]: it["id"] for it in added}

    # Also check base vocab for some tokens
    base_vocab = tjd.get("model", {}).get("vocab", {})

    def find(name: str) -> int:
        if name in tok_map:
            return int(tok_map[name])
        if name in base_vocab:
            return int(base_vocab[name])
        return -1

    out = {}

    # Control tokens
    for name, key in [
        ("<|im_start|>", "im_start_token"),
        ("<|im_end|>", "im_end_token"),
        ("<|text_start|>", "text_start_token"),
        ("<|text_end|>", "text_end_token"),
        ("<|audio_start|>", "audio_start_token"),
        ("<|audio_end|>", "audio_end_token"),
        ("<|space|>", "space_token"),
    ]:
        tid = find(name)
        if tid < 0:
            print(f"  WARN: {name} not found in tokenizer", file=sys.stderr)
        out[key] = tid

    # Audio code tokens: <|0|> through <|4099|>
    audio_ids = []
    for i in range(4100):
        name = f"<|{i}|>"
        tid = find(name)
        if tid >= 0:
            audio_ids.append((i, tid))

    if not audio_ids:
        sys.exit("no <|0|>..<|4099|> audio tokens found -- wrong tokenizer?")

    # Verify contiguous
    first_code_idx, first_code_id = audio_ids[0]
    if first_code_idx != 0:
        sys.exit(f"audio tokens don't start at <|0|>: first={first_code_idx}")

    out["audio_token_offset"] = first_code_id
    out["audio_token_count"] = len(audio_ids)

    # Check contiguity
    for k, (idx, tid) in enumerate(audio_ids):
        if idx != k or tid != first_code_id + k:
            print(f"  WARN: audio token block not contiguous at {k}", file=sys.stderr)
            break

    # Punctuation tokens (for prompt building)
    punct_tokens = {}
    for name in ["<|period|>", "<|comma|>", "<|exclamation_mark|>",
                 "<|question_mark|>", "<|double_quote|>", "<|ellipsis|>"]:
        tid = find(name)
        if tid >= 0:
            punct_tokens[name] = tid
    out["punct_tokens"] = punct_tokens

    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Convert OuteTTS-0.3-1B to GGUF")
    ap.add_argument("--input", required=True,
                    help="HF model ID (e.g. OuteAI/OuteTTS-0.3-1B) or local dir")
    ap.add_argument("--output", required=True, help="Output GGUF path")
    ap.add_argument("--outtype", default="f16", choices=["f32", "f16", "q8_0", "q4_k"])
    args = ap.parse_args()

    model_dir = load_model_dir(args.input)

    with open(model_dir / "config.json", encoding="utf-8") as f:
        cfg = json.load(f)

    n_layers = int(cfg["num_hidden_layers"])
    d_model = int(cfg["hidden_size"])
    n_heads = int(cfg["num_attention_heads"])
    n_kv_heads = int(cfg.get("num_key_value_heads", n_heads))
    head_dim = int(cfg.get("head_dim", d_model // n_heads))
    ff_dim = int(cfg["intermediate_size"])
    vocab_size = int(cfg["vocab_size"])
    rope_theta = float(cfg.get("rope_theta", 10000.0))
    rms_norm_eps = float(cfg.get("rms_norm_eps", 1e-5))
    max_pos = int(cfg.get("max_position_embeddings", 4096))
    tie_embeddings = cfg.get("tie_word_embeddings", True)

    print(f"\nOuteTTS -- OLMo backbone")
    print(f"  LLM:           {n_layers}L  hidden={d_model}  "
          f"heads={n_heads}/{n_kv_heads}  head_dim={head_dim}  "
          f"ff={ff_dim}  vocab={vocab_size}")
    print(f"  RoPE theta:    {rope_theta}  max_pos={max_pos}")
    print(f"  RMS eps:       {rms_norm_eps}")
    print(f"  Tie embed:     {tie_embeddings}")

    specials = discover_specials(model_dir)
    print(f"  Audio tokens:  offset={specials['audio_token_offset']}  "
          f"count={specials['audio_token_count']}")
    for k in ["im_start_token", "text_start_token", "text_end_token",
              "audio_start_token", "audio_end_token", "space_token"]:
        print(f"  {k}: {specials.get(k, '?')}")

    quant_map = {
        "f32": (np.float32, GGMLQuantizationType.F32, None),
        "f16": (np.float16, GGMLQuantizationType.F16, None),
        "q8_0": (np.float32, GGMLQuantizationType.F16, GGMLQuantizationType.Q8_0),
        "q4_k": (np.float32, GGMLQuantizationType.F16, GGMLQuantizationType.Q4_K),
    }
    out_dtype, out_qt, quant_type = quant_map[args.outtype]

    st_files = sorted(model_dir.glob("*.safetensors"))
    if not st_files:
        sys.exit(f"no safetensors in {model_dir}")
    handles = [safe_open(str(f), framework="pt") for f in st_files]
    name_to_idx = {}
    for i, h in enumerate(handles):
        for k in h.keys():
            name_to_idx[k] = i
    print(f"  Safetensors:   {len(name_to_idx)} tensors in {len(st_files)} file(s)")

    out_path = Path(args.output)
    w = GGUFWriter(str(out_path), arch="outetts", use_temp_file=False)

    # ----- metadata -----------------------------------------------------
    w.add_name("outetts-0.3-1b")

    def u32(k, v): w.add_uint32(k, int(v))
    def f32(k, v): w.add_float32(k, float(v))

    u32("outetts.talker.n_layers", n_layers)
    u32("outetts.talker.d_model", d_model)
    u32("outetts.talker.n_heads", n_heads)
    u32("outetts.talker.n_kv_heads", n_kv_heads)
    u32("outetts.talker.head_dim", head_dim)
    u32("outetts.talker.ff_dim", ff_dim)
    u32("outetts.talker.vocab_size", vocab_size)
    u32("outetts.talker.max_pos", max_pos)
    f32("outetts.talker.rope_theta", rope_theta)
    f32("outetts.talker.rms_norm_eps", rms_norm_eps)

    # OuteTTS special tokens
    for key in ["im_start_token", "im_end_token", "text_start_token",
                "text_end_token", "audio_start_token", "audio_end_token",
                "space_token"]:
        if specials.get(key, -1) >= 0:
            u32(f"outetts.{key}", specials[key])

    u32("outetts.audio_token_offset", specials["audio_token_offset"])
    u32("outetts.audio_token_count", specials["audio_token_count"])

    # WavTokenizer codec info (runtime reads this to know what to expect)
    u32("outetts.wavtok.codebook_size", 4096)
    u32("outetts.wavtok.sample_rate", 24000)

    # Tokenizer -- pull vocab + merges from tokenizer.json
    tj = model_dir / "tokenizer.json"
    if not tj.exists():
        sys.exit("no tokenizer.json")
    with open(tj, encoding="utf-8") as f:
        tjd = json.load(f)

    model_block = tjd.get("model", {})
    base_vocab = model_block.get("vocab", {})
    added = tjd.get("added_tokens", [])

    max_id = 0
    if base_vocab:
        max_id = max(int(v) for v in base_vocab.values())
    if added:
        max_id = max(max_id, max(int(it["id"]) for it in added))

    toks = [""] * (max_id + 1)
    for tok, idx in base_vocab.items():
        idx = int(idx)
        if idx < len(toks):
            toks[idx] = tok
    for it in added:
        idx = int(it["id"])
        if idx < len(toks):
            toks[idx] = str(it["content"])
    w.add_token_list(toks)
    print(f"  Tokens:        {len(toks)} entries")

    merges = model_block.get("merges", [])
    if merges:
        if merges and isinstance(merges[0], list):
            merges = [" ".join(p) for p in merges]
        w.add_token_merges(merges)
        print(f"  Merges:        {len(merges)} entries")

    # ----- tensors ------------------------------------------------------
    n_mapped = 0
    n_skipped = 0
    seen_lm_head = False
    for hf_name in sorted(name_to_idx.keys()):
        gn = map_tensor_name(hf_name)
        if gn is None:
            n_skipped += 1
            continue
        if not gn.startswith("talker."):
            n_skipped += 1
            continue

        # Handle tied embeddings: skip lm_head.weight if tie_word_embeddings
        if gn == "talker.output.weight" and tie_embeddings:
            seen_lm_head = True
            n_skipped += 1
            continue

        t = handles[name_to_idx[hf_name]].get_tensor(hf_name).to(torch.float32).numpy()
        if t.ndim <= 1:
            t = np.ascontiguousarray(t.astype(np.float32))
            w.add_tensor(gn, t, raw_dtype=GGMLQuantizationType.F32)
        elif quant_type is not None:
            t = np.ascontiguousarray(t.astype(np.float32))
            try:
                tq = gguf.quantize(t, quant_type)
                w.add_tensor(gn, tq, raw_dtype=quant_type)
            except Exception as e:
                print(f"  [WARN] {gn}: quantize failed ({e}), keeping F16", file=sys.stderr)
                t = np.ascontiguousarray(t.astype(np.float16))
                w.add_tensor(gn, t, raw_dtype=GGMLQuantizationType.F16)
        else:
            t = np.ascontiguousarray(t.astype(out_dtype))
            w.add_tensor(gn, t, raw_dtype=out_qt)
        n_mapped += 1
        if n_mapped <= 30 or n_mapped % 50 == 0:
            print(f"  [{n_mapped}] {gn:55s} {t.shape}  {t.dtype}")

    if tie_embeddings and seen_lm_head:
        print("  (lm_head.weight skipped -- tied to embed_tokens)")

    print(f"\nMapped: {n_mapped}, skipped: {n_skipped}")
    print(f"Writing {out_path}...")
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()

    sz = out_path.stat().st_size / 1e9
    print(f"Done: {out_path}  ({sz:.2f} GB, {n_mapped} tensors)")


if __name__ == "__main__":
    main()
