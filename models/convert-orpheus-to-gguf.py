#!/usr/bin/env python3
"""
Convert canopylabs/orpheus-3b-0.1-ft (and the Kartoffel_Orpheus family)
HuggingFace safetensors → GGUF F16/F32 for the CrispASR `orpheus`
backend.

Orpheus is Llama-3.2-3B-Instruct as the talker plus the SNAC 24 kHz
codec (3 codebooks, 4096 entries each). The codec lives in a separate
HF repo (hubertsiuzdak/snac_24khz) and gets its own converter.

Architecture (from canopylabs/orpheus-3b-0.1-ft/config.json):

  Top-level Llama-3.2-3B:
    architectures        = ["LlamaForCausalLM"]
    hidden_size          = 3072
    num_hidden_layers    = 28
    num_attention_heads  = 24
    num_key_value_heads  = 8
    head_dim             = 128
    intermediate_size    = 8192
    vocab_size           = 128256 (base) — Orpheus extends with
                           added_tokens.json: 128257..(128257 + 7*4096)
                           for the <custom_token_N> codec slots, plus
                           a handful of audio markers.
    rope_theta           = 500000
    rms_norm_eps         = 1e-5

  Orpheus-specific (from added_tokens.json + decoder.py):
    audio_start_token       = 128259  (start_of_human / audio BOS)
    audio_pre_end_marker    = 128009  (Llama <|eot_id|>)
    audio_end_marker_a      = 128260
    audio_end_marker_b      = 128261
    audio_end_token         = 128257  (end_of_audio)
    custom_token_offset     = 128266  ( <custom_token_0> token id )
    custom_token_count      = 7 * 4096 = 28672
    snac_super_frame_slots  = 7   (slots / SNAC super-frame)
    snac_codebooks          = [1, 2, 4]  (slot counts per codebook,
                                          summing to 7)
    snac_codebook_size      = 4096

  Fixed speakers (from the model card):
    English ft  : tara, leah, jess, leo, dan, mia, zac, zoe
    DE Kartoffel : Jakob, Anton, Julian, Jan, Alexander, Emil, Ben,
                   Elias, Felix, Frederik, Karl, Linus, Lukas,
                   Matteo, Maximilian, Sophie, Marie, Mia, Maria,
                   Sophia, Lina, Lea, Anna, Emilia, Mila, Clara
    These are baked into the LM training data as the literal `name:`
    prefix of the prompt; there is NO embedding-table dispatch (the
    Qwen3-TTS-CustomVoice trick does not apply).

Usage:

    python models/convert-orpheus-to-gguf.py \\
        --input canopylabs/orpheus-3b-0.1-ft \\
        --output orpheus-3b-base-f16.gguf

    # or local dir, or a Kartoffel fine-tune:
    python models/convert-orpheus-to-gguf.py \\
        --input SebastianBodza/Kartoffel_Orpheus-3B_german_natural-v0.1 \\
        --output kartoffel-orpheus-de-natural-f16.gguf \\
        --speakers Jakob,Anton,Julian,Jan,Alexander,Emil,Ben,Elias,Felix,\\
Frederik,Karl,Linus,Lukas,Matteo,Maximilian,Sophie,Marie,Mia,Maria,Sophia,\\
Lina,Lea,Anna,Emilia,Mila,Clara
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

try:
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


DEFAULT_EN_SPEAKERS = ["tara", "leah", "jess", "leo", "dan", "mia", "zac", "zoe"]


def load_model_dir(model_id: str) -> Path:
    p = Path(model_id)
    if p.is_dir():
        return p
    print(f"Downloading {model_id}…", file=sys.stderr)
    return Path(snapshot_download(model_id, allow_patterns=[
        "*.safetensors", "*.json", "*.txt", "tokenizer*", "*.model",
    ]))


# ---------------------------------------------------------------------------
# Tensor name remapping — HF Llama → GGUF talker.* (mirrors qwen3_tts)
# ---------------------------------------------------------------------------

def map_tensor_name(hf_name: str) -> str | None:
    if hf_name.endswith("num_batches_tracked"):
        return None
    if hf_name.endswith(".rotary_emb.inv_freq"):
        return None  # derived at runtime from rope_theta + head_dim

    n = hf_name

    # Top-level Llama prefix — no leading "talker." in HF names.
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
# Tokenizer + Orpheus special-token discovery
# ---------------------------------------------------------------------------

def discover_orpheus_specials(model_dir: Path) -> dict:
    """Locate the <custom_token_N> block and the audio markers in the
    tokenizer. Pin everything against actual token NAMES, not guessed
    offsets — the model card's added_tokens.json is the authority."""

    added = model_dir / "added_tokens.json"
    if not added.exists():
        # Fall back to tokenizer.json's added_tokens
        tj = model_dir / "tokenizer.json"
        if not tj.exists():
            sys.exit("no added_tokens.json or tokenizer.json — cannot resolve "
                     "<custom_token_N> block")
        with open(tj) as f:
            tjd = json.load(f)
        items = tjd.get("added_tokens", [])
        # Each entry: {"id": int, "content": str, ...}
        added_map = {it["content"]: it["id"] for it in items}
    else:
        with open(added) as f:
            added_map = json.load(f)

    custom_ids = []
    for name, tid in added_map.items():
        if name.startswith("<custom_token_") and name.endswith(">"):
            try:
                idx = int(name[len("<custom_token_"):-1])
            except ValueError:
                continue
            custom_ids.append((idx, int(tid)))

    if not custom_ids:
        sys.exit("no <custom_token_N> tokens in vocab — wrong tokenizer?")

    custom_ids.sort()
    # Verify the block is contiguous, starting at custom_token_0.
    first_idx, first_id = custom_ids[0]
    if first_idx != 0:
        sys.exit(f"<custom_token_N> block does not start at 0: first={first_idx}")
    for k, (idx, tid) in enumerate(custom_ids):
        if idx != k or tid != first_id + k:
            sys.exit(f"<custom_token_N> block not contiguous at idx {k}: "
                     f"got idx={idx} id={tid}, expected idx={k} id={first_id + k}")

    expected = 7 * 4096  # 7 super-frame slots × 4096 codebook entries
    if len(custom_ids) != expected:
        print(f"  WARN: <custom_token_N> count {len(custom_ids)} != expected "
              f"{expected} (7×4096) — Orpheus 0.1 ships exactly this. Verify model.",
              file=sys.stderr)

    out = {
        "custom_token_offset": first_id,
        "custom_token_count": len(custom_ids),
    }

    for marker_name, key in [
        ("<|audio|>", "marker_audio_alias"),
        ("<custom_token_0>", "marker_custom0"),
    ]:
        if marker_name in added_map:
            out[key] = int(added_map[marker_name])

    # Fixed numeric IDs from upstream decoder.py (engine_class.py): these
    # are the start/end wrappers around the prompt, NOT speaker tokens.
    out["audio_start_token"] = 128259
    out["audio_pre_end"] = 128009  # Llama <|eot_id|>
    out["audio_end_a"] = 128260
    out["audio_end_b"] = 128261
    out["audio_end_token"] = 128257

    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Convert Orpheus-3B (and finetunes) to GGUF")
    ap.add_argument("--input", required=True,
                    help="HF model ID (e.g. canopylabs/orpheus-3b-0.1-ft) or local dir")
    ap.add_argument("--output", required=True, help="Output GGUF path")
    ap.add_argument("--outtype", default="f16", choices=["f32", "f16"])
    ap.add_argument("--speakers", default=",".join(DEFAULT_EN_SPEAKERS),
                    help="Comma-separated fixed-speaker NAMES (used as the "
                         "`name: text` prefix at synthesis time). Defaults to "
                         "the canopylabs orpheus-3b-0.1-ft 8-voice EN set.")
    ap.add_argument("--variant", default="base", choices=["base", "fixed_speaker"],
                    help="orpheus.tts_model_type metadata. Most ft checkpoints "
                         "(canopylabs orpheus-3b-0.1-ft, Kartoffel_*) are "
                         "fixed_speaker — pass `--variant fixed_speaker` to "
                         "have the runtime require --voice / qwen3_tts-style "
                         "speaker selection. Use `base` for the unfinetuned "
                         "pretrained backbone (canopylabs/orpheus-3b-0.1-pretrained).")
    args = ap.parse_args()

    model_dir = load_model_dir(args.input)

    with open(model_dir / "config.json") as f:
        cfg = json.load(f)

    n_layers = int(cfg["num_hidden_layers"])
    d_model = int(cfg["hidden_size"])
    n_heads = int(cfg["num_attention_heads"])
    n_kv_heads = int(cfg.get("num_key_value_heads", n_heads))
    head_dim = int(cfg.get("head_dim", d_model // n_heads))
    ff_dim = int(cfg["intermediate_size"])
    vocab_size = int(cfg["vocab_size"])
    rope_theta = float(cfg.get("rope_theta", 500000.0))
    rms_norm_eps = float(cfg.get("rms_norm_eps", 1e-5))
    max_pos = int(cfg.get("max_position_embeddings", 131072))

    print(f"\nOrpheus — variant={args.variant}")
    print(f"  Talker:        {n_layers}L  hidden={d_model}  "
          f"heads={n_heads}/{n_kv_heads}  head_dim={head_dim}  "
          f"ff={ff_dim}  vocab={vocab_size}")
    print(f"  RoPE theta:    {rope_theta}  max_pos={max_pos}")
    print(f"  RMS eps:       {rms_norm_eps}")

    specials = discover_orpheus_specials(model_dir)
    print(f"  Custom tokens: offset={specials['custom_token_offset']}  "
          f"count={specials['custom_token_count']}")
    print(f"  Audio markers: start={specials['audio_start_token']}  "
          f"end={specials['audio_end_token']}  "
          f"pre_end={specials['audio_pre_end']}  "
          f"end_a={specials['audio_end_a']}  end_b={specials['audio_end_b']}")

    speakers = [s.strip() for s in args.speakers.split(",") if s.strip()]
    print(f"  Speakers:      {len(speakers)} ({', '.join(speakers)})")

    out_dtype = np.float16 if args.outtype == "f16" else np.float32
    out_qt = GGMLQuantizationType.F16 if args.outtype == "f16" else GGMLQuantizationType.F32

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
    # use_temp_file=False holds tensor bytes in RAM until write_tensors_to_file
    # so the conversion only needs ~6.6 GB of free disk at the output path
    # (instead of 13 GB temp+output). Llama-3.2-3B at f16 fits comfortably in
    # the ~16-32 GB RAM available on Apple Silicon dev machines.
    w = GGUFWriter(str(out_path), arch="orpheus", use_temp_file=False)

    # ----- metadata -----------------------------------------------------
    w.add_name(f"orpheus-{args.variant}")

    def u32(k, v): w.add_uint32(k, int(v))
    def f32(k, v): w.add_float32(k, float(v))

    u32("orpheus.talker.n_layers", n_layers)
    u32("orpheus.talker.d_model", d_model)
    u32("orpheus.talker.n_heads", n_heads)
    u32("orpheus.talker.n_kv_heads", n_kv_heads)
    u32("orpheus.talker.head_dim", head_dim)
    u32("orpheus.talker.ff_dim", ff_dim)
    u32("orpheus.talker.vocab_size", vocab_size)
    u32("orpheus.talker.max_pos", max_pos)
    f32("orpheus.talker.rope_theta", rope_theta)
    f32("orpheus.talker.rms_norm_eps", rms_norm_eps)

    # SNAC slot-layout constants (also embedded in the C++ runtime — keep
    # in sync). 7 LM tokens per super-frame, slot 0 → codes_0,
    # slots 1,4 → codes_1, slots 2,3,5,6 → codes_2.
    u32("orpheus.snac.super_frame_slots", 7)
    u32("orpheus.snac.codebook_count", 3)
    u32("orpheus.snac.codebook_size", 4096)
    w.add_array("orpheus.snac.slots_per_codebook", [1, 2, 4])

    # Audio wrapper / codec slot tokens
    u32("orpheus.audio_start_token", specials["audio_start_token"])
    u32("orpheus.audio_pre_end_token", specials["audio_pre_end"])
    u32("orpheus.audio_end_a_token", specials["audio_end_a"])
    u32("orpheus.audio_end_b_token", specials["audio_end_b"])
    u32("orpheus.audio_end_token", specials["audio_end_token"])
    u32("orpheus.custom_token_offset", specials["custom_token_offset"])
    u32("orpheus.custom_token_count", specials["custom_token_count"])

    # Variant: drives runtime --voice semantics
    w.add_string("orpheus.tts_model_type", args.variant)

    # Fixed-speaker name list (string prefix at synthesis time —
    # there is no embedding-table dispatch for Orpheus speakers).
    if speakers:
        w.add_array("orpheus.spk_names", speakers)

    # Tokenizer — pull vocab + merges. Llama-3.2 ships tokenizer.json
    # (a tokenizers-rs serialization), not vocab.json/merges.txt. The
    # cleanest path is to read the merges + vocab out of tokenizer.json.
    tj = model_dir / "tokenizer.json"
    if not tj.exists():
        sys.exit("no tokenizer.json — cannot persist BPE vocab")
    with open(tj) as f:
        tjd = json.load(f)

    model_block = tjd.get("model", {})
    if model_block.get("type") not in ("BPE",):
        print(f"  WARN: tokenizer.model.type = {model_block.get('type')!r}, "
              f"expected BPE — output may not load.", file=sys.stderr)

    base_vocab = model_block.get("vocab", {})
    added = tjd.get("added_tokens", [])
    # Build id → token map, allocating to fit the highest id we see.
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
    print(f"  Tokens:        {len(toks)} entries (base + added)")

    merges = model_block.get("merges", [])
    if merges:
        # tokenizer.json merges may be ["a b", ...] or [["a","b"], ...]
        if merges and isinstance(merges[0], list):
            merges = [" ".join(p) for p in merges]
        w.add_token_merges(merges)
        print(f"  Merges:        {len(merges)} entries")

    # ----- tensors ------------------------------------------------------
    n_mapped = 0
    n_skipped = 0
    skipped_examples = []
    for hf_name in sorted(name_to_idx.keys()):
        gn = map_tensor_name(hf_name)
        if gn is None:
            n_skipped += 1
            continue
        if not gn.startswith("talker."):
            if len(skipped_examples) < 20:
                skipped_examples.append(f"  [WARN unmapped] {hf_name}")
            n_skipped += 1
            continue

        t = handles[name_to_idx[hf_name]].get_tensor(hf_name).to(torch.float32).numpy()
        if t.ndim <= 1:
            t = np.ascontiguousarray(t.astype(np.float32))
            w.add_tensor(gn, t, raw_dtype=GGMLQuantizationType.F32)
        else:
            t = np.ascontiguousarray(t.astype(out_dtype))
            w.add_tensor(gn, t, raw_dtype=out_qt)
        n_mapped += 1
        if n_mapped <= 30 or n_mapped % 100 == 0:
            print(f"  [{n_mapped}] {gn:55s} {t.shape}  {t.dtype}")

    if skipped_examples:
        print("\n".join(skipped_examples), file=sys.stderr)
        print(f"\n  WARNING: {len(skipped_examples)} unmapped — re-check map_tensor_name()",
              file=sys.stderr)

    print(f"\nMapped: {n_mapped}, skipped: {n_skipped}")
    print(f"Writing {out_path}…")
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()

    sz = out_path.stat().st_size / 1e9
    print(f"Done: {out_path}  ({sz:.2f} GB, {n_mapped} tensors)")


if __name__ == "__main__":
    main()
