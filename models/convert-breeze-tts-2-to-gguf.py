#!/usr/bin/env python3
"""Convert BreezeBlue/Breeze-TTS-2 HF safetensors -> GGUF for the CrispASR
`breeze-tts-2` backend (issue #412, PHASE 1).

Breeze TTS 2 is a **CSM (Sesame) fork** with three swaps: a T5Gemma2 text
encoder in front, a genuine Qwen3 backbone, and the Qwen3-TTS-Tokenizer-12Hz
codec instead of Mimi. The GGUF layout below therefore MIRRORS
`models/convert-csm-to-gguf.py` wherever the architecture is literally the
same (`backbone.*` / `depth.*`), and adds a `te.*` block for the text encoder
in the naming style of `models/convert-gemma4-e2b-to-gguf.py`.

  ┌ text_encoder (T5Gemma2, 26L, 1.00 B)   -> te.*
  │   26 layers, d=1152, 4H / 1KVH, head_dim 256, ff 6912
  │   BIDIRECTIONAL (see "Attention pattern" below), alternating
  │   sliding(512, symmetric)/full, TWO RoPE configs
  ├ text_encoder_proj (Linear 1152->2048)  -> te_proj.weight
  ├ backbone (Qwen3, 28L, 1.41 B)          -> backbone.*
  │   hparams come from the NESTED config["backbone_config"], NOT the
  │   top-level decoys (see "Decoy hparams" below)
  ├ depth_decoder (12L, 434 M)             -> depth.*
  │   + codebooks_head [15, 1024, 2051] split into 15 2-D heads
  └ codec  -> NOT IN THIS GGUF. The bundled audio_tokenizer/ is
     bit-identical to Qwen/Qwen3-TTS-Tokenizer-12Hz, which CrispASR already
     ships as `cstr/qwen3-tts-tokenizer-12hz-GGUF`. Wire it in as a registry
     companion (cf. the `qwen3-tts` entry, crispasr_model_registry.cpp:726).

DROPPED (1.16 GB of inference-dead weight):
  embed_text_tokens.weight  262158 x 2048 = 537 M params (1.07 GB bf16)
      Used ONLY on the no-text-encoder fallback path (breeze.py:1578); with
      a text_encoder present, `convert_input_ids_to_embeds` builds
      inputs_embeds from zeros and writes the projected text-encoder output
      into the text positions (breeze.py:1452-1458).
  codec_model.*             350 tensors, 96 M params (Mimi training leftover)
      `codec_model.decode` is the `audio_tokenizer is None` fallback only
      (generation_breeze.py:1323-1336); runtime.py:94-105 always loads the
      bundled Qwen3TTSTokenizer.

Decoy hparams (config.json top level vs. what the code actually reads):
  top-level rope_theta 500000        -> DECOY. breeze_backbone_factory.py:129
  top-level rope_scaling llama3         `AutoConfig.for_model(**backbone_config)`
  top-level rms_norm_eps 1e-5           then Qwen3RMSNorm(eps=llm_config.
  top-level max_position_embeddings      rms_norm_eps) at :178 -> 1e-6, and
                                         Qwen3RotaryEmbedding(config=llm_config)
                                         at :181 -> theta 1e6, scaling null.
  The top-level values ARE live for the depth decoder's sibling classes and
  for BreezeBackboneModelEmbeddings (hidden_size / num_codebooks /
  vocab_size / audio_embed_size), so they are emitted separately.

Attention pattern (text encoder) — the single most load-bearing correction:
  `use_bidirectional_attention: false` in the config is a MISNOMER. The
  registered implementation (breeze_config.py:19-20 registers the LOCAL
  models/t5gemma2_compat.py, not HF's) sets `is_causal = False`
  (t5gemma2_compat.py:429) and passes `causal=False` to flash-attn
  (:395, :407). `_build_additive_attention_mask` (:686-731) returns None for
  full_attention layers with no padding — i.e. fully bidirectional — and for
  sliding layers builds a SYMMETRIC window:
      left_window  = (sliding_window + 1) // 2 = 256   (dist in [0, 255])
      right_window = sliding_window // 2 + 1  = 257    (dist in [-256, -1])
  so a query attends to [i-255, i+256]. The flash path uses the equivalent
  `window_size = (255, 256)` (:371-373).

Usage:
    python models/convert-breeze-tts-2-to-gguf.py \
        --input /mnt/storage/gguf-models/breeze-tts-2-src \
        --output /mnt/storage/gguf-models/breeze-tts-2-f16.gguf \
        --outtype f16 --tmpdir /mnt/volume1/tmp-overflow

Tensors are converted ONE AT A TIME straight from the safetensors mmap
(bf16 -> f16 with no f32 round trip), and GGUFWriter spills to --tmpdir, so
peak RSS stays ~1.3 GB and the box never needs to hold the 6.97 GB
checkpoint. NEVER let the temp file land on /tmp (tmpfs) — pass --tmpdir.

Expected artifacts:
    breeze-tts-2-f16.gguf   ~5.7 GB   (this script, --outtype f16)
    breeze-tts-2-q4_k.gguf  ~1.7 GB   (llama-quantize / tools/quantize-gguf,
                                       run on Kaggle; ~2.85 B live params)
License: NON-COMMERCIAL. See the license KVs written below; redistribution
requires the LICENSE copy + NOTICE + "Derived from Breeze TTS 2 ..." line.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

# EU AI Act Art. 50(4): whose voice this checkpoint's preset speakers are.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _speaker_identity_arg import add_speaker_identity_arg, stamp_speaker_identity  # noqa: E402

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


ARCH = "breeze-tts-2"

# ---------------------------------------------------------------------------
# License — BreezeBlue Research and Non-Commercial License Agreement v1.1.
# §4(c) requires the NOTICE verbatim; §4(d) requires the "Derived from" line.
# Both are embedded so a GGUF that escapes its repo still carries the terms.
# ---------------------------------------------------------------------------
LICENSE_ID = "other"
LICENSE_NAME = "BreezeBlue Research and Non-Commercial License Agreement (Version 1.1)"
LICENSE_LINK = "https://huggingface.co/BreezeBlue/Breeze-TTS-2/blob/main/LICENSE"
LICENSE_NOTICE = (
    "Breeze TTS 2 is licensed under the BreezeBlue Research and Non-Commercial "
    "License Agreement. Copyright (c) 2026 RESONIA, INC. All Rights Reserved."
)
LICENSE_DERIVED_FROM = (
    "Derived from Breeze TTS 2 by BreezeBlue and licensed for research and "
    "non-commercial use only."
)
LICENSE_SUMMARY = (
    "other - NON-COMMERCIAL use only (BreezeBlue Research and Non-Commercial "
    "License Agreement v1.1, RESONIA INC; quantization is a Derivative Model "
    "under LICENSE §1.3, so this GGUF inherits the NC terms; hosting-as-a-service "
    "is commercial under §1.7(b))"
)

# ---------------------------------------------------------------------------
# Tensor families that are inference-dead and must NOT reach the GGUF.
# ---------------------------------------------------------------------------
DROP_EXACT = {
    # 262158 x 2048, 537 M params / 1.07 GB bf16. Fallback-only (breeze.py:1578).
    "embed_text_tokens.weight",
}
DROP_PREFIX = (
    # 350 tensors, 96 M params. Mimi training leftover; decode goes through
    # the Qwen3-TTS tokenizer (generation_breeze.py:1323-1336).
    "codec_model.",
)


def is_dropped(name: str) -> bool:
    return name in DROP_EXACT or name.startswith(DROP_PREFIX)


# ---------------------------------------------------------------------------
# Tensor name remapping: HuggingFace -> GGUF
#
# `backbone.*` / `depth.*` mirror models/convert-csm-to-gguf.py exactly (CSM
# is the literal parent), with two additions Breeze needs and CSM does not:
#   - backbone q_norm / k_norm  (Qwen3 per-head QK-RMSNorm; the CSM backbone
#     is Llama and has none)
#   - depth.cb_head.{i}.weight  (15 separate 2-D heads instead of CSM's one
#     3-D [n_cb-1, d, vocab] tensor — see split_codebooks_head())
# `te.*` follows the gemma4-e2b converter's flattened-suffix style.
# ---------------------------------------------------------------------------

_ATTN_MLP_COMMON = (
    (".self_attn.q_proj.", ".attn_q."),
    (".self_attn.k_proj.", ".attn_k."),
    (".self_attn.v_proj.", ".attn_v."),
    (".self_attn.o_proj.", ".attn_output."),
    (".self_attn.q_norm.", ".attn_q_norm."),
    (".self_attn.k_norm.", ".attn_k_norm."),
    (".mlp.gate_proj.", ".ffn_gate."),
    (".mlp.up_proj.", ".ffn_up."),
    (".mlp.down_proj.", ".ffn_down."),
)

# Llama/Qwen3-style block norms (backbone + depth decoder).
_LLAMA_NORMS = (
    (".input_layernorm.", ".attn_norm."),
    (".post_attention_layernorm.", ".ffn_norm."),
)

# Gemma-3-shaped block norms (text encoder: pre AND post around both
# sublayers — t5gemma2_compat.py:510-554).
_GEMMA_NORMS = (
    (".pre_self_attn_layernorm.", ".attn_pre_norm."),
    (".post_self_attn_layernorm.", ".attn_post_norm."),
    (".pre_feedforward_layernorm.", ".ffn_pre_norm."),
    (".post_feedforward_layernorm.", ".ffn_post_norm."),
)


def _apply(name: str, rules) -> str:
    for a, b in rules:
        name = name.replace(a, b)
    return name


def map_tensor_name(hf_name: str) -> str | None:
    """HF tensor name -> GGUF name. None = skip (dropped or non-parameter)."""
    if is_dropped(hf_name):
        return None
    if ".rotary_emb." in hf_name or hf_name.endswith(".inv_freq"):
        return None
    if hf_name.endswith(".audio_tokens_offsets"):
        # Non-persistent buffer (breeze.py:792-796): arange(16) * 2051. The
        # runtime bakes it in; see `breeze.audio_tokens_offsets` KV below.
        return None

    n = hf_name

    # ---- Text encoder (T5Gemma2) ----
    if n == "text_encoder.embed_tokens.weight":
        return "te.token_embd.weight"
    if n == "text_encoder.embed_tokens.eoi_embedding":
        # [1152] override row substituted wherever input_id == eoi_token_index
        # (t5gemma2_compat.py:579-587). NOT an extra vocab row.
        return "te.eoi_embd"
    if n == "text_encoder.norm.weight":
        return "te.output_norm.weight"
    if n == "text_encoder_proj.weight":
        return "te_proj.weight"
    if n.startswith("text_encoder.layers."):
        n = n.replace("text_encoder.layers.", "te.blk.")
        return _apply(_apply(n, _GEMMA_NORMS), _ATTN_MLP_COMMON)

    # ---- Backbone (Qwen3) ----
    if n == "backbone_model.embed_tokens.embed_audio_tokens.weight":
        # Normally absent: tied to depth_decoder.model.embed_tokens by
        # `_tied_weights_keys` (breeze.py:913-917) under
        # config.tie_codebooks_embeddings. Handled in main().
        return "backbone.audio_embd.weight"
    if n == "backbone_model.norm.weight":
        return "backbone.output_norm.weight"
    if n == "lm_head.weight":
        # [2052, 2048] — vocab_size + 1; row 2051 is the backbone EOS class
        # (breeze.py:922-923).
        return "backbone.codebook0_head.weight"
    if n.startswith("backbone_model.layers."):
        n = n.replace("backbone_model.layers.", "backbone.blk.")
        return _apply(_apply(n, _LLAMA_NORMS), _ATTN_MLP_COMMON)

    # ---- Depth decoder ----
    if n == "depth_decoder.model.embed_tokens.weight":
        return "backbone.audio_embd.weight"  # tied; see main()
    if n == "depth_decoder.model.inputs_embeds_projector.weight":
        return "depth.projection.weight"  # 2048 -> 1024 (breeze.py:487-489)
    if n == "depth_decoder.model.norm.weight":
        return "depth.output_norm.weight"
    if n == "depth_decoder.codebooks_head.weight":
        return "depth.cb_head"  # sentinel; split in main()
    if n.startswith("depth_decoder.model.layers."):
        n = n.replace("depth_decoder.model.layers.", "depth.blk.")
        return _apply(_apply(n, _LLAMA_NORMS), _ATTN_MLP_COMMON)

    print(f"  WARN: unmapped tensor: {hf_name}", file=sys.stderr)
    return None


# ---------------------------------------------------------------------------
# Numeric helpers
# ---------------------------------------------------------------------------

F16_MAX = 65504.0


def to_out(t: "torch.Tensor", out_dtype, name: str):
    """bf16/f32 torch tensor -> contiguous numpy in the output dtype.

    1-D tensors (all the RMSNorm gains + eoi_embd) stay F32: they are tiny
    and the Gemma-style `1 + w` norm is sensitive to rounding.
    Anything that would overflow f16 falls back to F32 rather than emitting
    inf — a silent inf here would surface as NaN logits three stages later.
    """
    if t.ndim <= 1:
        return np.ascontiguousarray(t.to(torch.float32).numpy()), GGMLQuantizationType.F32
    if out_dtype is np.float32:
        return np.ascontiguousarray(t.to(torch.float32).numpy()), GGMLQuantizationType.F32
    amax = float(t.abs().max().to(torch.float32))
    if amax > F16_MAX:
        print(f"  NOTE: |{name}|max={amax:.1f} > f16 max; keeping F32", file=sys.stderr)
        return np.ascontiguousarray(t.to(torch.float32).numpy()), GGMLQuantizationType.F32
    return np.ascontiguousarray(t.to(torch.float16).numpy()), GGMLQuantizationType.F16


# ---------------------------------------------------------------------------
# codebooks_head split
# ---------------------------------------------------------------------------

def split_codebooks_head(t: "torch.Tensor"):
    """[num_codebooks-1, hidden, vocab] -> 15 x [vocab, hidden] linear weights.

    BreezeCodebooksHead.forward (breeze.py:612-628) does
        F.linear(h[:, i, :], weight[i].T)
    i.e. `out = h @ weight[i]` with weight[i] of shape (hidden, vocab). The
    standard nn.Linear storage for that map is (vocab, hidden), so each slice
    is TRANSPOSED here. Emitting it untransposed is the obvious way to get a
    silently wrong depth decoder that still produces plausible-sounding audio.
    """
    n_heads, hidden, vocab = t.shape
    for i in range(n_heads):
        yield i, t[i].transpose(0, 1).contiguous(), (vocab, hidden)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Convert BreezeBlue/Breeze-TTS-2 to GGUF")
    ap.add_argument("--input", required=True,
                    help="Local dir with config.json + model-*.safetensors")
    ap.add_argument("--output", required=True, help="Output GGUF path")
    ap.add_argument("--outtype", default="f16", choices=["f16", "f32"])
    ap.add_argument("--tmpdir", default=None,
                    help="Scratch dir for the GGUF writer spill file. MUST NOT be "
                         "/tmp on the CrispASR VPS (tmpfs). Default: output's dir.")
    ap.add_argument("--no-tokenizer", action="store_true",
                    help="Skip embedding the 262158-entry Gemma vocab (debug only)")
    add_speaker_identity_arg(ap)
    args = ap.parse_args()

    model_dir = Path(args.input)
    if not model_dir.is_dir():
        sys.exit(f"--input must be a local directory: {model_dir}")
    out_path = Path(args.output)

    # Keep the writer's spill file off tmpfs.
    tmpdir = Path(args.tmpdir) if args.tmpdir else out_path.parent
    tmpdir.mkdir(parents=True, exist_ok=True)
    tempfile.tempdir = str(tmpdir)

    with open(model_dir / "config.json", encoding="utf-8") as f:
        cfg = json.load(f)

    te = cfg["text_encoder_config"]
    bb = cfg["backbone_config"]          # <- the REAL backbone hparams
    dd = cfg["depth_decoder_config"]
    spec = cfg.get("text_encoder_special_tokens_config", {})

    # -- text encoder --------------------------------------------------------
    te_layer_types = list(te["layer_types"])          # 26 entries, config.json
    te_is_full = [1 if t == "full_attention" else 0 for t in te_layer_types]
    te_rope = te["rope_parameters"]
    te_rope_full = te_rope["full_attention"]
    te_rope_slide = te_rope["sliding_attention"]
    te_sw = int(te["sliding_window"])                  # 512
    # t5gemma2_compat.py:712-720 — SYMMETRIC, not causal.
    te_win_left = (te_sw + 1) // 2                     # 256 -> dist in [0, 255]
    te_win_right = te_sw // 2 + 1                      # 257 -> dist in [-256, -1]

    # -- depth decoder -------------------------------------------------------
    dd_rs = dd.get("rope_scaling") or {}

    print(f"\n{ARCH} conversion")
    print(f"  Text encoder:  {te['num_hidden_layers']}L  d={te['hidden_size']}  "
          f"{te['num_attention_heads']}H/{te['num_key_value_heads']}KVH  "
          f"head_dim={te['head_dim']}  ff={te['intermediate_size']}  "
          f"vocab={te['vocab_size']}")
    print(f"                 sliding_window={te_sw} (symmetric: -{te_win_left - 1}..+{te_win_right - 1})  "
          f"full layers at {[i for i, v in enumerate(te_is_full) if v]}")
    print(f"                 rope sliding: {te_rope_slide['rope_type']} theta="
          f"{te_rope_slide['rope_theta']} | full: {te_rope_full['rope_type']} "
          f"theta={te_rope_full['rope_theta']} factor={te_rope_full.get('factor')}")
    print(f"                 BIDIRECTIONAL (is_causal=False, t5gemma2_compat.py:429)")
    print(f"  Backbone:      {bb['num_hidden_layers']}L  d={bb['hidden_size']}  "
          f"{bb['num_attention_heads']}H/{bb['num_key_value_heads']}KVH  "
          f"head_dim={bb['head_dim']}  ff={bb['intermediate_size']}  "
          f"theta={bb['rope_theta']}  eps={bb['rms_norm_eps']}  "
          f"scaling={bb['rope_scaling']}  (nested config, NOT the top-level decoys)")
    print(f"  Depth decoder: {dd['num_hidden_layers']}L  d={dd['hidden_size']}  "
          f"{dd['num_attention_heads']}H/{dd['num_key_value_heads']}KVH  "
          f"head_dim={dd['head_dim']}  ff={dd['intermediate_size']}  "
          f"theta={dd['rope_theta']}  llama3 factor={dd_rs.get('factor')} "
          f"orig_max_pos={dd_rs.get('original_max_position_embeddings')}")
    print(f"  Audio:         {cfg['num_codebooks']} codebooks x vocab "
          f"{cfg['audio_vocab_size']}  (backbone emits cb0, depth runs "
          f"{cfg['num_codebooks'] - 1} steps)")
    print(f"  Codec:         NOT bundled — companion "
          f"cstr/qwen3-tts-tokenizer-12hz-GGUF (bit-identical weights)")

    out_dtype = np.float16 if args.outtype == "f16" else np.float32

    # --- open shards --------------------------------------------------------
    index_file = model_dir / "model.safetensors.index.json"
    if index_file.exists():
        with open(index_file, encoding="utf-8") as f:
            weight_map = json.load(f)["weight_map"]
        st_files = [model_dir / f for f in sorted(set(weight_map.values()))]
    else:
        st_files = sorted(model_dir.glob("*.safetensors"))
    st_files = [p for p in st_files if p.parent == model_dir]  # skip audio_tokenizer/
    if not st_files:
        sys.exit(f"no safetensors in {model_dir}")

    handles = [safe_open(str(p), framework="pt") for p in st_files]
    name_to_h = {}
    for h in handles:
        for k in h.keys():
            name_to_h[k] = h
    print(f"  Safetensors:   {len(name_to_h)} tensors in {len(st_files)} shard(s)")

    w = GGUFWriter(str(out_path), arch=ARCH, use_temp_file=True)

    # ================= metadata =================
    w.add_name("breeze-tts-2")
    w.add_description(
        "Breeze TTS 2 (BreezeBlue/RESONIA) — CSM fork: T5Gemma2 text encoder + "
        "Qwen3 backbone + 12L depth decoder over 16 codebooks. Codec is the "
        "separate qwen3-tts-tokenizer-12hz GGUF."
    )
    w.add_source_url("https://huggingface.co/BreezeBlue/Breeze-TTS-2")
    w.add_repo_url("https://github.com/breezeblue-ai/breeze-tts")
    w.add_languages(["en", "zh"])
    stamp_speaker_identity(w, args)

    # ---- license (NON-COMMERCIAL) ----
    w.add_license(LICENSE_ID)
    w.add_license_name(LICENSE_NAME)
    w.add_license_link(LICENSE_LINK)
    w.add_string("breeze.license.notice", LICENSE_NOTICE)
    w.add_string("breeze.license.derived_from", LICENSE_DERIVED_FROM)
    w.add_string("breeze.license.summary", LICENSE_SUMMARY)
    w.add_bool("breeze.license.non_commercial", True)

    def u32(k, v):
        w.add_uint32(k, int(v))

    def f32(k, v):
        w.add_float32(k, float(v))

    def b(k, v):
        w.add_bool(k, bool(v))

    def s(k, v):
        w.add_string(k, str(v))

    # ---- text encoder ----
    u32("breeze.te.n_layers", te["num_hidden_layers"])
    u32("breeze.te.d_model", te["hidden_size"])
    u32("breeze.te.n_heads", te["num_attention_heads"])
    u32("breeze.te.n_kv_heads", te["num_key_value_heads"])
    u32("breeze.te.head_dim", te["head_dim"])
    u32("breeze.te.ff_dim", te["intermediate_size"])
    u32("breeze.te.vocab_size", te["vocab_size"])
    u32("breeze.te.max_pos", te["max_position_embeddings"])
    f32("breeze.te.rms_norm_eps", te["rms_norm_eps"])
    s("breeze.te.hidden_activation", te["hidden_activation"])
    # RMSNorm is Gemma-style: out = x_normed * (1 + w). t5gemma2_compat.py:125.
    b("breeze.te.norm_unit_offset", True)
    # embed_scale = sqrt(hidden_size), applied on lookup. t5gemma2_compat.py:606.
    f32("breeze.te.embed_scale", float(te["hidden_size"]) ** 0.5)
    u32("breeze.te.eoi_token_index", te["eoi_token_index"])
    u32("breeze.te.pad_token_id", te["pad_token_id"])
    # attn scale = query_pre_attn_scalar ** -0.5, NOT head_dim ** -0.5
    # (t5gemma2_compat.py:427). Here they happen to coincide (both 256), but
    # do not collapse them — a future checkpoint can decouple them.
    u32("breeze.te.query_pre_attn_scalar", te["query_pre_attn_scalar"])
    f32("breeze.te.attn_scale", float(te["query_pre_attn_scalar"]) ** -0.5)
    b("breeze.te.qk_norm", True)         # per-head RMSNorm on q and k (:454-455)
    # --- attention pattern ---
    b("breeze.te.causal", False)         # is_causal=False (:431), causal=False (:394,:406)
    u32("breeze.te.sliding_window", te_sw)
    u32("breeze.te.sliding_window_left", te_win_left)    # 256
    u32("breeze.te.sliding_window_right", te_win_right)  # 257
    # 1 = full_attention, 0 = sliding_attention; indexed by layer.
    w.add_array("breeze.te.layer_types", te_is_full)
    u32("breeze.te.n_full_layers", sum(te_is_full))
    # --- dual RoPE ---
    s("breeze.te.rope_type_sliding", te_rope_slide.get("rope_type", "default"))
    f32("breeze.te.rope_theta_sliding", te_rope_slide["rope_theta"])
    f32("breeze.te.rope_factor_sliding", te_rope_slide.get("factor", 1.0))
    s("breeze.te.rope_type_full", te_rope_full.get("rope_type", "default"))
    f32("breeze.te.rope_theta_full", te_rope_full["rope_theta"])
    f32("breeze.te.rope_factor_full", te_rope_full.get("factor", 1.0))
    # "linear" here means inv_freq /= factor with attention_scaling == 1.0
    # (t5gemma2_compat.py:177, :182). It is NOT llama3/yarn scaling.
    b("breeze.te.rope_linear_divides_inv_freq", True)
    f32("breeze.te.rope_attn_scaling", 1.0)

    # ---- text encoder projection ----
    s("breeze.te_proj.type", cfg.get("text_encoder_proj_type", "linear"))
    u32("breeze.te_proj.in_dim", te["hidden_size"])
    u32("breeze.te_proj.out_dim", cfg["hidden_size"])
    # text_encoder_feature_layer_idx defaults to (-1,) and
    # text_encoder_layer_projs is absent from this checkpoint -> the DimFusion
    # path is inert. Recorded so the runtime can assert it.
    fli = cfg.get("text_encoder_feature_layer_idx", -1)
    fli = [fli] if isinstance(fli, int) else list(fli)
    w.add_array("breeze.te.feature_layer_idx", [int(x) for x in fli])
    b("breeze.te.dimfusion", False)
    f32("breeze.te.bucket_max_length_ratio",
        cfg.get("text_encoder_bucket_max_length_ratio", 4.0))
    # Each text SEGMENT is encoded as its own padded row — no cross-segment
    # attention (breeze.py:1418-1436). The C++ port must do the same.
    b("breeze.te.encode_segments_separately", True)

    # ---- backbone (Qwen3, NESTED config) ----
    u32("breeze.bb.n_layers", bb["num_hidden_layers"])
    u32("breeze.bb.d_model", bb["hidden_size"])
    u32("breeze.bb.n_heads", bb["num_attention_heads"])
    u32("breeze.bb.n_kv_heads", bb["num_key_value_heads"])
    u32("breeze.bb.head_dim", bb["head_dim"])
    u32("breeze.bb.ff_dim", bb["intermediate_size"])
    u32("breeze.bb.max_pos", bb["max_position_embeddings"])
    f32("breeze.bb.rope_theta", bb["rope_theta"])            # 1e6, not 5e5
    f32("breeze.bb.rms_norm_eps", bb["rms_norm_eps"])        # 1e-6, not 1e-5
    b("breeze.bb.rope_scaling", bool(bb.get("rope_scaling")))  # false
    b("breeze.bb.qk_norm", True)                             # Qwen3 q_norm/k_norm
    b("breeze.bb.causal", True)
    b("breeze.bb.attention_bias", bool(bb.get("attention_bias", False)))
    s("breeze.bb.model_type", cfg.get("backbone_model_type", "qwen3"))
    # Decoys kept ONLY as provenance so a reader can see they were considered.
    f32("breeze.bb.decoy_top_level_rope_theta", cfg.get("rope_theta", 0.0))
    f32("breeze.bb.decoy_top_level_rms_norm_eps", cfg.get("rms_norm_eps", 0.0))

    # ---- backbone audio embedding (top-level config drives this) ----
    u32("breeze.audio_embed_size", cfg["audio_embed_size"])
    # audio_embed_size == hidden_size == 2048 -> audio_embeds_projector is
    # ABSENT (breeze.py:786-791). Assert this at load.
    b("breeze.audio_embeds_projector",
      int(cfg["audio_embed_size"]) != int(cfg["hidden_size"]))
    # audio_tokens_offsets = arange(num_codebooks) * vocab_size (breeze.py:792-796)
    w.add_array("breeze.audio_tokens_offsets",
                [i * int(cfg["audio_vocab_size"]) for i in range(int(cfg["num_codebooks"]))])
    b("breeze.tie_codebooks_embeddings", bool(cfg.get("tie_codebooks_embeddings", True)))

    # ---- depth decoder ----
    u32("breeze.dd.n_layers", dd["num_hidden_layers"])
    u32("breeze.dd.d_model", dd["hidden_size"])
    u32("breeze.dd.n_heads", dd["num_attention_heads"])
    u32("breeze.dd.n_kv_heads", dd["num_key_value_heads"])
    u32("breeze.dd.head_dim", dd["head_dim"])
    u32("breeze.dd.ff_dim", dd["intermediate_size"])
    u32("breeze.dd.max_pos", dd["max_position_embeddings"])     # 33
    u32("breeze.dd.backbone_hidden", dd["backbone_hidden_size"])
    u32("breeze.dd.audio_embed_size", dd["audio_embed_size"])
    u32("breeze.dd.vocab_size", dd["vocab_size"])
    f32("breeze.dd.rope_theta", dd["rope_theta"])               # 5e5
    f32("breeze.dd.rms_norm_eps", dd["rms_norm_eps"])           # 1e-5
    b("breeze.dd.qk_norm", False)                               # no q_norm/k_norm
    s("breeze.dd.rope_scaling_type", dd_rs.get("rope_type", "none"))
    f32("breeze.dd.rope_scaling_factor", dd_rs.get("factor", 1.0))
    f32("breeze.dd.rope_low_freq_factor", dd_rs.get("low_freq_factor", 1.0))
    f32("breeze.dd.rope_high_freq_factor", dd_rs.get("high_freq_factor", 1.0))
    u32("breeze.dd.rope_orig_max_pos",
        dd_rs.get("original_max_position_embeddings", 16))
    # backbone_hidden_size == audio_embed_size -> backbone_hidden_state_projector
    # is None (breeze.py:492-496); the backbone hidden state is written into
    # inputs_embeds[:, 0] unprojected (breeze.py:556-561).
    b("breeze.dd.backbone_hidden_projector",
      int(dd["backbone_hidden_size"]) != int(dd["audio_embed_size"]))
    # embed lookup offset: codebook_idx = clamp(cache_position - 1, min=0),
    # offset = codebook_idx * vocab_size (breeze.py:550-551).
    b("breeze.dd.embed_offset_by_codebook", True)
    u32("breeze.dd.n_codebook_heads", int(cfg["num_codebooks"]) - 1)

    # ---- audio / vocab / special tokens ----
    u32("breeze.num_codebooks", cfg["num_codebooks"])
    u32("breeze.audio_vocab_size", cfg["audio_vocab_size"])
    u32("breeze.text_vocab_size", cfg["text_vocab_size"])
    u32("breeze.hidden_size", cfg["hidden_size"])
    u32("breeze.audio_token_id", cfg["audio_token_id"])           # 262144 <|AUDIO|>
    u32("breeze.audio_eos_token_id", cfg["audio_eos_token_id"])   # 262145 <|audio_eos|>
    u32("breeze.codebook_pad_token_id", cfg["codebook_pad_token_id"])  # 2050
    u32("breeze.codebook_eos_token_id", cfg["codebook_eos_token_id"])  # 0
    # lm_head has vocab_size + 1 rows; the extra class IS the backbone EOS
    # (breeze.py:922-923).
    u32("breeze.backbone_eos_token_id", cfg["audio_vocab_size"])
    u32("breeze.lm_head_out", int(cfg["audio_vocab_size"]) + 1)
    u32("breeze.bos_token_id", cfg["bos_token_id"])
    u32("breeze.eos_token_id", cfg["eos_token_id"])
    u32("breeze.pad_token_id", cfg["pad_token_id"])
    # Codec ids >= codec_config.codebook_size (2048) are RESERVED and masked
    # out of the sampler (generation_breeze.py:125-131).
    u32("breeze.codec_codebook_size", cfg["codec_config"]["codebook_size"])  # 2048
    u32("breeze.reserved_codec_id_lo", cfg["codec_config"]["codebook_size"])
    u32("breeze.reserved_codec_id_hi", cfg["audio_vocab_size"])              # exclusive

    # ---- prompt template special tokens (breeze_infer/templates.py) ----
    tok_ids = spec.get("token_ids", {})
    speaker_names = list(spec.get("speaker_tokens", []))
    if speaker_names:
        w.add_array("breeze.speaker_token_ids",
                    [int(tok_ids[t]) for t in speaker_names])
        w.add_array("breeze.speaker_token_names", speaker_names)
    ins = spec.get("instruction_tokens", {})
    if ins:
        u32("breeze.ins_bos_token_id", tok_ids[ins["bos"]])   # 262156
        u32("breeze.ins_eos_token_id", tok_ids[ins["eos"]])   # 262157
        s("breeze.ins_bos_token", ins["bos"])
        s("breeze.ins_eos_token", ins["eos"])
    s("breeze.audio_tag", "<|AUDIO|>")          # templates.py:12
    s("breeze.audio_eos_tag", "<|audio_eos|>")  # templates.py:13
    # LoRA r=8 + the 12 added tokens are already merged (config.json
    # text_encoder_lora_config.merged_into_base / special tokens .merged_into_base).
    b("breeze.lora_merged_into_base", True)

    # ---- codec companion ----
    s("breeze.codec.arch", "qwen3-tts-tokenizer-12hz")
    s("breeze.codec.repo", "cstr/qwen3-tts-tokenizer-12hz-GGUF")
    u32("breeze.codec.sample_rate", 24000)
    f32("breeze.codec.frame_rate", 12.5)
    u32("breeze.codec.n_quantizers", 16)
    u32("breeze.codec.downsample_rate", 1920)

    # ---- sampling defaults (generation_config.json + infer.py) ----
    gc_path = model_dir / "generation_config.json"
    gc = json.loads(gc_path.read_text(encoding="utf-8")) if gc_path.exists() else {}
    f32("breeze.sampling.temperature", gc.get("temperature", 0.9))
    f32("breeze.sampling.depth_temperature", gc.get("depth_decoder_temperature", 0.9))
    u32("breeze.sampling.top_k", gc.get("top_k", 50))
    f32("breeze.sampling.top_p", gc.get("top_p", 1.0))
    u32("breeze.sampling.max_new_tokens", gc.get("max_new_tokens", 750))
    f32("breeze.sampling.repetition_penalty", 1.1)   # infer.py:26
    u32("breeze.sampling.max_seq_len", 2048)         # infer.py:25
    # CFG: up to 3 branches (uncond / ref / ins) — templates.py:99-107,
    # generation_breeze.py:822-826. Combination is
    #   logits = uncond + s_ref*(ref - uncond) + s_ins*(ins - uncond)
    u32("breeze.cfg.max_branches", 3)
    s("breeze.cfg.combine",
      "uncond + cfg_scale_ref*(ref - uncond) + cfg_scale_ins*(ins - uncond)")
    f32("breeze.cfg.default_scale", 1.0)             # infer.py:23

    # ---- tokenizer (Gemma, 262158 entries) ----
    if not args.no_tokenizer:
        tj = model_dir / "tokenizer.json"
        if tj.exists():
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
            w.add_tokenizer_model(str(model_block.get("type", "BPE")).lower())
            w.add_token_list(toks)
            print(f"  Tokens:        {len(toks)} entries")
            merges = model_block.get("merges", [])
            if merges:
                if isinstance(merges[0], list):
                    merges = [" ".join(p) for p in merges]
                w.add_token_merges(merges)
                print(f"  Merges:        {len(merges)} entries")
            w.add_bos_token_id(int(cfg["bos_token_id"]))
            w.add_eos_token_id(int(cfg["eos_token_id"]))
            w.add_pad_token_id(int(cfg["pad_token_id"]))
            # The parsed 33 MB tokenizer.json is the RAM high-water mark of the
            # whole run; drop it before the tensor loop starts.
            del tjd, model_block, base_vocab, added, toks, merges
        else:
            print("  WARN: no tokenizer.json found", file=sys.stderr)

    # ================= tensors (streamed) =================
    n_written = 0
    n_dropped = 0
    dropped_params = 0
    written_params = 0
    seen_audio_embd = False

    def emit(gn: str, t: "torch.Tensor"):
        nonlocal n_written, written_params
        arr, qt = to_out(t, out_dtype, gn)
        w.add_tensor(gn, arr, raw_dtype=qt)
        n_written += 1
        written_params += int(arr.size)
        if n_written <= 24 or n_written % 60 == 0:
            print(f"  [{n_written:4d}] {gn:44s} {tuple(arr.shape)}  {arr.dtype}")
        del arr

    for hf_name in sorted(name_to_h.keys()):
        t_meta_dropped = is_dropped(hf_name)
        gn = map_tensor_name(hf_name)
        if gn is None:
            if t_meta_dropped:
                t = name_to_h[hf_name].get_slice(hf_name)
                shp = t.get_shape()
                n = 1
                for d in shp:
                    n *= d
                dropped_params += n
                n_dropped += 1
            continue

        t = name_to_h[hf_name].get_tensor(hf_name)

        if gn == "depth.cb_head":
            # [15, 1024, 2051] -> 15 x [2051, 1024]
            for i, sl, shp in split_codebooks_head(t):
                emit(f"depth.cb_head.{i}.weight", sl)
                del sl
            del t
            continue

        if gn == "backbone.audio_embd.weight":
            if seen_audio_embd:
                # The tie means only one physical copy is stored; if both
                # names ever appear, keep the first and skip the duplicate.
                print(f"  [tied] skipping duplicate of backbone.audio_embd.weight "
                      f"({hf_name})")
                del t
                continue
            seen_audio_embd = True

        emit(gn, t)
        del t

    if not seen_audio_embd:
        sys.exit("FATAL: neither backbone_model.embed_tokens.embed_audio_tokens.weight "
                 "nor depth_decoder.model.embed_tokens.weight was found — the summed "
                 "codebook embedding is missing")

    print(f"\n  Wrote {n_written} tensors ({written_params / 1e6:.1f} M params)")
    print(f"  Dropped {n_dropped} inference-dead tensors "
          f"({dropped_params / 1e6:.1f} M params, "
          f"{dropped_params * 2 / 1e9:.2f} GB bf16)")
    print(f"  Writing {out_path} (temp spill in {tempfile.tempdir}) ...")

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file(progress=True)
    w.close()

    sz = out_path.stat().st_size / 1e9
    print(f"\nDone: {out_path}  ({sz:.2f} GB, {n_written} tensors)")
    print("Expected Q4_K path: ~1.7-1.8 GB "
          f"(~{written_params / 1e9:.2f} B live params). Quantize on a bigger box:")
    print(f"  llama-quantize {out_path} "
          f"{out_path.with_name(out_path.stem.replace('-f16', '') + '-q4_k.gguf')} Q4_K")
    print("\nREDISTRIBUTION (LICENSE §4): ship the full LICENSE copy, a NOTICE file "
          "containing")
    print(f'  "{LICENSE_NOTICE}"')
    print("and a prominent model-card line reading")
    print(f'  "{LICENSE_DERIVED_FROM}"')
    print('§4 also bars "Breeze"/"BreezeBlue" as the derivative\'s PRIMARY name.')


if __name__ == "__main__":
    main()
