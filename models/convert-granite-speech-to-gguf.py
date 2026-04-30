#!/usr/bin/env python3
"""
Convert ibm-granite/granite-4.0-1b-speech (HF safetensors) → GGUF F16.

Architecture:
  Audio encoder (granite_speech_encoder): 16-layer Conformer
    input_linear: (1024, 160)
    16 × encoder block (Macaron-style: FFN + MHSA + Conv + FFN):
      ff1: pre_norm + up_proj + down_proj
      attn: pre_norm + to_q + to_kv + to_out + rel_pos_emb
      conv: up_conv + depth_conv + batch_norm + down_conv + norm
      ff2: pre_norm + up_proj + down_proj
      post_norm
    out/out_mid: CTC head (ignored at inference)

  Projector (BLIP-2 Q-Former): 2-layer cross-attention
    query: [1, 3, 1024] learned query tokens
    2 × qformer layer:
      self-attention (Q/K/V + dense + LayerNorm)
      cross-attention (Q/K/V + dense + LayerNorm)
      intermediate_query FFN + output_query FFN + LayerNorm
    linear: (2048, 1024) final projection

  LLM (Granite 4.0-1B): 40-layer decoder
    embed_tokens: (100353, 2048)
    40 × layer: input_layernorm + self_attn(Q/K/V/O) + post_attn_layernorm + mlp(gate/up/down)
    norm + lm_head: (100353, 2048)
    μP multipliers: embedding=12.0, attention=1/128, residual=0.22, logits=8.0
"""

from __future__ import annotations
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

try:
    import gguf
except ImportError:
    sys.exit("pip install gguf")
try:
    from safetensors import safe_open
except ImportError:
    sys.exit("pip install safetensors")


# ---------------------------------------------------------------------------
# Tensor name remapping
# ---------------------------------------------------------------------------

DIRECT = {
    "encoder.input_linear.weight": "enc.input.weight",
    "encoder.input_linear.bias": "enc.input.bias",
    # CTC head (kept for completeness, could be skipped)
    "encoder.out.weight": "enc.ctc_out.weight",
    "encoder.out.bias": "enc.ctc_out.bias",
    "encoder.out_mid.weight": "enc.ctc_mid.weight",
    "encoder.out_mid.bias": "enc.ctc_mid.bias",
    # Projector top-level
    "projector.query": "proj.query",
    "projector.linear.weight": "proj.linear.weight",
    "projector.linear.bias": "proj.linear.bias",
    "projector.qformer.layernorm.weight": "proj.ln.weight",
    "projector.qformer.layernorm.bias": "proj.ln.bias",
    # LLM top-level
    "language_model.model.embed_tokens.weight": "token_embd.weight",
    "language_model.model.norm.weight": "output_norm.weight",
    "language_model.lm_head.weight": "output.weight",
}

ENC_LAYER_PATTERNS = [
    # Attention
    (r"encoder\.layers\.(\d+)\.attn\.pre_norm\.weight", "enc.blk.{}.attn_norm.weight"),
    (r"encoder\.layers\.(\d+)\.attn\.pre_norm\.bias", "enc.blk.{}.attn_norm.bias"),
    (r"encoder\.layers\.(\d+)\.attn\.to_q\.weight", "enc.blk.{}.attn_q.weight"),
    (r"encoder\.layers\.(\d+)\.attn\.to_kv\.weight", "enc.blk.{}.attn_kv.weight"),
    (r"encoder\.layers\.(\d+)\.attn\.to_out\.weight", "enc.blk.{}.attn_out.weight"),
    (r"encoder\.layers\.(\d+)\.attn\.to_out\.bias", "enc.blk.{}.attn_out.bias"),
    (
        r"encoder\.layers\.(\d+)\.attn\.rel_pos_emb\.weight",
        "enc.blk.{}.attn_rel_pos.weight",
    ),
    # Conv module
    (r"encoder\.layers\.(\d+)\.conv\.up_conv\.weight", "enc.blk.{}.conv_up.weight"),
    (r"encoder\.layers\.(\d+)\.conv\.up_conv\.bias", "enc.blk.{}.conv_up.bias"),
    (
        r"encoder\.layers\.(\d+)\.conv\.depth_conv\.conv\.weight",
        "enc.blk.{}.conv_dw.weight",
    ),
    (r"encoder\.layers\.(\d+)\.conv\.batch_norm\.weight", "enc.blk.{}.conv_bn.weight"),
    (r"encoder\.layers\.(\d+)\.conv\.batch_norm\.bias", "enc.blk.{}.conv_bn.bias"),
    (
        r"encoder\.layers\.(\d+)\.conv\.batch_norm\.running_mean",
        "enc.blk.{}.conv_bn.running_mean",
    ),
    (
        r"encoder\.layers\.(\d+)\.conv\.batch_norm\.running_var",
        "enc.blk.{}.conv_bn.running_var",
    ),
    (r"encoder\.layers\.(\d+)\.conv\.batch_norm\.num_batches_tracked", None),  # skip
    (r"encoder\.layers\.(\d+)\.conv\.down_conv\.weight", "enc.blk.{}.conv_down.weight"),
    (r"encoder\.layers\.(\d+)\.conv\.down_conv\.bias", "enc.blk.{}.conv_down.bias"),
    (r"encoder\.layers\.(\d+)\.conv\.norm\.weight", "enc.blk.{}.conv_norm.weight"),
    (r"encoder\.layers\.(\d+)\.conv\.norm\.bias", "enc.blk.{}.conv_norm.bias"),
    # FFN1 (Macaron pre-FFN)
    (r"encoder\.layers\.(\d+)\.ff1\.pre_norm\.weight", "enc.blk.{}.ff1_norm.weight"),
    (r"encoder\.layers\.(\d+)\.ff1\.pre_norm\.bias", "enc.blk.{}.ff1_norm.bias"),
    (r"encoder\.layers\.(\d+)\.ff1\.up_proj\.weight", "enc.blk.{}.ff1_up.weight"),
    (r"encoder\.layers\.(\d+)\.ff1\.up_proj\.bias", "enc.blk.{}.ff1_up.bias"),
    (r"encoder\.layers\.(\d+)\.ff1\.down_proj\.weight", "enc.blk.{}.ff1_down.weight"),
    (r"encoder\.layers\.(\d+)\.ff1\.down_proj\.bias", "enc.blk.{}.ff1_down.bias"),
    # FFN2 (Macaron post-FFN)
    (r"encoder\.layers\.(\d+)\.ff2\.pre_norm\.weight", "enc.blk.{}.ff2_norm.weight"),
    (r"encoder\.layers\.(\d+)\.ff2\.pre_norm\.bias", "enc.blk.{}.ff2_norm.bias"),
    (r"encoder\.layers\.(\d+)\.ff2\.up_proj\.weight", "enc.blk.{}.ff2_up.weight"),
    (r"encoder\.layers\.(\d+)\.ff2\.up_proj\.bias", "enc.blk.{}.ff2_up.bias"),
    (r"encoder\.layers\.(\d+)\.ff2\.down_proj\.weight", "enc.blk.{}.ff2_down.weight"),
    (r"encoder\.layers\.(\d+)\.ff2\.down_proj\.bias", "enc.blk.{}.ff2_down.bias"),
    # Post-norm
    (r"encoder\.layers\.(\d+)\.post_norm\.weight", "enc.blk.{}.post_norm.weight"),
    (r"encoder\.layers\.(\d+)\.post_norm\.bias", "enc.blk.{}.post_norm.bias"),
]

PROJ_LAYER_PATTERNS = [
    # Self-attention
    (
        r"projector\.qformer\.encoder\.layer\.(\d+)\.attention\.attention\.(query|key|value)\.weight",
        "proj.blk.{}.sa_{}.weight",
    ),
    (
        r"projector\.qformer\.encoder\.layer\.(\d+)\.attention\.attention\.(query|key|value)\.bias",
        "proj.blk.{}.sa_{}.bias",
    ),
    (
        r"projector\.qformer\.encoder\.layer\.(\d+)\.attention\.output\.dense\.weight",
        "proj.blk.{}.sa_out.weight",
    ),
    (
        r"projector\.qformer\.encoder\.layer\.(\d+)\.attention\.output\.dense\.bias",
        "proj.blk.{}.sa_out.bias",
    ),
    (
        r"projector\.qformer\.encoder\.layer\.(\d+)\.attention\.output\.LayerNorm\.weight",
        "proj.blk.{}.sa_norm.weight",
    ),
    (
        r"projector\.qformer\.encoder\.layer\.(\d+)\.attention\.output\.LayerNorm\.bias",
        "proj.blk.{}.sa_norm.bias",
    ),
    # Cross-attention
    (
        r"projector\.qformer\.encoder\.layer\.(\d+)\.crossattention\.attention\.(query|key|value)\.weight",
        "proj.blk.{}.ca_{}.weight",
    ),
    (
        r"projector\.qformer\.encoder\.layer\.(\d+)\.crossattention\.attention\.(query|key|value)\.bias",
        "proj.blk.{}.ca_{}.bias",
    ),
    (
        r"projector\.qformer\.encoder\.layer\.(\d+)\.crossattention\.output\.dense\.weight",
        "proj.blk.{}.ca_out.weight",
    ),
    (
        r"projector\.qformer\.encoder\.layer\.(\d+)\.crossattention\.output\.dense\.bias",
        "proj.blk.{}.ca_out.bias",
    ),
    (
        r"projector\.qformer\.encoder\.layer\.(\d+)\.crossattention\.output\.LayerNorm\.weight",
        "proj.blk.{}.ca_norm.weight",
    ),
    (
        r"projector\.qformer\.encoder\.layer\.(\d+)\.crossattention\.output\.LayerNorm\.bias",
        "proj.blk.{}.ca_norm.bias",
    ),
    # FFN
    (
        r"projector\.qformer\.encoder\.layer\.(\d+)\.intermediate_query\.dense\.weight",
        "proj.blk.{}.ffn_up.weight",
    ),
    (
        r"projector\.qformer\.encoder\.layer\.(\d+)\.intermediate_query\.dense\.bias",
        "proj.blk.{}.ffn_up.bias",
    ),
    (
        r"projector\.qformer\.encoder\.layer\.(\d+)\.output_query\.dense\.weight",
        "proj.blk.{}.ffn_down.weight",
    ),
    (
        r"projector\.qformer\.encoder\.layer\.(\d+)\.output_query\.dense\.bias",
        "proj.blk.{}.ffn_down.bias",
    ),
    (
        r"projector\.qformer\.encoder\.layer\.(\d+)\.output_query\.LayerNorm\.weight",
        "proj.blk.{}.ffn_norm.weight",
    ),
    (
        r"projector\.qformer\.encoder\.layer\.(\d+)\.output_query\.LayerNorm\.bias",
        "proj.blk.{}.ffn_norm.bias",
    ),
]

LLM_LAYER_PATTERNS = [
    (
        r"language_model\.model\.layers\.(\d+)\.input_layernorm\.weight",
        "blk.{}.attn_norm.weight",
    ),
    (
        r"language_model\.model\.layers\.(\d+)\.self_attn\.q_proj\.weight",
        "blk.{}.attn_q.weight",
    ),
    (
        r"language_model\.model\.layers\.(\d+)\.self_attn\.k_proj\.weight",
        "blk.{}.attn_k.weight",
    ),
    (
        r"language_model\.model\.layers\.(\d+)\.self_attn\.v_proj\.weight",
        "blk.{}.attn_v.weight",
    ),
    (
        r"language_model\.model\.layers\.(\d+)\.self_attn\.o_proj\.weight",
        "blk.{}.attn_output.weight",
    ),
    (
        r"language_model\.model\.layers\.(\d+)\.post_attention_layernorm\.weight",
        "blk.{}.ffn_norm.weight",
    ),
    (
        r"language_model\.model\.layers\.(\d+)\.mlp\.gate_proj\.weight",
        "blk.{}.ffn_gate.weight",
    ),
    (
        r"language_model\.model\.layers\.(\d+)\.mlp\.up_proj\.weight",
        "blk.{}.ffn_up.weight",
    ),
    (
        r"language_model\.model\.layers\.(\d+)\.mlp\.down_proj\.weight",
        "blk.{}.ffn_down.weight",
    ),
]


def _normalize_rnn_tr_name(hf_name: str) -> str:
    """Rewrite granite-3.2-style ``encoder.rnn_tr.N.*`` tensor names into the
    canonical granite-3.3/4.0 ``encoder.layers.(N-1).*`` form.

    granite-speech 3.2-8b uses an older PyTorch module structure that wraps
    each Conformer sub-block in a functional-decorator nesting
    (``.attn.fn.*``, ``.ff1.fn.fn.net.N.*``, ``.conv.net.N.*``) and
    anchors the input-linear at index 0 of a single ``rnn_tr`` ModuleList,
    with the encoder blocks starting at index 1. granite-3.3 / 4.0
    flattened this into ``encoder.input_linear`` + ``encoder.layers.N``.

    The function takes an rnn_tr name and returns its granite-3.3
    equivalent. Non-rnn_tr names pass through unchanged.
    """
    if not hf_name.startswith("encoder.rnn_tr."):
        return hf_name

    # encoder.rnn_tr.0.weight/bias is the input linear.
    m = re.match(r"encoder\.rnn_tr\.0\.(weight|bias)$", hf_name)
    if m:
        return f"encoder.input_linear.{m.group(1)}"

    # encoder.rnn_tr.N.* for N>=1 → encoder.layers.(N-1).*  with sub-path
    # rewrites below. We shift the layer index down by one so the output
    # matches the granite-3.3 convention of layers.0 being the first block.
    m = re.match(r"encoder\.rnn_tr\.(\d+)\.(.*)$", hf_name)
    if not m:
        return hf_name
    layer = int(m.group(1)) - 1
    rest = m.group(2)

    # Sub-path rewrites inside one block.
    #   attn.norm.*                  → attn.pre_norm.*
    #   attn.fn.<x>                  → attn.<x>
    rest = re.sub(r"^attn\.norm\.", "attn.pre_norm.", rest)
    rest = re.sub(r"^attn\.fn\.", "attn.", rest)

    #   ff1.fn.norm.*                → ff1.pre_norm.*
    #   ff1.fn.fn.net.0.*            → ff1.up_proj.*     (first Linear)
    #   ff1.fn.fn.net.3.*            → ff1.down_proj.*   (Linear after GELU+dropout)
    rest = re.sub(r"^ff1\.fn\.norm\.", "ff1.pre_norm.", rest)
    rest = re.sub(r"^ff1\.fn\.fn\.net\.0\.", "ff1.up_proj.", rest)
    rest = re.sub(r"^ff1\.fn\.fn\.net\.3\.", "ff1.down_proj.", rest)

    rest = re.sub(r"^ff2\.fn\.norm\.", "ff2.pre_norm.", rest)
    rest = re.sub(r"^ff2\.fn\.fn\.net\.0\.", "ff2.up_proj.", rest)
    rest = re.sub(r"^ff2\.fn\.fn\.net\.3\.", "ff2.down_proj.", rest)

    # Conv module is a Sequential:
    #   net.0 = LayerNorm              → conv.norm
    #   net.2 = up pointwise 1×1       → conv.up_conv
    #   net.4.conv = depthwise         → conv.depth_conv.conv
    #   net.5 = BatchNorm              → conv.batch_norm
    #   net.7 = down pointwise 1×1     → conv.down_conv
    rest = re.sub(r"^conv\.net\.0\.", "conv.norm.", rest)
    rest = re.sub(r"^conv\.net\.2\.", "conv.up_conv.", rest)
    rest = re.sub(r"^conv\.net\.4\.conv\.", "conv.depth_conv.conv.", rest)
    rest = re.sub(r"^conv\.net\.5\.", "conv.batch_norm.", rest)
    rest = re.sub(r"^conv\.net\.7\.", "conv.down_conv.", rest)

    return f"encoder.layers.{layer}.{rest}"


def remap_name(hf_name: str) -> str | None:
    # Normalize granite-3.2 rnn_tr layout into granite-3.3/4.0 names first,
    # then run the single pattern set below (shared by all granite-speech
    # variants). This keeps the C++ runtime's tensor-name expectations
    # unchanged across the family.
    hf_name = _normalize_rnn_tr_name(hf_name)

    if hf_name in DIRECT:
        return DIRECT[hf_name]
    for patterns in [ENC_LAYER_PATTERNS, PROJ_LAYER_PATTERNS, LLM_LAYER_PATTERNS]:
        for pat, tmpl in patterns:
            m = re.match(pat, hf_name)
            if m:
                if tmpl is None:
                    return None  # skip
                groups = m.groups()
                if len(groups) == 2:
                    return tmpl.format(groups[0], groups[1])
                return tmpl.format(groups[0])
    return None


def is_f32_tensor(name: str, shape: tuple) -> bool:
    if name.endswith(".bias"):
        return True
    if "norm" in name or "ln" in name:
        return True
    if "rel_pos" in name:
        return True
    if "running_mean" in name or "running_var" in name:
        return True
    if "query" in name and len(shape) == 3:  # projector.query [1,3,1024]
        return True
    if len(shape) <= 1:
        return True
    # Keep encoder weights as F32 to avoid precision loss across 16 layers
    if name.startswith("enc."):
        return True
    # Keep projector weights as F32 (small, precision-sensitive)
    if name.startswith("proj."):
        return True
    return False


# ---------------------------------------------------------------------------
# Mel filter bank (80 bins, Slaney-style)
# ---------------------------------------------------------------------------


def build_htk_mel_filters(sr=16000, n_fft=512, n_mels=80, f_min=0.0, f_max=8000.0):
    """HTK mel filter bank matching torchaudio.transforms.MelSpectrogram defaults."""
    n_freqs = n_fft // 2 + 1

    def hz_to_mel(f):
        return 2595.0 * np.log10(1.0 + np.asarray(f, dtype=np.float64) / 700.0)

    def mel_to_hz(m):
        return 700.0 * (10.0 ** (np.asarray(m, dtype=np.float64) / 2595.0) - 1.0)

    fft_freqs = np.linspace(0, sr / 2, n_freqs)
    mel_min = hz_to_mel(f_min)
    mel_max = hz_to_mel(f_max)
    mel_freqs = np.linspace(mel_min, mel_max, n_mels + 2)
    filt_freqs = mel_to_hz(mel_freqs)
    filt_diff = np.diff(filt_freqs)
    slopes = filt_freqs[None, :] - fft_freqs[:, None]
    down = -slopes[:, :-2] / filt_diff[:-1]
    up = slopes[:, 2:] / filt_diff[1:]
    fb = np.maximum(0, np.minimum(down, up))
    # torchaudio default: norm=None (no area normalization)
    return fb.astype(np.float32)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _load_lora_adapter(input_dir: Path) -> tuple[dict, dict] | None:
    """Load the granite-speech LoRA adapter tensors if present.

    Returns (lora_weights, cfg) where `lora_weights` is a dict keyed by the
    base tensor name (e.g. "language_model.model.layers.3.self_attn.q_proj.weight")
    mapping to `(lora_A, lora_B, scale)` tuples ready to be merged via
    `W_merged = W + scale * (B @ A)`. Returns None if no adapter is present.

    granite-speech 3.x ships with an unmerged PEFT LoRA (r=64, alpha=32,
    target_modules=[q_proj, v_proj]) that must be folded into the base LLM
    weights at conversion time so the C++ runtime can stay linear-LoRA-free.
    """
    cfg_path = input_dir / "adapter_config.json"
    ada_path = input_dir / "adapter_model.safetensors"
    if not (cfg_path.exists() and ada_path.exists()):
        return None
    with open(cfg_path) as f:
        cfg = json.load(f)
    alpha = float(cfg.get("lora_alpha", 32))
    r = float(cfg.get("r", 64))
    scale = alpha / r
    targets = set(cfg.get("target_modules", ["q_proj", "v_proj"]))
    print(
        f"  lora: r={int(r)} alpha={alpha:g} scale={scale:g} targets={sorted(targets)}"
    )

    # Collect (A, B) pairs keyed by the base weight path. PEFT tensors
    # look like "...q_proj.lora_A.weight" / "...q_proj.lora_B.weight";
    # the base weight this LoRA modifies is the corresponding ".weight".
    pairs: dict[str, dict[str, np.ndarray]] = {}
    with safe_open(str(ada_path), framework="pt", device="cpu") as f:
        for name in sorted(f.keys()):
            if ".lora_A.weight" in name:
                base = name.replace(".lora_A.weight", ".weight")
                pairs.setdefault(base, {})["A"] = f.get_tensor(name).float().numpy()
            elif ".lora_B.weight" in name:
                base = name.replace(".lora_B.weight", ".weight")
                pairs.setdefault(base, {})["B"] = f.get_tensor(name).float().numpy()

    merged: dict[str, tuple] = {}
    for base, ab in pairs.items():
        if "A" in ab and "B" in ab:
            # Only merge onto targets the adapter actually matches
            if any(t in base for t in targets):
                merged[base] = (ab["A"], ab["B"], scale)
    print(f"  lora: merged deltas for {len(merged)} base tensors")
    return merged, cfg


def _apply_lora(name: str, arr, lora_map) -> any:
    """Return `arr + scale * (B @ A)` when `name` has a LoRA delta, else `arr`."""
    if lora_map is None or name not in lora_map:
        return arr
    A, B, scale = lora_map[name]
    # W shape: (out, in); B: (out, r); A: (r, in). delta = (out, in).
    import numpy as np

    delta = (B @ A).astype(arr.dtype) * np.array(scale, dtype=arr.dtype)
    return arr + delta


def convert(input_dir: Path, out_path: Path) -> None:
    print(f"Loading: {input_dir}")
    with open(input_dir / "config.json") as f:
        cfg = json.load(f)
    enc_cfg = cfg.get("encoder_config", {})
    text_cfg = cfg.get("text_config", {})
    proj_cfg = cfg.get("projector_config", {})

    # Some HF snapshots (granite-speech-3.3-*) ship BOTH a model-*-of-N
    # and a model-*-of-M shard set containing the same content under two
    # layouts. Trust the index file — it names the shards the model was
    # built against. Fall back to the naive glob for checkpoints that
    # don't have an index.
    idx_path = input_dir / "model.safetensors.index.json"
    if idx_path.exists():
        with open(idx_path) as f:
            idx = json.load(f)
        wanted = sorted(set(idx.get("weight_map", {}).values()))
        safetensor_files = [input_dir / n for n in wanted]
    else:
        safetensor_files = sorted(input_dir.glob("model-*.safetensors"))
        if not safetensor_files:
            safetensor_files = sorted(input_dir.glob("*.safetensors"))
    print(f"  shards: {[p.name for p in safetensor_files]}")

    # LoRA adapter merge (granite-speech 3.x has this; 4.0-1b does not).
    lora_merged = _load_lora_adapter(input_dir)
    lora_map = lora_merged[0] if lora_merged is not None else None

    print(f"Writing: {out_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # use_temp_file=True spills tensor data to a disk-backed temp file
    # as we call add_tensor(), so the resident set stays bounded at the
    # largest single tensor rather than growing with the whole model.
    # This matters for granite-speech-3.2-8b / 3.3-8b (~16 GB total in
    # F16) on machines with under ~16 GB of RAM.
    writer = gguf.GGUFWriter(str(out_path), arch="granite_speech", use_temp_file=True)

    # Metadata
    writer.add_uint32("granite_speech.sample_rate", 16000)
    writer.add_uint32("granite_speech.n_mels", 80)
    writer.add_uint32("granite_speech.enc.n_layers", enc_cfg.get("num_layers", 16))
    writer.add_uint32("granite_speech.enc.d_model", enc_cfg.get("hidden_dim", 1024))
    writer.add_uint32("granite_speech.enc.n_heads", enc_cfg.get("num_heads", 8))
    writer.add_uint32("granite_speech.enc.head_dim", enc_cfg.get("dim_head", 128))
    writer.add_uint32("granite_speech.enc.input_dim", enc_cfg.get("input_dim", 160))
    writer.add_uint32(
        "granite_speech.enc.conv_kernel", enc_cfg.get("conv_kernel_size", 15)
    )
    writer.add_uint32(
        "granite_speech.enc.ff_dim",
        enc_cfg.get("hidden_dim", 1024) * enc_cfg.get("feedforward_mult", 4),
    )
    writer.add_uint32(
        "granite_speech.enc.context_size", enc_cfg.get("context_size", 200)
    )
    writer.add_uint32(
        "granite_speech.enc.max_pos_emb", enc_cfg.get("max_pos_emb", 512)
    )

    writer.add_uint32(
        "granite_speech.proj.n_layers", proj_cfg.get("num_hidden_layers", 2)
    )
    writer.add_uint32("granite_speech.proj.d_model", proj_cfg.get("hidden_size", 1024))
    writer.add_uint32(
        "granite_speech.proj.n_heads", proj_cfg.get("num_attention_heads", 16)
    )
    writer.add_uint32(
        "granite_speech.proj.ff_dim", proj_cfg.get("intermediate_size", 4096)
    )
    # granite-speech-4.1-2b-plus: encoder feeds the projector a
    # concatenation of [layer_i for i in cat_hidden_layers] + final.
    # encoder_hidden_size in projector_config is the resulting width
    # (= hidden_dim * (len(cat_hidden_layers) + 1)).
    cat_layers = enc_cfg.get("cat_hidden_layers", [])
    proj_enc_hidden = proj_cfg.get(
        "encoder_hidden_size", enc_cfg.get("hidden_dim", 1024)
    )
    if cat_layers:
        # gguf doesn't have add_array_uint32 — pack as a length-prefixed list
        # via a single key per index. Keep it simple: ship the indices as
        # a comma-separated string. The runtime parses on load.
        writer.add_string(
            "granite_speech.proj.cat_layers",
            ",".join(str(i) for i in cat_layers),
        )
    writer.add_uint32(
        "granite_speech.proj.encoder_hidden_size", int(proj_enc_hidden)
    )

    writer.add_uint32(
        "granite_speech.llm.n_layers", text_cfg.get("num_hidden_layers", 40)
    )
    writer.add_uint32("granite_speech.llm.d_model", text_cfg.get("hidden_size", 2048))
    writer.add_uint32(
        "granite_speech.llm.n_heads", text_cfg.get("num_attention_heads", 16)
    )
    writer.add_uint32(
        "granite_speech.llm.n_kv_heads", text_cfg.get("num_key_value_heads", 4)
    )
    writer.add_uint32(
        "granite_speech.llm.head_dim",
        text_cfg.get("hidden_size", 2048) // text_cfg.get("num_attention_heads", 16),
    )
    writer.add_uint32(
        "granite_speech.llm.ff_dim", text_cfg.get("intermediate_size", 4096)
    )
    writer.add_float32(
        "granite_speech.llm.rope_theta", float(text_cfg.get("rope_theta", 10000))
    )
    writer.add_float32(
        "granite_speech.llm.rms_norm_eps", float(text_cfg.get("rms_norm_eps", 1e-5))
    )
    writer.add_uint32(
        "granite_speech.llm.vocab_size", text_cfg.get("vocab_size", 100353)
    )

    # μP multipliers
    writer.add_float32(
        "granite_speech.llm.embedding_multiplier",
        float(text_cfg.get("embedding_multiplier", 12.0)),
    )
    writer.add_float32(
        "granite_speech.llm.attention_multiplier",
        float(text_cfg.get("attention_multiplier", 0.0078125)),
    )
    writer.add_float32(
        "granite_speech.llm.residual_multiplier",
        float(text_cfg.get("residual_multiplier", 0.22)),
    )
    writer.add_float32(
        "granite_speech.llm.logits_scaling", float(text_cfg.get("logits_scaling", 8.0))
    )

    writer.add_uint32("granite_speech.downsample_rate", cfg.get("downsample_rate", 5))
    writer.add_uint32("granite_speech.window_size", cfg.get("window_size", 15))
    writer.add_uint32(
        "granite_speech.audio_token_index", cfg.get("audio_token_index", 100352)
    )

    # EOS / BOS — varies between releases (granite-4.0 uses 100257 from the
    # GPT-NeoX table, granite-3.x uses 0 from the granite tokenizer). Read
    # from generation_config.json when present, fall back to the LLM
    # text_config, then a reasonable default.
    gen_cfg_path = input_dir / "generation_config.json"
    gen_cfg = {}
    if gen_cfg_path.exists():
        with open(gen_cfg_path) as f:
            gen_cfg = json.load(f)

    def _first_int(*vals, default):
        for v in vals:
            if isinstance(v, list) and v:
                return int(v[0])
            if isinstance(v, int):
                return int(v)
        return default

    eos_id = _first_int(
        gen_cfg.get("eos_token_id"),
        text_cfg.get("eos_token_id"),
        default=text_cfg.get("eos_token_id", 0) or 0,
    )
    bos_id = _first_int(
        gen_cfg.get("bos_token_id"),
        text_cfg.get("bos_token_id"),
        default=text_cfg.get("bos_token_id", 0) or 0,
    )
    writer.add_uint32("granite_speech.llm.eos_token_id", eos_id)
    writer.add_uint32("granite_speech.llm.bos_token_id", bos_id)
    print(f"  eos_token_id={eos_id}  bos_token_id={bos_id}")

    # Tokenizer vocab (GPT-2 BPE — store token strings for detokenization)
    vocab_path = input_dir / "vocab.json"
    if vocab_path.exists():
        with open(vocab_path) as f:
            vocab_json = json.load(f)  # {token_str: id}
        # Build id→token list
        n_vocab = text_cfg.get("vocab_size", 100353)
        tokens = [""] * n_vocab
        for tok_str, tok_id in vocab_json.items():
            if tok_id < n_vocab:
                tokens[tok_id] = tok_str
        # Add special tokens from tokenizer_config.json
        tok_cfg_path = input_dir / "tokenizer_config.json"
        if tok_cfg_path.exists():
            with open(tok_cfg_path) as f:
                tok_cfg = json.load(f)
            for tid_str, info in tok_cfg.get("added_tokens_decoder", {}).items():
                tid = int(tid_str)
                if tid < n_vocab:
                    tokens[tid] = info["content"]
        # Store as GGUF string array
        writer.add_array("tokenizer.ggml.tokens", tokens)
        print(f"  tokenizer: {len(tokens)} tokens")

    # BPE merges — required by the C++ encoder side (granite_speech_tokenize)
    # for arbitrary text input. Older converted GGUFs will lack this key
    # and the C++ side falls back gracefully (single-token vocab lookup
    # only). Format: each line is "left right" — exactly the same content
    # as the original merges.txt minus the leading "#version" comment.
    merges_path = input_dir / "merges.txt"
    if merges_path.exists():
        with open(merges_path, encoding="utf-8") as f:
            merges = []
            for line in f:
                line = line.rstrip("\n")
                if not line or line.startswith("#"):
                    continue
                merges.append(line)
        writer.add_array("tokenizer.ggml.merges", merges)
        print(f"  merges:    {len(merges)} BPE merges")

    # Mel filterbank (80 bins)
    mel_filters = build_htk_mel_filters(sr=16000, n_fft=512, n_mels=80)
    writer.add_tensor("audio.mel_filters", mel_filters)
    # Hann window: win_length=400, zero-padded to n_fft=512
    win_length = 400
    n_fft = 512
    win = np.zeros(n_fft, dtype=np.float32)
    win[:win_length] = (
        0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(win_length) / win_length)
    ).astype(np.float32)
    writer.add_tensor("audio.mel_window", win)

    # Tensors
    n_written = n_f16 = n_f32 = n_skipped = 0
    skipped = []

    # Some granite releases tie the lm_head to the token embedding
    # (granite-3.2, granite-3.3) so the safetensors has no separate
    # `lm_head.weight`. Our C++ runtime expects `output.weight` anyway —
    # when tied, we alias the embedding as output.weight below after the
    # main loop finishes so we only write the tensor once.
    tie_word_embeddings = bool(text_cfg.get("tie_word_embeddings", False))
    if tie_word_embeddings:
        print(
            "  llm: tie_word_embeddings=true — aliasing token embedding as output.weight"
        )
    token_embd_arr = None  # captured during the loop for the tie-alias path

    n_lora_merged = 0
    for sf_path in safetensor_files:
        print(f"  reading {sf_path.name}")
        with safe_open(str(sf_path), framework="pt", device="cpu") as f:
            for hf_name in sorted(f.keys()):
                gguf_name = remap_name(hf_name)
                if gguf_name is None:
                    n_skipped += 1
                    skipped.append(hf_name)
                    continue
                t = f.get_tensor(hf_name)
                if "bfloat" in str(t.dtype):
                    t = t.float()
                arr = t.numpy().astype(np.float32)
                # Fold in the LoRA delta (no-op when this tensor isn't
                # a LoRA target — the map lookup simply misses).
                if lora_map is not None and hf_name in lora_map:
                    arr = _apply_lora(hf_name, arr, lora_map)
                    n_lora_merged += 1
                if is_f32_tensor(gguf_name, arr.shape):
                    arr = arr.astype(np.float32)
                    n_f32 += 1
                else:
                    arr = arr.astype(np.float16)
                    n_f16 += 1
                writer.add_tensor(gguf_name, arr)
                n_written += 1
                if gguf_name == "token_embd.weight":
                    token_embd_arr = arr
                if n_written <= 20 or n_written % 100 == 0:
                    print(f"    {gguf_name:50s} {str(arr.shape):25s} {arr.dtype}")

    if tie_word_embeddings and token_embd_arr is not None:
        # Write a second copy of the token embedding under the lm_head
        # name. GGUF tensor names must be unique — the base loop did not
        # yet write output.weight because the source safetensors omits
        # lm_head.weight when embeddings are tied.
        writer.add_tensor("output.weight", token_embd_arr)
        n_written += 1
        print(
            f"    output.weight (aliased from token_embd.weight) "
            f"{token_embd_arr.shape} {token_embd_arr.dtype}"
        )

    if lora_map is not None:
        print(
            f"  lora: folded {n_lora_merged}/{len(lora_map)} deltas into base weights"
        )

    print(
        f"\n  total: {n_written} tensors (F16: {n_f16}, F32: {n_f32}) skipped: {n_skipped}"
    )
    if skipped:
        print("  skipped:")
        for s in skipped:
            print(f"    {s}")

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"\nDone: {out_path} ({out_path.stat().st_size / 1e9:.2f} GB)")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Convert Granite Speech 4.0-1B → GGUF F16")
    p.add_argument("--input", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    convert(args.input, args.output)
