#!/usr/bin/env python3
"""Convert HojoAI/Hojo-ASR-Multi-V1 to GGUF format for CrispASR (issue #438).

Architecture (config.yaml + the `hojo-asr` PyPI package, Apache-2.0, which is
the driving inference code — `hojo_asr/hojo_asr_model.py::HOJO_ASR.infer`):

  Front end   WhisperFeatureExtractor(feature_size=128, n_fft=400, hop=160,
              chunk_length=40) called with padding=False → a VARIABLE-length
              128-bin log-mel at 100 fps. No 30 s / 40 s padding: chunk_length
              only moves the `padding="max_length"` cap, which is never used.

  Encoder     ModifyQwen3OmniMoeAudioEncoder — the STOCK Qwen3-Omni "AuT" audio
              tower with n_window=1500 / n_window_infer=3000 patched in.
              128-mel → 3× Conv2d(3x3, stride 2, pad 1, 480 ch, GELU) →
              conv_out Linear(480*16=7680 → 1280, NO bias) → + sinusoidal pos
              (per chunk, positions restart at 0) → 32 Whisper-style pre-LN
              layers (1280 d, 20 heads, FFN 5120, GELU) → ln_post →
              proj1(1280→1280) → gelu → proj2(1280→2048).
              Mel is cut into chunks of n_window*2 = 3000 frames; attention is
              block-diagonal over those chunks (no cross-chunk attention), so
              each chunk is mathematically independent. 8× time downsample →
              12.5 encoder frames per second.

  Adapter     WeNet `ConformerEncoder(2048, 2560, linear_units=640,
              num_blocks=2, input_layer="linear")` — all other arguments at
              their WeNet defaults: 4 heads, rel_pos, macaron, SiLU, cnn kernel
              15, non-causal, batch_norm, normalize_before, eps 1e-5.
              LinearNoSubsampling: Linear(2048→2560) → LayerNorm → ×sqrt(2560);
              RelPositionalEncoding supplies an ABSOLUTE sin/cos table (length
              T, interleaved sin/cos — NOT the concat-halves convention the
              audio encoder uses) that feeds `linear_pos`.
              ⚠ WeNet deletes `rel_shift`, so matrix_bd = (q+pos_bias_v)·p^T
              with NO shift — it is an absolute-position bias, not Transformer-XL.

  ln_speech   LayerNorm(2560) applied to the adapter output.

  Decoder     Qwen3-4B-Instruct-2507 (36 L, 2560 d, 32 Q / 8 KV heads,
              head_dim 128, SwiGLU 9728, QK-norm, RoPE θ=5e6, RMS eps 1e-6),
              embeddings resized to 151670 (Qwen3's 151669 + an added [PAD]).
              inputs_embeds = [embed(<|im_start|>)] ++ speech_embeddings.
              There is NO text prompt and no chat template.

`merged_full_model.safetensors` (11.96 GB) holds EVERY weight — encoder
(F32), adapter (F32) and decoder (BF16). The bare
`Qwen3-Omni-30B-A3B-Instruct/config.json` in the repo is used for the audio
hyper-parameters ONLY; no 30 B checkpoint is ever fetched.

Streams tensors one at a time via safe_open. ~2 GB peak RAM.

Usage:
    python models/convert-hojo-asr-to-gguf.py \
        --input HojoAI/Hojo-ASR-Multi-V1 \
        --output hojo-asr-multi-v1-f16.gguf

    # Then quantize (encoder + adapter + tied embedding stay F16):
    crispasr-quantize hojo-asr-multi-v1-f16.gguf \
                      hojo-asr-multi-v1-q4_k.gguf q4_k
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch

try:
    from gguf import GGUFWriter, GGMLQuantizationType
except ImportError:
    print("Error: gguf package not found. Install with: pip install gguf")
    sys.exit(1)

try:
    from safetensors import safe_open
except ImportError:
    print("Error: safetensors package not found. Install with: pip install safetensors")
    sys.exit(1)

try:
    from huggingface_hub import snapshot_download
except ImportError:
    print("Error: huggingface_hub package not found. Install with: pip install huggingface_hub")
    sys.exit(1)


ARCH = "hojo_asr"


def load_model_dir(model_id: str) -> Path:
    model_path = Path(model_id)
    if model_path.is_dir():
        return model_path
    print(f"Downloading model from HuggingFace: {model_id}")
    path = snapshot_download(model_id)
    return Path(path)


# ---------------------------------------------------------------------------
# Tensor name mapping
# ---------------------------------------------------------------------------

_ENC_LAYER_RE = re.compile(r"^speech_encoder\.layers\.(\d+)\.(.*)$")
_ADP_BLK_RE = re.compile(r"^bottleneck\.encoders\.(\d+)\.(.*)$")
_LLM_LAYER_RE = re.compile(r"^decoder_model\.model\.layers\.(\d+)\.(.*)$")

_ENC_SUFFIX = {
    "self_attn_layer_norm.weight": "attn_norm.weight",
    "self_attn_layer_norm.bias": "attn_norm.bias",
    "self_attn.q_proj.weight": "attn.q.weight",
    "self_attn.q_proj.bias": "attn.q.bias",
    "self_attn.k_proj.weight": "attn.k.weight",
    "self_attn.k_proj.bias": "attn.k.bias",
    "self_attn.v_proj.weight": "attn.v.weight",
    "self_attn.v_proj.bias": "attn.v.bias",
    "self_attn.out_proj.weight": "attn.o.weight",
    "self_attn.out_proj.bias": "attn.o.bias",
    "final_layer_norm.weight": "ffn_norm.weight",
    "final_layer_norm.bias": "ffn_norm.bias",
    "fc1.weight": "ffn.fc1.weight",
    "fc1.bias": "ffn.fc1.bias",
    "fc2.weight": "ffn.fc2.weight",
    "fc2.bias": "ffn.fc2.bias",
}

_ENC_TOP = {
    "speech_encoder.conv2d1.weight": "enc.conv1.weight",
    "speech_encoder.conv2d1.bias": "enc.conv1.bias",
    "speech_encoder.conv2d2.weight": "enc.conv2.weight",
    "speech_encoder.conv2d2.bias": "enc.conv2.bias",
    "speech_encoder.conv2d3.weight": "enc.conv3.weight",
    "speech_encoder.conv2d3.bias": "enc.conv3.bias",
    "speech_encoder.conv_out.weight": "enc.conv_out.weight",
    "speech_encoder.ln_post.weight": "enc.ln_post.weight",
    "speech_encoder.ln_post.bias": "enc.ln_post.bias",
    "speech_encoder.proj1.weight": "enc.proj1.weight",
    "speech_encoder.proj1.bias": "enc.proj1.bias",
    "speech_encoder.proj2.weight": "enc.proj2.weight",
    "speech_encoder.proj2.bias": "enc.proj2.bias",
}

# WeNet ConformerEncoderLayer submodule → CrispASR suffix.
_ADP_SUFFIX = {
    "norm_ff_macaron.weight": "norm_ff_macaron.weight",
    "norm_ff_macaron.bias": "norm_ff_macaron.bias",
    "feed_forward_macaron.w_1.weight": "ff_macaron.w1.weight",
    "feed_forward_macaron.w_1.bias": "ff_macaron.w1.bias",
    "feed_forward_macaron.w_2.weight": "ff_macaron.w2.weight",
    "feed_forward_macaron.w_2.bias": "ff_macaron.w2.bias",
    "norm_mha.weight": "norm_mha.weight",
    "norm_mha.bias": "norm_mha.bias",
    "self_attn.linear_q.weight": "attn.q.weight",
    "self_attn.linear_q.bias": "attn.q.bias",
    "self_attn.linear_k.weight": "attn.k.weight",
    "self_attn.linear_k.bias": "attn.k.bias",
    "self_attn.linear_v.weight": "attn.v.weight",
    "self_attn.linear_v.bias": "attn.v.bias",
    "self_attn.linear_out.weight": "attn.o.weight",
    "self_attn.linear_out.bias": "attn.o.bias",
    "self_attn.linear_pos.weight": "attn.pos.weight",
    "self_attn.pos_bias_u": "attn.pos_bias_u",
    "self_attn.pos_bias_v": "attn.pos_bias_v",
    "norm_conv.weight": "norm_conv.weight",
    "norm_conv.bias": "norm_conv.bias",
    "conv_module.pointwise_conv1.weight": "conv.pw1.weight",
    "conv_module.pointwise_conv1.bias": "conv.pw1.bias",
    "conv_module.depthwise_conv.weight": "conv.dw.weight",
    "conv_module.depthwise_conv.bias": "conv.dw.bias",
    "conv_module.pointwise_conv2.weight": "conv.pw2.weight",
    "conv_module.pointwise_conv2.bias": "conv.pw2.bias",
    "norm_ff.weight": "norm_ff.weight",
    "norm_ff.bias": "norm_ff.bias",
    "feed_forward.w_1.weight": "ff.w1.weight",
    "feed_forward.w_1.bias": "ff.w1.bias",
    "feed_forward.w_2.weight": "ff.w2.weight",
    "feed_forward.w_2.bias": "ff.w2.bias",
    "norm_final.weight": "norm_final.weight",
    "norm_final.bias": "norm_final.bias",
}

_ADP_TOP = {
    "bottleneck.embed.out.0.weight": "adapter.embed.proj.weight",
    "bottleneck.embed.out.0.bias": "adapter.embed.proj.bias",
    "bottleneck.embed.out.1.weight": "adapter.embed.norm.weight",
    "bottleneck.embed.out.1.bias": "adapter.embed.norm.bias",
    "bottleneck.after_norm.weight": "adapter.after_norm.weight",
    "bottleneck.after_norm.bias": "adapter.after_norm.bias",
    "ln_speech.weight": "ln_speech.weight",
    "ln_speech.bias": "ln_speech.bias",
}

_LLM_SUFFIX = {
    "input_layernorm.weight": "attn_norm.weight",
    "post_attention_layernorm.weight": "ffn_norm.weight",
    "self_attn.q_proj.weight": "attn.q.weight",
    "self_attn.k_proj.weight": "attn.k.weight",
    "self_attn.v_proj.weight": "attn.v.weight",
    "self_attn.o_proj.weight": "attn.o.weight",
    "self_attn.q_norm.weight": "attn.q_norm.weight",
    "self_attn.k_norm.weight": "attn.k_norm.weight",
    "mlp.gate_proj.weight": "ffn.gate.weight",
    "mlp.up_proj.weight": "ffn.up.weight",
    "mlp.down_proj.weight": "ffn.down.weight",
}


def map_tensor_name(hf_name: str):
    """HF tensor name → GGUF name. None = handled elsewhere / intentionally dropped."""
    if hf_name in _ENC_TOP:
        return _ENC_TOP[hf_name]
    m = _ENC_LAYER_RE.match(hf_name)
    if m:
        suf = _ENC_SUFFIX.get(m.group(2))
        if suf is None:
            raise KeyError(f"unmapped encoder tensor: {hf_name}")
        return f"enc.blk.{int(m.group(1))}.{suf}"

    if hf_name in _ADP_TOP:
        return _ADP_TOP[hf_name]
    if hf_name == "bottleneck.embed.pos_enc.pe":
        return "adapter.pe"
    m = _ADP_BLK_RE.match(hf_name)
    if m:
        tail = m.group(2)
        # BatchNorm1d running stats are folded into conv.norm.{scale,shift}
        # by fold_conv_batchnorm(); the raw buffers are not emitted.
        if tail.startswith("conv_module.norm."):
            return None
        suf = _ADP_SUFFIX.get(tail)
        if suf is None:
            raise KeyError(f"unmapped adapter tensor: {hf_name}")
        return f"adapter.blk.{int(m.group(1))}.{suf}"

    if hf_name == "decoder_model.model.embed_tokens.weight":
        return "llm.embed.weight"
    if hf_name == "decoder_model.model.norm.weight":
        return "llm.final_norm.weight"
    if hf_name == "decoder_model.lm_head.weight":
        # tie_word_embeddings=True — verified byte-identical to embed_tokens in
        # main(); dropping it halves the vocab-matrix cost of the GGUF.
        return None
    m = _LLM_LAYER_RE.match(hf_name)
    if m:
        suf = _LLM_SUFFIX.get(m.group(2))
        if suf is None:
            raise KeyError(f"unmapped decoder tensor: {hf_name}")
        return f"llm.blk.{int(m.group(1))}.{suf}"

    raise KeyError(f"unmapped tensor: {hf_name}")


# ---------------------------------------------------------------------------
# BatchNorm folding
# ---------------------------------------------------------------------------

def fold_conv_batchnorm(get, n_blocks, eps=1e-5):
    """eval-mode BatchNorm1d → a per-channel affine (scale, shift).

        y = (x - mean) / sqrt(var + eps) * w + b
          = x * (w / sqrt(var + eps)) + (b - mean * w / sqrt(var + eps))

    Exact algebra — no tolerance involved. Emitting scale/shift instead of the
    four raw buffers keeps the runtime graph to one mul + one add and makes the
    "did you remember this is eval mode?" question unaskable.
    """
    out = {}
    for i in range(n_blocks):
        p = f"bottleneck.encoders.{i}.conv_module.norm."
        w = get(p + "weight").float().numpy().astype(np.float64)
        b = get(p + "bias").float().numpy().astype(np.float64)
        mean = get(p + "running_mean").float().numpy().astype(np.float64)
        var = get(p + "running_var").float().numpy().astype(np.float64)
        scale = w / np.sqrt(var + eps)
        shift = b - mean * scale

        # Verify the identity on the REAL buffers, not on a toy. A probe with
        # production-realistic magnitude (the running stats say what scale the
        # activations live at) through both forms must agree to float64 epsilon.
        rng = np.random.default_rng(0)
        probe = mean[:, None] + np.sqrt(var)[:, None] * rng.standard_normal((w.size, 8))
        ref = (probe - mean[:, None]) / np.sqrt(var[:, None] + eps) * w[:, None] + b[:, None]
        got = probe * scale[:, None] + shift[:, None]
        err = float(np.max(np.abs(ref - got)))
        # Positive control: dropping the shift must be caught, so the tolerance
        # is known to be narrower than the defect it is guarding against.
        err_noshift = float(np.max(np.abs(ref - probe * scale[:, None])))
        if err > 1e-9 or err_noshift <= 1e-9:
            raise SystemExit(
                f"BatchNorm fold failed for block {i}: err={err:.3e}, "
                f"no-shift control={err_noshift:.3e}")
        if i == 0:
            print(f"  BN fold identity: max|delta| = {err:.3e} "
                  f"(no-shift control {err_noshift:.3e})")
        out[f"adapter.blk.{i}.conv.norm.scale"] = scale.astype(np.float32)
        out[f"adapter.blk.{i}.conv.norm.shift"] = shift.astype(np.float32)
    return out


def check_pe_table(pe, d_model):
    """The shipped RelPositionalEncoding table must match WeNet's CONVENTION.

    sin/cos are INTERLEAVED here (pe[:, 0::2] = sin, pe[:, 1::2] = cos) — the
    audio encoder's SinusoidsPositionEmbedding uses concat-halves instead. The
    two live 40 lines apart in the same forward pass, so the table is shipped
    verbatim and this check exists only to notice if the convention ever
    changes under us.

    It is deliberately NOT a precision check. torch builds the table in float32
    and `position * div_term` reaches ~5000, so sin/cos of it carries ~1e-4 of
    float32 error against a float64 recomputation — measured 4.1e-4 on the
    real checkpoint. A convention error (interleaved vs concat-halves, or a
    different base) moves entries by O(1). Gating anywhere between those two
    scales would be a test that fails on arithmetic and passes on the bug it
    was written for; both deltas are returned so the log shows which regime we
    are in.
    """
    max_len = pe.shape[0]
    out = {}
    for tag, dt in (("f64", np.float64), ("f32", np.float32)):
        pos = np.arange(max_len, dtype=dt)[:, None]
        div = np.exp(np.arange(0, d_model, 2, dtype=dt) * dt(-(np.log(10000.0) / d_model)))
        ref = np.zeros((max_len, d_model), dtype=dt)
        ref[:, 0::2] = np.sin(pos * div)
        ref[:, 1::2] = np.cos(pos * div)
        out[tag] = float(np.max(np.abs(ref.astype(np.float64) - pe.astype(np.float64))))
    return out


def main():
    parser = argparse.ArgumentParser(description="Convert Hojo-ASR-Multi-V1 to GGUF")
    parser.add_argument("--input", required=True, help="HF model ID or local directory")
    parser.add_argument("--output", required=True, help="Output GGUF path")
    parser.add_argument("--outtype", default="f16", choices=["f32", "f16"],
                        help="Output data type for 2D+ tensors (default: f16)")
    args = parser.parse_args()

    model_dir = load_model_dir(args.input)

    import yaml  # noqa: F401  (omegaconf/pyyaml both fine; yaml keeps deps light)
    with open(model_dir / "config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    mcfg = cfg["model"]
    gcfg = cfg.get("generate", {})

    enc_dir = model_dir / mcfg["encoder_path"]
    llm_dir = model_dir / mcfg["llama_path"]
    with open(enc_dir / "config.json", encoding="utf-8") as f:
        omni_cfg = json.load(f)
    ac = omni_cfg["thinker_config"]["audio_config"]
    with open(llm_dir / "config.json", encoding="utf-8") as f:
        lc = json.load(f)

    # hojo_asr_model.py patches these two in before constructing the encoder.
    n_window = 1500
    n_window_infer = 3000

    adapter_units = int(mcfg.get("linear_units", 1280))
    adapter_blocks = int(mcfg.get("num_blocks", 2))
    if mcfg.get("input_layer", "embed") != "linear":
        raise SystemExit(f"unsupported input_layer {mcfg.get('input_layer')!r} (only 'linear' is ported)")

    st_path = model_dir / "merged_full_model.safetensors"
    if not st_path.exists():
        cands = sorted(model_dir.glob("*.safetensors"))
        if not cands:
            raise SystemExit("no safetensors found")
        st_path = cands[0]

    h = safe_open(str(st_path), framework="pt")
    names = list(h.keys())
    print(f"\nHojo-ASR-Multi-V1  ({st_path.name}, {len(names)} tensors)")
    print(f"  Encoder: {ac['encoder_layers']}L d={ac['d_model']} heads={ac['encoder_attention_heads']} "
          f"ffn={ac['encoder_ffn_dim']} out={ac['output_dim']} "
          f"n_window={n_window} n_window_infer={n_window_infer}")
    print(f"  Adapter: {adapter_blocks} conformer blocks, d={lc['hidden_size']}, "
          f"linear_units={adapter_units}, kernel=15, heads=4")
    print(f"  LLM:     {lc['num_hidden_layers']}L d={lc['hidden_size']} "
          f"{lc['num_attention_heads']}Q/{lc['num_key_value_heads']}KV hd={lc['head_dim']} "
          f"ffn={lc['intermediate_size']} theta={lc['rope_theta']}")

    # tie_word_embeddings — prove it before dropping lm_head.
    if "decoder_model.lm_head.weight" in names:
        emb = h.get_tensor("decoder_model.model.embed_tokens.weight")
        lmh = h.get_tensor("decoder_model.lm_head.weight")
        if emb.shape != lmh.shape or not torch.equal(emb, lmh):
            raise SystemExit(
                "lm_head.weight differs from embed_tokens.weight — the runtime "
                "assumes tied embeddings; emit llm.lm_head.weight instead of dropping it")
        vocab_size = int(emb.shape[0])
        del emb, lmh
        print(f"  lm_head verified tied to embed_tokens (vocab {vocab_size})")
    else:
        vocab_size = int(h.get_slice("decoder_model.model.embed_tokens.weight").get_shape()[0])

    if args.outtype == "f16":
        out_dtype, ggml_type = np.float16, GGMLQuantizationType.F16
    else:
        out_dtype, ggml_type = np.float32, GGMLQuantizationType.F32

    outfile = Path(args.output)
    writer = GGUFWriter(str(outfile), ARCH, use_temp_file=True)
    writer.add_name("Hojo-ASR-Multi-V1")

    # ---- mel front end (WhisperFeatureExtractor, padding=False) ----
    n_mels, mel_n_fft, mel_hop = 128, 400, 160
    writer.add_uint32(f"{ARCH}.enc.num_mel_bins", n_mels)
    writer.add_uint32(f"{ARCH}.mel.n_fft", mel_n_fft)
    writer.add_uint32(f"{ARCH}.mel.hop_length", mel_hop)
    writer.add_uint32(f"{ARCH}.mel.sample_rate", 16000)

    # ---- audio encoder ----
    writer.add_uint32(f"{ARCH}.enc.encoder_layers", ac["encoder_layers"])
    writer.add_uint32(f"{ARCH}.enc.d_model", ac["d_model"])
    writer.add_uint32(f"{ARCH}.enc.encoder_attention_heads", ac["encoder_attention_heads"])
    writer.add_uint32(f"{ARCH}.enc.encoder_ffn_dim", ac["encoder_ffn_dim"])
    writer.add_uint32(f"{ARCH}.enc.downsample_hidden_size", ac["downsample_hidden_size"])
    writer.add_uint32(f"{ARCH}.enc.max_source_positions", ac["max_source_positions"])
    writer.add_uint32(f"{ARCH}.enc.output_dim", ac["output_dim"])
    writer.add_uint32(f"{ARCH}.enc.n_window", n_window)
    writer.add_uint32(f"{ARCH}.enc.n_window_infer", n_window_infer)
    writer.add_float32(f"{ARCH}.enc.layer_norm_eps", 1e-5)

    # ---- conformer adapter ----
    writer.add_uint32(f"{ARCH}.adapter.num_blocks", adapter_blocks)
    writer.add_uint32(f"{ARCH}.adapter.hidden_size", lc["hidden_size"])
    writer.add_uint32(f"{ARCH}.adapter.linear_units", adapter_units)
    writer.add_uint32(f"{ARCH}.adapter.attention_heads", 4)
    writer.add_uint32(f"{ARCH}.adapter.cnn_module_kernel", 15)
    writer.add_float32(f"{ARCH}.adapter.layer_norm_eps", 1e-5)
    writer.add_float32(f"{ARCH}.adapter.ff_scale", 0.5)

    # ---- LLM ----
    writer.add_uint32(f"{ARCH}.llm.hidden_size", lc["hidden_size"])
    writer.add_uint32(f"{ARCH}.llm.num_layers", lc["num_hidden_layers"])
    writer.add_uint32(f"{ARCH}.llm.num_heads", lc["num_attention_heads"])
    writer.add_uint32(f"{ARCH}.llm.num_kv_heads", lc["num_key_value_heads"])
    writer.add_uint32(f"{ARCH}.llm.head_dim", lc["head_dim"])
    writer.add_uint32(f"{ARCH}.llm.intermediate_size", lc["intermediate_size"])
    writer.add_uint32(f"{ARCH}.llm.vocab_size", vocab_size)
    writer.add_uint32(f"{ARCH}.llm.max_position_embeddings", lc["max_position_embeddings"])
    writer.add_float32(f"{ARCH}.llm.rope_theta", lc["rope_theta"])
    writer.add_float32(f"{ARCH}.llm.rms_norm_eps", lc["rms_norm_eps"])
    writer.add_bool(f"{ARCH}.llm.tied_embeddings", True)

    # ---- decode recipe (config.yaml `generate:`) ----
    # `HOJO_ASR.infer` passes every one of these to `generate` verbatim.
    writer.add_uint32(f"{ARCH}.bos_token_id", 151644)   # <|im_start|> == additional_special_tokens[0]
    writer.add_uint32(f"{ARCH}.eos_token_id", 151645)   # <|im_end|>
    writer.add_uint32(f"{ARCH}.gen.max_new_tokens", int(gcfg.get("max_new_tokens", 200)))
    writer.add_uint32(f"{ARCH}.gen.num_beams", int(gcfg.get("num_beams", 4)))
    writer.add_float32(f"{ARCH}.gen.repetition_penalty", float(gcfg.get("repetition_penalty", 1.0)))
    writer.add_float32(f"{ARCH}.gen.length_penalty", float(gcfg.get("length_penalty", 1.0)))

    # ---- mel filterbank + window, baked from the reference extractor ----
    try:
        from transformers import WhisperFeatureExtractor
        fe = WhisperFeatureExtractor(feature_size=n_mels, sampling_rate=16000,
                                     hop_length=mel_hop, n_fft=mel_n_fft, chunk_length=40)
        # transformers stores (num_frequency_bins, num_mel_filters) = (201, 128)
        # and multiplies by `mel_filters.T`. The runtime's core_mel wants the
        # MelsFreqs layout (128, 201), so the transpose happens HERE, once, next
        # to the source of truth -- not as a guess at load time.
        mel_filters = np.ascontiguousarray(
            np.asarray(fe.mel_filters, dtype=np.float32).T)
        assert mel_filters.shape == (n_mels, mel_n_fft // 2 + 1), mel_filters.shape
        writer.add_tensor("audio.mel_filters", mel_filters)
        # torch.hann_window defaults to PERIODIC: 0.5*(1 - cos(2*pi*i/N)).
        win = np.asarray(
            [0.5 * (1.0 - np.cos(2.0 * np.pi * i / mel_n_fft)) for i in range(mel_n_fft)],
            dtype=np.float32)
        writer.add_tensor("audio.mel_window", win)
        print(f"  mel_filters {mel_filters.shape} (mels x freqs), mel_window {win.shape}")
    except ImportError:
        print("  WARNING: transformers not available, skipping mel filter bake")

    # ---- tokenizer (Qwen3 byte-level BPE + the added [PAD]) ----
    with open(llm_dir / "tokenizer.json", encoding="utf-8") as f:
        tok_data = json.load(f)
    vocab = tok_data.get("model", {}).get("vocab", {})
    added = tok_data.get("added_tokens", [])
    tokens = [f"[PAD{i}]" for i in range(vocab_size)]
    for token, idx in vocab.items():
        if 0 <= idx < vocab_size:
            tokens[idx] = token
    for entry in added:
        tid, content = entry.get("id"), entry.get("content")
        if content and tid is not None and 0 <= tid < vocab_size:
            tokens[tid] = content
    # HOJO_ASR.__init__ does add_special_tokens({'pad_token': '[PAD]'}) then
    # resize_token_embeddings(len(tokenizer)) — one extra row past Qwen3's 151669.
    n_base = len(vocab) + len(added)
    for i in range(n_base, vocab_size):
        tokens[i] = "[PAD]" if i == n_base else f"[PAD{i}]"
    writer.add_tokenizer_model("gpt2")
    writer.add_token_list(tokens)
    print(f"  Tokenizer: {vocab_size} tokens ({len(vocab)} BPE + {len(added)} added + "
          f"{vocab_size - n_base} pad)")

    merges = [" ".join(m) if isinstance(m, list) else m
              for m in tok_data.get("model", {}).get("merges", [])]
    merges_path = llm_dir / "merges.txt"
    if merges_path.exists():
        merges = []
        with open(merges_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if line and not line.startswith("#"):
                    merges.append(line)
    if merges:
        writer.add_token_merges(merges)
        print(f"  Merges: {len(merges)}")

    # ---- folded conv BatchNorm ----
    folded = fold_conv_batchnorm(h.get_tensor, adapter_blocks)
    for name, arr in folded.items():
        writer.add_tensor(name, arr, raw_dtype=GGMLQuantizationType.F32)
    print(f"  Folded {len(folded) // 2} conv BatchNorm1d → scale/shift pairs")

    # ---- stream every remaining tensor ----
    mapped, skipped = 0, []
    for hf_name in sorted(names):
        gguf_name = map_tensor_name(hf_name)
        if gguf_name is None:
            skipped.append(hf_name)
            continue
        t = h.get_tensor(hf_name)
        arr = t.float().numpy() if t.dtype in (torch.bfloat16, torch.float16) else t.numpy()

        if gguf_name == "adapter.pe":
            # (1, 5000, 2560) → (5000, 2560); kept F32 (a 51 MB sin/cos table
            # whose values feed linear_pos, not the residual stream).
            arr = np.ascontiguousarray(arr[0])
            err = check_pe_table(arr, arr.shape[1])
            print(f"  adapter.pe {arr.shape}: max|shipped - WeNet formula| = "
                  f"{err['f32']:.3e} (f32 recompute), {err['f64']:.3e} (f64 recompute)")
            # O(1) means a different convention/base; ~1e-4 is float32 arithmetic.
            if min(err["f32"], err["f64"]) > 1e-2:
                raise SystemExit(
                    "adapter.pe does not match the RelPositionalEncoding convention "
                    f"(min delta {min(err['f32'], err['f64']):.3e}) — check interleaved "
                    "vs concat-halves and the rotary base before shipping this")
            writer.add_tensor(gguf_name, arr.astype(np.float32),
                              raw_dtype=GGMLQuantizationType.F32)
            mapped += 1
            del t, arr
            continue

        # pos_bias_u / pos_bias_v are 2-D (n_heads, head_dim) but are ADDED to
        # an F32 Q in the graph, and ggml's broadcast ops reject F32 (+) F16 on
        # every backend. They are 2560 floats each, so keep them F32 rather
        # than paying for an in-graph cast on every layer of every call.
        force_f32 = gguf_name.endswith(".pos_bias_u") or gguf_name.endswith(".pos_bias_v")
        if arr.ndim >= 2 and not force_f32:
            arr = np.ascontiguousarray(arr.astype(out_dtype))
            dtype = ggml_type
        else:
            arr = np.ascontiguousarray(arr.astype(np.float32))
            dtype = GGMLQuantizationType.F32
        writer.add_tensor(gguf_name, arr, raw_dtype=dtype)
        mapped += 1
        del t, arr

    print(f"\n  Mapped {mapped} tensors, skipped {len(skipped)}")
    for s in skipped[:8]:
        print(f"    SKIP: {s}")
    if len(skipped) > 8:
        print(f"    ... and {len(skipped) - 8} more")

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"\n  Written: {outfile} ({outfile.stat().st_size / 1024**3:.2f} GB)")


if __name__ == "__main__":
    main()
