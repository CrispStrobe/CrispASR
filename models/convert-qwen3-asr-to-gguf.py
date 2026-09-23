#!/usr/bin/env python3
"""
Convert Qwen3-ASR (HuggingFace safetensors) → GGUF F16.

Supports both model formats:
  - Non-HF (thinker.*):  Qwen/Qwen3-ASR-0.6B, etc.
  - HF (-hf):            Qwen/Qwen3-ASR-0.6B-hf, Qwen/Qwen3-ASR-1.7B-hf, etc.
  - Fine-tunes:          jaykwok/Qwen3-ASR-1.7B-JA-Anime-Galgame-hf, etc.

Architecture (all sizes share the same structure):

  Audio tower:
    conv2d{1,2,3}                 stride-2 freq subsampling
    conv_out                      linear: conv_ch × 16freq → audio_d_model
    N × encoder block (Whisper-block style, pre-LN, biased)
    ln_post
    projector (proj1/GELU/proj2)  audio_d_model → llm_d_model

  Text decoder (Qwen3 LLM, GQA + SwiGLU):
    embed_tokens
    M × decoder block (RMSNorm, QKV + q/k RMSNorm, SwiGLU MLP)
    norm + lm_head

  Model sizes:
    0.6B: audio 18L d=896, LLM 28L d=1024
    1.7B: audio 24L d=1024, LLM 28L d=2048

GGUF tensor naming (C++ loader expects):
  audio.conv.{1,2,3}.{weight,bias}
  audio.conv_out.{weight,bias}
  audio.blk.{i}.attn_norm.{weight,bias}
  audio.blk.{i}.attn_{q,k,v,out}.{weight,bias}
  audio.blk.{i}.ffn_norm.{weight,bias}
  audio.blk.{i}.ffn_up.{weight,bias}   (fc1)
  audio.blk.{i}.ffn_down.{weight,bias} (fc2)
  audio.ln_post.{weight,bias}
  audio.proj1.{weight,bias}
  audio.proj2.{weight,bias}
  token_embd.weight
  blk.{i}.attn_norm.weight
  blk.{i}.attn_{q,k,v,output}.weight
  blk.{i}.attn_{q,k}_norm.weight
  blk.{i}.ffn_norm.weight
  blk.{i}.ffn_{gate,up,down}.weight
  output_norm.weight
  output.weight
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
# Tensor name remapping — supports both non-hf (thinker.*) and hf (model.*)
# ---------------------------------------------------------------------------

# Direct mappings for non-hf format (thinker.* prefix)
DIRECT_THINKER = {
    "thinker.audio_tower.conv2d1.weight": "audio.conv.1.weight",
    "thinker.audio_tower.conv2d1.bias": "audio.conv.1.bias",
    "thinker.audio_tower.conv2d2.weight": "audio.conv.2.weight",
    "thinker.audio_tower.conv2d2.bias": "audio.conv.2.bias",
    "thinker.audio_tower.conv2d3.weight": "audio.conv.3.weight",
    "thinker.audio_tower.conv2d3.bias": "audio.conv.3.bias",
    "thinker.audio_tower.conv_out.weight": "audio.conv_out.weight",
    "thinker.audio_tower.conv_out.bias": "audio.conv_out.bias",
    "thinker.audio_tower.ln_post.weight": "audio.ln_post.weight",
    "thinker.audio_tower.ln_post.bias": "audio.ln_post.bias",
    "thinker.audio_tower.proj1.weight": "audio.proj1.weight",
    "thinker.audio_tower.proj1.bias": "audio.proj1.bias",
    "thinker.audio_tower.proj2.weight": "audio.proj2.weight",
    "thinker.audio_tower.proj2.bias": "audio.proj2.bias",
    "thinker.model.embed_tokens.weight": "token_embd.weight",
    "thinker.model.norm.weight": "output_norm.weight",
    "thinker.lm_head.weight": "output.weight",
}

# Direct mappings for HF format (model.* prefix)
DIRECT_HF = {
    "model.audio_tower.conv2d1.weight": "audio.conv.1.weight",
    "model.audio_tower.conv2d1.bias": "audio.conv.1.bias",
    "model.audio_tower.conv2d2.weight": "audio.conv.2.weight",
    "model.audio_tower.conv2d2.bias": "audio.conv.2.bias",
    "model.audio_tower.conv2d3.weight": "audio.conv.3.weight",
    "model.audio_tower.conv2d3.bias": "audio.conv.3.bias",
    "model.audio_tower.conv_out.weight": "audio.conv_out.weight",
    "model.audio_tower.conv_out.bias": "audio.conv_out.bias",
    "model.audio_tower.ln_post.weight": "audio.ln_post.weight",
    "model.audio_tower.ln_post.bias": "audio.ln_post.bias",
    # HF format: projector is under multi_modal_projector
    "model.multi_modal_projector.linear_1.weight": "audio.proj1.weight",
    "model.multi_modal_projector.linear_1.bias": "audio.proj1.bias",
    "model.multi_modal_projector.linear_2.weight": "audio.proj2.weight",
    "model.multi_modal_projector.linear_2.bias": "audio.proj2.bias",
    "model.language_model.embed_tokens.weight": "token_embd.weight",
    "model.language_model.norm.weight": "output_norm.weight",
    "model.language_model.lm_head.weight": "output.weight",
    # Some HF models have lm_head at top level
    "lm_head.weight": "output.weight",
}

# #455 KRAFTON/Raon-Speech-9B (model_type "raon"): only the speech-to-text
# subset is converted. Talker, code predictor, Mimi codec, output adaptor and
# speaker encoder are skipped (they only serve speech output).
DIRECT_RAON = {
    "audio_encoder.encoder.conv2d1.weight": "audio.conv.1.weight",
    "audio_encoder.encoder.conv2d1.bias": "audio.conv.1.bias",
    "audio_encoder.encoder.conv2d2.weight": "audio.conv.2.weight",
    "audio_encoder.encoder.conv2d2.bias": "audio.conv.2.bias",
    "audio_encoder.encoder.conv2d3.weight": "audio.conv.3.weight",
    "audio_encoder.encoder.conv2d3.bias": "audio.conv.3.bias",
    "audio_encoder.encoder.conv_out.weight": "audio.conv_out.weight",
    "audio_encoder.encoder.conv_out.bias": "audio.conv_out.bias",
    "audio_encoder.encoder.ln_post.weight": "audio.ln_post.weight",
    "audio_encoder.encoder.ln_post.bias": "audio.ln_post.bias",
    "audio_encoder.encoder.proj1.weight": "audio.proj1.weight",
    "audio_encoder.encoder.proj1.bias": "audio.proj1.bias",
    "audio_encoder.encoder.proj2.weight": "audio.proj2.weight",
    "audio_encoder.encoder.proj2.bias": "audio.proj2.bias",
    "input_adaptor.proj.0.weight": "adaptor.fc1.weight",
    "input_adaptor.proj.2.weight": "adaptor.fc2.weight",
    "input_adaptor.post_norm.weight": "adaptor.norm.weight",
    "text_model.embed_tokens.weight": "token_embd.weight",
    "text_model.norm.weight": "output_norm.weight",
    "lm_head.weight": "output.weight",
}

# Audio layer patterns for both formats
AUDIO_LAYER_PATTERNS_THINKER = [
    (r"thinker\.audio_tower\.layers\.(\d+)\.", "audio.blk.{}."),
]
AUDIO_LAYER_PATTERNS_HF = [
    (r"model\.audio_tower\.layers\.(\d+)\.", "audio.blk.{}."),
]

# Sub-patterns within an audio layer (shared between both formats)
AUDIO_SUB = {
    "self_attn_layer_norm.weight": "attn_norm.weight",
    "self_attn_layer_norm.bias": "attn_norm.bias",
    "self_attn.q_proj.weight": "attn_q.weight",
    "self_attn.q_proj.bias": "attn_q.bias",
    "self_attn.k_proj.weight": "attn_k.weight",
    "self_attn.k_proj.bias": "attn_k.bias",
    "self_attn.v_proj.weight": "attn_v.weight",
    "self_attn.v_proj.bias": "attn_v.bias",
    "self_attn.out_proj.weight": "attn_out.weight",
    "self_attn.out_proj.bias": "attn_out.bias",
    "final_layer_norm.weight": "ffn_norm.weight",
    "final_layer_norm.bias": "ffn_norm.bias",
    "fc1.weight": "ffn_up.weight",
    "fc1.bias": "ffn_up.bias",
    "fc2.weight": "ffn_down.weight",
    "fc2.bias": "ffn_down.bias",
}

# Text layer patterns for both formats
TEXT_LAYER_PATTERNS_THINKER = [
    (r"thinker\.model\.layers\.(\d+)\.", "blk.{}."),
]
TEXT_LAYER_PATTERNS_HF = [
    (r"model\.language_model\.layers\.(\d+)\.", "blk.{}."),
]

# Sub-patterns within a text layer (shared between both formats)
TEXT_SUB = {
    "input_layernorm.weight": "attn_norm.weight",
    "self_attn.q_proj.weight": "attn_q.weight",
    "self_attn.k_proj.weight": "attn_k.weight",
    "self_attn.v_proj.weight": "attn_v.weight",
    "self_attn.o_proj.weight": "attn_output.weight",
    "self_attn.q_norm.weight": "attn_q_norm.weight",
    "self_attn.k_norm.weight": "attn_k_norm.weight",
    "post_attention_layernorm.weight": "ffn_norm.weight",
    "mlp.gate_proj.weight": "ffn_gate.weight",
    "mlp.up_proj.weight": "ffn_up.weight",
    "mlp.down_proj.weight": "ffn_down.weight",
}


def detect_format(tensor_names: list[str]) -> str:
    """Detect whether safetensors use thinker.* or model.* prefix."""
    for name in tensor_names:
        if name.startswith(("audio_encoder.encoder.", "text_model.", "input_adaptor.")):
            return "raon"
        if name.startswith("thinker."):
            return "thinker"
        if name.startswith("model."):
            return "hf"
    return "hf"  # default to hf


def build_remap(fmt: str) -> callable:
    """Build a remap function for the detected format."""
    direct = DIRECT_THINKER if fmt == "thinker" else DIRECT_HF
    audio_pats = AUDIO_LAYER_PATTERNS_THINKER if fmt == "thinker" else AUDIO_LAYER_PATTERNS_HF
    text_pats = TEXT_LAYER_PATTERNS_THINKER if fmt == "thinker" else TEXT_LAYER_PATTERNS_HF
    if fmt == "raon":
        direct = DIRECT_RAON
        audio_pats = [(r"audio_encoder\.encoder\.layers\.(\d+)\.", "audio.blk.{}.")]
        text_pats = [(r"text_model\.layers\.(\d+)\.", "blk.{}.")]

    def remap_name(hf_name: str) -> str | None:
        if hf_name in direct:
            return direct[hf_name]
        # Audio layer patterns
        for pat, tmpl in audio_pats:
            m = re.match(pat, hf_name)
            if m:
                layer_idx = m.group(1)
                suffix = hf_name[m.end():]
                if suffix in AUDIO_SUB:
                    return tmpl.format(layer_idx) + AUDIO_SUB[suffix]
                return None
        # Text layer patterns
        for pat, tmpl in text_pats:
            m = re.match(pat, hf_name)
            if m:
                layer_idx = m.group(1)
                suffix = hf_name[m.end():]
                if suffix in TEXT_SUB:
                    return tmpl.format(layer_idx) + TEXT_SUB[suffix]
                return None
        return None

    return remap_name


def is_f32_tensor(gguf_name: str, shape: tuple[int, ...]) -> bool:
    """Norms, biases, 1-D tensors stay F32 for accuracy."""
    if gguf_name.endswith(".bias"):
        return True
    if "norm" in gguf_name or "ln_post" in gguf_name:
        return True
    if len(shape) <= 1:
        return True
    return False


def load_tokenizer(input_dir: Path, vocab_size: int) -> tuple[list[str], list[str]]:
    """Load vocab tokens and BPE merges from either vocab.json or tokenizer.json."""
    vocab_path = input_dir / "vocab.json"
    merges_path = input_dir / "merges.txt"
    tokenizer_json_path = input_dir / "tokenizer.json"

    tokens: list[str] = []
    merges: list[str] = []

    if vocab_path.exists():
        # Non-hf format: separate vocab.json + merges.txt
        with open(vocab_path, "r", encoding="utf-8") as f:
            vocab_dict = json.load(f)
        sorted_vocab = sorted(vocab_dict.items(), key=lambda kv: kv[1])
        tokens = [tok for tok, _ in sorted_vocab]
    elif tokenizer_json_path.exists():
        # HF format: tokenizer.json (fast tokenizer)
        with open(tokenizer_json_path, "r", encoding="utf-8") as f:
            tok_data = json.load(f)
        # Extract vocab from model.vocab
        model_data = tok_data.get("model", {})
        vocab_dict = model_data.get("vocab", {})
        sorted_vocab = sorted(vocab_dict.items(), key=lambda kv: kv[1])
        tokens = [tok for tok, _ in sorted_vocab]
        # Extract merges
        raw_merges = model_data.get("merges", [])
        # Merges may be strings ("a b") or lists (["a", "b"])
        for m in raw_merges:
            if isinstance(m, list):
                merges.append(" ".join(m))
            elif isinstance(m, str) and m and not m.startswith("#"):
                merges.append(m)
        print(f"  tokenizer.json: {len(tokens)} vocab, {len(merges)} merges")
    else:
        sys.exit(f"no vocab.json or tokenizer.json found in {input_dir}")

    # Pad to vocab_size
    while len(tokens) < vocab_size:
        tokens.append(f"[PAD{len(tokens)}]")

    # Patch special tokens from tokenizer_config.json
    tcfg_path = input_dir / "tokenizer_config.json"
    if tcfg_path.exists():
        with open(tcfg_path, "r", encoding="utf-8") as f:
            tcfg = json.load(f)
        added = tcfg.get("added_tokens_decoder", {})
        for tid_str, info in added.items():
            tid = int(tid_str)
            content = info.get("content")
            if content and 0 <= tid < len(tokens):
                tokens[tid] = content
        print(f"  patched {len(added)} added/special tokens from tokenizer_config.json")

    # Also patch from tokenizer.json added_tokens
    if tokenizer_json_path.exists():
        with open(tokenizer_json_path, "r", encoding="utf-8") as f:
            tok_data = json.load(f)
        for at in tok_data.get("added_tokens", []):
            tid = at.get("id")
            content = at.get("content")
            if tid is not None and content and 0 <= tid < len(tokens):
                tokens[tid] = content

    print(f"  vocab: {len(tokens)} tokens")

    # Load merges if not yet loaded (from merges.txt)
    if not merges and merges_path.exists():
        with open(merges_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if line and not line.startswith("#"):
                    merges.append(line)
    if merges:
        print(f"  merges: {len(merges)}")

    return tokens, merges


# ---------------------------------------------------------------------------
# Mel filterbank (Slaney-normalized, matching WhisperFeatureExtractor)
# ---------------------------------------------------------------------------


def _hz_to_mel_slaney(hz: np.ndarray) -> np.ndarray:
    """Slaney/O'Shaughnessy mel scale (matches librosa default htk=False).
    Linear below 1000 Hz, logarithmic above."""
    f_sp = 200.0 / 3.0  # ~66.67 Hz per mel below 1kHz
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp  # = 15.0
    logstep = np.log(6.4) / 27.0

    mel = np.where(
        hz < min_log_hz,
        hz / f_sp,
        min_log_mel + np.log(hz / min_log_hz) / logstep,
    )
    return mel


def _mel_to_hz_slaney(mel: np.ndarray) -> np.ndarray:
    f_sp = 200.0 / 3.0
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = np.log(6.4) / 27.0

    hz = np.where(
        mel < min_log_mel,
        mel * f_sp,
        min_log_hz * np.exp(logstep * (mel - min_log_mel)),
    )
    return hz


def _compute_mel_filters(sr: int = 16000, n_fft: int = 400, n_mels: int = 128) -> np.ndarray:
    """Compute Slaney-normalized mel filterbank matching WhisperFeatureExtractor
    (i.e. librosa.filters.mel with htk=False, norm='slaney')).
    Returns shape (n_freqs, n_mels) in F32, where n_freqs = n_fft//2 + 1."""
    n_freqs = n_fft // 2 + 1
    f_min, f_max = 0.0, float(sr) / 2.0

    mel_min = _hz_to_mel_slaney(np.array([f_min]))[0]
    mel_max = _hz_to_mel_slaney(np.array([f_max]))[0]
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = _mel_to_hz_slaney(mel_points)
    bin_freqs = np.linspace(0.0, f_max, n_freqs)

    fb = np.zeros((n_freqs, n_mels), dtype=np.float32)
    for m in range(n_mels):
        f_left, f_center, f_right = hz_points[m], hz_points[m + 1], hz_points[m + 2]
        for k in range(n_freqs):
            if f_left <= bin_freqs[k] < f_center and f_center > f_left:
                fb[k, m] = (bin_freqs[k] - f_left) / (f_center - f_left)
            elif f_center <= bin_freqs[k] <= f_right and f_right > f_center:
                fb[k, m] = (f_right - bin_freqs[k]) / (f_right - f_center)
        # Slaney normalization: 2 / bandwidth
        bw = hz_points[m + 2] - hz_points[m]
        if bw > 0:
            fb[:, m] *= 2.0 / bw

    return np.ascontiguousarray(fb)


# ---------------------------------------------------------------------------
# Main conversion
# ---------------------------------------------------------------------------


def convert(input_dir: Path, out_path: Path, streaming_recipe: str = "") -> None:
    print(f"Loading: {input_dir}")
    with open(input_dir / "config.json", "r", encoding="utf-8") as f:
        cfg = json.load(f)

    raon = cfg.get("model_type") == "raon"
    # Detect config format: non-hf has thinker_config wrapper, hf has flat structure
    if raon:
        audio = cfg["audio_encoder_config"]
        text = cfg["text_model_config"]
        adaptor_cfg = cfg.get("input_adaptor_config") or {}
        if int(adaptor_cfg.get("output_time_scale", 1)) != 1 or not adaptor_cfg.get("use_post_norm", True):
            sys.exit(f"unsupported Raon input adaptor config: {adaptor_cfg}")
        if cfg.get("aut_is_causal"):
            sys.exit("causal AuT encoders are not supported (aut_is_causal=true)")
        # modeling_raon.py: <|audio_start|> 151669, <|audio_end|> 151670, the
        # STT placeholder is <|audio_input_placeholder|> 151676.
        audio_start_id, audio_end_id, audio_pad_id = 151669, 151670, 151676
        tie_word_embeddings = bool(text.get("tie_word_embeddings", False))
        print("  config format: raon (speech-to-text subset)")
    elif "thinker_config" in cfg:
        thinker = cfg["thinker_config"]
        audio = thinker["audio_config"]
        text = thinker["text_config"]
        # Special tokens from thinker level
        audio_start_id = thinker.get("audio_start_token_id", 151669)
        audio_end_id = thinker.get("audio_end_token_id", 151670)
        audio_pad_id = thinker.get("audio_token_id", 151676)
        # The flag can sit on the thinker OR on its text_config (Confucius4-R2T2,
        # #445, sets only text_config.tie_word_embeddings and ships no lm_head).
        tie_word_embeddings = thinker.get("tie_word_embeddings", text.get("tie_word_embeddings", False))
        print("  config format: non-hf (thinker_config)")
    else:
        audio = cfg["audio_config"]
        text = cfg["text_config"]
        # Special tokens from top level for hf format
        audio_start_id = cfg.get("audio_start_token_id", 151669)
        audio_end_id = cfg.get("audio_end_token_id", 151670)
        audio_pad_id = cfg.get("audio_token_id", cfg.get("audio_token_id", 151676))
        tie_word_embeddings = cfg.get("tie_word_embeddings", text.get("tie_word_embeddings", False))
        print("  config format: hf (flat)")

    # Handle rope_theta: may be nested under rope_parameters or at text level
    rope_theta = text.get("rope_theta", None)
    if rope_theta is None:
        rp = text.get("rope_parameters", {})
        rope_theta = rp.get("rope_theta", 1000000)

    vocab_size = text.get("vocab_size", 151936)

    safetensor_files = sorted(input_dir.glob("*.safetensors"))
    if not safetensor_files:
        sys.exit(f"no safetensors files found in {input_dir}")

    # Detect tensor naming format
    with safe_open(str(safetensor_files[0]), framework="pt", device="cpu") as f:
        sample_names = list(f.keys())[:10]
    fmt = detect_format(sample_names)
    print(f"  tensor format: {fmt}")

    # Tokenizer
    tokens, merges = load_tokenizer(input_dir, vocab_size)

    # ----- write GGUF -----
    print(f"Writing: {out_path}")
    writer = gguf.GGUFWriter(str(out_path), arch="qwen3asr", use_temp_file=True)

    # Audio params
    writer.add_uint32("qwen3asr.sample_rate", 16000)
    if streaming_recipe:
        # Selects the streaming schedule the CLI/server realtime session uses
        # by default (examples/cli/crispasr_backend_qwen3.cpp). "r2t2" is
        # Confucius4-R2T2's example.py driver: 160 ms steps, rollback 1 token.
        writer.add_string("qwen3asr.streaming_recipe", streaming_recipe)
    writer.add_uint32("qwen3asr.n_mels", audio.get("num_mel_bins", 128))
    writer.add_uint32("qwen3asr.n_fft", 400)
    writer.add_uint32("qwen3asr.win_length", 400)
    writer.add_uint32("qwen3asr.hop_length", 160)
    writer.add_uint32("qwen3asr.audio.n_layers", audio.get("encoder_layers", 18))
    writer.add_uint32("qwen3asr.audio.d_model", audio.get("d_model", 896))
    writer.add_uint32(
        "qwen3asr.audio.n_heads", audio.get("encoder_attention_heads", 14)
    )
    writer.add_uint32(
        "qwen3asr.audio.head_dim",
        audio.get("d_model", 896) // audio.get("encoder_attention_heads", 14),
    )
    writer.add_uint32("qwen3asr.audio.ff_dim", audio.get("encoder_ffn_dim", 3584))
    writer.add_uint32(
        "qwen3asr.audio.conv_channels", audio.get("downsample_hidden_size", 480)
    )
    writer.add_uint32("qwen3asr.audio.proj_dim", audio.get("output_dim", 1024))
    writer.add_uint32(
        "qwen3asr.audio.max_source_pos", audio.get("max_source_positions", 1500)
    )

    # LLM params
    writer.add_uint32("qwen3asr.llm.n_layers", text.get("num_hidden_layers", 28))
    writer.add_uint32("qwen3asr.llm.d_model", text.get("hidden_size", 1024))
    writer.add_uint32("qwen3asr.llm.n_heads", text.get("num_attention_heads", 16))
    writer.add_uint32("qwen3asr.llm.n_kv_heads", text.get("num_key_value_heads", 8))
    writer.add_uint32("qwen3asr.llm.head_dim", text.get("head_dim", 128))
    writer.add_uint32("qwen3asr.llm.ff_dim", text.get("intermediate_size", 3072))
    writer.add_float32("qwen3asr.llm.rope_theta", float(rope_theta))
    writer.add_float32(
        "qwen3asr.llm.rms_norm_eps", float(text.get("rms_norm_eps", 1e-6))
    )
    writer.add_uint32("qwen3asr.llm.vocab_size", vocab_size)
    writer.add_uint32(
        "qwen3asr.llm.max_pos", text.get("max_position_embeddings", 65536)
    )

    # Special tokens
    writer.add_uint32("qwen3asr.audio_start_token_id", audio_start_id)
    writer.add_uint32("qwen3asr.audio_end_token_id", audio_end_id)
    writer.add_uint32("qwen3asr.audio_pad_token_id", audio_pad_id)
    writer.add_uint32("qwen3asr.eos_token_id", 151645)
    writer.add_uint32("qwen3asr.pad_token_id", 151643)
    if raon:
        writer.add_string("qwen3asr.variant", "raon-speech")
        writer.add_float32("qwen3asr.adaptor.norm_eps", float(adaptor_cfg.get("norm_eps", 1e-6)))
        # 8 s STT chunks at the 24 kHz processor rate (RaonPipeline.stt), and
        # the 12.5 Hz frame grid (24000 / 1920) the placeholders count in.
        writer.add_uint32("qwen3asr.raon.chunk_samples", 192000)
        writer.add_uint32("qwen3asr.raon.samples_per_frame", 1920)
        writer.add_uint32("qwen3asr.raon.audio_output_pad_id", 151677)

    # Tokenizer
    writer.add_tokenizer_model("gpt2")
    writer.add_token_list(tokens)
    if merges:
        writer.add_token_merges(merges)

    # ----- Mel filterbank + Hann window (baked from WhisperFeatureExtractor) -----
    # Compute Slaney mel filterbank matching WhisperFeatureExtractor without importing
    # heavy transformers/librosa — critical for 8GB VPS where memory is tight.
    mel_filters = _compute_mel_filters(sr=16000, n_fft=400, n_mels=128)
    print(f"  mel_filters shape: {mel_filters.shape}")
    writer.add_tensor("audio.mel_filters", mel_filters)

    n_fft_w = 400
    win = np.ascontiguousarray(
        (0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(n_fft_w) / n_fft_w)).astype(
            np.float32
        )
    )
    writer.add_tensor("audio.mel_window", win)
    print(f"  mel_window shape: {win.shape}")

    # ----- tensors -----
    import gc
    import torch

    remap_name = build_remap(fmt)
    n_written = 0
    n_f16 = 0
    n_f32 = 0
    n_skipped = 0
    skipped_names: list[str] = []
    has_output_weight = False
    token_embd_data = None  # saved for tie_word_embeddings

    for sf_path in safetensor_files:
        print(f"  reading {sf_path.name}")
        with safe_open(str(sf_path), framework="pt", device="cpu") as f:
            for hf_name in sorted(f.keys()):
                gguf_name = remap_name(hf_name)
                if gguf_name is None:
                    n_skipped += 1
                    skipped_names.append(hf_name)
                    continue

                t = f.get_tensor(hf_name)
                # safetensors gives torch tensors; bf16 needs float32 detour
                if t.dtype == torch.bfloat16:
                    arr = t.float().numpy()
                else:
                    arr = t.numpy()
                del t

                if arr.dtype == np.float64:
                    arr = arr.astype(np.float32)

                if is_f32_tensor(gguf_name, arr.shape):
                    arr = arr.astype(np.float32)
                    n_f32 += 1
                else:
                    arr = arr.astype(np.float16)
                    n_f16 += 1

                if not arr.flags["C_CONTIGUOUS"]:
                    arr = np.ascontiguousarray(arr)

                if gguf_name == "output.weight":
                    has_output_weight = True
                if gguf_name == "token_embd.weight" and tie_word_embeddings:
                    token_embd_data = arr.copy()

                writer.add_tensor(gguf_name, arr)
                n_written += 1
                if n_written <= 25 or n_written % 100 == 0:
                    print(f"    {gguf_name:50s} {str(arr.shape):24s} {arr.dtype}")

        # Free memory between safetensor files
        gc.collect()

    # Handle tie_word_embeddings: if no separate lm_head, copy embed_tokens
    if not has_output_weight and tie_word_embeddings and token_embd_data is not None:
        print("  tie_word_embeddings: copying token_embd.weight → output.weight")
        writer.add_tensor("output.weight", token_embd_data)
    elif not has_output_weight:
        # The runtime requires output.weight; a GGUF without it cannot load.
        raise SystemExit("error: checkpoint has no lm_head and the config does not set "
                         "tie_word_embeddings — refusing to write a GGUF without output.weight")
        n_written += 1
        n_f16 += 1

    print(
        f"\n  total: {n_written} tensors  (F16: {n_f16}, F32: {n_f32})  "
        f"skipped: {n_skipped}"
    )
    if skipped_names:
        print("  skipped tensors (first 10):")
        for n in skipped_names[:10]:
            print(f"    {n}")

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"\nDone: {out_path}  ({out_path.stat().st_size / 1e9:.2f} GB)")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert Qwen3-ASR HF safetensors → GGUF F16"
    )
    p.add_argument("--input", required=True, type=Path, help="HF model directory")
    p.add_argument("--output", required=True, type=Path, help="output GGUF path")
    p.add_argument("--streaming-recipe", default="", choices=["", "r2t2"],
                   help="record the model's streaming recipe (r2t2 = Confucius4-R2T2, #445)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    convert(args.input, args.output, args.streaming_recipe)
