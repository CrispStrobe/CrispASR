#!/usr/bin/env python3
"""
Convert ResembleAI/chatterbox safetensors → GGUF for the CrispASR
`chatterbox` backend.

Chatterbox pipeline:
  1. T3 (520M Llama AR) — text + conditioning → speech tokens @25 Hz
  2. S3Gen — speech tokens → mel-spectrogram via CFM (flow matching)
     - UpsampleConformerEncoder (6+4 blocks, 512D, 8 heads)
     - ConditionalDecoder (UNet1D, 1 down + 12 mid + 1 up, 256 ch)
     - Euler ODE solver, 10 steps, cosine schedule
  3. HiFTGenerator — mel → 24 kHz waveform (F0 predictor + iSTFT)
  4. VoiceEncoder — 3-layer LSTM, 256D speaker embedding
  5. CAMPPlus — 192D x-vector for S3Gen speaker conditioning
  6. S3Tokenizer — tokenizes reference audio (WhisperV2-style)

Produces TWO GGUFs:
  - chatterbox-t3.gguf   — T3 model + VE + character tokenizer + precomputed conds
  - chatterbox-s3gen.gguf — S3Gen flow + HiFTGenerator + CAMPPlus + S3Tokenizer

Usage:
    python models/convert-chatterbox-to-gguf.py \\
        --input /mnt/storage/chatterbox \\
        --output-dir /mnt/storage/chatterbox

    # or from HuggingFace:
    python models/convert-chatterbox-to-gguf.py \\
        --input ResembleAI/chatterbox \\
        --output-dir .
"""

from __future__ import annotations

import argparse
import json
import struct
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
    snapshot_download = None


# ── Architecture constants ──────────────────────────────────────────

T3_HPARAMS = dict(
    arch="chatterbox",
    n_layers=30,
    n_heads=16,
    n_kv_heads=16,
    hidden_size=1024,
    intermediate_size=4096,
    head_dim=64,
    rms_norm_eps=1e-5,
    rope_theta=500000.0,
    rope_factor=8.0,
    rope_high_freq_factor=4.0,
    rope_low_freq_factor=1.0,
    rope_original_max_pos=8192,
    text_vocab_size=704,
    speech_vocab_size=8194,
    text_pos_emb_size=2050,
    speech_pos_emb_size=4100,
    start_text_token=255,
    stop_text_token=0,
    start_speech_token=6561,
    stop_speech_token=6562,
    speech_cond_prompt_len=150,
    speaker_embed_size=256,
    perceiver_n_queries=32,
    perceiver_n_heads=4,
)

S3GEN_HPARAMS = dict(
    # Conformer encoder
    enc_n_layers=6,
    enc_up_n_layers=4,
    enc_hidden=512,
    enc_heads=8,
    enc_ffn=2048,
    enc_head_dim=64,
    # CFM decoder (UNet1D)
    dec_in_channels=320,
    dec_out_channels=80,
    dec_channels=256,
    dec_n_down=1,
    dec_n_mid=12,
    dec_n_up=1,
    dec_n_blocks=4,  # transformer blocks per UNet block
    dec_n_heads=8,
    dec_head_dim=64,
    # CFM solver
    cfm_n_steps=10,
    cfm_sigma_min=1e-6,
    cfm_inference_cfg_rate=0.7,
    # Vocoder (HiFTGenerator)
    voc_upsample_rates="8,5,3",
    voc_upsample_kernels="16,11,7",
    voc_resblock_kernels="3,7,11",
    voc_source_resblock_kernels="7,7,11",
    voc_base_channels=512,
    voc_istft_n_fft=16,
    voc_istft_hop_len=4,
    voc_nb_harmonics=8,
    # Speaker encoder (CAMPPlus)
    spk_enc_dim=192,
    # S3Tokenizer
    s3_vocab_size=6561,
    s3_token_rate=25,
    # mel
    mel_channels=80,
    sample_rate=24000,
)

VE_HPARAMS = dict(
    ve_num_mels=40,
    ve_hidden_size=256,
    ve_n_layers=3,
    ve_embed_size=256,
    ve_sample_rate=16000,
)

KARTOFFELBOX_T3_HPARAMS = dict(
    arch="kartoffelbox",
    n_layers=24,
    n_heads=16,
    hidden_size=1024,
    intermediate_size=4096,
    head_dim=64,
    text_vocab_size=50276,
    speech_vocab_size=6563,
    wpe_max_positions=8196,
)


# ── Helpers ─────────────────────────────────────────────────────────

def load_model_dir(model_id: str) -> Path:
    p = Path(model_id)
    if p.is_dir():
        return p
    if snapshot_download is None:
        sys.exit(f"Directory {model_id} not found and huggingface_hub not installed")
    print(f"Downloading {model_id}…", file=sys.stderr)
    return Path(snapshot_download(model_id, allow_patterns=[
        "*.safetensors", "*.json", "*.pt", "tokenizer*",
    ]))


def denorm_weight_norm(original0, original1):
    """Denormalize PyTorch weight_norm parametrization.
    original0 is the magnitude (g), original1 is the direction (v).
    weight = g * v / ||v||
    """
    # original0: (out_ch, 1, 1) or (out_ch,) — magnitude
    # original1: (out_ch, in_ch, kernel) — direction
    g = original0
    v = original1
    # Compute norm over all dims except first
    norm_dims = tuple(range(1, v.ndim))
    v_norm = torch.norm(v, dim=norm_dims, keepdim=True)
    return (g * v / (v_norm + 1e-12))


def load_safetensors(path: Path) -> dict:
    """Load all tensors from a safetensors file, denormalizing weight_norm."""
    tensors = {}
    with safe_open(str(path), framework='pt') as f:
        keys = list(f.keys())
        # First pass: collect all tensors
        raw = {}
        for k in keys:
            raw[k] = f.get_tensor(k)

    # Second pass: denormalize weight_norm pairs
    wn_bases = set()
    for k in raw:
        if '.parametrizations.weight.original0' in k:
            base = k.replace('.parametrizations.weight.original0', '')
            wn_bases.add(base)

    for base in wn_bases:
        o0 = raw.pop(f'{base}.parametrizations.weight.original0')
        o1 = raw.pop(f'{base}.parametrizations.weight.original1')
        raw[f'{base}.weight'] = denorm_weight_norm(o0, o1)

    # Filter out non-weight tensors
    for k, v in raw.items():
        if k.endswith('.num_batches_tracked'):
            continue
        tensors[k] = v

    return tensors


def to_f16(t: torch.Tensor) -> np.ndarray:
    """Convert tensor to float16 numpy array."""
    return t.detach().to(torch.float16).numpy()


def to_f32(t: torch.Tensor) -> np.ndarray:
    """Convert tensor to float32 numpy array."""
    return t.detach().to(torch.float32).numpy()


def choose_dtype(name: str, shape: list, t: torch.Tensor):
    """Choose F16 vs F32 for a tensor. Keep small/1D/embedding tensors as F32."""
    n = np.prod(shape)
    if t.ndim <= 1 or n < 256:
        return to_f32(t), GGMLQuantizationType.F32
    # Keep embedding/position tables and conditioning tensors as F32
    # since they are read on CPU for embedding construction
    keep_f32 = (
        'emb.weight' in name or 'pos_emb.weight' in name or
        'cond.' in name or 'conds.' in name or
        'perceiver.' in name or 've.' in name or
        'input_embedding' in name or 'embed_affine' in name or
        'encoder_proj' in name or 'embed.out' in name or
        # Keep time MLP weights as F32 since they're read on CPU
        'time_mlp' in name or 'time_emb' in name or
        'tm.linear' in name or
        # Keep vocoder conv weights as F32 for precision
        'cpre' in name or 'cpost' in name or
        'ups.' in name or 'rb.' in name or 'srb.' in name or
        'sd.' in name
    )
    if keep_f32:
        return to_f32(t), GGMLQuantizationType.F32
    return to_f16(t), GGMLQuantizationType.F16


# ── T3 tensor name remapping ───────────────────────────────────────

def map_t3_name(hf_name: str) -> str | None:
    """Map T3 HF tensor name → GGUF tensor name."""
    n = hf_name

    # Llama backbone
    n = n.replace("tfmr.embed_tokens.", "t3.llama_embd.")  # unused dummy
    n = n.replace("tfmr.norm.", "t3.output_norm.")
    n = n.replace("tfmr.layers.", "t3.blk.")
    n = n.replace(".self_attn.q_proj.", ".attn_q.")
    n = n.replace(".self_attn.k_proj.", ".attn_k.")
    n = n.replace(".self_attn.v_proj.", ".attn_v.")
    n = n.replace(".self_attn.o_proj.", ".attn_output.")
    n = n.replace(".input_layernorm.", ".attn_norm.")
    n = n.replace(".post_attention_layernorm.", ".ffn_norm.")
    n = n.replace(".mlp.gate_proj.", ".ffn_gate.")
    n = n.replace(".mlp.up_proj.", ".ffn_up.")
    n = n.replace(".mlp.down_proj.", ".ffn_down.")

    # Custom embeddings
    n = n.replace("text_emb.", "t3.text_emb.")
    n = n.replace("speech_emb.", "t3.speech_emb.")
    n = n.replace("text_pos_emb.emb.", "t3.text_pos_emb.")
    n = n.replace("speech_pos_emb.emb.", "t3.speech_pos_emb.")

    # Heads
    n = n.replace("text_head.", "t3.text_head.")
    n = n.replace("speech_head.", "t3.speech_head.")

    # Conditioning encoder
    n = n.replace("cond_enc.spkr_enc.", "t3.cond.spkr_enc.")
    n = n.replace("cond_enc.emotion_adv_fc.", "t3.cond.emotion_adv.")
    n = n.replace("cond_enc.perceiver.", "t3.cond.perceiver.")

    return n


# ── Kartoffelbox T3 tensor name remapping ─────────────────────────

def map_kartoffelbox_t3_name(hf_name: str) -> str | None:
    """Map Kartoffelbox GPT-2 tensor name → GGUF tensor name."""
    import re

    n = hf_name

    # Transformer backbone (GPT-2 naming)
    n = n.replace("tfmr.wpe.weight", "t3.wpe.weight")
    n = n.replace("tfmr.ln_f.weight", "t3.output_norm.weight")
    n = n.replace("tfmr.ln_f.bias", "t3.output_norm.bias")

    # Per-layer mappings
    m = re.match(r'^tfmr\.h\.(\d+)\.(.+)$', n)
    if m:
        layer = m.group(1)
        rest = m.group(2)
        rest = rest.replace("ln_1.weight", "attn_norm.weight")
        rest = rest.replace("ln_1.bias", "attn_norm.bias")
        rest = rest.replace("attn.c_attn.weight", "attn_qkv.weight")
        rest = rest.replace("attn.c_attn.bias", "attn_qkv.bias")
        rest = rest.replace("attn.c_proj.weight", "attn_output.weight")
        rest = rest.replace("attn.c_proj.bias", "attn_output.bias")
        rest = rest.replace("ln_2.weight", "ffn_norm.weight")
        rest = rest.replace("ln_2.bias", "ffn_norm.bias")
        rest = rest.replace("mlp.c_fc.weight", "ffn_fc.weight")
        rest = rest.replace("mlp.c_fc.bias", "ffn_fc.bias")
        rest = rest.replace("mlp.c_proj.weight", "ffn_proj.weight")
        rest = rest.replace("mlp.c_proj.bias", "ffn_proj.bias")
        n = f"t3.blk.{layer}.{rest}"

    # Custom embeddings
    n = n.replace("text_emb.weight", "t3.text_emb.weight")
    n = n.replace("speech_emb.weight", "t3.speech_emb.weight")

    # Heads
    n = n.replace("text_head.weight", "t3.text_head.weight")
    if n == hf_name.replace("speech_head.weight", "t3.speech_head.weight"):
        n = n  # already replaced or not matching
    n = n.replace("speech_head.weight", "t3.speech_head.weight")
    n = n.replace("speech_head.bias", "t3.speech_head.bias")

    # Conditioning encoder
    n = n.replace("cond_enc.spkr_enc.weight", "t3.cond.spkr_enc.weight")
    n = n.replace("cond_enc.spkr_enc.bias", "t3.cond.spkr_enc.bias")

    return n


# ── S3Gen tensor name remapping ────────────────────────────────────

def map_s3gen_name(hf_name: str) -> str | None:
    """Map S3Gen HF tensor name → GGUF tensor name.
    Abbreviate deep paths to stay under the 64-byte GGUF name limit."""
    if hf_name.endswith('.num_batches_tracked'):
        return None

    n = hf_name
    # Abbreviate the deep decoder paths
    n = n.replace("flow.decoder.estimator.", "fd.")
    n = n.replace("down_blocks", "db")
    n = n.replace("mid_blocks", "mb")
    n = n.replace("up_blocks", "ub")
    n = n.replace("final_block", "fb")
    n = n.replace("final_proj", "fp")
    n = n.replace("time_embeddings", "te")
    n = n.replace("time_mlp", "tm")
    n = n.replace("time_embed_mixer", "tmx")
    n = n.replace("block1.block", "b1")
    n = n.replace("block2.block", "b2")
    n = n.replace("res_conv", "rc")
    n = n.replace("transformer_blocks", "tb")  # not used in current arch
    n = n.replace("attn1.to_out.0", "attn1.o")
    n = n.replace("attn1.to_", "attn1.")
    n = n.replace("ff.net.0.proj", "ff.up")
    n = n.replace("ff.net.2", "ff.down")
    # Abbreviate speaker encoder paths
    n = n.replace("speaker_encoder.", "se.")
    n = n.replace("xvector.", "xv.")
    n = n.replace("nonlinear.", "nl.")
    n = n.replace("batchnorm", "bn")
    n = n.replace("cam_layer.", "cam.")
    n = n.replace("linear_local", "ll")
    n = n.replace("linear1", "l1")
    n = n.replace("linear2", "l2")
    # Abbreviate conformer encoder paths
    n = n.replace("flow.encoder.", "fe.")
    n = n.replace("self_attn.", "sa.")
    n = n.replace("feed_forward.", "ff.")
    n = n.replace("pre_lookahead_layer.", "pla.")
    n = n.replace("up_encoders", "ue")
    n = n.replace("encoders", "enc")
    n = n.replace("up_embed", "uemb")
    n = n.replace("up_layer", "ul")
    n = n.replace("after_norm", "an")
    n = n.replace("norm_mha", "nmha")
    n = n.replace("norm_ff", "nff")
    n = n.replace("linear_out", "lo")
    n = n.replace("linear_pos", "lp")
    n = n.replace("linear_q", "lq")
    n = n.replace("linear_k", "lk")
    n = n.replace("linear_v", "lv")
    n = n.replace("pos_bias_u", "pbu")
    n = n.replace("pos_bias_v", "pbv")
    # Abbreviate vocoder paths
    n = n.replace("mel2wav.", "v.")
    n = n.replace("conv_pre", "cpre")
    n = n.replace("conv_post", "cpost")
    n = n.replace("source_downs", "sd")
    n = n.replace("source_resblocks", "srb")
    n = n.replace("resblocks", "rb")
    n = n.replace("activations1", "a1")
    n = n.replace("activations2", "a2")
    n = n.replace("convs1", "c1")
    n = n.replace("convs2", "c2")
    n = n.replace("condnet", "cn")
    n = n.replace("f0_predictor.", "f0.")
    n = n.replace("m_source.", "ms.")
    n = n.replace("l_sin_gen.", "sg.")
    n = n.replace("l_linear", "ll")
    n = n.replace("l_tanh", "lt")
    n = n.replace("classifier", "cls")
    # S3Tokenizer paths
    n = n.replace("tokenizer.", "tok.")
    n = n.replace("quantizer.", "quant.")
    n = n.replace("_codebook.", "cb.")
    n = n.replace("encoder.blocks", "enc.b")
    n = n.replace("fsmn_block", "fsmn")
    n = n.replace("project_down", "pd")
    n = n.replace("project_up", "pu")
    # Common
    n = n.replace("parametrizations.weight.", "pw.")

    name = "s3." + n
    if len(name) >= 64:
        # Last resort: hash the overflow
        import hashlib
        h = hashlib.md5(hf_name.encode()).hexdigest()[:8]
        name = f"s3.h.{h}"
    return name


# ── VE tensor name remapping ──────────────────────────────────────

def map_ve_name(hf_name: str) -> str | None:
    """Map VoiceEncoder HF tensor name → GGUF tensor name."""
    return "ve." + hf_name


# ── Write T3 GGUF ─────────────────────────────────────────────────

def write_t3_gguf(
    model_dir: Path,
    output_path: Path,
    conds_path: Path | None,
    tokenizer_path: Path | None,
):
    print(f"\n=== Writing T3 GGUF: {output_path} ===")

    writer = GGUFWriter(str(output_path), "chatterbox")

    # ── Hyperparameters ──
    for k, v in T3_HPARAMS.items():
        key = f"chatterbox.t3.{k}"
        if isinstance(v, int):
            writer.add_uint32(key, v)
        elif isinstance(v, float):
            writer.add_float32(key, v)
        elif isinstance(v, str):
            writer.add_string(key, v)

    # VE hparams (VE is in the T3 GGUF since it's tiny)
    for k, v in VE_HPARAMS.items():
        key = f"chatterbox.ve.{k}"
        if isinstance(v, int):
            writer.add_uint32(key, v)
        elif isinstance(v, float):
            writer.add_float32(key, v)

    # ── Load and write character tokenizer ──
    if tokenizer_path and tokenizer_path.exists():
        with open(tokenizer_path, 'r') as f:
            tok_data = json.load(f)
        # Extract vocabulary
        if 'model' in tok_data and 'vocab' in tok_data['model']:
            vocab = tok_data['model']['vocab']
            tokens = [""] * len(vocab)
            for token, idx in vocab.items():
                if idx < len(tokens):
                    tokens[idx] = token
            writer.add_array("chatterbox.t3.text_tokens", tokens)
            print(f"  Tokenizer: {len(tokens)} text tokens")

    # ── Load and write precomputed conditioning ──
    if conds_path and conds_path.exists():
        conds = torch.load(conds_path, map_location='cpu', weights_only=True)
        t3_cond = conds['t3']
        gen_cond = conds['gen']

        # T3 conditioning
        if t3_cond.get('speaker_emb') is not None:
            writer.add_tensor("conds.t3.speaker_emb",
                              to_f32(t3_cond['speaker_emb']),
                              raw_dtype=GGMLQuantizationType.F32)
        if t3_cond.get('cond_prompt_speech_tokens') is not None:
            tokens = t3_cond['cond_prompt_speech_tokens'].to(torch.int32).numpy()
            writer.add_tensor("conds.t3.speech_prompt_tokens", tokens,
                              raw_dtype=GGMLQuantizationType.I32)
        if t3_cond.get('emotion_adv') is not None:
            writer.add_float32("chatterbox.conds.emotion_adv",
                               float(t3_cond['emotion_adv'].item()))

        # S3Gen conditioning (stored in T3 GGUF for convenience)
        if gen_cond.get('prompt_token') is not None:
            tokens = gen_cond['prompt_token'].to(torch.int32).numpy()
            writer.add_tensor("conds.gen.prompt_token", tokens,
                              raw_dtype=GGMLQuantizationType.I32)
        if gen_cond.get('prompt_token_len') is not None:
            writer.add_uint32("chatterbox.conds.gen_prompt_token_len",
                              int(gen_cond['prompt_token_len'].item()))
        if gen_cond.get('prompt_feat') is not None:
            writer.add_tensor("conds.gen.prompt_feat",
                              to_f32(gen_cond['prompt_feat']),
                              raw_dtype=GGMLQuantizationType.F32)
        if gen_cond.get('embedding') is not None:
            writer.add_tensor("conds.gen.embedding",
                              to_f32(gen_cond['embedding']),
                              raw_dtype=GGMLQuantizationType.F32)
        print(f"  Precomputed conds loaded")

    # ── Load T3 weights ──
    t3_path = model_dir / "t3_cfg.safetensors"
    if not t3_path.exists():
        sys.exit(f"Missing {t3_path}")
    t3_tensors = load_safetensors(t3_path)
    n_t3 = 0
    for hf_name, tensor in sorted(t3_tensors.items()):
        gguf_name = map_t3_name(hf_name)
        if gguf_name is None:
            continue
        data, dtype = choose_dtype(gguf_name, list(tensor.shape), tensor)
        writer.add_tensor(gguf_name, data, raw_dtype=dtype)
        n_t3 += 1
    print(f"  T3: {n_t3} tensors")

    # ── Load VE weights ──
    ve_path = model_dir / "ve.safetensors"
    if ve_path.exists():
        ve_tensors = load_safetensors(ve_path)
        n_ve = 0
        for hf_name, tensor in sorted(ve_tensors.items()):
            gguf_name = map_ve_name(hf_name)
            if gguf_name is None:
                continue
            data, dtype = choose_dtype(gguf_name, list(tensor.shape), tensor)
            writer.add_tensor(gguf_name, data, raw_dtype=dtype)
            n_ve += 1
        print(f"  VE: {n_ve} tensors")

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"  Written: {output_path} ({output_path.stat().st_size / 1e6:.1f} MB)")


# ── Write Kartoffelbox T3 GGUF ────────────────────────────────────

def write_kartoffelbox_t3_gguf(
    model_dir: Path,
    output_path: Path,
):
    print(f"\n=== Writing Kartoffelbox T3 GGUF: {output_path} ===")

    writer = GGUFWriter(str(output_path), "chatterbox")

    # ── Hyperparameters ──
    for k, v in KARTOFFELBOX_T3_HPARAMS.items():
        key = f"chatterbox.t3.{k}"
        if isinstance(v, int):
            writer.add_uint32(key, v)
        elif isinstance(v, float):
            writer.add_float32(key, v)
        elif isinstance(v, str):
            writer.add_string(key, v)

    # ── Embed GPT-2 BPE tokenizer ──
    try:
        from transformers import GPT2TokenizerFast
        gpt2_tok = GPT2TokenizerFast.from_pretrained('gpt2')
        vocab = gpt2_tok.get_vocab()
        # Build token list ordered by ID
        max_id = max(vocab.values())
        tokens = [""] * (max_id + 1)
        for tok_str, tok_id in vocab.items():
            tokens[tok_id] = tok_str
        writer.add_array("tokenizer.ggml.tokens", tokens)
        # Get merges from the tokenizer's backend
        merges = []
        try:
            import json as _json
            tok_json = _json.loads(gpt2_tok.backend_tokenizer.to_str())
            raw_merges = tok_json.get('model', {}).get('merges', [])
            # Merges are [left, right] pairs — join to "left right" strings
            for m in raw_merges:
                if isinstance(m, list) and len(m) == 2:
                    merges.append(f"{m[0]} {m[1]}")
                elif isinstance(m, str):
                    merges.append(m)
        except Exception:
            pass
        if merges:
            writer.add_array("tokenizer.ggml.merges", merges)
            print(f"  GPT-2 tokenizer: {len(tokens)} tokens, {len(merges)} merges")
        else:
            print(f"  GPT-2 tokenizer: {len(tokens)} tokens (no merges — BPE will fall back to char-level)")
    except ImportError:
        print("  WARNING: transformers not installed — no GPT-2 tokenizer embedded")

    # ── Load weights from model.pt ──
    model_path = model_dir / "model.pt"
    if not model_path.exists():
        sys.exit(f"Missing {model_path}")

    print(f"  Loading {model_path}...")
    raw = torch.load(str(model_path), map_location='cpu', weights_only=True)

    n_t3 = 0
    for hf_name, tensor in sorted(raw.items()):
        gguf_name = map_kartoffelbox_t3_name(hf_name)
        if gguf_name is None:
            continue
        # Skip unmapped keys (still equal to original)
        if gguf_name == hf_name:
            print(f"  SKIP (unmapped): {hf_name}")
            continue
        data, dtype = choose_dtype(gguf_name, list(tensor.shape), tensor)
        writer.add_tensor(gguf_name, data, raw_dtype=dtype)
        n_t3 += 1
    print(f"  T3 (Kartoffelbox GPT-2): {n_t3} tensors")

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"  Written: {output_path} ({output_path.stat().st_size / 1e6:.1f} MB)")


# ── Write S3Gen GGUF ──────────────────────────────────────────────

def write_s3gen_gguf(
    model_dir: Path,
    output_path: Path,
):
    print(f"\n=== Writing S3Gen GGUF: {output_path} ===")

    writer = GGUFWriter(str(output_path), "chatterbox-s3gen")

    # ── Hyperparameters ──
    for k, v in S3GEN_HPARAMS.items():
        key = f"chatterbox.s3gen.{k}"
        if isinstance(v, int):
            writer.add_uint32(key, v)
        elif isinstance(v, float):
            writer.add_float32(key, v)
        elif isinstance(v, str):
            writer.add_string(key, v)

    # ── Load S3Gen weights ──
    s3gen_path = model_dir / "s3gen.safetensors"
    if not s3gen_path.exists():
        sys.exit(f"Missing {s3gen_path}")
    s3gen_tensors = load_safetensors(s3gen_path)

    # Group by component for reporting
    counts = {"flow": 0, "mel2wav": 0, "speaker_encoder": 0, "tokenizer": 0}

    for hf_name, tensor in sorted(s3gen_tensors.items()):
        gguf_name = map_s3gen_name(hf_name)
        if gguf_name is None:
            continue

        # Track counts
        for prefix in counts:
            if hf_name.startswith(prefix):
                counts[prefix] += 1
                break

        data, dtype = choose_dtype(gguf_name, list(tensor.shape), tensor)
        writer.add_tensor(gguf_name, data, raw_dtype=dtype)

    for comp, n in counts.items():
        print(f"  {comp}: {n} tensors")
    print(f"  Total: {sum(counts.values())} tensors")

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"  Written: {output_path} ({output_path.stat().st_size / 1e6:.1f} MB)")


# ── Main ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Convert Chatterbox to GGUF")
    parser.add_argument("--input", required=True,
                        help="HF repo ID or local directory with safetensors")
    parser.add_argument("--output-dir", required=True,
                        help="Output directory for GGUF files")
    parser.add_argument("--variant", default=None, choices=["kartoffelbox"],
                        help="Model variant (kartoffelbox = GPT-2 T3)")
    parser.add_argument("--t3-only", action="store_true",
                        help="Only convert T3 model")
    parser.add_argument("--s3gen-only", action="store_true",
                        help="Only convert S3Gen model")
    args = parser.parse_args()

    model_dir = load_model_dir(args.input)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.variant == "kartoffelbox":
        write_kartoffelbox_t3_gguf(
            model_dir,
            out_dir / "kartoffelbox-turbo-t3-f16.gguf",
        )
        print("\nDone!")
        return

    conds_path = model_dir / "conds.pt"
    tokenizer_path = model_dir / "tokenizer.json"

    if not args.s3gen_only:
        write_t3_gguf(
            model_dir,
            out_dir / "chatterbox-t3-f16.gguf",
            conds_path if conds_path.exists() else None,
            tokenizer_path if tokenizer_path.exists() else None,
        )

    if not args.t3_only:
        write_s3gen_gguf(
            model_dir,
            out_dir / "chatterbox-s3gen-f16.gguf",
        )

    print("\nDone!")


if __name__ == "__main__":
    main()
