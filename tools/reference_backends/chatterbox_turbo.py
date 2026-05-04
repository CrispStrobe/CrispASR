"""Chatterbox Turbo TTS reference dump backend for crispasr-diff.

Captures per-stage activations from the Chatterbox Turbo (GPT-2 T3 +
meanflow S3Gen) Python model for element-wise C++ validation.

Pipeline stages captured:
  s3gen_encoder_out   — UpsampleConformerEncoder output (after proj to 80D)
  s3gen_enc0_pre_attn — LayerNorm'd input to enc.0 attention
  s3gen_enc0_q        — Q projection output in enc.0 (before multi-head split)
  s3gen_enc0_k        — K projection output in enc.0
  s3gen_enc0_matrix_bd — rel-pos attention scores (post rel_shift) in enc.0
  s3gen_enc0_attn_out — attention output of enc.0 (after output proj)
  s3gen_pos_emb_pre   — positional encoding for pre-upsample blocks
  s3gen_gen_mel       — CFM denoiser output mel (generation portion only)
  hift_pcm            — final 24 kHz waveform

Usage:
    python tools/dump_reference.py --backend chatterbox_turbo \\
        --model-dir /mnt/storage/chatterbox_turbo_base \\
        --audio samples/jfk.wav \\
        --output /mnt/storage/chatterbox_turbo_base/turbo-enc-ref.gguf \\
        --max-new-tokens 200
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Set

import numpy as np

DEFAULT_STAGES = [
    "s3gen_encoder_out",
    "s3gen_enc0_pre_attn",
    "s3gen_enc0_q",
    "s3gen_enc0_matrix_bd",
    "s3gen_enc0_attn_out",
    "s3gen_pos_emb_pre",
    "s3gen_gen_mel",
    "hift_pcm",
]


def dump(*, model_dir: Path, audio: np.ndarray, stages: Set[str],
         max_new_tokens: int) -> Dict[str, np.ndarray]:
    import torch
    import math

    out: Dict[str, np.ndarray] = {}

    # ── Load Chatterbox Turbo ──
    from chatterbox.tts_turbo import ChatterboxTurboTTS
    from chatterbox.models.s3gen.flow import CausalMaskedDiffWithXvec
    from chatterbox.models.s3gen.transformer.attention import RelPositionMultiHeadedAttention

    print(f"  loading Chatterbox Turbo from {model_dir}")
    model = ChatterboxTurboTTS.from_local(str(model_dir), device="cpu")

    assert model.conds is not None, "conds.pt not found in model_dir"

    # ── Hook into encoder internals for enc.0 ──
    hooks = []
    captured = {}

    flow = model.s3gen.flow

    # Hook: enc.0 pre-attention norm output
    enc0 = flow.encoder.encoders[0]
    def hook_pre_attn(module, args, output):
        # ConformerEncoderLayer.forward() calls self_attn after norm
        # We need to hook into the norm_mha output
        pass  # We'll hook the attention module directly

    # Hook the self-attention forward to capture Q, K, matrix_bd
    enc0_attn = enc0.self_attn
    original_forward = enc0_attn.forward

    def hooked_attn_forward(query, key, value, mask=torch.ones((0,0,0), dtype=torch.bool),
                            pos_emb=torch.empty(0), cache=torch.zeros((0,0,0,0))):
        # query = LayerNorm'd x = pre_attn_norm
        if "s3gen_enc0_pre_attn" in stages:
            captured["s3gen_enc0_pre_attn"] = query.detach().squeeze(0).cpu().float().numpy()

        # Run Q/K/V projections manually to capture Q and K
        q, k, v = enc0_attn.forward_qkv(query, key, value)

        if "s3gen_enc0_q" in stages:
            # q shape: (B, H, T, d_k) after forward_qkv
            captured["s3gen_enc0_q"] = q.detach().squeeze(0).cpu().float().numpy()
        if "s3gen_enc0_k" in stages:
            captured["s3gen_enc0_k"] = k.detach().squeeze(0).cpu().float().numpy()

        # Continue with the actual attention computation
        q_t = q.transpose(1, 2)  # (B, T, H, d_k)

        if cache.size(0) > 0:
            key_cache, value_cache = torch.split(cache, cache.size(-1) // 2, dim=-1)
            k = torch.cat([key_cache, k], dim=2)
            v = torch.cat([value_cache, v], dim=2)
        new_cache = torch.cat((k, v), dim=-1)

        n_batch_pos = pos_emb.size(0)
        p = enc0_attn.linear_pos(pos_emb).view(n_batch_pos, -1, enc0_attn.h, enc0_attn.d_k)
        p = p.transpose(1, 2)

        q_with_bias_u = (q_t + enc0_attn.pos_bias_u).transpose(1, 2)
        q_with_bias_v = (q_t + enc0_attn.pos_bias_v).transpose(1, 2)

        matrix_ac = torch.matmul(q_with_bias_u, k.transpose(-2, -1))
        matrix_bd = torch.matmul(q_with_bias_v, p.transpose(-2, -1))

        if matrix_ac.shape != matrix_bd.shape:
            matrix_bd = enc0_attn.rel_shift(matrix_bd)

        if "s3gen_enc0_matrix_bd" in stages:
            captured["s3gen_enc0_matrix_bd"] = matrix_bd.detach().squeeze(0).cpu().float().numpy()

        scores = (matrix_ac + matrix_bd) / math.sqrt(enc0_attn.d_k)
        attn_out = enc0_attn.forward_attention(v, scores, mask)

        if "s3gen_enc0_attn_out" in stages:
            captured["s3gen_enc0_attn_out"] = attn_out.detach().squeeze(0).cpu().float().numpy()

        return attn_out, new_cache

    enc0_attn.forward = hooked_attn_forward

    # Hook positional encoding
    original_embed_forward = flow.encoder.embed.forward
    def hooked_embed_forward(xs, masks, offset=0):
        result = original_embed_forward(xs, masks, offset)
        xs_out, pos_emb, masks_out = result
        if "s3gen_pos_emb_pre" in stages:
            captured["s3gen_pos_emb_pre"] = pos_emb.detach().squeeze(0).cpu().float().numpy()
        return result
    flow.encoder.embed.forward = hooked_embed_forward

    # ── Run turbo inference with fixed tokens ──
    # Use Python's greedy tokens for "Hello world."
    speech_tokens = torch.tensor([[3700, 4218, 4218, 3891, 5832, 5105, 2828, 2747, 2216, 5096,
                                   5830, 4372, 3644, 1919, 1676, 1649, 596, 2816, 5088, 5067,
                                   1275, 5400, 732, 2031, 4218, 4218, 4218, 4218]])

    ref_dict = model.conds.gen

    print(f"  running S3Gen with {speech_tokens.size(1)} speech tokens (meanflow, 2 steps)")

    with torch.inference_mode():
        mel = model.s3gen.flow_inference(
            speech_tokens=speech_tokens,
            ref_dict=ref_dict,
            n_cfm_timesteps=2,
            finalize=True,
        )

    # Extract encoder output separately (need to re-run encoder only)
    prompt_token = ref_dict['prompt_token'].to("cpu")
    prompt_token_len = ref_dict['prompt_token_len']
    token_len = torch.LongTensor([speech_tokens.size(1)]).to("cpu")

    full_tokens = torch.cat([prompt_token, speech_tokens], dim=1)
    full_token_len = prompt_token_len + token_len
    mask_enc = (~_make_pad_mask(full_token_len)).unsqueeze(-1).to(torch.float32)
    emb_input = flow.input_embedding(torch.clamp(full_tokens, min=0).long()) * mask_enc

    # Restore original forward for clean encoder run to get encoder_out
    enc0_attn.forward = original_forward
    flow.encoder.embed.forward = original_embed_forward

    with torch.inference_mode():
        h, h_masks = flow.encoder(emb_input, full_token_len)
        h = flow.encoder_proj(h)

    if "s3gen_encoder_out" in stages:
        out["s3gen_encoder_out"] = h.detach().squeeze(0).cpu().float().numpy()

    # Copy captured values to output
    for k_name, v_val in captured.items():
        if k_name in stages:
            out[k_name] = v_val

    # Gen mel
    if "s3gen_gen_mel" in stages and mel is not None:
        # mel shape: (B, 80, T_gen)
        out["s3gen_gen_mel"] = mel.detach().squeeze(0).permute(1, 0).contiguous().cpu().float().numpy()

    # Vocoder
    if "hift_pcm" in stages:
        with torch.inference_mode():
            wav, _ = model.s3gen.inference(
                speech_tokens=speech_tokens,
                ref_dict=ref_dict,
                n_cfm_timesteps=2,
            )
        out["hift_pcm"] = wav.detach().squeeze(0).cpu().float().numpy()

    print(f"  captured {len(out)} stages: {list(out.keys())}")
    return out


def _make_pad_mask(lengths, max_len=None):
    """Create a boolean mask where True = padding."""
    import torch
    if max_len is None:
        max_len = lengths.max().item()
    batch_size = lengths.size(0)
    seq_range = torch.arange(0, max_len, device=lengths.device)
    return seq_range.unsqueeze(0) >= lengths.unsqueeze(1)
