"""Chatterbox TTS reference dump backend for crispasr-diff.

Captures per-stage activations from the ResembleAI/chatterbox Python
model so the C++ implementation can be validated tensor-by-tensor.

Pipeline stages captured:
  t3_cond_emb         — conditioning embedding (spkr + perceiver + emotion)
  t3_prefill_emb      — full prefill input embeddings [cond | text | speech_start]
  t3_speech_tokens    — T3 AR output (speech token IDs as float32)
  s3gen_token_emb     — S3Gen flow.input_embedding lookup
  s3gen_encoder_out   — UpsampleConformerEncoder output (after proj to 80D)
  s3gen_mel           — CFM denoiser output mel-spectrogram
  hift_f0             — HiFTGenerator F0 prediction
  hift_pcm            — final 24 kHz waveform

Usage:
    python tools/dump_reference.py --backend chatterbox \\
        --model-dir /mnt/storage/chatterbox \\
        --audio samples/jfk.wav \\
        --output /mnt/storage/chatterbox/chatterbox-ref.gguf \\
        --stages t3_speech_tokens,s3gen_encoder_out,s3gen_mel,hift_pcm \\
        --max-new-tokens 200
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Set

import numpy as np

DEFAULT_STAGES = [
    "t3_cond_emb",
    "t3_prefill_emb",
    "t3_speech_tokens",
    "s3gen_token_emb",
    "s3gen_encoder_out",
    "s3gen_mel",
    "hift_pcm",
]


def dump(*, model_dir: Path, audio: np.ndarray, stages: Set[str],
         max_new_tokens: int) -> Dict[str, np.ndarray]:
    import torch
    import librosa

    out: Dict[str, np.ndarray] = {}

    # ── Load Chatterbox ──
    from chatterbox.tts import ChatterboxTTS, punc_norm
    from chatterbox.models.s3gen import S3GEN_SR
    from chatterbox.models.s3tokenizer import S3_SR
    from chatterbox.models.t3.modules.cond_enc import T3Cond

    print(f"  loading Chatterbox from {model_dir}")
    model = ChatterboxTTS.from_local(model_dir, device="cpu")

    # Use built-in voice (conds.pt)
    assert model.conds is not None, "conds.pt not found in model_dir"

    # ── Text tokenization ──
    test_text = "Hello world."
    text_norm = punc_norm(test_text)
    text_tokens = model.tokenizer.text_to_tokens(text_norm).to("cpu")

    import torch.nn.functional as F
    sot = model.t3.hp.start_text_token
    eot = model.t3.hp.stop_text_token
    text_tokens = F.pad(text_tokens, (1, 0), value=sot)
    text_tokens = F.pad(text_tokens, (0, 1), value=eot)

    # ── T3 conditioning ──
    t3_cond = model.conds.t3
    t3_cond_emb = model.t3.prepare_conditioning(t3_cond)
    if "t3_cond_emb" in stages:
        out["t3_cond_emb"] = t3_cond_emb.detach().squeeze(0).cpu().float().numpy()

    # ── T3 prefill embeddings ──
    speech_start = model.t3.hp.start_speech_token * torch.ones_like(text_tokens[:, :1])
    embeds, len_cond = model.t3.prepare_input_embeds(
        t3_cond=t3_cond,
        text_tokens=text_tokens,
        speech_tokens=speech_start,
        cfg_weight=0.0,  # no CFG for diff testing
    )
    if "t3_prefill_emb" in stages:
        out["t3_prefill_emb"] = embeds.detach().squeeze(0).cpu().float().numpy()

    # ── T3 AR decode (greedy, no CFG for reproducibility) ──
    with torch.inference_mode():
        speech_tokens = model.t3.inference_turbo(
            t3_cond=t3_cond,
            text_tokens=text_tokens,
            temperature=0.0,  # greedy for reproducibility
            top_k=1,
            repetition_penalty=1.0,
            max_gen_len=max_new_tokens,
        )
    # Remove EOS
    if speech_tokens.size(1) > 0 and speech_tokens[0, -1] == model.t3.hp.stop_speech_token:
        speech_tokens = speech_tokens[:, :-1]
    # Filter to valid range
    valid_mask = speech_tokens[0] < 6561
    speech_tokens_valid = speech_tokens[0][valid_mask]

    if "t3_speech_tokens" in stages:
        out["t3_speech_tokens"] = speech_tokens_valid.cpu().float().numpy()

    print(f"  T3 generated {speech_tokens_valid.size(0)} speech tokens")

    # ── S3Gen: tokens → mel ──
    speech_tokens_2d = speech_tokens_valid.unsqueeze(0).to("cpu")

    # Token embedding
    flow = model.s3gen.flow
    token_emb = flow.input_embedding(torch.clamp(speech_tokens_2d, min=0).long())
    if "s3gen_token_emb" in stages:
        out["s3gen_token_emb"] = token_emb.detach().squeeze(0).cpu().float().numpy()

    # Full S3Gen inference
    with torch.inference_mode():
        wav, _ = model.s3gen.inference(
            speech_tokens=speech_tokens_2d,
            ref_dict=model.conds.gen,
            n_cfm_timesteps=10,
        )

    # Extract mel from flow_inference
    with torch.inference_mode():
        mel = model.s3gen.flow_inference(
            speech_tokens=speech_tokens_2d,
            ref_dict=model.conds.gen,
            n_cfm_timesteps=10,
            finalize=True,
        )
    if "s3gen_mel" in stages:
        # mel shape: (B, 80, T) → (T, 80)
        out["s3gen_mel"] = mel.detach().squeeze(0).permute(1, 0).contiguous().cpu().float().numpy()

    # Extract encoder output
    ref_dict = model.conds.gen
    prompt_token = ref_dict['prompt_token'].to("cpu")
    prompt_token_len = ref_dict['prompt_token_len']
    token_len = torch.LongTensor([speech_tokens_valid.size(0)]).to("cpu")

    # Concat prompt + speech tokens
    full_tokens = torch.cat([prompt_token, speech_tokens_2d], dim=1)
    full_token_len = prompt_token_len + token_len
    mask = (~_make_pad_mask(full_token_len)).unsqueeze(-1).to(torch.float32)
    emb_input = flow.input_embedding(torch.clamp(full_tokens, min=0).long()) * mask

    h, h_masks = flow.encoder(emb_input, full_token_len)
    h = flow.encoder_proj(h)  # (B, T*2, 80)

    if "s3gen_encoder_out" in stages:
        out["s3gen_encoder_out"] = h.detach().squeeze(0).cpu().float().numpy()

    # ── HiFT vocoder ──
    if "hift_pcm" in stages:
        out["hift_pcm"] = wav.detach().squeeze(0).cpu().float().numpy()

    # F0 prediction
    if "hift_f0" in stages:
        with torch.inference_mode():
            f0 = model.s3gen.mel2wav.f0_predictor(mel)
        out["hift_f0"] = f0.detach().squeeze(0).cpu().float().numpy()

    return out


def _make_pad_mask(lengths, max_len=None):
    """Create a boolean mask where True = padding."""
    if max_len is None:
        max_len = lengths.max().item()
    batch_size = lengths.size(0)
    seq_range = torch.arange(0, max_len, device=lengths.device)
    return seq_range.unsqueeze(0) >= lengths.unsqueeze(1)


import torch
