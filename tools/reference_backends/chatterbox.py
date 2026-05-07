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

import os
import sys
import inspect
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
    "hift_source",
    "hift_source_stft",
    "voc_conv_pre",
    "voc_ups_0",
    "voc_rb_0",
    "voc_ups_1",
    "voc_rb_1",
    "voc_ups_2",
    "voc_rb_2",
    "voc_conv_post",
    "hift_pcm",
]

DEFAULT_SYN_TEXT = "Hello world."
DEFAULT_CFG_WEIGHT = 0.5
DEFAULT_TEMPERATURE = 0.8
DEFAULT_REPETITION_PENALTY = 1.2
DEFAULT_MIN_P = 0.05
DEFAULT_TOP_P = 1.0


def dump(*, model_dir: Path, audio: np.ndarray, stages: Set[str],
         max_new_tokens: int) -> Dict[str, np.ndarray]:
    import torch
    import librosa

    out: Dict[str, np.ndarray] = {}

    upstream_src = os.environ.get("RESEMBLE_CHATTERBOX_SRC")
    if upstream_src:
        src_path = str(Path(upstream_src).resolve())
        if src_path not in sys.path:
            sys.path.insert(0, src_path)

    # ── Load Chatterbox ──
    from chatterbox.tts import ChatterboxTTS, punc_norm
    from chatterbox.models.s3gen import S3GEN_SR
    from chatterbox.models.s3tokenizer import S3_SR, drop_invalid_tokens

    print(f"  loading Chatterbox from {model_dir}")
    model = ChatterboxTTS.from_local(model_dir, device="cpu")

    # Use built-in voice (conds.pt)
    assert model.conds is not None, "conds.pt not found in model_dir"

    # ── Text tokenization ──
    test_text = os.environ.get("CHATTERBOX_SYN_TEXT", DEFAULT_SYN_TEXT)
    text_norm = punc_norm(test_text)
    text_tokens = model.tokenizer.text_to_tokens(text_norm).to("cpu")
    text_tokens_infer = text_tokens
    if DEFAULT_CFG_WEIGHT > 0.0:
        text_tokens_infer = torch.cat([text_tokens_infer, text_tokens_infer], dim=0)

    import torch.nn.functional as F
    sot = model.t3.hp.start_text_token
    eot = model.t3.hp.stop_text_token
    text_tokens = F.pad(text_tokens, (1, 0), value=sot)
    text_tokens = F.pad(text_tokens, (0, 1), value=eot)
    text_tokens_infer = F.pad(text_tokens_infer, (1, 0), value=sot)
    text_tokens_infer = F.pad(text_tokens_infer, (0, 1), value=eot)

    # ── T3 conditioning ──
    t3_cond = model.conds.t3
    t3_cond_emb = model.t3.prepare_conditioning(t3_cond)
    if "t3_cond_emb" in stages:
        out["t3_cond_emb"] = t3_cond_emb.detach().squeeze(0).cpu().float().numpy()

    # ── T3 prefill embeddings ──
    speech_start = model.t3.hp.start_speech_token * torch.ones_like(text_tokens_infer[:, :1])
    embeds, len_cond = model.t3.prepare_input_embeds(
        t3_cond=t3_cond,
        text_tokens=text_tokens_infer,
        speech_tokens=speech_start,
        cfg_weight=DEFAULT_CFG_WEIGHT,
    )
    if "t3_prefill_emb" in stages:
        out["t3_prefill_emb"] = embeds.detach().cpu().float().numpy()

    # ── T3 AR decode — exact upstream path ──
    with torch.inference_mode():
        speech_tokens = model.t3.inference(
            t3_cond=t3_cond,
            text_tokens=text_tokens_infer,
            max_new_tokens=max_new_tokens,
            temperature=DEFAULT_TEMPERATURE,
            cfg_weight=DEFAULT_CFG_WEIGHT,
            repetition_penalty=DEFAULT_REPETITION_PENALTY,
            min_p=DEFAULT_MIN_P,
            top_p=DEFAULT_TOP_P,
        )
    speech_tokens = speech_tokens[0]
    speech_tokens = drop_invalid_tokens(speech_tokens)
    speech_tokens_valid = speech_tokens[speech_tokens < 6561]

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

    s3gen_infer_sig = inspect.signature(model.s3gen.inference)
    s3gen_flow_sig = inspect.signature(model.s3gen.flow_inference)

    infer_kwargs = {
        "speech_tokens": speech_tokens_2d,
        "ref_dict": model.conds.gen,
    }
    flow_kwargs = {
        "speech_tokens": speech_tokens_2d,
        "ref_dict": model.conds.gen,
        "finalize": True,
    }
    if "n_cfm_timesteps" in s3gen_infer_sig.parameters:
        infer_kwargs["n_cfm_timesteps"] = 10
    if "n_cfm_timesteps" in s3gen_flow_sig.parameters:
        flow_kwargs["n_cfm_timesteps"] = 10

    # Extract mel from flow_inference
    with torch.inference_mode():
        mel = model.s3gen.flow_inference(**flow_kwargs)
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

    # ── HiFT vocoder — exact upstream decode path ──
    hift = model.s3gen.mel2wav
    with torch.inference_mode():
        f0 = hift.f0_predictor(mel)
        s = hift.f0_upsamp(f0[:, None]).transpose(1, 2)
        s, _, _ = hift.m_source(s)
        s = s.transpose(1, 2)

        if "hift_source" in stages:
            out["hift_source"] = s.detach().squeeze(0).transpose(0, 1).contiguous().cpu().float().numpy()

        s_stft_real, s_stft_imag = hift._stft(s.squeeze(1))
        s_stft = torch.cat([s_stft_real, s_stft_imag], dim=1)
        if "hift_source_stft" in stages:
            out["hift_source_stft"] = s_stft.detach().squeeze(0).permute(1, 0).contiguous().cpu().float().numpy()

        x = hift.conv_pre(mel)
        if "voc_conv_pre" in stages:
            out["voc_conv_pre"] = x.detach().squeeze(0).permute(1, 0).contiguous().cpu().float().numpy()

        for i in range(hift.num_upsamples):
            x = torch.nn.functional.leaky_relu(x, hift.lrelu_slope)
            x = hift.ups[i](x)
            if i == hift.num_upsamples - 1:
                x = hift.reflection_pad(x)
            if f"voc_ups_{i}" in stages:
                out[f"voc_ups_{i}"] = x.detach().squeeze(0).permute(1, 0).contiguous().cpu().float().numpy()

            si = hift.source_downs[i](s_stft)
            si = hift.source_resblocks[i](si)
            x = x + si

            xs = None
            for j in range(hift.num_kernels):
                rb = hift.resblocks[i * hift.num_kernels + j](x)
                xs = rb if xs is None else xs + rb
            x = xs / hift.num_kernels
            if f"voc_rb_{i}" in stages:
                out[f"voc_rb_{i}"] = x.detach().squeeze(0).permute(1, 0).contiguous().cpu().float().numpy()

        x = torch.nn.functional.leaky_relu(x)
        x = hift.conv_post(x)
        if "voc_conv_post" in stages:
            out["voc_conv_post"] = x.detach().squeeze(0).permute(1, 0).contiguous().cpu().float().numpy()

        magnitude = torch.exp(x[:, :hift.istft_params["n_fft"] // 2 + 1, :])
        phase = torch.sin(x[:, hift.istft_params["n_fft"] // 2 + 1:, :])
        wav = hift._istft(magnitude, phase)
        wav = torch.clamp(wav, -hift.audio_limit, hift.audio_limit)
        trim_fade = model.s3gen.trim_fade.to(device=wav.device, dtype=wav.dtype)
        wav = wav.clone()
        wav[:, :trim_fade.numel()] *= trim_fade

    if "hift_pcm" in stages:
        out["hift_pcm"] = wav.detach().squeeze(0).cpu().float().numpy()

    if "hift_f0" in stages:
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
