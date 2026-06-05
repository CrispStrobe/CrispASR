"""Qwen3-TTS-Tokenizer-12Hz codec decoder reference dump backend.

Captures stage-by-stage activations from the official PyTorch
Qwen3TTSTokenizerV2Decoder to diff against the CrispASR C++ codec decoder.

Input: deterministic codes (T=10 frames × 16 codebooks, all zeros by default).
This avoids the voice-pack + talker dependency — only the codec tokenizer model
directory is needed.

Stages dumped:
  codec_input_codes    — (T_codec, n_q) int32 as float32
  codec_rvq_out        — after SplitRVQ decode: (512, T_codec) channels-first
  codec_pre_conv_out   — after pre_conv CausalConvNet: (1024, T_codec)
  codec_xfmr_out       — after transformer + output_proj: (1024, T_codec)
  codec_up0_out        — after first ConvNeXt upsample: (1024, 2*T_codec)
  codec_up1_out        — after second ConvNeXt upsample: (1024, 4*T_codec)
  codec_in_conv_out    — after in_conv: (1536, 4*T_codec)
  codec_blk0_out       — after DecoderBlock 0 (stride=8): (768, 32*T_codec)
  codec_pcm            — final clamp output: (T_pcm,) = (1920*T_codec,)

The "audio" arg is unused (required by the dispatcher but ignored here).
Set QWEN3_TTS_CODEC_T=N to use N codec frames (default 10).
Set QWEN3_TTS_CODEC_CODE=K to use K as the constant code value (default 0).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, Set

import numpy as np

from . import _hooks

DEFAULT_STAGES = [
    "codec_input_codes",
    "codec_rvq_out",
    "codec_pre_conv_out",
    "codec_xfmr_out",
    "codec_up0_out",
    "codec_up1_out",
    "codec_in_conv_out",
    "codec_blk0_out",
    "codec_pcm",
]


def dump(*, model_dir: Path, audio: np.ndarray, stages: Set[str],
         max_new_tokens: int) -> Dict[str, np.ndarray]:
    """Run the codec decoder forward and return captured stage tensors."""
    import torch

    # Prefer the local ref tree (same commit as the C++ code) over the pip package.
    ref_path = Path(__file__).resolve().parents[2] / "ref" / "Qwen3-TTS"
    if ref_path.is_dir() and str(ref_path) not in sys.path:
        sys.path.insert(0, str(ref_path))

    from qwen_tts.core.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 import (
        Qwen3TTSTokenizerV2Model,
    )

    T = int(os.environ.get("QWEN3_TTS_CODEC_T", "10"))
    code_val = int(os.environ.get("QWEN3_TTS_CODEC_CODE", "0"))
    n_q = 16

    print(f"  loading Qwen3-TTS-Tokenizer-12Hz from {model_dir} (CPU, fp32)")
    model = Qwen3TTSTokenizerV2Model.from_pretrained(
        str(model_dir), dtype=torch.float32, device_map="cpu"
    )
    model.eval()
    decoder = model.decoder

    # Build deterministic input codes: [1, n_q, T] int64 for PyTorch
    codes_np = np.full((n_q, T), code_val, dtype=np.int32)  # [n_q, T]
    codes_pt = torch.tensor(codes_np, dtype=torch.long).unsqueeze(0)  # [1, n_q, T]

    out: Dict[str, np.ndarray] = {}

    if "codec_input_codes" in stages:
        # codes_np is [n_q, T]. Store as [T, n_q] (time-first) → GGUF ne[0]=n_q.
        # The C++ graph stores codes in [T, n_q] layout (ne[0]=T innermost? no,
        # actually the input tensor has ne[0]=T ne[1]=n_q). This stage is
        # informational only (not compared element-by-element in the diff).
        out["codec_input_codes"] = codes_np.T.astype(np.float32)  # [T, n_q]

    # ── Hooks to capture intermediate activations ──────────────────────────
    captures: Dict[str, Any] = {}
    handles = []

    # Standard forward_hook captures (first_call_only=True is a safety net
    # since the decoder runs once per dump, but kept for parity with sibling
    # backends that share modules across multiple calls).
    handles.extend(_hooks.capture_modules(captures, [
        ("codec_pre_conv_out", decoder.pre_conv),
        ("codec_up0_out",      decoder.upsample[0][1]),
        ("codec_up1_out",      decoder.upsample[1][1]),
        ("codec_in_conv_out",  decoder.decoder[0]),
        ("codec_blk0_out",     decoder.decoder[1]),
    ], first_call_only=True))

    # Two pre-hooks that _hooks.py doesn't cover (it only registers
    # post-hooks). Inline first-call-only logic mirrors the helper.
    #
    # RVQ output: Decoder calls self.quantizer.decode() directly, so a
    # forward_hook on the quantizer wouldn't fire — we hook pre_conv's
    # PRE-hook to grab the quantizer output (= pre_conv's input).
    if "codec_rvq_out" in stages:
        def cap_rvq(_mod, args):
            if "codec_rvq_out" not in captures and args:
                captures["codec_rvq_out"] = args[0].detach().cpu().float()
        handles.append(decoder.pre_conv.register_forward_pre_hook(cap_rvq))
    # Transformer-output-after-permute: the transformer's own last_hidden_state
    # is (B, T, 1024); after `.permute(0, 2, 1)` in Decoder.forward it's
    # (B, 1024, T). Hook upsample[0]'s pre-hook to see that permuted tensor.
    if "codec_xfmr_out" in stages:
        def cap_xfmr(_mod, args, _kw):
            h = args[0] if args else None
            if h is not None and "codec_xfmr_out" not in captures:
                captures["codec_xfmr_out"] = h.detach().cpu().float()
        handles.append(decoder.upsample[0][0].register_forward_pre_hook(
            cap_xfmr, with_kwargs=True))

    # Final PCM — the full forward captures it directly
    with torch.no_grad():
        pcm_pt = decoder(codes_pt)  # (1, 1, T_pcm)

    _hooks.drop_hooks(handles)

    if "codec_pcm" in stages:
        out["codec_pcm"] = pcm_pt.detach().cpu().float().squeeze().numpy()  # (T_pcm,)

    # Squeeze batch + transpose 2D (C, T) → (T, C) for the time-first ggml
    # convention. Captures from `_hooks.capture_modules` are torch tensors;
    # the inline pre-hooks above also store torch tensors so the same
    # post-process works for both.
    import torch
    for name, t in captures.items():
        if name not in stages:
            continue
        arr = t.squeeze(0).numpy() if isinstance(t, torch.Tensor) else np.asarray(t)
        out[name] = arr.T if arr.ndim == 2 else arr

    return out
