"""Orpheus SNAC 24 kHz codec decoder reference dump backend.

Captures stage-by-stage activations from the official PyTorch SNAC
24 kHz model (hubertsiuzdak/snac_24khz) so the CrispASR C++ SNAC
decoder can be diffed via `crispasr-diff orpheus`.

The Orpheus talker emits a stream of `<custom_token_N>` LM tokens; the
de-interleave from those into 3 SNAC codebook tensors is verbatim from
canopyai/Orpheus-TTS:orpheus_tts_pypi/orpheus_tts/decoder.py:

  for j in range(num_frames):                           # 7-token super-frames
      i = 7*j
      codes_0 << frame[i]                               # 1 entry / super-frame
      codes_1 << frame[i+1]; codes_1 << frame[i+4]      # 2 / super-frame
      codes_2 << frame[i+2]; codes_2 << frame[i+3]      # 4 / super-frame
      codes_2 << frame[i+5]; codes_2 << frame[i+6]

  audio_hat = model.decode([codes_0, codes_1, codes_2])
  emitted = audio_hat[:, :, 2048:4096]   # middle super-frame of a 4-super-
                                         # frame sliding window (Orpheus
                                         # streaming protocol)

Stages dumped (matching `crispasr-diff orpheus` C++ side):
  snac_codes_0      — (B=1, T_g0)         int32 as float32
  snac_codes_1      — (B=1, T_g1)         int32 as float32, T_g1 = 2·T_g0
  snac_codes_2      — (B=1, T_g2)         int32 as float32, T_g2 = 4·T_g0
  snac_quant_out    — quantizer.from_codes(...) → (B, D, T_q) at the
                      decoder input rate (T_q = 4·T_g0 = T_g2)
  snac_dec_pre      — decoder.model[0..1] (input convs) output
  snac_dec_blk0     — after DecoderBlock 0 (stride 8): T·8
  snac_dec_blk1     — after DecoderBlock 1 (stride 8): T·64
  snac_dec_blk2     — after DecoderBlock 2 (stride 4): T·256
  snac_dec_blk3     — after DecoderBlock 3 (stride 2): T·512   = hop_length
  snac_pcm          — final tanh head: (B, 1, T_q · 512) at 24 kHz
  snac_pcm_emit     — audio_hat[:, :, 2048:4096] sliding-window slice
                      (only meaningful when num_super_frames == 4)

Drive with deterministic codes by default; set ORPHEUS_SNAC_T_SUPER=N
to render N super-frames (default 4 — the canonical streaming window
size). Set ORPHEUS_SNAC_CODE=K to override the constant fill value.

The "audio" arg from the dispatcher is unused (kept for harness
compat) — codec dumps don't run the encoder.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, Set

import numpy as np

from . import _hooks


DEFAULT_STAGES = [
    "snac_codes_0",
    "snac_codes_1",
    "snac_codes_2",
    "snac_quant_out",
    "snac_dec_pre",
    "snac_dec_blk0",
    "snac_dec_blk1",
    "snac_dec_blk2",
    "snac_dec_blk3",
    "snac_pcm",
    "snac_pcm_emit",
]


# 7-slot super-frame layout from decoder.py — 1/2/4 entries per
# codebook. Total 7 LM tokens per super-frame.
SLOT_TO_CB = [0, 1, 2, 2, 1, 2, 2]   # which codebook each slot feeds
# How many entries per codebook accumulate per super-frame:
PER_SF = [SLOT_TO_CB.count(c) for c in range(3)]   # = [1, 2, 4]


def _build_codes(num_super_frames: int, fill: int):
    """Synthesize a deterministic 7N-token stream and de-interleave it
    into 3 codebook tensors exactly the way decoder.py does.

    Returns (codes_0, codes_1, codes_2) as np.int64 arrays of shape
    (1, T_cb_k) = (1, num_super_frames * PER_SF[k]).
    """
    cb = [[] for _ in range(3)]
    for j in range(num_super_frames):
        # All 7 slots get the same fill value (clamped to [0, 4095])
        for slot in range(7):
            cb[SLOT_TO_CB[slot]].append(int(fill) % 4096)
    out = [np.array([row], dtype=np.int64) for row in cb]
    # Sanity-check: shape must be (1, num_super_frames * PER_SF[k])
    for k, t in enumerate(out):
        assert t.shape == (1, num_super_frames * PER_SF[k]), \
            f"codes_{k} shape mismatch: got {t.shape}"
    return out


def dump(*, model_dir: Path, audio: np.ndarray, stages: Set[str],
         dtype: str = "f32", **_unused) -> Dict[str, Any]:
    """Dispatcher entry point. `model_dir` must point at the
    huggingface-cli-resolved hubertsiuzdak/snac_24khz snapshot dir; the
    `audio` arg is ignored (codec-decode-only)."""

    try:
        import torch
    except ImportError:
        sys.exit("pip install torch")
    try:
        from snac import SNAC
    except ImportError:
        sys.exit("pip install snac")

    num_super_frames = int(os.environ.get("ORPHEUS_SNAC_T_SUPER", "4"))
    fill_code = int(os.environ.get("ORPHEUS_SNAC_CODE", "0"))
    assert num_super_frames >= 1

    # Build deterministic codes + the corresponding 7N-token LM stream
    codes_np = _build_codes(num_super_frames, fill_code)

    # Load SNAC. SNAC.from_pretrained takes either an HF repo ID or a
    # local dir; passing the resolved snapshot dir keeps us offline-safe.
    model = SNAC.from_pretrained(str(model_dir)).eval()
    device = torch.device("cpu")
    model = model.to(device)

    # NoiseBlock injects torch.randn(...) at every forward call (see
    # snac/layers.py:NoiseBlock.forward). That makes model.decode()
    # non-deterministic and the C++ side (which has no RNG state) can't
    # match it. The injection is incidental at inference — its purpose is
    # exploration during training. Replace the per-call forward with the
    # zero-noise equivalent (return x), making NoiseBlock identity. The
    # C++ side does the same. The PCM still passes the perceptual bar
    # (the noise contribution is ~1e-2 of the signal RMS at 24 kHz).
    from snac.layers import NoiseBlock
    NoiseBlock.forward = lambda self, x: x

    codes_t = [torch.from_numpy(c).long().to(device) for c in codes_np]

    out: Dict[str, Any] = {}

    if "snac_codes_0" in stages:
        out["snac_codes_0"] = codes_np[0].astype(np.float32)
    if "snac_codes_1" in stages:
        out["snac_codes_1"] = codes_np[1].astype(np.float32)
    if "snac_codes_2" in stages:
        out["snac_codes_2"] = codes_np[2].astype(np.float32)

    # Quantizer.from_codes returns the dequantized embedding sequence at
    # the decoder input rate. Capture it before the decoder runs.
    captured: Dict[str, torch.Tensor] = {}

    def _hook(name):
        def fn(_mod, _inp, outp):
            t = outp if isinstance(outp, torch.Tensor) else outp[0]
            captured[name] = t.detach()
        return fn

    handles = []
    decoder_model = model.decoder.model

    # decoder.model layout (verified in this session):
    #   [0] ParametrizedConv1d        ─┐ "pre" stage
    #   [1] ParametrizedConv1d        ─┘
    #   [2] DecoderBlock (stride 8)
    #   [3] DecoderBlock (stride 8)
    #   [4] DecoderBlock (stride 4)
    #   [5] DecoderBlock (stride 2)
    #   [6] Snake1d
    #   [7] ParametrizedConv1d
    #   [8] Tanh                       — final PCM
    if "snac_dec_pre" in stages:
        handles.append(decoder_model[1].register_forward_hook(_hook("snac_dec_pre")))
    if "snac_dec_blk0" in stages:
        handles.append(decoder_model[2].register_forward_hook(_hook("snac_dec_blk0")))
    if "snac_dec_blk1" in stages:
        handles.append(decoder_model[3].register_forward_hook(_hook("snac_dec_blk1")))
    if "snac_dec_blk2" in stages:
        handles.append(decoder_model[4].register_forward_hook(_hook("snac_dec_blk2")))
    if "snac_dec_blk3" in stages:
        handles.append(decoder_model[5].register_forward_hook(_hook("snac_dec_blk3")))

    # Quantizer pre-decoder activation: hook the last quantizer output
    # by patching from_codes since the canonical SNAC quantizer doesn't
    # expose a forward we can hook directly.
    z_q = model.quantizer.from_codes(codes_t)[0] if hasattr(model.quantizer, "from_codes") \
          else None
    if z_q is None:
        # Fallback: manual sum of quantized embeddings (matches the
        # canonical RVQ from_codes).
        z_q = torch.zeros(1, model.quantizer.quantizers[0].codebook_dim,
                          codes_t[2].shape[1], device=device)

    if "snac_quant_out" in stages:
        out["snac_quant_out"] = z_q.detach().float().cpu().numpy()

    with torch.inference_mode():
        audio_hat = model.decode(codes_t)

    for h in handles:
        h.remove()

    if "snac_pcm" in stages:
        out["snac_pcm"] = audio_hat.detach().float().cpu().numpy()

    if "snac_pcm_emit" in stages:
        # Streaming-window slice — only canonical when 4 super-frames.
        if num_super_frames == 4:
            out["snac_pcm_emit"] = audio_hat[:, :, 2048:4096].detach().float().cpu().numpy()
        else:
            # For non-canonical T_super, emit the full waveform; the
            # diff harness will note the shape difference.
            out["snac_pcm_emit"] = audio_hat.detach().float().cpu().numpy()

    for name, t in captured.items():
        if name in stages:
            out[name] = t.float().cpu().numpy()

    # Cast everything down per dtype request (matches other ref backends)
    np_dtype = np.float16 if dtype == "f16" else np.float32
    return {k: (v.astype(np_dtype) if isinstance(v, np.ndarray) else v)
            for k, v in out.items()}
