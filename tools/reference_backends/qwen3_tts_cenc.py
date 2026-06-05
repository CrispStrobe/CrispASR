"""Qwen3-TTS-Tokenizer-12Hz codec ENCODER reference dump.

Hooks the PyTorch Qwen3TTSTokenizerV2Encoder forward at three points so
crispasr-diff can verify the C++ encoder numerically against the
reference (the same diff-harness approach that took ECAPA from cos=0.74
to 0.999999 and brought all 8 codec-decoder stages to PASS).

Stages dumped (all stored in (T, C) time-first to match ggml flat layout):
  cenc_input_audio   — fixed deterministic 24kHz PCM (3s of clone.wav)
  cenc_seanet_out    — output of self.encoder (SEANet) [T_enc, 512]
  cenc_xfmr_out      — output of self.encoder_transformer [T_enc, 512]
  cenc_ds_out        — after self.downsample [T_frames, 512]
  cenc_codes         — final RVQ codes [T_frames, 16] as float32

The audio arg is unused — we use a fixed slice of clone.wav so both
sides see identical inputs.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, Set

import numpy as np

from . import _hooks

DEFAULT_STAGES = [
    "cenc_input_audio",
    "cenc_se_init",
    "cenc_se_s0",
    "cenc_se_s1",
    "cenc_se_s2",
    "cenc_se_s3",
    "cenc_seanet_out",
    "cenc_xfmr_out",
    "cenc_ds_out",
    "cenc_codes",
]


def _load_clone_audio() -> np.ndarray:
    """Load a 24kHz mono float32 WAV.

    Defaults to samples/qwen3_tts/clone.wav and trims to 3 seconds for speed.
    Override with:
      QWEN3_TTS_CENC_WAV=/path/to/file.wav
      QWEN3_TTS_CENC_FULL=1   # keep full duration
    """
    repo_root = Path(__file__).resolve().parents[2]
    wav_override = os.environ.get("QWEN3_TTS_CENC_WAV", "").strip()
    wav_path = Path(wav_override) if wav_override else (repo_root / "samples" / "qwen3_tts" / "clone.wav")
    if not wav_path.exists():
        raise FileNotFoundError(f"clone.wav not at {wav_path}")
    # Read RIFF/WAVE — IEEE Float, mono, 24kHz
    with open(wav_path, "rb") as f:
        data = f.read()
    if data[0:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise ValueError("not a RIFF WAV")
    pos = 12
    sr = 0; n_ch = 0; bps = 0; fmt = 0
    samples = None
    while pos < len(data) - 8:
        cid = data[pos:pos+4]
        csz = int.from_bytes(data[pos+4:pos+8], "little")
        if cid == b"fmt ":
            fmt = int.from_bytes(data[pos+8:pos+10], "little")
            n_ch = int.from_bytes(data[pos+10:pos+12], "little")
            sr = int.from_bytes(data[pos+12:pos+16], "little")
            bps = int.from_bytes(data[pos+22:pos+24], "little")
        elif cid == b"data":
            raw = data[pos+8:pos+8+csz]
            if fmt == 3 and bps == 32:
                samples = np.frombuffer(raw, dtype="<f4")
            elif fmt == 1 and bps == 16:
                samples = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
            else:
                raise ValueError(f"unsupported WAV format fmt={fmt} bps={bps}")
            break
        pos += 8 + csz + (csz % 2)
    if samples is None or sr != 24000 or n_ch != 1:
        raise ValueError(f"expected 24kHz mono, got sr={sr} ch={n_ch}")
    if os.environ.get("QWEN3_TTS_CENC_FULL", "") not in ("", "0"):
        return samples.astype(np.float32)
    # Default: take first 3 seconds (smaller diff = faster, frame count divisible by all strides)
    return samples[:24000 * 3].astype(np.float32)


def dump(*, model_dir: Path, audio: np.ndarray, stages: Set[str],
         max_new_tokens: int) -> Dict[str, np.ndarray]:
    import torch

    ref_path = Path(__file__).resolve().parents[2] / "ref" / "Qwen3-TTS"
    if ref_path.is_dir() and str(ref_path) not in sys.path:
        sys.path.insert(0, str(ref_path))

    from qwen_tts.core.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 import (
        Qwen3TTSTokenizerV2Model,
    )

    print(f"  loading Qwen3-TTS-Tokenizer-12Hz from {model_dir}")
    model = Qwen3TTSTokenizerV2Model.from_pretrained(
        str(model_dir), dtype=torch.float32, device_map="cpu"
    )
    model.eval()
    encoder = model.encoder

    # Fixed input: first 3 seconds of clone.wav at 24kHz
    audio_np = _load_clone_audio()
    print(f"  input audio: {len(audio_np)} samples ({len(audio_np)/24000:.2f}s)")

    out: Dict[str, np.ndarray] = {}
    if "cenc_input_audio" in stages:
        out["cenc_input_audio"] = audio_np

    captures: Dict[str, "torch.Tensor"] = {}

    # Modules whose outputs need a (B, C, T) → (T, C) transpose after squeeze.
    # SEANet (encoder.encoder) returns [B, 512, T_enc]; intra-SEANet layers
    # return [B, C, T]; downsample returns [B, 512, T_frames]. We collect them
    # all into one capture set, transpose at the end.
    needs_transpose = {
        "cenc_seanet_out": encoder.encoder,
        "cenc_se_init":    encoder.encoder.layers[0],
        "cenc_se_s0":      encoder.encoder.layers[3],
        "cenc_se_s1":      encoder.encoder.layers[6],
        "cenc_se_s2":      encoder.encoder.layers[9],
        "cenc_se_s3":      encoder.encoder.layers[12],
        "cenc_ds_out":     encoder.downsample,
    }
    handles = _hooks.capture_modules(captures, [
        (name, mod) for name, mod in needs_transpose.items() if name in stages
    ], first_call_only=True)

    # encoder_transformer returns BaseModelOutputWithPast — _hooks._hook_factory
    # picks `output.last_hidden_state` automatically. Shape stays (B, T, 512),
    # NO transpose needed (already time-first).
    if "cenc_xfmr_out" in stages:
        handles.extend(_hooks.capture_modules(
            captures, [("cenc_xfmr_out", encoder.encoder_transformer)],
            first_call_only=True,
        ))

    # Run the encoder
    audio_pt = torch.from_numpy(audio_np).unsqueeze(0).unsqueeze(0)  # [1, 1, T]
    with torch.no_grad():
        result = encoder.encode(audio_pt)

    _hooks.drop_hooks(handles)

    # result is MimiEncoderOutput with audio_codes [B, n_q, T_codec]
    codes = result.audio_codes if hasattr(result, "audio_codes") else result[0]
    codes_16 = codes[:, :16, :]  # take first 16 quantizers
    if "cenc_codes" in stages:
        out["cenc_codes"] = codes_16[0].T.detach().cpu().numpy().astype(np.float32)

    # Squeeze batch + transpose channels-first stages to time-first.
    for name, t in captures.items():
        if name not in stages:
            continue
        arr = t.squeeze(0).numpy()
        if name in needs_transpose:
            # (C, T) → (T, C)
            arr = arr.T if arr.ndim == 2 else arr
        out[name] = arr

    return out
