"""Qwen3-TTS speaker encoder (ECAPA-TDNN) reference dump backend.

Runs the PyTorch Qwen3TTSSpeakerEncoder on a fixed deterministic audio
input (440 Hz sine, 3 s, 24 kHz) and dumps the intermediate activations
so crispasr-diff can verify the C++ ECAPA forward numerically.

Stages dumped:
  spk_mel          (T_mel, 128)    log-mel input
  spk_blk0_out     (512, T_mel)    after initial TDNN
  spk_mfa_out      (1536, T_mel)   after multi-frame aggregation
  spk_emb          (1024,)         final speaker embedding

The "audio" arg is ignored — the input is always the fixed sine wave so
both sides see identical inputs without file-path dependencies.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Set

import numpy as np

from . import _hooks

DEFAULT_STAGES = [
    "spk_mel",
    "spk_blk0_out",
    "spk_mfa_out",
    "spk_emb",
]


def dump(*, model_dir: Path, audio: np.ndarray, stages: Set[str],
         max_new_tokens: int) -> Dict[str, np.ndarray]:
    import torch

    ref_path = Path(__file__).resolve().parents[2] / "ref" / "Qwen3-TTS"
    if ref_path.is_dir() and str(ref_path) not in sys.path:
        sys.path.insert(0, str(ref_path))

    from qwen_tts.core.models.modeling_qwen3_tts import (
        Qwen3TTSSpeakerEncoder, mel_spectrogram)
    from qwen_tts.core.models.configuration_qwen3_tts import (
        Qwen3TTSSpeakerEncoderConfig, Qwen3TTSConfig)
    from transformers import AutoModel

    # Fixed 440 Hz sine at 24 kHz, 3 s — matches C++ test
    sr = 24000
    t = np.arange(sr * 3) / sr
    audio_fixed = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    audio_pt = torch.from_numpy(audio_fixed).unsqueeze(0)  # [1, N]

    print(f"  loading speaker encoder from {model_dir} (CPU, fp32)")
    # Load via the full model config
    import json
    with open(model_dir / "config.json") as f:
        cfg_dict = json.load(f)
    spk_cfg = Qwen3TTSSpeakerEncoderConfig(**cfg_dict.get("speaker_encoder_config", {}))

    # Load speaker encoder weights from the safetensors checkpoint
    from safetensors import safe_open
    import glob
    st_files = sorted(glob.glob(str(model_dir / "*.safetensors")))
    tensors = {}
    for fp in st_files:
        h = safe_open(fp, framework="pt")
        for k in h.keys():
            tensors[k] = h.get_tensor(k)

    spk = Qwen3TTSSpeakerEncoder(spk_cfg).float()
    # Load weights: speaker_encoder.* → model state dict
    sd = {}
    for k, v in tensors.items():
        if k.startswith("speaker_encoder."):
            sd[k[len("speaker_encoder."):]] = v.float()
    missing, unexpected = spk.load_state_dict(sd, strict=False)
    if missing: print(f"  [WARN] missing: {missing[:5]}")
    spk.eval()

    out: Dict[str, np.ndarray] = {}
    captures: Dict[str, "torch.Tensor"] = {}

    # Mel spectrogram (24kHz, n_fft=1024, hop=256, fmin=0, fmax=12000)
    with torch.no_grad():
        mel = mel_spectrogram(audio_pt, n_fft=1024, num_mels=128,
                              sampling_rate=24000, hop_size=256,
                              win_size=1024, fmin=0, fmax=12000)  # [1, 128, T]
        mel_t = mel.transpose(1, 2)  # [1, T, 128] — input to speaker_encoder

    if "spk_mel" in stages:
        # Store as (T, 128) time-first matching ggml convention
        out["spk_mel"] = mel_t[0].numpy().astype(np.float32)  # [T, 128]

    # Each module fires once per forward, but we use first_call_only=True for
    # consistency with sibling backends and to be safe if the encoder is ever
    # called more than once per dump in the future.
    handles = _hooks.capture_modules(captures, [
        ("spk_blk0_out", spk.blocks[0]),
        ("spk_mfa_out",  spk.mfa),
    ], first_call_only=True)

    with torch.no_grad():
        emb = spk(mel_t)  # [1, 1024]

    _hooks.drop_hooks(handles)

    if "spk_emb" in stages:
        out["spk_emb"] = emb[0].detach().cpu().float().numpy()  # [1024]

    # Captures are torch tensors with batch dim. Squeeze batch and transpose
    # 2D (C, T) -> (T, C) for the time-first ggml convention.
    for name, t in captures.items():
        if name not in stages:
            continue
        arr = t.squeeze(0).numpy()
        out[name] = arr.T if arr.ndim == 2 else arr

    return out
