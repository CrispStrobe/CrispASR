"""Moonshine (UsefulSensors) reference dump backend.

Loads `usefulsensors/moonshine-tiny` or `usefulsensors/moonshine-base`
via HuggingFace transformers and captures the encoder output for
crispasr-diff comparison against the C++ runtime.

Stages:

  raw_audio        (N,)              input PCM at 16 kHz
  encoder_output   (T_enc, d_model)  conv-stem + rotary-attention encoder
                                     output.  Matches the tensor returned by
                                     `moonshine_encode()` in src/moonshine.h:
                                     (T_enc, hidden_dim) row-major float32.

Usage:

  python tools/dump_reference.py --backend moonshine \\
      --model-dir usefulsensors/moonshine-tiny \\
      --audio samples/jfk.wav \\
      --output /tmp/moonshine-tiny-ref.gguf

  python tools/dump_reference.py --backend moonshine \\
      --model-dir usefulsensors/moonshine-base \\
      --audio samples/jfk.wav \\
      --output /tmp/moonshine-base-ref.gguf
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Set

import numpy as np

DEFAULT_STAGES = [
    "raw_audio",
    "encoder_output",
]


def dump(*, model_dir: Path, audio: np.ndarray, stages: Set[str],
         max_new_tokens: int) -> Dict[str, np.ndarray]:
    """Run Moonshine reference forward and return stage captures.

    `model_dir` may be either a local HF cache path or a HuggingFace
    pretrained name (e.g. "usefulsensors/moonshine-tiny").

    The encoder used here is `MoonshineForConditionalGeneration`
    available in transformers >= 4.46. The model encodes raw PCM directly
    (no explicit mel step) — the audio front-end is the first few conv
    layers of the encoder. We capture the final encoder output before the
    cross-attention key/value projection.
    """
    import torch
    try:
        from transformers import MoonshineForConditionalGeneration
    except ImportError as e:
        raise SystemExit(
            "transformers >= 4.46 required for Moonshine.\n"
            "Install: pip install --upgrade transformers\n"
            f"(import error: {e})")

    pretrained = str(model_dir)
    print(f"  loading Moonshine model from {pretrained}")
    model = MoonshineForConditionalGeneration.from_pretrained(pretrained, torch_dtype=torch.float32)
    model.eval()
    dev = next(model.parameters()).device

    out: Dict[str, np.ndarray] = {}

    if "raw_audio" in stages:
        out["raw_audio"] = audio.astype(np.float32)

    if "encoder_output" in stages:
        # Moonshine's encoder takes raw PCM as a (B, N) float32 tensor.
        # No separate preprocessor step — the first conv layers act as the
        # audio front-end. `attention_mask` is None for a single utterance.
        # The encoder lives at model.model.encoder (not model.encoder).
        encoder = model.model.encoder
        input_values = torch.from_numpy(audio.astype(np.float32)).unsqueeze(0).to(dev)
        with torch.no_grad():
            enc_out = encoder(input_values)
            # HF MoonshineEncoder returns a BaseModelOutput;
            # last_hidden_state has shape (B, T_enc, hidden_dim).
            h = enc_out.last_hidden_state  # (1, T_enc, hidden_dim)
        # Drop batch dim → (T_enc, hidden_dim), matching moonshine_encode()
        arr = h[0].detach().cpu().float().numpy()
        out["encoder_output"] = arr

    return out
