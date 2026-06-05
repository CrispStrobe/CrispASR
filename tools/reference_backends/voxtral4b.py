"""Voxtral-Mini-4B-Realtime-2602 reference dump backend.

Port of `models/voxtral4b-dump-ref.py` into the modular
tools/dump_reference.py interface. Captures:

  mel_spectrogram    after the processor's feature extractor
  encoder_output     after model.get_audio_features (post-projector)
  t_cond             after model.time_embedding(delay_tokens=6)
  llm_logits         from model.generate() (best-effort — the
                     realtime pipeline doesn't expose per-step scores
                     in all transformers versions, so this may be
                     an empty tensor placeholder)
  llm_argmax         generated token IDs
  generated_text     decoded transcript
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Set

import numpy as np

DEFAULT_STAGES = [
    "raw_audio",
    "mel_spectrogram",
    "encoder_output",
    "t_cond",
    "llm_argmax",
    "generated_text",
]


def dump(*, model_dir: Path, audio: np.ndarray, stages: Set[str],
         max_new_tokens: int) -> Dict[str, np.ndarray]:
    """Run Voxtral 4B Realtime reference forward and return stage captures."""
    import torch
    try:
        from transformers import VoxtralRealtimeForConditionalGeneration, AutoProcessor
    except ImportError as e:
        raise SystemExit(
            "transformers with VoxtralRealtime support required.\n"
            "Install: pip install 'transformers>=5.2.0'\n"
            f"(import error: {e})")

    print(f"  loading Voxtral 4B Realtime from {model_dir}")
    processor = AutoProcessor.from_pretrained(str(model_dir))
    model = VoxtralRealtimeForConditionalGeneration.from_pretrained(
        str(model_dir), torch_dtype=torch.float32, device_map="cpu",
    ).eval()

    # ---- Processor: mel + input_ids from a raw PCM array ----
    # The Voxtral processor takes raw audio directly and internally
    # computes mel + builds the BOS/STREAMING_PAD prompt.
    inputs = processor(audio, return_tensors="pt")

    out: Dict[str, np.ndarray] = {}
    if "mel_spectrogram" in stages:
        # shape (1, 1, n_mels, T) -> (n_mels, T)
        mel = inputs["input_features"][0, 0]
        out["mel_spectrogram"] = mel.detach().cpu().float().numpy()

    with torch.no_grad():
        # ---- Encoder (+ projector) ----
        if "encoder_output" in stages:
            audio_outputs = model.get_audio_features(
                input_features=inputs["input_features"],
                return_dict=True,
            )
            enc = audio_outputs.pooler_output  # shape (B, N, proj_dim)
            out["encoder_output"] = enc[0].detach().cpu().float().numpy()

        # ---- Time embedding (delay_tokens=6 is the default 4B config) ----
        if "t_cond" in stages:
            time_tensor = torch.full((1,), 6.0)
            t_cond = model.time_embedding(time_tensor)
            out["t_cond"] = t_cond[0].detach().cpu().float().numpy()

        # ---- Full generation for argmax + text ----
        want_argmax = "llm_argmax" in stages
        want_text   = "generated_text" in stages
        if want_argmax or want_text:
            gen = model.generate(**inputs, max_new_tokens=max_new_tokens)
            if want_argmax:
                out["llm_argmax"] = gen[0].detach().cpu().int().numpy().astype(np.int32)
            if want_text:
                decoded = processor.batch_decode(gen, skip_special_tokens=True)
                out["generated_text"] = decoded[0] if decoded else ""

    return out
