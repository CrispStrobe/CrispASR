"""NVIDIA Nemotron-3-Diarization reference dump backend (#466).

transformers' Nemotron3DiarizationForAudioFrameClassification in offline mode
(the model chunks the recording itself, with the Arrival-Order Speaker Cache),
fp32. Needs a transformers with nemotron3_diarization (5.18.0.dev0 / main).

Stages (layouts match what crispasr-diff nemotron3-diar reads):
  raw_audio      (N,)
  mel            (T, n_mels)   processor input_features (masked, as the model sees it)
  embeds         (Ne, d)       audio_tower.embedder(input_features): 8x stacking + projection
  logits         (T, S)        pre-sigmoid speaker logits, one row per 10 ms
  probs          (T, S)        sigmoid(logits)
  segments_text  str           processor.extract_speaker_dict, "start end speaker" lines
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Set

import numpy as np

DEFAULT_STAGES = ["raw_audio", "mel", "embeds", "logits", "probs", "segments_text"]


def dump(*, model_dir: Path, audio: np.ndarray, stages: Set[str], max_new_tokens: int = 0) -> Dict[str, np.ndarray]:
    import torch
    from transformers import AutoProcessor, Nemotron3DiarizationForAudioFrameClassification

    md = str(model_dir)
    processor = AutoProcessor.from_pretrained(md)
    model = Nemotron3DiarizationForAudioFrameClassification.from_pretrained(md, dtype=torch.float32).eval()

    out: Dict[str, np.ndarray] = {}
    if "raw_audio" in stages:
        out["raw_audio"] = audio.astype(np.float32)
    inputs = processor(audio.astype(np.float32), sampling_rate=16000, return_tensors="pt")
    feats, mask = inputs["input_features"], inputs.get("attention_mask")
    with torch.no_grad():
        if "mel" in stages:
            out["mel"] = feats[0].detach().clone().float().numpy()
        if "embeds" in stages:
            out["embeds"] = model.model.audio_tower.embedder(feats)[0].detach().clone().float().numpy()
        logits = model(input_features=feats, attention_mask=mask).logits
    if "logits" in stages:
        out["logits"] = logits[0].detach().clone().float().numpy()
    if "probs" in stages:
        out["probs"] = logits[0].sigmoid().detach().clone().float().numpy()
    if "segments_text" in stages:
        segs = processor.extract_speaker_dict(logits, mask)[0]
        out["segments_text"] = "\n".join(f"{s['Start']:.2f} {s['End']:.2f} {s['Speaker']}" for s in segs)
        print(f"  nemotron3-diar: {len(segs)} segments, {len({s['Speaker'] for s in segs})} speakers")
    return out
