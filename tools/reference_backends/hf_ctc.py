"""Transformers *ForCTC reference dump backend (wav2vec2 / hubert / data2vec /
omniASR-CTC).

All four regression sources are plain transformers CTC checkpoints
(Wav2Vec2ForCTC, HubertForCTC, Data2VecAudioForCTC - aadel4/omniASR-CTC-1B-v2
is a Wav2Vec2ForCTC conversion), so one dumper covers them. The model's OWN
feature extractor prepares the input (do_normalize etc.), never a hand-built
front-end.

Stages:

  raw_audio   (N,)    input PCM at 16 kHz
  ctc_logits  (T, V)  CTC head output, pre-softmax - the grid returned by
                      wav2vec2_compute_logits() / omniasr_transcribe_with_logits()
                      (frame-major, row-major float32)

Usage:

  python tools/dump_reference.py --backend wav2vec2 \\
      --model-dir jonatasgrosman/wav2vec2-large-xlsr-53-english \\
      --audio samples/jfk.wav --output /tmp/w2v-ref.gguf
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Set

import numpy as np

DEFAULT_STAGES = [
    "raw_audio",
    "ctc_logits",
]


def dump(*, model_dir: Path, audio: np.ndarray, stages: Set[str], max_new_tokens: int = 0,
         **kwargs) -> Dict[str, np.ndarray]:
    import torch
    from transformers import AutoFeatureExtractor, AutoModelForCTC

    src = str(model_dir)
    fe = AutoFeatureExtractor.from_pretrained(src)
    model = AutoModelForCTC.from_pretrained(src, torch_dtype=torch.float32).eval()

    out: Dict[str, np.ndarray] = {}
    if "raw_audio" in stages:
        out["raw_audio"] = np.asarray(audio, dtype=np.float32).copy()

    inputs = fe(np.asarray(audio, dtype=np.float32), sampling_rate=16000, return_tensors="pt")
    with torch.no_grad():
        logits = model(inputs.input_values).logits  # (1, T, V)
    if "ctc_logits" in stages:
        out["ctc_logits"] = logits[0].float().cpu().numpy().copy()
    return out
