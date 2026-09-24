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

NEMOTRON3_DIAR_MODE=low_latency|very_low_latency|ultra_low_latency dumps a
streaming session instead: the model card's chunk-by-chunk driver, the model's
speaker_cache carried between forwards. mel / embeds are then the scored part
of every chunk (its look-ahead is the next chunk's head), concatenated.
"""

from __future__ import annotations

import os
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
    mode = os.environ.get("NEMOTRON3_DIAR_MODE", "")
    if mode and mode != "offline":
        return _dump_streaming(processor, model, audio.astype(np.float32), stages, mode, out)
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


def _dump_streaming(processor, model, audio, stages, mode, out):
    """The model card's streaming example, verbatim in its chunking."""
    import torch

    sr = processor.feature_extractor.sampling_rate
    processor.set_streaming_mode(mode)
    F = processor.subsampling_factor

    def chunks():
        yield processor(audio[: processor.num_samples_first_audio_chunk], sampling_rate=sr, is_streaming=True,
                        is_first_audio_chunk=True)
        mel_frame_idx = processor.num_mel_frames_per_step
        start_idx = processor.audio_chunk_start(mel_frame_idx)
        while (end_idx := start_idx + processor.num_samples_per_audio_chunk) <= audio.shape[0]:
            yield processor(audio[start_idx:end_idx], sampling_rate=sr, is_streaming=True, is_first_audio_chunk=False)
            mel_frame_idx += processor.num_mel_frames_per_step
            start_idx = processor.audio_chunk_start(mel_frame_idx)
        yield processor(audio[start_idx:], sampling_rate=sr, is_streaming=True, is_first_audio_chunk=False,
                        is_last_audio_chunk=True)

    cache, logits, mels, embs = None, [], [], []
    with torch.no_grad():
        for inputs in chunks():
            feats = inputs["input_features"]
            la = int(inputs.get("num_lookahead_frames", 0) or 0)
            o = model(**inputs, speaker_cache=cache)
            cache = o.speaker_cache
            n = o.logits.shape[1]  # scored 10 ms frames of this chunk
            logits.append(o.logits)
            mels.append(feats[0, :n].detach().clone().float())
            e = model.model.audio_tower.embedder(feats)[0]
            embs.append(e[: e.shape[0] - la].detach().clone().float())
    logits = torch.cat(logits, dim=1)
    print(f"  nemotron3-diar streaming {mode}: {len(mels)} chunks, {logits.shape[1]} frames")
    if "mel" in stages:
        out["mel"] = torch.cat(mels).numpy()
    if "embeds" in stages:
        out["embeds"] = torch.cat(embs).numpy()
    if "logits" in stages:
        out["logits"] = logits[0].detach().clone().float().numpy()
    if "probs" in stages:
        out["probs"] = logits[0].sigmoid().detach().clone().float().numpy()
    if "segments_text" in stages:
        segs = processor.extract_speaker_dict(logits)[0]
        out["segments_text"] = "\n".join(f"{s['Start']:.2f} {s['End']:.2f} {s['Speaker']}" for s in segs)
        print(f"  nemotron3-diar: {len(segs)} segments, {len({s['Speaker'] for s in segs})} speakers")
    return out
