"""Hooks on the ORIGINAL Inworld Voice Isolator Python blueprint.

Upstream code: github.com/inworld-ai/voice-isolator (Apache-2.0), imported
unchanged from ``VOICE_ISOLATOR_SRC`` (the repo's ``src`` directory, which
holds the ``inworld.voice_isolation`` package). ``--model-dir`` is a directory
holding ``denoiser.safetensors`` (offline) or, with
``VOICE_ISOLATOR_VARIANT=realtime``, ``denoiser_realtime.safetensors``.

The 16 kHz harness input is resampled to 48 kHz with the upstream
``audio.resample`` (torchaudio kaiser_best), exactly as ``VoiceIsolator``
does, and the result is stored as ``input_48k`` so the C++ diff feeds the
identical samples. The network then runs as ``VoiceIsolatorNetwork.enhance``
(offline) or its full-clip causal forward with the trained read-ahead
(realtime), in float32 on the CPU.

Layouts are NumPy row-major (GGML reverses them):
  input_48k     [samples]
  stft          [bins, frames, 2]            complex spectrum (re, im)
  encode        [bands, frames, width]       band tokens + embeddings
  block{i}      [bands, frames, width]       output of block i
  decoder_raw   [bins, frames, 4]            per-bin head output (after read-ahead)
  voice         [samples]                    isolated voice at 48 kHz
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

N_BLOCKS = 7
DEFAULT_STAGES = ["input_48k", "stft", "encode", *[f"block{i}" for i in range(N_BLOCKS)], "decoder_raw", "voice"]


def _import_upstream():
    src = os.environ.get("VOICE_ISOLATOR_SRC")
    if src:
        sys.path.insert(0, src)
    from inworld.voice_isolation import audio as audio_lib  # noqa: PLC0415
    from inworld.voice_isolation import isolator  # noqa: PLC0415
    from inworld.voice_isolation import model as modeling  # noqa: PLC0415
    return audio_lib, isolator, modeling


def load(model_dir: str | Path, variant: str = "offline"):
    _, isolator, _ = _import_upstream()
    name = "denoiser_realtime.safetensors" if variant == "realtime" else "denoiser.safetensors"
    model = isolator.load_model(str(Path(model_dir) / name), "cpu")
    if variant == "realtime":
        lookahead = int(os.environ.get("VOICE_ISOLATOR_LOOKAHEAD", "2"))
        model.set_lookahead(lookahead)
    return model


def dump_48k(model, wave48, stages=None) -> dict:
    """Capture every stage of one run over 48 kHz mono float32 samples."""
    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415

    stages = set(DEFAULT_STAGES if not stages else stages)
    out: dict = {}
    x = torch.from_numpy(np.ascontiguousarray(wave48, dtype=np.float32))
    if "input_48k" in stages:
        out["input_48k"] = x.numpy().copy()

    handles = []

    def keep(name, value):
        if name in stages:
            out[name] = value.detach().float().cpu().numpy().copy()

    for i, block in enumerate(model.blocks):
        handles.append(block.register_forward_hook(lambda _m, _i, o, i=i: keep(f"block{i}", o[0])))
    handles.append(model.denoise_decoder.register_forward_hook(lambda _m, _i, o: keep("decoder_raw_pre", o[0])))
    original_encode = model.encode
    original_stft = model.stft

    def encode(spectrum):
        y = original_encode(spectrum)
        keep("encode", y[0])
        return y

    def stft(audio):
        s = original_stft(audio)
        keep("stft", torch.view_as_real(s[0]))
        return s

    model.encode = encode
    model.stft = stft
    try:
        with torch.inference_mode():
            n = x.shape[-1]
            chunk = int(model.config.chunk_seconds * model.config.sample_rate)
            if model.offline and n > chunk:
                raise ValueError("reference stages are captured on a single chunk; pass <= chunk_seconds of audio")
            voice = model.enhance(x) if model.offline else model(x[None])[0]
    finally:
        for h in handles:
            h.remove()
        model.encode = original_encode
        model.stft = original_stft
    if "decoder_raw_pre" in out:
        raw = out.pop("decoder_raw_pre")  # [bins, frames, 4]
        if not model.offline and model.lookahead_frames:
            idx = np.minimum(np.arange(raw.shape[1]) + model.lookahead_frames, raw.shape[1] - 1)
            raw = raw[:, idx]
        if "decoder_raw" in stages:
            out["decoder_raw"] = np.ascontiguousarray(raw)
    if "voice" in stages:
        out["voice"] = voice.float().numpy().copy()
    return out


def dump(model_dir, audio, stages, max_new_tokens=None, **kwargs):
    """Unified harness entry: 16 kHz mono in, upstream resample to 48 kHz."""
    import torch  # noqa: PLC0415

    audio_lib, _, _ = _import_upstream()
    variant = os.environ.get("VOICE_ISOLATOR_VARIANT", "offline")
    model = load(model_dir, variant)
    source_rate = int(kwargs.get("sample_rate", 16000))
    wave = audio_lib.resample(torch.from_numpy(audio).float(), source_rate, model.config.sample_rate)
    return dump_48k(model, wave.numpy(), stages)
