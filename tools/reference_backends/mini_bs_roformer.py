"""Hooks on the ORIGINAL pinned Mini-BS-RoFormer V2.5 Python blueprint.

This module imports the upstream Python files without altering their forward
methods. Every numerical stage is captured from the original method or module
that produced it. In-memory captures own their memory; an optional synchronous
sink can serialize each stage before the forward continues. The stereo/native-rate entry
point is ``dump_waveform``; ``dump`` adapts the unified 16-kHz-mono harness.

Layouts are NumPy row-major (reverse the dimensions for GGML):
  input_audio             [channels, samples]
  stft_input              [frames, input_bins * channels * 2]
  stft_output             [channels, output_bins, frames, 2]
  band_split              [frames, bands, band_proj_size]
  time_compressed         [compressed_frames, bands, hidden_size]
  layer{i}_{time,freq}     [compressed_frames + registers, bands + registers, hidden_size]
  encoder_output          [compressed_frames, bands, hidden_size]
  freq_pooled             [compressed_frames, hidden_size]
  stem_embeddings         [stems, hidden_size]
  mask_raw                [stems, frames, output_bins * channels * 2]
  output_{stem}           [channels, samples]

The registered row/column positions in axial layers are retained: they are
part of the original computation and cannot be dropped to improve a score.
"""

from __future__ import annotations

import functools
import hashlib
import importlib.util
import json
import math
import os
import sys
import types
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

MODEL_ID = "HiDolen/Mini-BS-RoFormer-V2.5-46.2M"
MODEL_REVISION = "38dafc78834607aec57d8c0cb3568523f9ea5563"
SOURCE_HASHES = {
    "configuration_bs_roformer.py": "0731875d28b6db7a28d435172ea0969eafd91dac44ab18c487d88750441138ee",
    "modeling_bs_roformer.py": "b3175a8c9577a71642e2d5bc90cbcc53a16ddc6450f30c3d4b316264449ee98f",
    "config.json": "5f228ea7ec785a3cc8b465b74aadc1cf1f62a94dbf3afe3c3dfde121ddc263ab",
    "model.safetensors": "0f173c42b871f849905db96c4197488b683e0398af4673b902e60dd347785180",
}
SOURCES = ("bass", "drums", "other", "vocals")
DEFAULT_STAGES = [
    "input_audio", "stft_input", "stft_output", "band_split", "time_compressed",
    *[f"layer{i}_{axis}" for i in range(6) for axis in ("time", "freq")],
    "encoder_output", "freq_pooled", "stem_embeddings", "mask_raw",
    *[f"output_{source}" for source in SOURCES],
]


def _snapshot(model_dir: str | Path) -> Path:
    path = Path(model_dir).expanduser()
    if path.is_dir():
        return path
    if str(model_dir) != MODEL_ID:
        raise FileNotFoundError(f"expected a local model snapshot or {MODEL_ID}")
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(
        MODEL_ID, revision=MODEL_REVISION,
        allow_patterns=["config.json", "model.safetensors", "*.py", "README.md"],
    ))


def verify_source_files(path: Path) -> None:
    """Verify identity before any upstream source code is executed."""
    for filename, expected in SOURCE_HASHES.items():
        digest = hashlib.sha256()
        with (path / filename).open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        actual = digest.hexdigest()
        if actual != expected:
            raise ValueError(f"{filename} does not match original revision {MODEL_REVISION}: "
                             f"expected SHA256 {expected}, got {actual}")


def _load(model_dir: str | Path, device: str = "cpu", dtype: str = "float32"):
    """Return the unmodified upstream model with a strict checkpoint load.

    A local package import avoids network-dependent AutoModel dispatch and
    loading a different cached revision. Source hashes are checked before the
    model files execute. Transformers' actual version is recorded in captures.
    """
    import torch
    from safetensors.torch import load_file

    path = _snapshot(model_dir)
    verify_source_files(path)
    namespace = "_crispasr_original_mini_bs_roformer_" + MODEL_REVISION
    if namespace not in sys.modules:
        package = types.ModuleType(namespace)
        package.__path__ = [str(path.resolve())]
        sys.modules[namespace] = package
        for basename in ("configuration_bs_roformer", "modeling_bs_roformer"):
            name = f"{namespace}.{basename}"
            spec = importlib.util.spec_from_file_location(name, path / f"{basename}.py")
            if spec is None or spec.loader is None:
                raise ImportError(f"cannot import {name}")
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
    config_cls = sys.modules[namespace + ".configuration_bs_roformer"].BSRoformerConfig
    model_cls = sys.modules[namespace + ".modeling_bs_roformer"].BSRoformerForMaskedEstimation
    config = config_cls(**json.loads((path / "config.json").read_text(encoding="utf-8")))
    model = model_cls(config)
    state = load_file(str(path / "model.safetensors"), device="cpu")
    model.load_state_dict(state, strict=True)
    del state
    torch_dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[dtype]
    model = model.eval().to(device=device, dtype=torch_dtype)
    return model


def _load_published(model_dir: str | Path, device: str = "cpu"):
    """Load with the original published ``from_pretrained`` dtype behavior.

    No dtype argument or post-load dtype cast is supplied. Transformers uses
    the pinned configuration's dtype during construction as well as weight
    loading; this also preserves original nonpersistent buffer initialization.
    The separate ``_load`` path remains the F32 numerical reference oracle.
    """
    path = _snapshot(model_dir)
    verify_source_files(path)
    namespace = "_crispasr_original_mini_bs_roformer_" + MODEL_REVISION
    if namespace not in sys.modules:
        package = types.ModuleType(namespace)
        package.__path__ = [str(path.resolve())]
        sys.modules[namespace] = package
        for basename in ("configuration_bs_roformer", "modeling_bs_roformer"):
            name = f"{namespace}.{basename}"
            spec = importlib.util.spec_from_file_location(name, path / f"{basename}.py")
            if spec is None or spec.loader is None:
                raise ImportError(f"cannot import {name}")
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
    config_cls = sys.modules[namespace + ".configuration_bs_roformer"].BSRoformerConfig
    model_cls = sys.modules[namespace + ".modeling_bs_roformer"].BSRoformerForMaskedEstimation
    config = config_cls(**json.loads((path / "config.json").read_text(encoding="utf-8")))
    model, info = model_cls.from_pretrained(
        str(path), config=config, local_files_only=True, output_loading_info=True,
    )
    if not isinstance(info, dict):
        raise RuntimeError("original from_pretrained did not return checkpoint loading information")
    failures = {key: info[key] for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
                if info.get(key)}
    if failures:
        raise RuntimeError(f"original from_pretrained did not load the exact checkpoint: {failures}")
    return model.eval().to(device=device)


def _array(tensor) -> np.ndarray:
    import numpy as np

    # A CPU float .numpy() is usually a view. Explicit ownership protects the
    # archive from subsequent in-place operations in the original model.
    return np.array(tensor.detach().float().cpu().numpy(), dtype=np.float32, order="C", copy=True)


def stage_shapes(config, samples: int, stages=None) -> dict[str, tuple[int, ...]]:
    """Describe capture layouts without allocating activations or running math."""
    if samples <= config.stft_n_fft // 2:
        raise ValueError("forward input must exceed half the input FFT for original reflect padding")
    frames = samples // config.stft_hop_length + 1
    compressed = (frames + config.time_conv_length - 1) // config.time_conv_length
    bands = len(config.freqs_per_bands)
    channels, hidden = config.num_input_channels, config.hidden_size
    registers, stems = config.register_token_num, config.num_stems
    shapes = {
        "input_audio": (channels, samples),
        "raw_audio": (channels, samples),
        "stft_input": (frames, (config.stft_n_fft // 2 + 1) * channels * 2),
        "stft_output": (channels, config.stft_n_fft_out // 2 + 1, frames, 2),
        "band_split": (frames, bands, config.band_proj_size),
        "time_compressed": (compressed, bands, hidden),
        **{f"layer{i}_{axis}": (compressed + registers, bands + registers, hidden)
           for i in range(config.num_hidden_layers) for axis in ("time", "freq")},
        "encoder_output": (compressed, bands, hidden),
        "freq_pooled": (compressed, hidden),
        "stem_embeddings": (stems, hidden),
        "mask_raw": (stems, frames, (config.stft_n_fft_out // 2 + 1) * channels * 2),
        **{f"output_{stem}": (channels, samples) for stem in SOURCES},
    }
    requested = set(DEFAULT_STAGES if not stages else stages)
    unknown = requested - set(shapes)
    if unknown:
        raise ValueError(f"unknown Mini-BS-RoFormer stages: {sorted(unknown)}")
    return {name: shapes[name] for name in sorted(requested)}


def reference_metadata(model) -> dict[str, str]:
    import torch
    import transformers

    return {
        "source_model": MODEL_ID,
        "source_revision": MODEL_REVISION,
        "blueprint_sha256": SOURCE_HASHES["modeling_bs_roformer.py"],
        "checkpoint_sha256": SOURCE_HASHES["model.safetensors"],
        "config_sha256": SOURCE_HASHES["config.json"],
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "reference_dtype": str(next(model.parameters()).dtype),
        "reference_mode": "original_forward",
    }


def dump_waveform(model, audio: np.ndarray, stages=None, *, capture=None) -> dict:
    """Run one original forward on native-rate channel-first stereo audio.

    No silent padding, trimming, resampling, source mixing, or energy filtering
    is performed here. This intentionally exercises the exact caller's samples.
    Use the model's original ``separate`` for whole-recording overlap/add tests.

    If ``capture(name, value)`` is provided, it must synchronously consume the
    tensor/array: the original forward may subsequently reuse its storage.
    This path returns provenance only and retains no activation copies.
    """
    import numpy as np
    import torch

    requested = set(DEFAULT_STAGES if not stages else stages)
    known = set(DEFAULT_STAGES) | {"raw_audio"}
    unknown = requested - known
    if unknown:
        raise ValueError(f"unknown Mini-BS-RoFormer stages: {sorted(unknown)}")
    wave = np.asarray(audio, dtype=np.float32)
    if wave.ndim != 2 or wave.shape[0] != model.config.num_input_channels:
        raise ValueError("audio must have shape [2, samples] at the model's 44100-Hz sample rate")
    if wave.shape[1] <= model.config.stft_n_fft // 2:
        raise ValueError("forward input must exceed half the input FFT for original reflect padding")
    if not np.isfinite(wave).all():
        raise ValueError("audio contains nonfinite samples")
    parameter = next(model.parameters())
    x = torch.from_numpy(np.ascontiguousarray(wave)).to(device=parameter.device).unsqueeze(0)
    out = {}
    seen = set()

    def emit(name, value):
        if name in seen:
            raise RuntimeError(f"original forward produced duplicate stage {name}")
        if capture is None:
            out[name] = value.copy() if isinstance(value, np.ndarray) else _array(value)
        else:
            capture(name, value)
        seen.add(name)

    if "input_audio" in requested:
        emit("input_audio", wave)
    if "raw_audio" in requested:
        emit("raw_audio", wave)
    handles = []
    originals = []

    def module_capture(name, module, transform=lambda value: value[0]):
        if name not in requested:
            return

        def hook(_module, _inputs, output):
            emit(name, transform(output))

        handles.append(module.register_forward_hook(hook))

    def method_capture(name, method_name, transform):
        if name not in requested:
            return
        original = getattr(model, method_name)

        @functools.wraps(original)
        def capture(*args, **kwargs):
            value = original(*args, **kwargs)
            emit(name, transform(value))
            return value

        originals.append((method_name, original))
        setattr(model, method_name, capture)

    freq = model.freq_domain_model
    backbone = freq.mask_model
    method_capture("stft_input", "_audio_to_stft", lambda value: value[0])
    method_capture("stft_output", "_audio_to_complex_stft",
                   lambda value: torch.view_as_real(value[0, 0]))
    module_capture("band_split", backbone.band_split)
    module_capture("time_compressed", backbone.time_conv)
    for index, (time_layer, freq_layer) in enumerate(backbone.encoder.layers):
        module_capture(f"layer{index}_time", time_layer)
        module_capture(f"layer{index}_freq", freq_layer)
    module_capture("encoder_output", backbone.encoder)
    module_capture("freq_pooled", freq.freq_pooling, lambda value: value[:, 0])
    module_capture("stem_embeddings", freq.time_pooling)
    module_capture("mask_raw", freq)
    try:
        with torch.inference_mode():
            output = model(x)
        for index, stem in enumerate(SOURCES):
            name = f"output_{stem}"
            if name in requested:
                emit(name, output[0, index])
    finally:
        for handle in handles:
            handle.remove()
        for name, method in originals:
            setattr(model, name, method)
    missing = requested - seen
    if missing:
        raise RuntimeError(f"original forward did not produce stages {sorted(missing)}")
    out.update(reference_metadata(model))
    return out


def dump(model_dir, audio, stages, max_new_tokens=None, **kwargs):
    """Adapt the shared harness's 16-kHz mono input and retain exact PCM."""
    import numpy as np
    from scipy.signal import resample_poly

    device = os.environ.get("MINI_BS_ROFORMER_DEVICE", "cpu")
    model = _load(model_dir, device=device)
    source_rate = int(kwargs.get("sample_rate", 16000))
    target_rate = model.config.wave_sample_rate
    wave = np.asarray(audio, dtype=np.float32)
    if wave.ndim == 1:
        wave = np.stack([wave, wave])
    if source_rate != target_rate:
        gcd = math.gcd(source_rate, target_rate)
        wave = resample_poly(wave, target_rate // gcd, source_rate // gcd, axis=-1).astype(np.float32)
    return dump_waveform(model, wave, stages=stages)
