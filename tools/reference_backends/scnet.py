"""Pinned, unmodified SCNet model and inference-method reference.

The original ``SCNet.forward`` and ``Seperator.separate_music_file`` execute
unchanged. Hooks capture module boundaries; a line observer reads the original
normalization locals before they are consumed. It never computes a substitute
STFT, statistic, FFT, recurrent layer, chunk, shift, or overlap-add result.

Capture tensors use NumPy order: [1, channels, frequency, time] for network
stages, [channels, samples] for audio, and [mean, sample_std] for statistics.
Reverse these dimensions for ggml. The first actual forward inside the
original overlapping-chunk inference is captured, followed by the complete
original four-stem output. All checkpoint tensor names remain unchanged.

Upstream: https://github.com/starrytong/SCNet (MIT, copyright 2024 starrytong).
The complete upstream notice is in models/licenses/SCNet-MIT.txt.
"""

from __future__ import annotations

import ast
import hashlib
import importlib
import importlib.util
import inspect
import json
import random
import re
import sys
import textwrap
import threading
from pathlib import Path

SOURCE_REVISION = "5d95bf96b19c3eede63248d171efeca8e3abb948"
SOURCE_URL = "https://github.com/starrytong/SCNet"
LICENSE_URL = SOURCE_URL + "/issues/35"
SOURCE_HASHES = {
    "scnet/SCNet.py": "85a15ea5d28285a0cf0a24d6266a28d043c5a655d47aa41684ef256d84e7bc4a",
    "scnet/separation.py": "43402dc6579436d3b5abb921990572684beed8fa10b377a112892b438f40713b",
    "scnet/apply.py": "68b60535ad50b9b6cb0da4e4a0e7a269ff1349d345de30e666bdbda3d8fc8a31",
    "scnet/inference.py": "06c5c8cf4aeb4b7a0edf7ac698d2f454ead0bd3b0411d0b519815a50068597b8",
    "scnet/utils.py": "03a6311760dad60d7ccbd1a5416a3c10f613c8b9eb9f08691019589a2f692fdf",
    "scnet/__init__.py": "21b5f8fae849055f3b428d2353de72b477a722cc44e87c6fa0d3b22ac5f36fa2",
    "conf/config.yaml": "bc16895f69551cd8becbba384ae33e6e615a4792649b866036884f4a9a4bb4b0",
    "LICENSE": "0bdf1b69335198118ab16cfc50d337b496b8c6d90e83beeaba4643781ab62513",
}
CHECKPOINT_URLS = {
    "scnet": "https://drive.google.com/file/d/1CdEIIqsoRfHn1SJ7rccPfyYioW3BlXcW/view",
    "scnet-large": "https://drive.google.com/file/d/1s7QvQwn8ag9oVstGDBQ6KZvacJkvyK7t/view",
}
SOURCES = ("drums", "bass", "other", "vocals")
DEFAULT_STAGES = (
    "input_audio", "waveform_stats", "raw_input_audio", "stft", "stft_stats", "normalized",
    *(name for i in range(3) for name in (f"skip{i}", f"encoder{i}", f"decoder{i}")),
    *(name for i in range(6) for name in (f"dp{i}", f"feature{i}")),
    *(name for stem in SOURCES for name in (f"raw_output_{stem}", f"output_{stem}")),
)
_INFERENCE_LOCK = threading.RLock()


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def model_config(variant: str = "scnet") -> dict:
    if variant not in CHECKPOINT_URLS:
        raise ValueError(f"unsupported SCNet variant {variant!r}")
    return {
        "sources": list(SOURCES), "audio_channels": 2,
        "dims": [4, 32, 64, 128] if variant == "scnet" else [4, 64, 128, 256],
        "nfft": 4096, "hop_size": 1024, "win_size": 4096, "normalized": True,
        # The separately published SCNet-large author configuration changes
        # frequency-band proportions as well as channel widths.
        "band_SR": [0.175, 0.392, 0.433] if variant == "scnet" else [0.225, 0.372, 0.403],
        "band_stride": [1, 4, 16],
        "band_kernel": [3, 4, 16], "conv_depths": [3, 2, 1],
        "compress": 4, "conv_kernel": 3, "num_dplayer": 6, "expand": 1,
    }


def expected_shapes(variant: str = "scnet") -> dict[str, tuple[int, ...]]:
    """Complete original PyTorch checkpoint schema, before layout conversion."""
    cfg = model_config(variant)
    shapes = {}

    def affine(name, shape):
        shapes[name + ".weight"] = tuple(shape)
        shapes[name + ".bias"] = (shape[0],)

    for level, (cin, cout) in enumerate(zip(cfg["dims"], cfg["dims"][1:])):
        p = f"encoder.{level}"
        hidden = cout // cfg["compress"]
        for band, (kernel, depth) in enumerate(zip(cfg["band_kernel"], cfg["conv_depths"])):
            affine(f"{p}.SDlayer.convs.{band}", (cout, cin, kernel, 1))
            for layer in range(depth):
                q = f"{p}.conv_modules.{band}.layers.{layer}"
                affine(q + ".0", (cout,))
                affine(q + ".1", (2 * hidden, cout, 3))
                affine(q + ".3", (hidden, 1, 3))
                affine(q + ".4", (hidden,))
                affine(q + ".6", (cout, hidden, 1))
        affine(p + ".globalconv", (cout, cout, 3, 3))
        p = f"decoder.{2 - level}"
        affine(p + ".0.conv", (2 * cout, 2 * cout, 3, 3))
        output = cin if level else cin * len(SOURCES)
        for band, kernel in enumerate(cfg["band_kernel"]):
            name = f"{p}.1.convtrs.{band}"
            shapes[name + ".weight"] = (cout, output, kernel, 1)
            shapes[name + ".bias"] = (output,)
    for layer in range(cfg["num_dplayer"]):
        channels = cfg["dims"][-1] * (2 if layer % 2 else 1)
        p = f"separation_net.dp_modules.{layer}"
        for axis in range(2):
            affine(f"{p}.norm_layers.{axis}", (channels,))
            affine(f"{p}.linear_layers.{axis}", (channels, 2 * channels))
            for reverse in ("", "_reverse"):
                for projection in ("ih", "hh"):
                    q = f"{p}.lstm_layers.{axis}"
                    shapes[f"{q}.weight_{projection}_l0{reverse}"] = (4 * channels, channels)
                    shapes[f"{q}.bias_{projection}_l0{reverse}"] = (4 * channels,)
    return shapes


def validate_state(state, variant="scnet") -> None:
    expected = expected_shapes(variant)
    actual = set(state)
    if actual != set(expected):
        raise ValueError(f"checkpoint keys differ: missing={sorted(set(expected) - actual)}, "
                         f"unexpected={sorted(actual - set(expected))}")
    for name, shape in expected.items():
        value = state[name]
        if not hasattr(value, "shape") or tuple(value.shape) != shape:
            raise ValueError(f"{name}: expected original shape {shape}, got {getattr(value, 'shape', None)}")


def verify_source_files(source_dir: str | Path) -> Path:
    root = Path(source_dir).expanduser().resolve()
    for relative, expected in SOURCE_HASHES.items():
        actual = sha256(root / relative)
        if actual != expected:
            raise ValueError(f"{relative}: source differs from pinned revision {SOURCE_REVISION}; "
                             f"expected {expected}, got {actual}")
    return root


def verify_checkpoint(checkpoint: str | Path, expected_sha256: str | None = None) -> str:
    if expected_sha256 is not None and not re.fullmatch(r"[0-9a-fA-F]{64}", expected_sha256):
        raise ValueError("expected checkpoint SHA-256 must contain exactly 64 hexadecimal digits")
    actual = sha256(checkpoint)
    if expected_sha256 is not None and actual != expected_sha256.lower():
        raise ValueError(f"checkpoint SHA-256 mismatch: expected {expected_sha256.lower()}, got {actual}")
    return actual


def load_checkpoint(checkpoint: str | Path, variant="scnet", *, expected_sha256=None):
    """Read only the author's best_state, exactly as original load_model selects.

    PyTorch's restricted tensor loader is mandatory. A failed restricted load
    is never retried with arbitrary pickle execution. A full training checkpoint
    may contain optimizer/EMA state; none of those substitutes for best_state.
    """
    import torch

    digest = verify_checkpoint(checkpoint, expected_sha256)
    package = torch.load(str(checkpoint), map_location="cpu", weights_only=True)
    if not isinstance(package, dict) or not isinstance(package.get("best_state"), dict):
        raise ValueError("official SCNet checkpoint must contain the original best_state mapping")
    state = {}
    for key, value in package["best_state"].items():
        if not isinstance(key, str):
            raise ValueError("checkpoint tensor names must be strings")
        name = key[7:] if key.startswith("module.") else key
        if name in state:
            raise ValueError(f"duplicate checkpoint key after module-prefix removal: {name}")
        if not isinstance(value, torch.Tensor) or not value.is_floating_point():
            raise ValueError(f"checkpoint value must be a floating tensor: {key}")
        if not torch.isfinite(value).all().item():
            raise ValueError(f"nonfinite checkpoint tensor: {key}")
        state[name] = value
    validate_state(state, variant)
    return state, digest


def _source_package(source_dir: str | Path):
    root = verify_source_files(source_dir)
    namespace = "_crispasr_original_scnet_" + SOURCE_REVISION
    if namespace not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            namespace, root / "scnet/__init__.py", submodule_search_locations=[str(root / "scnet")])
        if spec is None or spec.loader is None:
            raise ImportError("cannot import pinned SCNet package")
        package = importlib.util.module_from_spec(spec)
        sys.modules[namespace] = package
        try:
            spec.loader.exec_module(package)
        except BaseException:
            sys.modules.pop(namespace, None)
            raise
    return namespace


def load_original(source_dir, checkpoint, variant="scnet", device="cpu", *, expected_sha256=None):
    """Load the pinned original model, with complete shape and file validation."""
    import torch

    namespace = _source_package(source_dir)
    model_cls = importlib.import_module(namespace + ".SCNet").SCNet
    state, digest = load_checkpoint(checkpoint, variant, expected_sha256=expected_sha256)
    model = model_cls(**model_config(variant))
    # Strict loading also cross-checks our complete schema against the original
    # class, rather than relying solely on a port-authored list of tensor names.
    model.load_state_dict(state, strict=True)
    del state
    model = model.eval().to(device=torch.device(device), dtype=torch.float32)
    model._scnet_source_namespace = namespace
    model._scnet_provenance = {
        "source_revision": SOURCE_REVISION,
        "source_url": SOURCE_URL,
        "source_hashes": dict(SOURCE_HASHES),
        "checkpoint_sha256": digest,
        "checkpoint_url": CHECKPOINT_URLS[variant],
        "variant": variant,
        "torch_version": torch.__version__,
    }
    return model


def validate_profile(segment_seconds=20, shifts=1, shift_seed=0):
    if isinstance(segment_seconds, bool) or not isinstance(segment_seconds, int) or not 1 <= segment_seconds <= 119:
        raise ValueError("segment_seconds must be an integer in [1,119]")
    if isinstance(shifts, bool) or not isinstance(shifts, int) or not 0 <= shifts <= 16:
        raise ValueError("shifts must be an integer in [0,16]")
    if isinstance(shift_seed, bool) or not isinstance(shift_seed, int) or not 0 <= shift_seed < 2**64:
        raise ValueError("shift_seed must be an unsigned 64-bit integer")


def _audio(audio):
    import numpy as np

    value = np.asarray(audio, dtype=np.float32)
    if value.ndim != 2 or value.shape[0] != 2 or not 0 < value.shape[1] <= 120 * 44100:
        raise ValueError("reference audio must have planar shape [2,N], with 0 < N <= 120*44100")
    if not np.isfinite(value).all():
        raise ValueError("reference input contains nonfinite values")
    return np.ascontiguousarray(value)


def _inference_modules(model):
    namespace = model._scnet_source_namespace
    return (importlib.import_module(namespace + ".inference"),
            importlib.import_module(namespace + ".apply"))


def separate_original(model, audio_planar_np, segment_seconds=20, shifts=1, shift_seed=0, no_segment=False):
    """Execute the original full inference helper, or diagnostic raw forward.

    The only configuration adaptation is passing explicit segment/shifts to
    the genuine apply_model and supplying its Python Random instance. The
    original normalization and complete chunk/shift/overlap implementation run
    unchanged. Temporary configuration is restored even when inference fails.
    """
    import numpy as np
    import torch

    validate_profile(segment_seconds, shifts, shift_seed)
    audio = _audio(audio_planar_np)
    device = next(model.parameters()).device
    if no_segment:
        with torch.no_grad():
            result = model(torch.from_numpy(audio)[None].to(device))[0]
        return np.array(result.detach().float().cpu().numpy(), dtype=np.float32, order="C", copy=True)
    inference, apply = _inference_modules(model)
    if apply.accelerator.device != device:
        raise RuntimeError(f"original apply_model selected {apply.accelerator.device}, but model is on {device}; "
                           "set ACCELERATE_USE_CPU=true before importing for an explicit CPU run")
    separator = inference.Seperator.__new__(inference.Seperator)
    separator.separator = model
    separator.device = device
    with _INFERENCE_LOCK:
        previous_apply, previous_random = inference.apply_model, apply.random

        def configured_apply(original_model, mix, **kwargs):
            return apply.apply_model(original_model, mix, segment=segment_seconds, shifts=shifts, **kwargs)

        try:
            inference.apply_model = configured_apply
            apply.random = random.Random(shift_seed)
            stems, rates = separator.separate_music_file(audio.T, 44100)
        finally:
            inference.apply_model, apply.random = previous_apply, previous_random
    if set(stems) != set(SOURCES) or any(rates.get(stem) != 44100 for stem in SOURCES):
        raise RuntimeError("original helper returned unexpected stems or sample rates")
    result = np.stack([stems[stem].T for stem in SOURCES]).astype(np.float32, copy=False)
    if result.shape != (4, 2, audio.shape[1]) or not np.isfinite(result).all():
        raise ValueError("original helper produced an invalid or nonfinite four-stem waveform")
    return np.ascontiguousarray(result)


def _normalization_line(function, variable):
    """Find the original assignment whose locals we observe, without editing it."""
    source, first = inspect.getsourcelines(function)
    parsed = ast.parse(textwrap.dedent("".join(source)))
    matches = []
    for node in ast.walk(parsed):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        if not isinstance(node.targets[0], ast.Name) or node.targets[0].id != variable:
            continue
        if not isinstance(node.value, ast.BinOp) or not isinstance(node.value.op, ast.Div):
            continue
        names = {value.id for value in ast.walk(node.value) if isinstance(value, ast.Name)}
        if {variable, "mean", "std"}.issubset(names):
            matches.append(first + node.lineno - 1)
    if len(matches) != 1:
        raise ValueError(f"cannot uniquely identify original {variable} normalization assignment")
    return matches[0]


def capture_original(model, audio_planar_np, segment_seconds=20, shifts=1, shift_seed=0,
                     no_segment=False, *, capture=None):
    """Capture all 35 stages from one genuine inference; an optional sink is synchronous."""
    import numpy as np

    validate_profile(segment_seconds, shifts, shift_seed)
    audio = _audio(audio_planar_np)
    values, seen, handles = {}, set(), []
    active, calls = False, 0

    def emit(name, value):
        if name in seen:
            raise RuntimeError(f"duplicate original stage: {name}")
        if capture is None:
            if not isinstance(value, np.ndarray):
                value = value.detach().float().cpu().numpy()
            value = np.array(value, dtype=np.float32, order="C", copy=True)
            if not np.isfinite(value).all():
                raise ValueError(f"nonfinite original stage: {name}")
            values[name] = value
        else:
            capture(name, value)
        seen.add(name)

    def pre_forward(_module, inputs):
        nonlocal active, calls
        calls += 1
        active = calls == 1
        if active:
            emit("raw_input_audio", inputs[0][0])

    def post_forward(_module, _inputs, output):
        nonlocal active
        if active:
            for index, stem in enumerate(SOURCES):
                emit(f"raw_output_{stem}", output[0, index])
        active = False

    def stage(name):
        def hook(_module, _inputs, output):
            if active:
                emit(name, output)
        return hook

    def encoder_hook(index):
        def hook(_module, _inputs, output):
            if active:
                emit(f"encoder{index}", output[0])
                emit(f"skip{index}", output[1])
        return hook

    def normalized(_module, inputs):
        if active:
            emit("normalized", inputs[0])

    inference, _apply = _inference_modules(model)
    forward_code = type(model).forward.__code__
    pipeline_code = inference.Seperator.separate_music_file.__code__
    forward_line = _normalization_line(type(model).forward, "x")
    pipeline_line = _normalization_line(inference.Seperator.separate_music_file, "mix")

    def observe(frame, event, _arg):
        if frame.f_code not in (forward_code, pipeline_code):
            return None
        if event == "line":
            local = frame.f_locals
            if frame.f_code is forward_code and active and frame.f_lineno == forward_line:
                emit("stft", local["x"])
                emit("stft_stats", np.asarray([local["mean"].item(), local["std"].item()], dtype=np.float32))
            elif frame.f_code is pipeline_code and frame.f_lineno == pipeline_line:
                emit("waveform_stats", np.asarray([local["mean"].item(), local["std"].item()], dtype=np.float32))
        return observe

    with _INFERENCE_LOCK:
        old_trace = sys.gettrace()
        if old_trace is not None:
            raise RuntimeError("original stage capture cannot replace an active Python trace/debugger")
        try:
            emit("input_audio", audio)
            if no_segment:
                emit("waveform_stats", np.array([0.0, 1.0], dtype=np.float32))
            handles.append(model.register_forward_pre_hook(pre_forward))
            handles.append(model.register_forward_hook(post_forward))
            handles.append(model.encoder[0].register_forward_pre_hook(normalized))
            for index, encoder in enumerate(model.encoder):
                handles.append(encoder.register_forward_hook(encoder_hook(index)))
            for index, decoder in enumerate(model.decoder):
                handles.append(decoder[1].register_forward_hook(stage(f"decoder{index}")))
            for index, module in enumerate(model.separation_net.dp_modules):
                handles.append(module.register_forward_hook(stage(f"dp{index}")))
            for index, module in enumerate(model.separation_net.feature_conversion):
                handles.append(module.register_forward_hook(stage(f"feature{index}")))
            sys.settrace(observe)
            output = separate_original(model, audio, segment_seconds, shifts, shift_seed, no_segment)
            sys.settrace(old_trace)
            for index, stem in enumerate(SOURCES):
                emit(f"output_{stem}", output[index])
        finally:
            sys.settrace(old_trace)
            for handle in reversed(handles):
                handle.remove()
    missing = set(DEFAULT_STAGES) - seen
    if missing or seen - set(DEFAULT_STAGES):
        raise RuntimeError(f"incomplete original capture: missing={sorted(missing)}, unexpected={sorted(seen - set(DEFAULT_STAGES))}")
    return values


def reference_metadata(model, audio_planar_np, segment_seconds=20, shifts=1, shift_seed=0, no_segment=False):
    validate_profile(segment_seconds, shifts, shift_seed)
    audio = _audio(audio_planar_np)
    provenance = model._scnet_provenance
    return {
        "scnet.reference.source_revision": SOURCE_REVISION,
        "scnet.reference.source_url": SOURCE_URL,
        "scnet.reference.source_hashes_json": json.dumps(SOURCE_HASHES, sort_keys=True),
        "scnet.reference.checkpoint_sha256": provenance["checkpoint_sha256"],
        "scnet.reference.checkpoint_url": provenance["checkpoint_url"],
        "scnet.reference.variant": provenance["variant"],
        "scnet.reference.mode": "raw" if no_segment else "pipeline",
        "scnet.reference.segment_seconds": segment_seconds,
        "scnet.reference.shifts": shifts,
        "scnet.reference.shift_seed": shift_seed,
        "scnet.reference.overlap": 0.5,
        "scnet.reference.sample_rate": 44100,
        "scnet.reference.input_pcm_sha256": hashlib.sha256(audio.astype("<f4", copy=False).tobytes()).hexdigest(),
        "scnet.reference.torch_version": provenance["torch_version"],
        "scnet.reference.oracle": "original SCNet.forward and Seperator.separate_music_file; unmodified math",
        "scnet.reference.required_stage_count": len(DEFAULT_STAGES),
    }
