"""Observe the pinned old-checkpoint Dia generate() path, including real DAC.

DIA_SOURCE is a clean checkout at SOURCE_PIN. DIA_TEXT must already have a
speaker tag and final punctuation to align with native text preparation.
DIA_MAX_TOKENS defaults to 128; full speech may use 3072. CPU is explicit.
Captures retain original [unconditional, conditional] order. Only captured
encoder/cache views trim padding; original inference, masks and math remain
unchanged. Native diff replays decoder inputs and DAC codes separately, so it
does not establish free-running sampler equality or replace speech acceptance.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

SOURCE_PIN = "4a9e29b1bdbfe1be721353ec034a8c66c9f0a1a8"
MODEL_PIN = "257bc72f9b78182ccc6fa07675a9ae4c1a44e2cd"
BOUNDARY_STAGES = [
    "encoder_tokens", "encoder_embedding", "encoder_output", "cross_k0",
    "cross_v0", "decoder_ca0", "decoder_hidden", "decoder_input_tokens",
    "decoder_step_logits", "decoder_codes_rev", "decoded_audio",
]
CODEC_STAGES = ["dac_latent", "dac_input_conv"] + [f"dac_block_{i}" for i in range(4)] + ["dac_output_snake", "dac_output_conv"]
DEFAULT_STAGES = BOUNDARY_STAGES + [f"encoder_layer_{i}" for i in range(12)] + [f"decoder_layer_{i}" for i in range(18)] + CODEC_STAGES


def _codec_array(value, name):
    if value.ndim != 3 or value.shape[0] != 1:
        raise ValueError(f"Unexpected actual DAC shape for {name}: {tuple(value.shape)}")
    return value[0].detach().cpu().float().numpy().copy()


def capture_codec(codec, codes):
    """Observe actual DAC modules on an exact code grid; no reconstructed math.

    Arrays retain original channel/time order (C,T); each comparison row is
    one complete channel waveform. Hooks own their arrays and always restore.
    This bounded replay is separate from Dia's free-running generation.
    """
    import torch
    results, handles = {}, []

    def save(name, value):
        results[name] = _codec_array(value, name)

    def hook(name):
        def observe(_module, _args, output):
            save(name, output)
        return observe

    modules = codec.decoder.model
    if len(modules) != 8:
        raise ValueError("DAC trace requires the pinned four-block decoder")
    try:
        for name, module in zip(CODEC_STAGES[1:], list(modules)[:7]):
            handles.append(module.register_forward_hook(hook(name)))
        with torch.inference_mode():
            latent = codec.quantizer.from_codes(codes)[0]
            save("dac_latent", latent)
            pcm = codec.decode(latent)
        results["decoded_audio"] = pcm.detach().cpu().float().numpy().reshape(-1).copy()
        if any(value.size == 0 or not np.isfinite(value).all() for value in results.values()):
            raise ValueError("DAC trace contains empty or nonfinite arrays")
        if set(results) != set(CODEC_STAGES) | {"decoded_audio"}:
            raise ValueError("Actual DAC did not execute all requested trace modules")
        return results
    finally:
        for handle in handles:
            handle.remove()


def required_packages():
    # Original package is nari-tts, not dia. Import the verified checkout;
    # avoid installing its UI/Triton dependencies or replacing shared torch.
    return ["torch", "torchaudio", "pydantic>=2.11.3", "safetensors",
            "huggingface_hub", "descript-audio-codec>=1.0.0"]


def verify_source(source):
    source = Path(source).resolve()
    head = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if head != SOURCE_PIN:
        raise ValueError(f"Dia source must be {SOURCE_PIN}, got {head}")
    hashes = {}
    for file in sorted((source / "dia").glob("*.py")):
        relative = file.relative_to(source).as_posix()
        expected = subprocess.check_output(["git", "-C", str(source), "show", f"{SOURCE_PIN}:{relative}"])
        actual = file.read_bytes()
        if actual != expected:
            raise ValueError(f"Modified Dia blueprint: {relative}")
        hashes[relative] = hashlib.sha256(actual).hexdigest()
    for name, module in list(sys.modules.items()):
        if name == "dia" or name.startswith("dia."):
            file = getattr(module, "__file__", None)
            if file is None or not Path(file).resolve().is_relative_to(source / "dia"):
                raise ValueError(f"Already imported Dia from another source: {name}")
    return source, hashes


def validate_text(text):
    # Native trims leading ASCII whitespace and adds punctuation using the
    # actual last byte; original generate() tokenizes the text unchanged.
    # Reject disagreement instead of silently normalizing the blueprint input.
    if (not text.startswith(("[S1]", "[S2]")) or text[-1:] not in (".", "!", "?")
            or "\x00" in text):
        raise ValueError("DIA_TEXT must start with [S1]/[S2], end in .!?, and contain no NUL for native alignment")


def capture(dia, text, *, max_tokens=128, seed=42, temperature=1.2, top_p=.95,
            top_k=45, cfg_scale=3., stages=None, capture_step=0, attention_layers=(), boundary_layers=()):
    """Capture actual original calls, owning every array and restoring hooks."""
    import torch
    from dia_attention_capture import attention_stages, install_attention_hooks, validate_attention_layers
    attention_layers = validate_attention_layers(attention_layers, len(dia.model.decoder.layers))
    from dia_decoder_capture import boundary_stages, install_boundary_hooks
    boundary_layers = validate_attention_layers(boundary_layers, len(dia.model.decoder.layers))
    available = BOUNDARY_STAGES + [f"encoder_layer_{i}" for i in range(len(dia.model.encoder.layers))]
    available += [f"decoder_layer_{i}" for i in range(len(dia.model.decoder.layers))]
    codec_modules = getattr(getattr(dia.dac_model, "decoder", None), "model", None)
    if codec_modules is not None and len(codec_modules) == 8:
        available += CODEC_STAGES
    available += attention_stages(attention_layers)
    available += boundary_stages(boundary_layers)
    stages = set(available if stages is None else stages)
    stages.update(attention_stages(attention_layers))
    stages.update(boundary_stages(boundary_layers))
    if stages - set(available):
        raise ValueError(f"Unsupported Dia stages: {sorted(stages - set(available))}")
    validate_text(text)
    if not max(dia.config.data.delay_pattern) < max_tokens <= dia.config.data.audio_length:
        raise ValueError("DIA_MAX_TOKENS must exceed the delay and fit the model capacity")
    if type(capture_step) is not int or not 0 <= capture_step < max_tokens:
        raise ValueError("Dia capture step must be an integer inside the generation capacity")
    results, inputs, logits, handles, replacements = {}, [], [], [], []
    length = None

    def save(name, value):
        results[name] = value.detach().cpu().float().numpy().copy()

    def replace(obj, name, wrapper):
        original = getattr(obj, name)
        replacements.append((obj, name, original))
        setattr(obj, name, wrapper(original))

    def prepare(original):
        def wrapped(text):
            nonlocal length
            tokens = original(text)
            nonpad = tokens[0] != dia.config.data.text_pad_value
            length = int(nonpad.sum().item())
            if length == 0 or not bool(nonpad[:length].all()) or bool(nonpad[length:].any()):
                raise ValueError("Dia diff requires a contiguous non-padding prompt")
            save("encoder_tokens", torch.cat([torch.zeros_like(tokens), tokens])[:, :length])
            return tokens
        return wrapped

    def encoder_hook(name):
        def hook(_module, _args, output):
            save(name, output[:, :length])
        return hook

    def decoder_hook(name):
        def hook(_module, _args, output):
            if len(inputs) - 1 == capture_step:
                save(name, output[:, 0])
        return hook

    def cross(original):
        def wrapped(*args, **kwargs):
            caches = original(*args, **kwargs)
            for name, value in (("cross_k0", caches[0].k), ("cross_v0", caches[0].v)):
                value = value.transpose(1, 2)[:, :length]
                save(name, value.reshape(value.shape[0], length, -1))
            return caches
        return wrapped

    def step(original):
        def wrapped(tokens, state):
            inputs.append(tokens[0, 0].detach().cpu().numpy().copy())
            output = original(tokens, state)
            logits.append(output[:, -1].detach().cpu().float().numpy().copy())
            return output
        return wrapped

    def codes(original):
        def wrapped(value):
            save("decoder_codes_rev", value[0].transpose(0, 1))
            output = original(value)
            if "dac_latent" in available:
                results["dac_latent"] = _codec_array(output[0], "dac_latent")
            return output
        return wrapped

    def codec_hook(name):
        def observe(_module, _args, output):
            results[name] = _codec_array(output, name)
        return observe

    try:
        replace(dia, "_prepare_text_input", prepare)
        replace(dia.model.decoder, "precompute_cross_attn_cache", cross)
        replace(dia.model.decoder, "decode_step", step)
        replace(dia.dac_model.quantizer, "from_codes", codes)
        if "dac_latent" in available:
            for name, module in zip(CODEC_STAGES[1:], list(codec_modules)[:7]):
                handles.append(module.register_forward_hook(codec_hook(name)))
        handles.append(dia.model.encoder.embedding.register_forward_hook(encoder_hook("encoder_embedding")))
        handles.append(dia.model.encoder.register_forward_hook(encoder_hook("encoder_output")))
        handles.append(dia.model.decoder.layers[0].cross_attention.register_forward_hook(decoder_hook("decoder_ca0")))
        handles.append(dia.model.decoder.norm.register_forward_hook(decoder_hook("decoder_hidden")))
        for i, layer in enumerate(dia.model.encoder.layers):
            handles.append(layer.register_forward_hook(encoder_hook(f"encoder_layer_{i}")))
        for i, layer in enumerate(dia.model.decoder.layers):
            handles.append(layer.register_forward_hook(decoder_hook(f"decoder_layer_{i}")))
        handles.extend(install_attention_hooks(dia.model.decoder.layers, attention_layers, capture_step,
                                               lambda: len(inputs) - 1 == capture_step, save))
        handles.extend(install_boundary_hooks(dia.model.decoder.layers, boundary_layers, capture_step,
                                              lambda: len(inputs) - 1 == capture_step, save))
        torch.manual_seed(seed)
        with torch.inference_mode():
            pcm = dia.generate(text, max_tokens=max_tokens, cfg_scale=cfg_scale,
                               temperature=temperature, top_p=top_p,
                               cfg_filter_top_k=top_k, use_torch_compile=False)
        if pcm is None or not inputs:
            raise RuntimeError("Original Dia generated no audio/logits")
        if len(inputs) <= capture_step:
            raise RuntimeError("Original Dia did not reach the requested capture step")
        results["decoder_input_tokens"] = np.stack(inputs).astype(np.float32)
        results["decoder_step_logits"] = np.stack(logits).astype(np.float32)
        results["decoded_audio"] = np.asarray(pcm, dtype=np.float32).reshape(-1).copy()
        missing = stages - results.keys()
        if missing:
            raise RuntimeError(f"Missing required Dia stages: {sorted(missing)}")
        if any(not np.isfinite(results[name]).all() or results[name].size == 0 for name in stages):
            raise RuntimeError("Dia reference contains empty or nonfinite stages")
        selected = stages | {"encoder_tokens", "decoder_input_tokens", "decoder_codes_rev"}
        return {name: results[name] for name in selected}
    finally:
        for handle in handles:
            handle.remove()
        for obj, name, original in reversed(replacements):
            setattr(obj, name, original)


def dump(model_dir, audio, stages=None, **kwargs):
    """Generic dispatcher entry; audio is unused for text-only synthesis."""
    from dia_attention_capture import parse_attention_layers, validate_attention_layers
    attention_layers = parse_attention_layers(os.environ.get("DIA_CAPTURE_ATTENTION_LAYERS", ""))
    from dia_decoder_capture import parse_boundary_layers
    boundary_layers = parse_boundary_layers(os.environ.get("DIA_CAPTURE_BOUNDARY_LAYERS", ""))
    text = os.environ.get("DIA_TEXT", "[S1] We are committed to making it secure.")
    validate_text(text)
    capture_index = os.environ.get("DIA_CAPTURE_STEP", "0")
    if (not capture_index.isascii() or not capture_index.isdecimal()
            or int(capture_index) >= int(os.environ.get("DIA_MAX_TOKENS", "128"))):
        raise ValueError("DIA_CAPTURE_STEP must be an exact index inside DIA_MAX_TOKENS")
    if "DIA_SOURCE" not in os.environ:
        raise ValueError("Set DIA_SOURCE to the clean pinned original checkout")
    source, source_hashes = verify_source(os.environ["DIA_SOURCE"])
    sys.path.insert(0, str(source))
    import torch
    import dac
    from safetensors.torch import load_file
    from huggingface_hub import snapshot_download
    from dia.config import DiaConfig
    from dia.layers import DiaModel, RotaryEmbedding
    from dia.model import Dia

    location = Path(model_dir)
    revision = "local-files-hashed"
    if not location.is_dir():
        if str(model_dir) != "nari-labs/Dia-1.6B":
            raise ValueError("Use nari-labs/Dia-1.6B or an explicit local checkpoint directory")
        location = Path(snapshot_download(str(model_dir), revision=MODEL_PIN,
                                         allow_patterns=["config.json", "model.safetensors"]))
        revision = MODEL_PIN
    config_path, weights_path = location / "config.json", location / "model.safetensors"
    config = DiaConfig.load(str(config_path))
    validate_attention_layers(attention_layers, config.model.decoder.n_layer)
    validate_attention_layers(boundary_layers, config.model.decoder.n_layer)
    device = torch.device(os.environ.get("DIA_DEVICE", "cpu"))
    with torch.device("meta"):
        model = DiaModel(config, torch.float32)
    state = load_file(str(weights_path), device="cpu")
    model.load_state_dict(state, assign=True, strict=True)
    del state
    for module in model.modules():
        if isinstance(module, RotaryEmbedding):
            module.timescale = RotaryEmbedding(module.embedding_dims, module.min_timescale,
                                               module.max_timescale, module.compute_dtype).timescale
    model.to(device=device, dtype=torch.float32).eval()
    dia = Dia.__new__(Dia)
    dia.config, dia.model, dia.device, dia.compute_dtype = config, model, device, torch.float32
    checkpoint = Path(os.environ["DIA_DAC_CHECKPOINT"]) if "DIA_DAC_CHECKPOINT" in os.environ else Path(dac.utils.download())
    dia.dac_model = dac.DAC.load(str(checkpoint)).to(device).eval()
    settings = dict(max_tokens=int(os.environ.get("DIA_MAX_TOKENS", "128")),
                    seed=int(os.environ.get("DIA_SEED", "42")),
                    temperature=float(os.environ.get("DIA_TEMPERATURE", "1.2")),
                    top_p=float(os.environ.get("DIA_TOP_P", ".95")),
                    top_k=int(os.environ.get("DIA_TOP_K", "45")),
                    cfg_scale=float(os.environ.get("DIA_CFG_SCALE", "3")),
                    capture_step=int(os.environ.get("DIA_CAPTURE_STEP", "0")),
                    attention_layers=attention_layers, boundary_layers=boundary_layers)
    result = capture(dia, text, stages=stages, **settings)
    if set(CODEC_STAGES) <= result.keys():
        result["dia_dac_stage_trace"] = "rvq+decoder-v1"
    def digest(path):
        sha = hashlib.sha256()
        with path.open("rb") as file:
            for chunk in iter(lambda: file.read(4 * 1024 * 1024), b""):
                sha.update(chunk)
        return sha.hexdigest()
    result.update(dia_text=text, dia_source_pin=SOURCE_PIN, dia_model_revision=revision,
                  dia_source_hashes=json.dumps(source_hashes, sort_keys=True),
                  dia_weights_sha256=digest(weights_path), dia_config_sha256=digest(config_path),
                  dia_dac_sha256=digest(checkpoint), dia_torch_version=str(torch.__version__),
                  dia_dac_package=importlib.metadata.version("descript-audio-codec"),
                  dia_generation_settings=json.dumps(settings, sort_keys=True),
                  dia_decoder_capture_step=str(settings["capture_step"]),
                  dia_attention_capture_layers=",".join(map(str, attention_layers)),
                  dia_boundary_capture_layers=",".join(map(str, boundary_layers)),
                  dia_device=str(device))
    return result
