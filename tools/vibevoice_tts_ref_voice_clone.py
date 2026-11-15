#!/usr/bin/env python3
"""Dump base VibeVoice serving voice conditioning from local source weights.

The serving method supplies acoustic posterior sampling, scaling and connection.
Only encoder/connector tensors are read, never whole shards or the LM/decoder.
Requires an upstream checkout containing modeling_vibevoice_inference.py:

  CRISPASR_VIBEVOICE_SRC=/path/to/VibeVoice python \
    tools/vibevoice_tts_ref_voice_clone.py --model /path/to/checkpoint \
    --voice reference.wav --output-dir voice-ref

--mean-acoustic disables posterior noise for a controlled native mean-path
comparison. It is a diagnostic variation, not the default serving recipe.
Compare dumps with tools/diff_vibevoice_tts.py REF_DIR NATIVE_DIR --voice-only;
the input PCM and all four conditioning stages are required by that mode.
"""
from __future__ import annotations

import argparse
import ast
import gc
import json
import logging
import math
import os
import typing
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import Optional, Union

import numpy as np


def source_definition(path, name, namespace, parent=None):
    """Execute an actual upstream definition without package import side effects."""
    body = ast.parse(path.read_text(), filename=str(path)).body
    if parent:
        body = next(n for n in body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name == parent).body
    node = next(n for n in body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name == name)
    module = ast.Module(body=[node], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace[name]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--voice", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--source", type=Path, default=os.environ.get("CRISPASR_VIBEVOICE_SRC"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mean-acoustic", action="store_true")
    parser.add_argument("--semantic-diagnostics", action="store_true")
    parser.add_argument("--no-normalize", action="store_true")
    parser.add_argument("--gguf-weights", type=Path,
                        help="controlled comparison using dequantized native GGUF encoder/connector weights")
    parser.add_argument("--effective-pcm", type=Path,
                        help="controlled comparison using native effective 24kHz mono raw float32 PCM")
    args = parser.parse_args()
    if args.source is None:
        parser.error("--source or CRISPASR_VIBEVOICE_SRC must point to the upstream source checkout")

    import torch
    import librosa
    from safetensors import safe_open
    from reference_backends import vibevoice as ref
    import importlib.util

    print("loading upstream encoder definitions", flush=True)
    modular = args.source / "vibevoice" / "modular"
    tokenizer_path = modular / "modular_vibevoice_tokenizer.py"
    # Execute the original nn.Module definitions without importing the unrelated
    # Transformers model/AutoModel registry or optional GPU compiler plugins.
    namespace = {"torch": torch, "nn": torch.nn, "F": torch.nn.functional,
                 "np": np, "math": math, "tp": typing, "partial": partial,
                 "Optional": Optional, "Union": Union, "Dict": dict, "List": list,
                 "Tuple": tuple, "APEX_AVAILABLE": False,
                 "logger": logging.getLogger("vibevoice-reference"),
                 "ACT2FN": {"gelu": torch.nn.functional.gelu},
                 "CONV_NORMALIZATIONS": frozenset(("none", "weight_norm", "spectral_norm",
                                                  "time_layer_norm", "layer_norm", "time_group_norm"))}
    definitions = []
    for node in ast.parse(tokenizer_path.read_text()).body:
        if isinstance(node, ast.ClassDef) and node.name == "TokenizerDecoder":
            break
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)):
            definitions.append(node)
    exec(compile(ast.fix_missing_locations(ast.Module(body=definitions, type_ignores=[])),
                 str(tokenizer_path), "exec"), namespace)
    enc_cls = namespace["TokenizerEncoder"]
    transformers_dir = Path(importlib.util.find_spec("transformers").origin).parent
    LlamaRMSNorm = source_definition(transformers_dir / "models/llama/modeling_llama.py", "LlamaRMSNorm",
                                     {"nn": torch.nn, "torch": torch,
                                      "use_kernel_forward_from_hub": lambda name: lambda cls: cls})
    connector_cls = source_definition(modular / "modeling_vibevoice.py", "SpeechConnector",
                                      {"nn": torch.nn, "LlamaRMSNorm": LlamaRMSNorm})
    posterior_cls = source_definition(modular / "modular_vibevoice_tokenizer.py",
                                      "VibeVoiceTokenizerEncoderOutput",
                                      {"torch": torch, "Optional": Optional, "Union": Union,
                                       "dataclass": __import__("dataclasses").dataclass})
    serving_method = source_definition(modular / "modeling_vibevoice_inference.py",
                                       "_process_speech_inputs", {"torch": torch},
                                       "VibeVoiceForConditionalGenerationInference")
    normalizer_cls = source_definition(args.source / "vibevoice" / "processor" / "audio_utils.py",
                                       "AudioNormalizer", {"np": np, "Optional": Optional})
    cfg = json.loads((args.model / "config.json").read_text())
    output_dim = cfg["decoder_config"]["hidden_size"]

    gguf_tensors = None
    if args.gguf_weights:
        import gguf
        from gguf.quants import dequantize
        reader = gguf.GGUFReader(str(args.gguf_weights))
        gguf_tensors = {tensor.name: tensor for tensor in reader.tensors}
        converter = Path(__file__).resolve().parents[1] / "models/convert-vibevoice-to-gguf.py"
        shorten = source_definition(converter, "shorten", {}, parent="main")
        source_keys = json.loads((args.model / "model.safetensors.index.json").read_text())["weight_map"]

    def gguf_tensor(key):
        tensor = gguf_tensors[shorten(key)]
        array = dequantize(tensor.data, tensor.tensor_type).reshape(tuple(reversed(tensor.shape)))
        return torch.from_numpy(np.array(array, dtype=np.float32, copy=True))

    def load_state(prefix):
        if gguf_tensors is None:
            return ref._lazy_state(args.model, prefix)
        return {key[len(prefix):]: gguf_tensor(key) for key in source_keys if key.startswith(prefix)}

    def scalar(key):
        if gguf_tensors is not None:
            return gguf_tensor(key)
        index_path = args.model / "model.safetensors.index.json"
        shard = (json.loads(index_path.read_text())["weight_map"][key] if index_path.exists()
                 else "model.safetensors")
        with safe_open(str(args.model / shard), framework="pt") as f:
            return f.get_tensor(key).float()

    audio, _ = librosa.load(str(args.voice), sr=24000, mono=True)
    if not args.no_normalize:
        audio = normalizer_cls()(audio)
    original_normalized_audio = audio.copy()
    if args.effective_pcm:
        audio = np.fromfile(args.effective_pcm, dtype="<f4")
    at_cfg = cfg["acoustic_tokenizer_config"]

    def run_encoder(raw, prefix):
        # These fields mirror the upstream acoustic/semantic tokenizer constructors.
        config = SimpleNamespace(**raw)
        config.dimension = raw["vae_dim"]
        config.n_filters = raw["encoder_n_filters"]
        config.ratios = raw["encoder_ratios"]
        config.depths = ([int(d) for d in raw["encoder_depths"].split("-")]
                         if isinstance(raw["encoder_depths"], str) else raw["encoder_depths"])
        config.norm = raw["conv_norm"]
        config.bias = raw["conv_bias"]
        encoder = enc_cls(config).eval()
        encoder.load_state_dict(load_state(prefix), strict=True)
        with torch.no_grad():
            result = encoder(torch.from_numpy(audio)[None, None]).permute(0, 2, 1)[0].numpy().copy()
        del encoder
        gc.collect()
        return result

    print("running upstream acoustic encoder", flush=True)
    mean = run_encoder(at_cfg, "model.acoustic_tokenizer.encoder.")
    acoustic_connector = connector_cls(at_cfg["vae_dim"], output_dim).eval()
    acoustic_connector.load_state_dict(load_state("model.acoustic_connector."), strict=True)

    class AcousticTokenizer:
        std_dist_type = "none" if args.mean_acoustic else at_cfg.get("std_dist_type", "fix")

        def encode(self, audio):
            # The actual upstream encoder has already run, then been freed.
            return posterior_cls(mean=torch.from_numpy(mean)[None], std=torch.tensor(at_cfg["fix_std"]))

    model = SimpleNamespace(acoustic_tokenizer=AcousticTokenizer(),
                            acoustic_connector=acoustic_connector,
                            speech_bias_factor=scalar("model.speech_bias_factor"),
                            speech_scaling_factor=scalar("model.speech_scaling_factor"))
    torch.manual_seed(args.seed)
    with torch.no_grad():
        scaled, connected = serving_method(SimpleNamespace(model=model),
                                            torch.from_numpy(audio)[None],
                                            torch.ones((1, math.ceil(len(audio) / 3200)), dtype=torch.bool))
    stages = {"tts_voice_original_normalized_pcm": original_normalized_audio,
              "tts_voice_encoder_input": audio,
              "tts_voice_acoustic_mean": mean,
              "tts_voice_acoustic_scaled": scaled[0].numpy(),
              "tts_voice_acoustic_connected": connected.numpy(),
              "tts_voice_embeds": connected.numpy()}
    if args.semantic_diagnostics:
        st_cfg = cfg["semantic_tokenizer_config"]
        semantic_mean = run_encoder(st_cfg, "model.semantic_tokenizer.encoder.")
        semantic_connector = connector_cls(st_cfg["vae_dim"], output_dim).eval()
        semantic_connector.load_state_dict(load_state("model.semantic_connector."), strict=True)
        with torch.no_grad():
            semantic = semantic_connector(torch.from_numpy(semantic_mean)[None])[0].numpy()
        stages["diagnostic_semantic_mean"] = semantic_mean
        stages["diagnostic_semantic_connected"] = semantic
        stages["diagnostic_legacy_dual_sum"] = connected.numpy() + semantic
    args.output_dir.mkdir(parents=True, exist_ok=True)
    shapes = {}
    for name, value in stages.items():
        array = np.ascontiguousarray(value, dtype="<f4")
        array.tofile(args.output_dir / f"{name}.bin")
        shapes[name] = list(array.shape)
        print(f"{name}: {array.shape}", flush=True)
    (args.output_dir / "shapes.json").write_text(json.dumps({
        "conditioning": "acoustic-only", "posterior": AcousticTokenizer.std_dist_type,
        "seed": args.seed, "sample_rate": 24000,
        "weights": "native-gguf" if args.gguf_weights else "original-checkpoint",
        "pcm": "native-effective" if args.effective_pcm else "original-preprocessing", "stages": shapes}, indent=2) + "\n")


if __name__ == "__main__":
    main()
