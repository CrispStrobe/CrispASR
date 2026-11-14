#!/usr/bin/env python3
"""Exercise conversion task metadata without weights, downloads or Torch."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch


class Writer:
    latest = None

    def __init__(self, output, arch):
        self.output = output
        self.values = {"general.architecture": arch}
        Writer.latest = self

    def add_uint32(self, key, value):
        self.values[key] = value

    add_float32 = add_uint32
    add_string = add_uint32
    add_array = add_uint32

    def add_name(self, value):
        self.values["general.name"] = value

    def write_header_to_file(self):
        Path(self.output).touch()

    def write_kv_data_to_file(self):
        pass

    write_tensors_to_file = write_kv_data_to_file
    close = write_kv_data_to_file


class ConverterTaskTest(unittest.TestCase):
    def convert(self, architecture, context_length=65536):
        tokenizer = types.SimpleNamespace(
            get_vocab=lambda: {"x": 0}, unk_token_id=-1,
            convert_tokens_to_ids=lambda token: {"<|text_chunk_end|>": 151665,
                                                 "<|object_ref_start|>": 151646,
                                                 "<|object_ref_end|>": 151647}[token])
        modules = {
            "torch": types.ModuleType("torch"),
            "numpy": types.ModuleType("numpy"),
            "gguf": types.SimpleNamespace(GGUFWriter=Writer),
            "safetensors": types.SimpleNamespace(safe_open=None),
            "transformers": types.SimpleNamespace(AutoTokenizer=types.SimpleNamespace(
                from_pretrained=lambda *args, **kwargs: tokenizer)),
        }
        with tempfile.TemporaryDirectory() as tmp, patch.dict(sys.modules, modules):
            model = Path(tmp)
            config = {"architectures": architecture if isinstance(architecture, list) else [architecture], "decoder_config": {
                "hidden_size": 8, "num_hidden_layers": 1, "num_attention_heads": 2,
                "num_key_value_heads": 1, "intermediate_size": 16, "vocab_size": 4,
                "max_position_embeddings": context_length}}
            if architecture is None:
                config.pop("architectures")
            (model / "config.json").write_text(json.dumps(config))
            (model / "tokenizer.json").write_text('{"model": {"merges": []}}')
            script = Path(__file__).resolve().parents[1] / "models/convert-vibevoice-to-gguf.py"
            spec = importlib.util.spec_from_file_location("vibevoice_converter", script)
            converter = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(converter)
            with patch.object(sys, "argv", [str(script), "--input", tmp, "--output", str(model / "out.gguf")]), contextlib.redirect_stdout(io.StringIO()):
                converter.main()
            return Writer.latest.values

    def test_unknown_task_is_rejected_instead_of_silently_labeled(self):
        for architecture in (None, "UnknownVibeVoiceModel",
                             ["VibeVoiceForASRTraining", "VibeVoiceForConditionalGeneration"]):
            with self.subTest(architecture=architecture), self.assertRaisesRegex(ValueError, "architectures"):
                self.convert(architecture)

    def test_base_tts_has_encoders_but_is_not_an_asr_checkpoint(self):
        values = self.convert("VibeVoiceForConditionalGeneration")
        self.assertEqual(values["general.architecture"], "vibevoice-tts")
        self.assertEqual(values["vibevoice.supports_asr"], 0)

    def test_base_tts_preserves_decoder_context(self):
        for context_length in (32768, 65536):
            with self.subTest(context_length=context_length):
                values = self.convert("VibeVoiceForConditionalGeneration", context_length)
                self.assertEqual(values["vibevoice.context_length"], context_length)

    def test_realtime_tts_keeps_its_trained_task(self):
        values = self.convert("VibeVoiceStreamingForConditionalGenerationInference")
        self.assertEqual(values["general.architecture"], "vibevoice-tts")
        self.assertEqual(values["vibevoice.supports_asr"], 0)

    def test_asr_checkpoint_declares_transcription(self):
        values = self.convert("VibeVoiceForASRTraining")
        self.assertEqual(values["general.architecture"], "vibevoice-asr")
        self.assertEqual(values["vibevoice.supports_asr"], 1)

    def test_streaming_asr_keeps_task_and_streaming_metadata(self):
        values = self.convert("VibeVoiceForASRStreamingTraining")
        self.assertEqual(values["general.architecture"], "vibevoice-asr")
        self.assertEqual(values["vibevoice.supports_asr"], 1)
        self.assertEqual(values["vibevoice.asr_streaming"], 1)


if __name__ == "__main__":
    unittest.main()
