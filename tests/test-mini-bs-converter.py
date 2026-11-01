#!/usr/bin/env python3
"""Model-free guards against silently converting an incompatible blueprint."""

import copy
from contextlib import contextmanager
import importlib.util
import math
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "mini_bs_converter", ROOT / "models" / "convert-mini-bs-roformer-to-gguf.py"
)
CONVERTER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CONVERTER)
REFERENCE_SPEC = importlib.util.spec_from_file_location(
    "mini_bs_original_reference", ROOT / "tools" / "reference_backends" / "mini_bs_roformer.py"
)
REFERENCE = importlib.util.module_from_spec(REFERENCE_SPEC)
REFERENCE_SPEC.loader.exec_module(REFERENCE)


def original_config():
    # Pinned upstream config values, independent of the converter's defaults.
    bands = (
        [2] * 17 + [4] * 16 + [6] * 5 + [8] * 4 + [10] * 3 + [12] * 3 + [14]
        + [16] * 3 + [20] * 3 + [24] * 2 + [28] * 2
        + [32, 34, 36, 38, 42, 46, 50, 52, 58, 62, 68, 72, 80, 86, 92, 100, 110, 118, 130, 140, 151]
    )
    return {
        "model_type": "bs_roformer", "hidden_size": 512, "band_proj_size": 256,
        "intermediate_size": 1024, "num_hidden_layers": 6,
        "num_attention_heads": 8, "num_key_value_heads": 4, "head_dim": 64,
        "num_input_channels": 2, "num_stems": 4, "register_token_num": 4,
        "time_conv_length": 4, "time_transformer_depth": 1, "freq_transformer_depth": 1,
        "stft_n_fft": 4096, "stft_n_fft_out": 2048, "stft_hop_length": 512,
        "wave_chunk_size": 352800, "wave_sample_rate": 44100,
        "rms_norm_eps": 1e-6, "rope_base": 10000.0,
        "freqs_per_bands": bands,
        "freqs_per_bands_out": [value // 2 for value in bands[:-1]] + [76],
    }


class ConverterContract(unittest.TestCase):
    def test_pinned_checkpoint_shapes_and_parameter_count(self):
        config = original_config()
        CONVERTER.validate_config(config)
        shapes = CONVERTER.expected_shapes(config)
        # Counts read from the actual upstream safetensors header.
        self.assertEqual(len(shapes), 456)
        self.assertEqual(sum(math.prod(shape) for shape in shapes.values()), 46173188)
        self.assertEqual(shapes["freq_domain_model.mask_model.encoder.register_tokens"], (4, 4, 512))
        self.assertEqual(shapes["freq_domain_model.mask_model.time_conv.gate_proj.weight"], (2048, 1024))
        self.assertEqual(shapes["freq_domain_model.mask_model.time_deconv.down_proj.weight"], (1024, 2048))
        self.assertEqual(sum(CONVERTER.keep_f32(name) for name in shapes), 112)

    def test_rejects_architecturally_incompatible_metadata(self):
        cases = [
            ("model_type", "mel_band_roformer"),
            ("num_key_value_heads", 3),
            ("num_attention_heads", 7),
            ("head_dim", 63),
            ("num_stems", 1),
            ("num_input_channels", 1),
            ("wave_sample_rate", 48000),
            ("wave_chunk_size", 264600),
            ("num_hidden_layers", 8),
            ("rms_norm_eps", float("nan")),
            ("rms_norm_eps", -1),
            ("rms_norm_eps", "1e-6"),
            ("rope_base", float("inf")),
            ("rope_base", 500000),
            ("time_conv_length", True),
        ]
        for key, value in cases:
            with self.subTest(key=key, value=value):
                config = original_config()
                config[key] = value
                with self.assertRaises(ValueError):
                    CONVERTER.validate_config(config)

    def test_rejects_missing_or_corrupt_band_layout(self):
        for name in ("freqs_per_bands", "freqs_per_bands_out"):
            for mutation in ([], [1], None, [1] * 80):
                with self.subTest(name=name, mutation=mutation):
                    config = original_config()
                    config[name] = copy.deepcopy(mutation)
                    with self.assertRaises(ValueError):
                        CONVERTER.validate_config(config)
            config = original_config()
            config[name][0] = 0
            with self.assertRaises(ValueError):
                CONVERTER.validate_config(config)

    def test_probe_register_and_delta_norm_precision(self):
        for name in (
            "freq_domain_model.freq_pooling.probe",
            "freq_domain_model.time_pooling.probe",
            "freq_domain_model.mask_model.encoder.register_tokens",
            "freq_domain_model.mask_model.time_conv_norm.scale",
        ):
            self.assertTrue(CONVERTER.keep_f32(name), name)

    def test_original_reference_rejects_missing_and_changed_source(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            with self.assertRaises(FileNotFoundError):
                REFERENCE.verify_source_files(path)
            for filename in REFERENCE.SOURCE_HASHES:
                (path / filename).write_text("raise RuntimeError('must never execute')\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "does not match original revision"):
                REFERENCE.verify_source_files(path)


@contextmanager
def published_loader_stub(info):
    """Observe loader arguments and error handling without importing Torch."""
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory)
        (path / "config.json").write_text('{"dtype": "bfloat16"}', encoding="utf-8")
        model = mock.Mock()
        model.eval.return_value = model
        model.to.return_value = model
        loader = mock.Mock(return_value=(model, info))
        namespace = "_crispasr_original_mini_bs_roformer_" + REFERENCE.MODEL_REVISION
        modules = {
            namespace: SimpleNamespace(),
            namespace + ".configuration_bs_roformer": SimpleNamespace(BSRoformerConfig=SimpleNamespace),
            namespace + ".modeling_bs_roformer": SimpleNamespace(
                BSRoformerForMaskedEstimation=SimpleNamespace(from_pretrained=loader)),
        }
        with mock.patch.dict(sys.modules, modules), \
                mock.patch.object(REFERENCE, "_snapshot", return_value=path), \
                mock.patch.object(REFERENCE, "verify_source_files") as verify:
            yield path, model, loader, verify


class PublishedLoaderContract(unittest.TestCase):
    def test_preserves_original_dtype_and_buffer_initialization_path(self):
        info = {key: [] for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")}
        with published_loader_stub(info) as (path, model, loader, verify):
            actual = REFERENCE._load_published(path, device="test-device")
            self.assertIs(actual, model)
            verify.assert_called_once_with(path)
            loader.assert_called_once_with(str(path), config=SimpleNamespace(dtype="bfloat16"),
                                           local_files_only=True, output_loading_info=True)
            model.eval.assert_called_once_with()
            model.to.assert_called_once_with(device="test-device")

    def test_rejects_inexact_checkpoint_loads(self):
        for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"):
            with self.subTest(key=key), published_loader_stub({key: ["invalid"]}) as (path, model, _, _verify):
                with self.assertRaisesRegex(RuntimeError, "exact checkpoint"):
                    REFERENCE._load_published(path)
                model.to.assert_not_called()

    def test_verifies_source_before_published_loader_executes(self):
        with published_loader_stub({}) as (path, _model, loader, verify):
            verify.side_effect = ValueError("source identity mismatch")
            with self.assertRaisesRegex(ValueError, "source identity mismatch"):
                REFERENCE._load_published(path)
            loader.assert_not_called()


if __name__ == "__main__":
    unittest.main()
