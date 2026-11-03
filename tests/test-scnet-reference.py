#!/usr/bin/env python3
"""Reference-archive corruption and original-method observer regression tests.

Archive tests require only numpy + gguf. Set SCNET_SOURCE_DIR to the pinned
official checkout, with requirements-scnet.txt + torch installed, to exercise
the genuine model and pipeline using synthetic weights (not a fidelity claim).
"""

import contextlib
import importlib.util
import io
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from dump_scnet_reference import ReferenceArchive  # noqa: E402
from reference_backends import scnet  # noqa: E402

HAS_GGUF = importlib.util.find_spec("gguf") is not None
HAS_ORIGINAL = bool(os.environ.get("SCNET_SOURCE_DIR")) and importlib.util.find_spec("torch") is not None


def fixture_stage(name):
    if name.endswith("_stats"):
        return np.array([0, 1], dtype=np.float32)
    if name.endswith("audio") or name.startswith(("output_", "raw_output_")):
        return np.arange(8, dtype=np.float32).reshape(2, 4)
    return np.arange(24, dtype=np.float32).reshape(1, 2, 3, 4)


class ProfileTests(unittest.TestCase):
    def test_complete_stage_set(self):
        self.assertEqual(len(scnet.DEFAULT_STAGES), 35)
        self.assertEqual(len(set(scnet.DEFAULT_STAGES)), 35)

    def test_profile_rejects_truncation_and_unsupported_ranges(self):
        scnet.validate_profile(20, 1, 2**64 - 1)
        for profile in ((0, 1, 0), (120, 1, 0), (20.5, 1, 0), (20, -1, 0),
                        (20, 17, 0), (20, 1, -1), (20, 1, 2**64), (20, True, 0)):
            with self.assertRaises(ValueError):
                scnet.validate_profile(*profile)


@unittest.skipUnless(HAS_GGUF, "gguf package unavailable")
class ArchiveTests(unittest.TestCase):
    def test_complete_archive_retains_values_and_uint64_seed(self):
        import gguf

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reference.gguf"
            with ReferenceArchive(path, {"scnet.reference.shift_seed": 2**63 + 5}) as archive:
                for name in reversed(scnet.DEFAULT_STAGES):
                    source = fixture_stage(name)
                    archive.capture(name, source)
                    source[:] = -100  # no reference may alias a later in-place mutation
            reader = gguf.GGUFReader(path)
            self.assertEqual(len(reader.tensors), 35)
            seed = reader.get_field("scnet.reference.shift_seed")
            self.assertEqual(seed.types, [gguf.GGUFValueType.UINT64])
            self.assertEqual(int(seed.parts[seed.data[0]][0]), 2**63 + 5)
            for tensor in reader.tensors:
                np.testing.assert_array_equal(tensor.data, fixture_stage(tensor.name))

    def test_missing_stage_preserves_previous_output(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reference.gguf"
            path.write_bytes(b"previous valid artifact")
            with self.assertRaisesRegex(ValueError, "missing required"):
                with ReferenceArchive(path, {}) as archive:
                    archive.capture("input_audio", fixture_stage("input_audio"))
            self.assertEqual(path.read_bytes(), b"previous valid artifact")
            self.assertEqual(list(Path(directory).glob("*.partial")), [])

    def test_duplicate_nonfinite_and_wrong_layout_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            for bad in ("duplicate", "nonfinite", "layout"):
                with self.assertRaises(ValueError):
                    with ReferenceArchive(Path(directory) / f"{bad}.gguf", {}) as archive:
                        if bad == "duplicate":
                            archive.capture("waveform_stats", np.array([0, 1], dtype=np.float32))
                            archive.capture("waveform_stats", np.array([0, 1], dtype=np.float32))
                        elif bad == "nonfinite":
                            archive.capture("waveform_stats", np.array([0, np.nan], dtype=np.float32))
                        else:
                            archive.capture("input_audio", np.zeros((4, 2), dtype=np.float32))


@unittest.skipUnless(HAS_ORIGINAL, "set SCNET_SOURCE_DIR with original runtime dependencies")
class OriginalObserverTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import importlib
        import torch

        torch.set_num_threads(2)
        torch.manual_seed(12)
        namespace = scnet._source_package(os.environ["SCNET_SOURCE_DIR"])
        cls.model = importlib.import_module(namespace + ".SCNet").SCNet(**scnet.model_config()).eval()
        cls.model._scnet_source_namespace = namespace
        scnet.validate_state(cls.model.state_dict())
        cls.namespace = namespace

    def test_capture_keeps_original_raw_output_and_code(self):
        rng = np.random.default_rng(30)
        audio = rng.normal(0, 0.1, (2, 4096)).astype(np.float32)
        code = type(self.model).forward.__code__
        expected = scnet.separate_original(self.model, audio, no_segment=True)
        capture = scnet.capture_original(self.model, audio, no_segment=True)
        self.assertIs(type(self.model).forward.__code__, code)
        self.assertEqual(set(capture), set(scnet.DEFAULT_STAGES))
        for index, stem in enumerate(scnet.SOURCES):
            np.testing.assert_array_equal(capture[f"output_{stem}"], expected[index])
            np.testing.assert_array_equal(capture[f"raw_output_{stem}"], expected[index])
        self.assertIsNone(sys.gettrace())
        self.assertFalse(self.model._forward_hooks)

    def test_hooked_original_overlapping_pipeline_is_identical(self):
        rng = np.random.default_rng(31)
        audio = rng.normal(0.02, 0.1, (2, 48000)).astype(np.float32)
        profile = dict(segment_seconds=1, shifts=1, shift_seed=2**63 + 5)
        with contextlib.redirect_stdout(io.StringIO()):
            expected = scnet.separate_original(self.model, audio, **profile)
            capture = scnet.capture_original(self.model, audio, **profile)
        for index, stem in enumerate(scnet.SOURCES):
            np.testing.assert_array_equal(capture[f"output_{stem}"], expected[index])
        self.assertEqual(capture["raw_input_audio"].shape[0], 2)
        self.assertGreater(capture["raw_input_audio"].shape[1], 44100)
        self.assertNotEqual(capture["waveform_stats"][0], 0)
        self.assertIsNone(sys.gettrace())

    def test_failed_sink_removes_observers(self):
        audio = np.ones((2, 4096), dtype=np.float32)

        def fail_at_stft(name, _value):
            if name == "stft":
                raise RuntimeError("injected storage failure")

        with self.assertRaisesRegex(RuntimeError, "injected storage"):
            scnet.capture_original(self.model, audio, no_segment=True, capture=fail_at_stft)
        self.assertIsNone(sys.gettrace())
        self.assertFalse(self.model._forward_hooks)
        self.assertFalse(self.model._forward_pre_hooks)
        self.assertFalse(self.model.encoder[0]._forward_pre_hooks)


if __name__ == "__main__":
    unittest.main()
