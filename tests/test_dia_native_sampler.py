#!/usr/bin/env python3
"""Tiny actual-header/original-neural contracts; no real-model PCM claim."""
import ast
import importlib
import os
from pathlib import Path
import sys
import types
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from dia_native_sampler import NativeSampler, install_native_sampler, native_sample_rows


class NativeTraceTest(unittest.TestCase):
    def test_complete_trace_and_false_success_controls(self):
        row = ' 1 2 3 4 5 6 7 8 1024'
        log = 'backend loaded\nDIA_TOK step 0:' + row + '\nDIA_TOK step 1:' + row + '\nfinished\n'
        self.assertEqual(native_sample_rows(log, 2).tolist(), [[1, 2, 3, 4, 5, 6, 7, 8, 1024]] * 2)
        for invalid in (log.replace('step 1', 'step 0'), log.replace('step 0', 'step 1'),
                        log.replace(' 1024', ''), log.replace(' 1024', ' 1025'),
                        log.replace(' 1024', ' nan'), log + 'DIA_TOK step 2:' + row + '\n', ''):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                native_sample_rows(invalid, 2)


class NativeSamplerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        torch.set_num_threads(1)
        cls.torch = torch
        cls.library = os.environ['DIA_SAMPLER_BRIDGE']

    def test_state_is_identical_across_batch_and_single_row_calls(self):
        torch = self.torch
        logits = torch.arange(63).float().reshape(9, 7).sin() * 3
        with NativeSampler(self.library) as batch, NativeSampler(self.library) as rows:
            for _ in range(4):
                expected = batch(logits, 1.2, .95, 5)
                actual = torch.cat([rows(row[None], 1.2, .95, 5) for row in logits])
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            self.assertEqual(expected.dtype, torch.int64)
            frozen = batch.samples[0].copy()
            expected.fill_(0)
            self.assertTrue((batch.samples[0] == frozen).all())
        with self.assertRaisesRegex(RuntimeError, 'closed'):
            batch(logits, 1.2, .95, 5)

    def test_portable_candidate_matches_original_batched_draws(self):
        torch = self.torch
        source = Path(os.environ['DIA_SOURCE']) / 'dia/model.py'
        tree = ast.parse(source.read_text())
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                        and node.name == '_sample_next_token')
        namespace = dict(torch=torch)
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), 'exec'), namespace)
        original = namespace['_sample_next_token']
        logits = torch.arange(9 * 1028).float().reshape(9, 1028).sin() * 3
        logits[:, 1025:] = -torch.inf
        with mock.patch.dict(os.environ, {'CRISPASR_DIA_PORTABLE_RNG': '1'}):
            with NativeSampler(self.library) as candidate:
                torch.manual_seed(42)
                for _ in range(4):
                    expected = original(logits, temperature=1.2, top_p=.95, cfg_filter_top_k=45)
                    actual = candidate(logits, 1.2, .95, 45)
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_masked_dominant_token_and_invalid_input_state(self):
        torch = self.torch
        valid = torch.tensor([[-torch.inf, -20, 20.]])
        with NativeSampler(self.library) as sampler, NativeSampler(self.library) as untouched:
            for values in ([float('nan'), 0, 1], [float('inf'), 0, 1], [-float('inf')] * 3):
                with self.assertRaisesRegex(ValueError, 'rejected'):
                    sampler(torch.tensor([values]), 1.2, .95, 2)
            for temperature in (0., 1.2):
                self.assertEqual(sampler(valid, temperature, .95, 2).tolist(), [2])
                self.assertEqual(untouched(valid, temperature, .95, 2).tolist(), [2])
            logits = torch.tensor([[1., 2, 3, 4]])
            torch.testing.assert_close(sampler(logits, 1., 1., None),
                                       untouched(logits, 1., 1., None), rtol=0, atol=0)

    def test_failure_restores_original_function_and_releases_handle(self):
        original = lambda *args: None
        module = types.SimpleNamespace(_sample_next_token=original)
        with self.assertRaisesRegex(RuntimeError, 'deliberate'):
            with install_native_sampler(module, self.library) as sampler:
                self.assertIs(module._sample_next_token, sampler)
                raise RuntimeError('deliberate')
        self.assertIs(module._sample_next_token, original)
        self.assertIsNone(sampler.handle)

    def test_actual_tiny_original_generate_and_capture_are_identical(self):
        # Reuse the existing actual pinned tiny encoder/decoder/generate
        # fixture. Only its final dummy DAC is excluded from quality proof.
        from test_dia_reference_capture import DiaCaptureTest
        from reference_backends import dia_tts_reference as reference
        DiaCaptureTest.setUpClass()
        module = importlib.import_module('dia.model')
        original = module._sample_next_token
        with install_native_sampler(module, self.library):
            plain = DiaCaptureTest.dia.generate('[S1] Test.', max_tokens=32, temperature=1.2,
                                               top_p=.95, cfg_filter_top_k=45)
        with install_native_sampler(module, self.library) as sampler:
            captured = reference.capture(DiaCaptureTest.dia, '[S1] Test.', max_tokens=32)
            self.assertGreater(len(sampler.samples), 0)
        self.assertIs(module._sample_next_token, original)
        import numpy as np
        np.testing.assert_array_equal(captured['decoded_audio'], plain)


if __name__ == '__main__':
    unittest.main()
