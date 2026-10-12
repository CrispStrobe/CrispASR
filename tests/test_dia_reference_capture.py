#!/usr/bin/env python3
"""Tiny original neural model tests capture transparency, not speech quality.

Run in a torch environment with DIA_SOURCE pointing to the pinned checkout.
Only the final DAC is a stub; encoder/decoder/masks/generation are original.
"""
import os
from pathlib import Path
import sys
import types
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from reference_backends import dia_tts_reference as reference


class DiaCaptureTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.torch = torch
        torch.set_num_threads(1)
        source, _ = reference.verify_source(os.environ["DIA_SOURCE"])
        sys.path.insert(0, str(source))
        # Loading the original neural source does not require a real DAC here.
        # Restore the import namespace after importing the source.
        # torchaudio is also unused: no audio-prompt loading/resampling runs.
        previous = {name: sys.modules.get(name) for name in ("dac", "torchaudio")}
        for name in previous:
            sys.modules[name] = types.ModuleType(name)
        try:
            from dia.config import DiaConfig
            from dia.layers import DiaModel
            from dia.model import Dia
        finally:
            for name, module in previous.items():
                if module is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = module
        config = DiaConfig.model_validate({
            "data": {"text_length": 128, "audio_length": 128},
            "model": {
                "encoder": {"n_layer": 1, "n_embd": 8, "n_hidden": 16, "n_head": 2, "head_dim": 4},
                "decoder": {"n_layer": 1, "n_embd": 8, "n_hidden": 16, "gqa_query_heads": 2,
                            "kv_heads": 1, "gqa_head_dim": 4, "cross_query_heads": 2, "cross_head_dim": 4},
                "src_vocab_size": 256, "tgt_vocab_size": 1028,
            },
        })
        torch.manual_seed(17)
        model = DiaModel(config, torch.float32).eval()
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if "norm" in name:
                    parameter.fill_(1)
                else:
                    parameter.normal_(0, .02)
        dia = Dia.__new__(Dia)
        dia.config, dia.model, dia.device, dia.compute_dtype = config, model, torch.device("cpu"), torch.float32
        # Codes -> deterministic dummy output to inspect capture ownership;
        # this is expressly not a reference DAC implementation or PCM proof.
        dia.dac_model = types.SimpleNamespace(
            quantizer=types.SimpleNamespace(from_codes=lambda codes: (codes.float(),)),
            decode=lambda z: z.sum(dim=1, keepdim=True),
        )
        cls.dia = dia

    def test_actual_generate_is_unchanged_and_captures_own_storage(self):
        text = "[S1] Tiny capture test."
        torch = self.torch
        torch.manual_seed(42)
        plain = self.dia.generate(text, max_tokens=32, temperature=1.2, top_p=.95, cfg_filter_top_k=45)
        original = self.dia.model.decoder.decode_step
        captured = reference.capture(self.dia, text, max_tokens=32)
        np.testing.assert_array_equal(captured["decoded_audio"], plain)
        self.assertEqual(self.dia.model.decoder.decode_step, original)
        self.assertEqual(set(captured), set(reference.BOUNDARY_STAGES) | {"encoder_layer_0", "decoder_layer_0"})
        tokens = captured["encoder_tokens"]
        self.assertEqual(tokens.shape[0], 2)
        self.assertTrue((tokens[0] == 0).all())
        self.assertEqual(tokens[1, 0], 1)
        self.assertEqual(captured["decoder_step_logits"].shape[1:], (2, 9, 1028))
        frozen = {name: value.copy() for name, value in captured.items()}
        self.dia.generate("[S2] A different prompt.", max_tokens=32, temperature=0)
        for name in captured:
            np.testing.assert_array_equal(captured[name], frozen[name])

    def test_failure_restores_original_methods_and_hooks(self):
        original = self.dia.model.decoder.decode_step
        generate = self.dia.generate
        def fail(*args, **kwargs):
            raise RuntimeError("deliberate generation failure")
        self.dia.generate = fail
        try:
            with self.assertRaisesRegex(RuntimeError, "deliberate"):
                reference.capture(self.dia, "[S1] Test.", max_tokens=32)
            self.assertEqual(self.dia.model.decoder.decode_step, original)
            self.assertEqual(len(self.dia.model.encoder._forward_hooks), 0)
            self.assertEqual(len(self.dia.model.encoder.embedding._forward_hooks), 0)
        finally:
            self.dia.generate = generate

    def test_unaligned_text_and_unknown_stage_are_rejected(self):
        for text in ["", "Test.", "[S1] Test", "[S1] A\x00B.",
                     " [S1] Test.", "\t[S1] Test.", "\n[S1] Test.",
                     "[S1] Test. ", "[S1] Test.\n"]:
            with self.subTest(text=repr(text)):
                with mock.patch.object(self.dia, "generate", side_effect=AssertionError("unaligned prompt reached generation")):
                    with self.assertRaises(ValueError):
                        reference.capture(self.dia, text, max_tokens=32)
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            reference.capture(self.dia, "[S1] Test.", max_tokens=32, stages={"invented_stage"})

    def test_dump_rejects_unaligned_text_before_source_or_model_loading(self):
        with mock.patch.dict(os.environ, {"DIA_TEXT": "[S1] Test. "}):
            with mock.patch.object(reference, "verify_source", side_effect=AssertionError("invalid prompt reached source loading")) as verify:
                with self.assertRaisesRegex(ValueError, "DIA_TEXT"):
                    reference.dump("nari-labs/Dia-1.6B", None)
                verify.assert_not_called()


if __name__ == "__main__":
    unittest.main()
