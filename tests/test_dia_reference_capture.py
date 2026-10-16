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
                "decoder": {"n_layer": 2, "n_embd": 8, "n_hidden": 16, "gqa_query_heads": 2,
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
        self.assertEqual(set(captured), set(reference.BOUNDARY_STAGES) | {"encoder_layer_0", "decoder_layer_0", "decoder_layer_1"})
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

    def test_late_capture_matches_actual_call_and_preserves_generation(self):
        text = "[S1] Tiny capture test."
        layer_calls, hidden_calls = [], []
        handles = [
            self.dia.model.decoder.layers[0].register_forward_hook(
                lambda _m, _a, output: layer_calls.append(output[:, 0].detach().numpy().copy())),
            self.dia.model.decoder.norm.register_forward_hook(
                lambda _m, _a, output: hidden_calls.append(output[:, 0].detach().numpy().copy())),
        ]
        self.torch.manual_seed(42)
        try:
            plain = self.dia.generate(text, max_tokens=32, temperature=1.2, top_p=.95, cfg_filter_top_k=45)
        finally:
            for handle in handles:
                handle.remove()
        first = reference.capture(self.dia, text, max_tokens=32)
        late = reference.capture(self.dia, text, max_tokens=32, capture_step=7)
        repeat = reference.capture(self.dia, text, max_tokens=32)
        np.testing.assert_array_equal(late["decoder_layer_0"], layer_calls[7])
        np.testing.assert_array_equal(late["decoder_hidden"], hidden_calls[7])
        np.testing.assert_array_equal(late["decoded_audio"], plain)
        for name in first:
            np.testing.assert_array_equal(first[name], repeat[name])
        for name in ("decoder_step_logits", "decoder_input_tokens", "decoder_codes_rev", "decoded_audio"):
            np.testing.assert_array_equal(first[name], late[name])
        self.assertTrue(all(not module._forward_hooks for module in self.dia.model.decoder.modules()))

    def test_invalid_capture_step_is_rejected_before_generation(self):
        for step in (-1, 32, True, "7", 7.5):
            with self.subTest(step=step):
                with mock.patch.object(self.dia, "generate", side_effect=AssertionError("invalid index reached generation")):
                    with self.assertRaisesRegex(ValueError, "capture step"):
                        reference.capture(self.dia, "[S1] Test.", max_tokens=32, capture_step=step)

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

    def test_attention_capture_matches_actual_calls_and_keeps_full_trajectory(self):
        from dia_attention_capture import attention_stages
        text, step = "[S1] Tiny capture test.", 5
        attention = self.dia.model.decoder.layers[0].self_attention
        calls, projected = [], []
        def observe(_m, _args, kwargs, _output):
            cache = kwargs["cache"]
            if int(cache.current_idx) == step + 1:
                calls.append((cache.k[:, :, :step + 1].transpose(1, 2).detach().numpy().copy(),
                              cache.v[:, :, :step + 1].transpose(1, 2).detach().numpy().copy()))
        handles = [attention.register_forward_hook(observe, with_kwargs=True),
                   attention.o_proj.register_forward_pre_hook(
                       lambda _m, args: projected.append(args[0].detach().numpy().copy()))]
        self.torch.manual_seed(42)
        try:
            plain = self.dia.generate(text, max_tokens=32, temperature=1.2, top_p=.95, cfg_filter_top_k=45)
        finally:
            for h in handles:
                h.remove()
        off = reference.capture(self.dia, text, max_tokens=32, capture_step=step)
        on = reference.capture(self.dia, text, max_tokens=32, capture_step=step, attention_layers=(0, 1))
        repeat = reference.capture(self.dia, text, max_tokens=32, capture_step=step)
        for key in off:
            self.assertEqual(off[key].tobytes(), on[key].tobytes(), key)
            self.assertEqual(off[key].tobytes(), repeat[key].tobytes(), key)
        np.testing.assert_array_equal(on["decoded_audio"], plain)
        self.assertTrue(set(attention_stages((0,))) <= on.keys())
        np.testing.assert_array_equal(on["decoder_sa0_k"], calls[0][0])
        np.testing.assert_array_equal(on["decoder_sa0_v"], calls[0][1])
        np.testing.assert_array_equal(on["decoder_sa0_attention"], projected[step])
        self.assertEqual(on["decoder_sa0_q"].shape, (2, 1, 2, 4))
        self.assertEqual(on["decoder_sa0_k"].shape, (2, step + 1, 1, 4))
        self.assertTrue((on["decoder_sa1_q"][0] != on["decoder_sa1_q"][1]).any())
        frozen = {k: on[k].copy() for k in attention_stages((0, 1))}
        self.dia.generate("[S2] Different.", max_tokens=32, temperature=0)
        for k in frozen:
            self.assertEqual(on[k].tobytes(), frozen[k].tobytes())
        for module in (attention, attention.rotary_emb, attention.o_proj):
            self.assertFalse(module._forward_hooks)
            self.assertFalse(module._forward_pre_hooks)

    def test_explicit_stage_subset_still_returns_requested_optional_captures(self):
        from dia_attention_capture import attention_stages
        from dia_decoder_capture import boundary_stages
        trace = reference.capture(self.dia, "[S1] Tiny capture test.", max_tokens=32,
                                  stages={"decoder_hidden"}, capture_step=5,
                                  attention_layers=(1,), boundary_layers=(1,))
        self.assertTrue(set(attention_stages((1,))) <= trace.keys())
        self.assertTrue(set(boundary_stages((1,))) <= trace.keys())
        self.assertEqual(trace["decoder_sa1_k"].shape[1], 6)

    def test_attention_capture_rejects_bad_layers_and_restores_on_inner_failure(self):
        for layers in ((-1,), (2,), (0, 0), (True,)):
            with self.subTest(layers=layers), self.assertRaisesRegex(ValueError, "layer"):
                reference.capture(self.dia, "[S1] Test.", max_tokens=32, attention_layers=layers)
        attention = self.dia.model.decoder.layers[0].self_attention
        original = self.dia.model.decoder.decode_step
        with mock.patch.object(attention.o_proj, "forward", side_effect=RuntimeError("projection failure")):
            with self.assertRaisesRegex(RuntimeError, "projection failure"):
                reference.capture(self.dia, "[S1] Test.", max_tokens=32, attention_layers=(0,))
        self.assertEqual(self.dia.model.decoder.decode_step, original)
        for module in (attention, attention.rotary_emb, attention.o_proj):
            self.assertFalse(module._forward_hooks)
            self.assertFalse(module._forward_pre_hooks)

    def test_dump_rejects_bad_attention_indices_before_loading(self):
        for value in ("-1", "0,0", "0,", "0, 1", "１", "true"):
            with self.subTest(value=value), mock.patch.dict(os.environ, {"DIA_CAPTURE_ATTENTION_LAYERS": value}):
                with mock.patch.object(reference, "verify_source", side_effect=AssertionError("bad layer reached loading")):
                    with self.assertRaisesRegex(ValueError, "DIA_CAPTURE_ATTENTION_LAYERS"):
                        reference.dump("nari-labs/Dia-1.6B", None)

    def test_boundary_hooks_observe_fused_original_mlp_and_preserve_trajectory(self):
        from dia_decoder_capture import PARTS, boundary_stages
        text, step = "[S1] Tiny capture test.", 5
        layer = self.dia.model.decoder.layers[0]
        fused, hidden, residual, outputs = [], [], [], []
        def owned(value):
            return value.detach().numpy().copy()
        handles = [
            layer.mlp.wi_fused.register_forward_hook(lambda _m, _a, y: fused.append(owned(y))),
            layer.mlp.wo.register_forward_pre_hook(lambda _m, a: hidden.append(owned(a[0]))),
            layer.pre_ca_norm.register_forward_pre_hook(lambda _m, a: residual.append(owned(a[0]))),
            layer.register_forward_hook(lambda _m, _a, y: outputs.append(owned(y))),
        ]
        self.torch.manual_seed(42)
        try:
            plain = self.dia.generate(text, max_tokens=32, temperature=1.2, top_p=.95, cfg_filter_top_k=45)
        finally:
            for handle in handles:
                handle.remove()
        off = reference.capture(self.dia, text, max_tokens=32, capture_step=step)
        on = reference.capture(self.dia, text, max_tokens=32, capture_step=step,
                               boundary_layers=(0, 1), attention_layers=(0,))
        repeat = reference.capture(self.dia, text, max_tokens=32, capture_step=step)
        for name in off:
            self.assertEqual(off[name].tobytes(), on[name].tobytes(), name)
            self.assertEqual(off[name].tobytes(), repeat[name].tobytes(), name)
        np.testing.assert_array_equal(on['decoded_audio'], plain)
        self.assertEqual(len(PARTS), 16)
        self.assertTrue(set(boundary_stages((0, 1))) <= on.keys())
        np.testing.assert_array_equal(on['decoder_block0_mlp_gate'], fused[step][..., 0, :])
        np.testing.assert_array_equal(on['decoder_block0_mlp_up'], fused[step][..., 1, :])
        np.testing.assert_array_equal(on['decoder_block0_mlp_hidden'], hidden[step])
        np.testing.assert_array_equal(on['decoder_block0_sa_residual'], residual[step])
        np.testing.assert_array_equal(on['decoder_block0_output'], outputs[step])
        np.testing.assert_array_equal(on['decoder_block0_sa_attention'], on['decoder_sa0_attention'])
        frozen = {name: on[name].copy() for name in boundary_stages((0, 1))}
        self.dia.generate('[S2] Different.', max_tokens=32, temperature=0)
        for name in frozen:
            self.assertEqual(on[name].tobytes(), frozen[name].tobytes())
        for module in layer.modules():
            self.assertFalse(module._forward_hooks)
            self.assertFalse(module._forward_pre_hooks)

    def test_boundary_hooks_reject_bad_layers_and_clean_up_after_inner_failure(self):
        for layers in ((-1,), (2,), (0, 0), (True,)):
            with self.subTest(layers=layers), self.assertRaisesRegex(ValueError, 'layer'):
                reference.capture(self.dia, '[S1] Test.', max_tokens=32, boundary_layers=layers)
        layer = self.dia.model.decoder.layers[0]
        original = self.dia.model.decoder.decode_step
        with mock.patch.object(layer.mlp.wo, 'forward', side_effect=RuntimeError('original MLP failure')):
            with self.assertRaisesRegex(RuntimeError, 'original MLP failure'):
                reference.capture(self.dia, '[S1] Test.', max_tokens=32, boundary_layers=(0,))
        self.assertEqual(self.dia.model.decoder.decode_step, original)
        for module in layer.modules():
            self.assertFalse(module._forward_hooks)
            self.assertFalse(module._forward_pre_hooks)

    def test_dump_rejects_bad_boundary_indices_before_loading(self):
        for value in ('-1', '0,0', '0,', '0, 1', '１', 'true'):
            with self.subTest(value=value), mock.patch.dict(os.environ, {'DIA_CAPTURE_BOUNDARY_LAYERS': value}):
                with mock.patch.object(reference, 'verify_source', side_effect=AssertionError('invalid layer reached loading')):
                    with self.assertRaisesRegex(ValueError, 'DIA_CAPTURE_BOUNDARY_LAYERS'):
                        reference.dump('nari-labs/Dia-1.6B', None)

    def test_dump_rejects_unaligned_text_before_source_or_model_loading(self):
        with mock.patch.dict(os.environ, {"DIA_TEXT": "[S1] Test. "}):
            with mock.patch.object(reference, "verify_source", side_effect=AssertionError("invalid prompt reached source loading")) as verify:
                with self.assertRaisesRegex(ValueError, "DIA_TEXT"):
                    reference.dump("nari-labs/Dia-1.6B", None)
                verify.assert_not_called()

    def test_dump_rejects_bad_capture_step_before_model_loading(self):
        for step in ("-1", "128", "1x", "", "１"):
            with self.subTest(step=step), mock.patch.dict(os.environ, {"DIA_TEXT": "[S1] Test.", "DIA_CAPTURE_STEP": step, "DIA_MAX_TOKENS": "128"}):
                with mock.patch.object(reference, "verify_source", side_effect=AssertionError("bad index reached model loading")):
                    with self.assertRaisesRegex(ValueError, "DIA_CAPTURE_STEP"):
                        reference.dump("nari-labs/Dia-1.6B", None)

    def test_actual_generate_observes_all_declared_codec_modules(self):
        torch = self.torch
        class SumChannels(torch.nn.Module):
            def forward(self, value):
                return value.sum(dim=1, keepdim=True)
        modules = torch.nn.Sequential(SumChannels(), *[torch.nn.Identity() for _ in range(6)], torch.nn.Tanh())
        original = self.dia.dac_model
        codec = types.SimpleNamespace(decoder=types.SimpleNamespace(model=modules),
                                      quantizer=types.SimpleNamespace(from_codes=lambda codes: (codes.float(),)),
                                      decode=modules)
        self.dia.dac_model = codec
        try:
            torch.manual_seed(42)
            plain = self.dia.generate("[S1] Test.", max_tokens=32, temperature=1.2,
                                      top_p=.95, cfg_filter_top_k=45)
            trace = reference.capture(self.dia, "[S1] Test.", max_tokens=32)
            self.assertTrue(set(reference.CODEC_STAGES) <= trace.keys())
            np.testing.assert_array_equal(trace["decoded_audio"], plain)
            np.testing.assert_array_equal(trace["dac_latent"], trace["decoder_codes_rev"].T)
            self.assertTrue(all(not module._forward_hooks for module in modules))
        finally:
            self.dia.dac_model = original


class CodecCaptureOwnershipTest(unittest.TestCase):
    """Small module plumbing tests; these are not original DAC PCM parity."""
    def codec(self):
        import torch
        decoder = torch.nn.Sequential(torch.nn.Conv1d(2, 2, 3, padding=1),
                                      *[torch.nn.Identity() for _ in range(6)], torch.nn.Tanh())
        codec = types.SimpleNamespace(decoder=types.SimpleNamespace(model=decoder),
                                      quantizer=types.SimpleNamespace(from_codes=lambda x: (x.float(),)),
                                      decode=decoder)
        return codec, torch.arange(14).reshape(1, 2, 7)

    def test_trace_owns_complete_channels_and_preserves_actual_calls(self):
        codec, codes = self.codec()
        plain = codec.decode(codes.float()).detach().numpy().reshape(-1)
        trace = reference.capture_codec(codec, codes)
        self.assertEqual(set(trace), set(reference.CODEC_STAGES) | {"decoded_audio"})
        np.testing.assert_array_equal(trace["decoded_audio"], plain)
        np.testing.assert_array_equal(trace["dac_latent"], codes[0].numpy())
        self.assertEqual(trace["dac_output_conv"].shape, (2, 7))
        frozen = {name: value.copy() for name, value in trace.items()}
        codec.decode(codes.float() * 2)
        for name, value in trace.items():
            np.testing.assert_array_equal(value, frozen[name])
        self.assertTrue(all(not module._forward_hooks for module in codec.decoder.model))

    def test_failure_removes_all_observers(self):
        codec, codes = self.codec()
        def fail(*args):
            raise RuntimeError("deliberate codec failure")
        codec.decode = fail
        with self.assertRaisesRegex(RuntimeError, "deliberate codec failure"):
            reference.capture_codec(codec, codes)
        self.assertTrue(all(not module._forward_hooks for module in codec.decoder.model))


if __name__ == "__main__":
    unittest.main()
