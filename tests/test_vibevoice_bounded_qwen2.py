#!/usr/bin/env python3
"""Qualify bounded upstream extraction against the full pinned implementation."""
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class BoundedUpstream(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import transformers
        if transformers.__version__ != '4.57.6':
            raise unittest.SkipTest('requires audited Transformers4.57.6 reference environment')
        from transformers.models.qwen2 import modeling_qwen2
        cls.temp = tempfile.TemporaryDirectory(prefix='vv-bounded-source-')
        cls.source = Path(cls.temp.name)
        original = Path(modeling_qwen2.__file__).parents[2]
        for path in [Path(modeling_qwen2.__file__), original/'modeling_rope_utils.py', original/'masking_utils.py']:
            shutil.copyfile(path, cls.source/path.name)
        cls.adapter = load('bounded_adapter', ROOT/'tools/vibevoice_qwen2_bounded_upstream.py')
        cls.replay = load('bounded_replay', ROOT/'tools/vibevoice_prefill_replay.py')

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_full_and_extracted_math_match_for_f32_and_f16_weights_and_kv(self):
        import numpy as np
        import torch
        import gguf
        from transformers import Qwen2Config, Qwen2Model
        from transformers.models.qwen2 import modeling_qwen2 as full
        torch.set_num_threads(1)
        for weight_precision in ('f32', 'f16'):
            for kv_precision in ('f32', 'f16'):
                with self.subTest(weights=weight_precision, kv=kv_precision), tempfile.TemporaryDirectory(prefix='vv-bounded-') as tmp:
                    folder = Path(tmp)
                    torch.manual_seed(589)
                    cfg = Qwen2Config(hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                        num_attention_heads=4, num_key_value_heads=2, vocab_size=16,
                        max_position_embeddings=64, rope_theta=1000000., rms_norm_eps=1e-6,
                        attention_dropout=0.)
                    cfg._attn_implementation = 'eager'
                    model = Qwen2Model(cfg).eval()
                    with torch.no_grad():
                        for name, value in model.named_parameters():
                            if 'norm' not in name:
                                value.mul_(4)
                        if weight_precision == 'f16':
                            # Round learned GGUF weights, retaining the original
                            # F32 RoPE frequencies (they are not checkpoint weights).
                            for value in model.parameters():
                                value.copy_(value.half().float())
                    bounded = self.adapter.load(self.source)
                    resolved = self.adapter.config_from_dict(cfg.to_dict())
                    resolved._attn_implementation = 'eager'
                    ids = np.arange(39, dtype=np.int64) % 16
                    inputs = model.embed_tokens(torch.from_numpy(ids).unsqueeze(0)).detach()
                    positions = torch.arange(39).unsqueeze(0)
                    kwargs = dict(input_embeds=inputs, attention_mask=None, cache_position=positions[0],
                                  past_key_values=None, position_ids=positions)
                    self.assertTrue(torch.equal(full.create_causal_mask(config=cfg, **kwargs),
                                                bounded.create_causal_mask(config=resolved, **kwargs)))
                    for expected, actual in zip(full.Qwen2RotaryEmbedding(cfg)(inputs, positions),
                                                bounded.Qwen2RotaryEmbedding(resolved)(inputs, positions)):
                        self.assertTrue(torch.equal(expected, actual))
                    path = folder/'tiny.gguf'
                    writer = gguf.GGUFWriter(str(path), 'vibevoice-tts')
                    for name, value in dict(d_lm=32, d_ffn=64, n_lm_layers=2, n_heads=4,
                                             n_kv_heads=2, head_dim=8, vocab_size=16).items():
                        writer.add_uint32('vibevoice.'+name, value)
                    writer.add_float32('vibevoice.rope_theta', cfg.rope_theta)
                    for name, tensor in model.state_dict().items():
                        key = 'lm.tok_emb.weight' if name == 'embed_tokens.weight' else 'lm.'+self.replay.native_key(name)
                        writer.add_tensor(key, tensor.numpy().astype(np.float16 if weight_precision == 'f16' else np.float32))
                    writer.write_header_to_file(); writer.write_kv_data_to_file(); writer.write_tensors_to_file(); writer.close()
                    raw = []
                    hooks = [layer.register_forward_hook(lambda module, values, output: raw.append(output.detach().clone()))
                             for layer in model.layers]
                    with torch.no_grad():
                        cache = self.replay.rounded_kv_cache(cfg) if kv_precision == 'f16' else None
                        expected = model(inputs_embeds=inputs, use_cache=cache is not None,
                                         past_key_values=cache).last_hidden_state
                    for hook in hooks:
                        hook.remove()
                    original = self.replay.replay(path, cfg, inputs.numpy(), folder/'full', ids, kv_precision)
                    extracted = self.replay.replay(path, resolved, inputs.numpy(), folder/'bounded', ids, kv_precision, bounded)
                    for payload in (folder/'full').glob('*.bin'):
                        self.assertEqual(payload.read_bytes(), (folder/'bounded'/payload.name).read_bytes())
                    for index, value in enumerate(raw):
                        actual = np.fromfile(folder/'bounded'/f'prefill_block_{index:03}.bin', dtype='<f4')
                        self.assertTrue(np.array_equal(actual.reshape(value.shape), value.numpy()))
                    self.assertTrue(np.array_equal(np.fromfile(folder/'bounded/tts_prefill_hidden.bin', dtype='<f4'),
                                                   expected[0, -1].numpy()))
                    self.assertEqual(original['embedding_gate']['max_abs'], 0.)
                    self.assertEqual(extracted['embedding_gate']['max_abs'], 0.)
                    self.assertEqual(extracted['upstream_source_hashes'], self.adapter.PINNED)
                    print(json.dumps(dict(weights=weight_precision, kv=kv_precision,
                        raw_blocks_exact=2, full_streamed_and_extracted_payloads_exact=True)), flush=True)

    def test_scope_and_source_guards(self):
        import torch
        from transformers import Qwen2Config
        cfg = Qwen2Config(hidden_size=32, intermediate_size=64, num_hidden_layers=2,
            num_attention_heads=4, num_key_value_heads=2, vocab_size=16).to_dict()
        for change in [dict(attention_dropout=.1), dict(rope_scaling={'rope_type':'dynamic','factor':2.}),
                       dict(use_sliding_window=True), dict(partial_rotary_factor=.5),
                       dict(layer_types=['sliding_attention']*2)]:
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.adapter.config_from_dict({**cfg, **change})
        bounded = self.adapter.load(self.source)
        resolved = self.adapter.config_from_dict(cfg)
        positions = torch.arange(39)
        with self.assertRaises(ValueError):
            bounded.create_causal_mask(config=resolved, input_embeds=torch.zeros(1,38,32),
                attention_mask=None, cache_position=positions, past_key_values=None, position_ids=positions[None])
        cache = bounded.rounded_cache(resolved)
        value = torch.zeros(1,2,39,8)
        cache.update(value, value, 0)
        with self.assertRaises(ValueError):
            cache.update(value, value, 0)
        with tempfile.TemporaryDirectory(prefix='vv-bad-source-') as tmp:
            for name in self.adapter.PINNED:
                shutil.copyfile(self.source/name, Path(tmp)/name)
            with (Path(tmp)/'modeling_qwen2.py').open('a') as stream:
                stream.write('\n# changed source\n')
            with self.assertRaisesRegex(ValueError, 'source hash'):
                self.adapter.load(tmp)


if __name__ == '__main__':
    unittest.main()
