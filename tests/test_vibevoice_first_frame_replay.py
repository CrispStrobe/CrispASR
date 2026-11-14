#!/usr/bin/env python3
"""Tiny actual upstream-head/scheduler and same-GGUF replay plumbing control."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]


class ReplayControl(unittest.TestCase):
    def test_tiny_same_weight_upstream_replay(self):
        source = os.environ.get('CRISPASR_VIBEVOICE_SRC')
        if not source:
            self.skipTest('set CRISPASR_VIBEVOICE_SRC to audited upstream checkout')
        import numpy as np
        import torch
        import gguf
        spec = importlib.util.spec_from_file_location('frame_replay', ROOT / 'tools/vibevoice_first_frame_replay.py')
        replay = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(replay)
        source = Path(source)
        cls = replay.upstream_head(source / 'vibevoice/modular/modular_vibevoice_diffusion_head.py', torch)
        cfg = SimpleNamespace(hidden_size=8, latent_size=4, head_ffn_ratio=2, head_layers=1, rms_norm_eps=1e-5)
        torch.manual_seed(42)
        head = cls(cfg)
        # Force a nonzero output so a missing/zero stage cannot masquerade as parity.
        with torch.no_grad():
            head.final_layer.linear.weight.fill_(.1)
        with tempfile.TemporaryDirectory(prefix='vv-frame-control-') as tmp:
            path = Path(tmp)
            model = path / 'tiny.gguf'
            writer = gguf.GGUFWriter(str(model), 'vibevoice')
            def short(n):
                return ('pred.' + n).replace('noisy_images_proj.', 'noisy_proj.').replace('cond_proj.', 'cond.').replace('t_embedder.mlp.', 't_emb.').replace('adaLN_modulation.1.', 'adaln.').replace('final_layer.', 'final.')
            for name, tensor in head.state_dict().items():
                writer.add_tensor(short(name), tensor.detach().numpy())
            writer.write_header_to_file(); writer.write_kv_data_to_file(); writer.write_tensors_to_file(); writer.close()
            capture = path / 'capture'
            capture.mkdir()
            np.array([4, 8, 2], dtype='<i4').tofile(capture / 'capture_complete.bin')
            np.array([999, 500], dtype='<i4').tofile(capture / 'tts_frame0_timesteps.bin')
            for name, count in [('tts_pos_condition_frame0',8), ('tts_neg_condition_frame0',8), ('tts_noise_frame0',4), ('tts_latent_frame0',4)]:
                np.ones(count,dtype='<f4').tofile(capture / (name+'.bin'))
            for i in range(2):
                for name, count in [('v_both',8), ('v_cfg',4), ('z',4)]:
                    np.zeros(count,dtype='<f4').tofile(capture / f'tts_frame0_{name}_s{i:03}.bin')
            old = sys.argv
            try:
                sys.argv = ['replay', '--gguf',str(model),'--capture',str(capture),'--source',str(source),
                            '--output',str(path/'out'),'--expected-sha256',hashlib.sha256(model.read_bytes()).hexdigest()]
                replay.main()
            finally:
                sys.argv = old
            result = json.loads((path/'out/comparison.json').read_text())
            self.assertEqual(len(result['stages']), 7)
            self.assertGreater(result['stages'][0]['norm_ref'], 0)
            self.assertGreater(result['stages'][0]['max_abs'], 0)
            self.assertIn('no prefill parity', result['scope'])


if __name__ == '__main__':
    unittest.main()
