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
            head.final_layer.linear.weight.normal_(std=.1)
            head.final_layer.adaLN_modulation[1].weight.normal_(std=.1)
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
            positive = torch.linspace(-1., 1., 8)
            negative = torch.linspace(.5, -.5, 8)
            noise = torch.linspace(-.3, .7, 4)
            for name, tensor in [('tts_pos_condition_frame0', positive),
                                 ('tts_neg_condition_frame0', negative), ('tts_noise_frame0', noise)]:
                tensor.numpy().astype('<f4').tofile(capture / (name + '.bin'))
            # Build matching captures with the original upstream F32 module,
            # before GGUF serialization/dequantization and replay reconstruction.
            scheduler_spec = importlib.util.spec_from_file_location(
                'tiny_original_scheduler', source / 'vibevoice/schedule/dpm_solver.py')
            scheduler_module = importlib.util.module_from_spec(scheduler_spec)
            scheduler_spec.loader.exec_module(scheduler_module)
            scheduler = scheduler_module.DPMSolverMultistepScheduler(
                num_train_timesteps=1000, beta_schedule='squaredcos_cap_v2', prediction_type='v_prediction')
            scheduler.set_timesteps(2)
            z = noise.repeat(2, 1)
            condition = torch.stack([positive, negative])
            with torch.no_grad():
                for i, t in enumerate(scheduler.timesteps):
                    both = head(z[:1].repeat(2, 1), t.repeat(2).to(z), condition)
                    velocity = both[1] + 1.3 * (both[0] - both[1])
                    z = scheduler.step(velocity.repeat(2, 1), t, z).prev_sample
                    for label, tensor in [('v_both', both), ('v_cfg', velocity), ('z', z[0])]:
                        tensor.numpy().astype('<f4').tofile(capture / f'tts_frame0_{label}_s{i:03}.bin')
                z[0].numpy().astype('<f4').tofile(capture / 'tts_latent_frame0.bin')
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
            self.assertTrue(all(row['max_abs'] == 0 for row in result['stages']))
            self.assertFalse(np.array_equal(both[0].numpy(), both[1].numpy()))
            self.assertIn('no prefill parity', result['scope'])
            # A changed boundary must appear in the report, even when the
            # remaining stages continue to match the original upstream run.
            boundary = capture / 'tts_frame0_v_both_s000.bin'
            values = np.fromfile(boundary, dtype='<f4')
            values[0] += .25
            values.tofile(boundary)
            try:
                sys.argv = ['replay', '--gguf', str(model), '--capture', str(capture), '--source', str(source),
                            '--output', str(path/'perturbed'), '--expected-sha256', hashlib.sha256(model.read_bytes()).hexdigest()]
                replay.main()
            finally:
                sys.argv = old
            perturbed = json.loads((path/'perturbed/comparison.json').read_text())
            divergent = [row for row in perturbed['stages'] if row['max_abs'] > 0]
            self.assertEqual([row['stage'] for row in divergent], ['tts_frame0_v_both_s000'])
            self.assertAlmostEqual(divergent[0]['max_abs'], .25, places=6)



if __name__ == '__main__':
    unittest.main()
