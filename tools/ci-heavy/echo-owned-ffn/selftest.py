#!/usr/bin/env python3
"""Actual released MLP + tiny BF16 safetensors: owned-input, scale and source guards."""
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'tools/ci-heavy'))
import echo_owned_ffn as worker


class OwnedFFN(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from safetensors.torch import save_file
        torch.set_num_threads(1)
        cls.torch = torch
        cls.directory = tempfile.TemporaryDirectory(prefix='echo-owned-ffn-')
        cls.addClassCleanup(cls.directory.cleanup)
        cls.path = Path(cls.directory.name)
        state = {}
        for projection, shape in [('gate',(8,4)),('up',(8,4)),('down',(4,8))]:
            values = torch.arange(32).float().reshape(shape)/37 - .4
            if projection == 'up':
                values = values.flip(0)*.7
            state[f'model.layers.0.mlp.{projection}_proj.weight'] = values.to(torch.bfloat16)
        save_file(state, str(cls.path/'tiny.safetensors'))
        cls.config = dict(hidden_size=4, intermediate_size=8, hidden_act='silu')

    def load(self):
        from safetensors import safe_open
        with safe_open(str(self.path/'tiny.safetensors'), framework='pt', device='cpu') as reader:
            return worker.load_layer(reader, 0, self.config)

    def test_actual_original_owned_input_and_attribution(self):
        mlp, ledger = self.load()
        self.assertEqual(ledger['resident_parameter_bytes'], 384)
        x = np.array([.25,-.75,1.5,-2.],dtype='f4')
        native = worker.capture(mlp,x)
        result = worker.attribute(mlp,native,worker.array_sha(x),self.path/'positive')
        self.assertTrue(result['original_off_on_off_exact'])
        for metric in result['metrics'].values():
            self.assertLess(metric['relative_l2'], 1e-7)
            self.assertAlmostEqual(metric['cosine'],1.,places=12)
            self.assertAlmostEqual(metric['norm_ratio'],1.,places=12)
        saved=np.load(self.path/'positive/mlp_in.npy',allow_pickle=False)
        native['mlp_in'].fill(0)
        np.testing.assert_array_equal(saved,x)

    def test_scale_blind_cosine_negative(self):
        a=np.array([1.,2.,-3.,4.],dtype='f4')
        def strict_control(mine,reference):
            row=worker.metric(mine,reference)
            assert row['cosine']>.999999 and row['relative_l2']<1e-5
        strict_control(a,a)
        row=worker.metric(2*a,a)
        self.assertAlmostEqual(row['cosine'],1.)
        self.assertAlmostEqual(row['norm_ratio'],2.)
        with self.assertRaises(AssertionError):strict_control(2*a,a)

    def test_changed_input_ledger_rejected(self):
        mlp,_=self.load();x=np.array([.25,-.75,1.5,-2.],dtype='f4');native=worker.capture(mlp,x)
        expected=worker.array_sha(native['mlp_in']);native['mlp_in'][0]+=.125
        with self.assertRaisesRegex(AssertionError,'Owned input ledger'):
            worker.attribute(mlp,native,expected,self.path/'bad-input')
        self.assertFalse((self.path/'bad-input').exists())

    def test_changed_source_rejected_before_load(self):
        old=worker.MODELING_SHA
        try:
            worker.MODELING_SHA='0'*64
            with self.assertRaisesRegex(RuntimeError,'modeling file SHA mismatch'):self.load()
        finally:worker.MODELING_SHA=old

    def test_malformed_shape_rejected(self):
        mlp,_=self.load();native=worker.capture(mlp,np.ones(4,dtype='f4'));native['gate_out']=np.ones(7,dtype='f4')
        with self.assertRaises(AssertionError):worker.attribute(mlp,native,worker.array_sha(native['mlp_in']),self.path/'bad-shape')

    def test_block_observer_survives_ffn_off_on_off(self):
        environment = {}
        for arm in ['off', 'on', 'off2']:
            environment.pop('CRISPASR_INDEX_ECHO_DUMP_FFN', None)
            environment.update(worker.native_observer_environment(arm))
            worker.validate_native_observers(arm, environment)
            self.assertEqual(environment['CRISPASR_INDEX_ECHO_DUMP_STAGES'], '1')
            self.assertEqual(environment.get('CRISPASR_INDEX_ECHO_DUMP_FFN'), '1' if arm == 'on' else None)
        # The first hosted worker cleared both observers in its OFF arm.
        with self.assertRaisesRegex(AssertionError, 'Block observer'):
            worker.validate_native_observers('off', {})
        with self.assertRaisesRegex(AssertionError, 'Wrong FFN observer'):
            worker.validate_native_observers('off2', worker.native_observer_environment('on'))


if __name__=='__main__':
    unittest.main()
