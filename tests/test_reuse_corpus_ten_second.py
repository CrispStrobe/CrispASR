"""Functional controls for original-only ten-second context comparisons."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
runner=load('ten_corpus',ROOT/'tools/kaggle/reuse-corpus-ten-second-20261113/quality.py')
chunking=load('ten_chunking',ROOT/'tools/ci-heavy/issue579_reuse/chunked.py')


class TenSecondControls(unittest.TestCase):
    def test_genuine_ten_seconds_uses_exact_three_global_input_slices(self):
        audio=np.random.default_rng(579).standard_normal(160000).astype('float32');seen=[]
        def identity(block,index,start):
            seen.append((index,start));np.testing.assert_array_equal(block,audio[start:start+80000]);return block
        output,starts=chunking.chunked(audio,identity,window_samples=80000,hop_samples=40000,max_window_samples=80000)
        self.assertEqual(starts,[0,40000,80000]);self.assertEqual(seen,[(0,0),(1,40000),(2,80000)])
        np.testing.assert_array_equal(output,audio)

    def test_waveform_difference_keeps_scale_visible(self):
        x=np.arange(1,100,dtype='float32');value=runner.difference(x,x/2)
        self.assertAlmostEqual(value['rel_rms'],.5);self.assertAlmostEqual(value['norm_ratio'],.5)
        self.assertAlmostEqual(value['cosine'],1)
        for a,b in (([],[]),([0],[0]),([1],[float('nan')]),([1],[1,2])):
            with self.assertRaises(ValueError):runner.difference(a,b)

    def test_selected_references_retain_actual_arrays_and_reject_missing_stage(self):
        captures={name:np.arange(3,dtype='float32') for name in runner.STAGES}
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'reference.npz';runner.stage_archive(path,captures)
            with np.load(path) as archive:
                self.assertEqual(set(archive.files),runner.STAGES)
                for name in captures:np.testing.assert_array_equal(archive[name],captures[name])
            del captures['block_29']
            with self.assertRaises(ValueError):runner.stage_archive(path,captures)


if __name__=='__main__':unittest.main()
