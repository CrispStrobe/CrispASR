#!/usr/bin/env python3
"""Independent scalar controls of the pinned CUDA DS4 conversion semantics."""
import importlib.util
import math
from pathlib import Path
import struct
import unittest

import numpy as np

path = Path(__file__).resolve().parents[1]/'tools/vibevoice_block0_precision_replay.py'
spec = importlib.util.spec_from_file_location('block0_precision',path)
probe = importlib.util.module_from_spec(spec);spec.loader.exec_module(probe)


def f32(x):return struct.unpack('<f',struct.pack('<f',x))[0]
def half(x):return struct.pack('<e',x)


def scalar(group):
    # Source thread i owns four consecutive F32s. XOR offsets are 4,2,1.
    maximum = max(abs(float(x)) for x in group)
    inverse = f32(127/maximum)
    q = [int(math.copysign(math.floor(abs(f32(float(x)*inverse))+.5),float(x))) for x in group]
    lanes = [f32(f32(f32(float(group[4*i])+float(group[4*i+1]))+float(group[4*i+2]))+float(group[4*i+3])) for i in range(8)]
    for offset in (4,2,1):lanes=[f32(lanes[i]+lanes[i^offset]) for i in range(8)]
    return q,half(f32(1/inverse)),half(lanes[0])


class DS4Control(unittest.TestCase):
    def test_original_cuda_scale_rounding_and_partial_sum(self):
        rng=np.random.default_rng(589)
        x=rng.standard_normal((3,256)).astype(np.float32)
        # Half-integer ties, adjacent non-ties and mixed signs are exactly
        # representable here: maximum127 makes the inverse exactly one.
        x[0,:8]=[127.,.5,-.5,1.5,-1.5,np.nextafter(np.float32(.5),np.float32(0)),2.5,-2.5]
        q,d,s=probe.q8_ds4(x)
        for row in range(3):
            for block in range(8):
                wanted=scalar(x[row,block*32:(block+1)*32])
                self.assertEqual(q[row,block].tolist(),wanted[0])
                self.assertEqual(d[row,block].tobytes(),wanted[1])
                self.assertEqual(s[row,block].tobytes(),wanted[2])
        self.assertEqual(q[0,0,1:8].tolist(),[1,-1,2,-2,0,3,-3])
        # Using sums of reconstructed Q8 values loses the original min-term
        # partial sum; retain a counterexample against that tempting shortcut.
        reconstructed=(q.astype(np.float32)*d.astype(np.float32)[...,None]).sum(-1).astype(np.float16)
        self.assertTrue(np.any(reconstructed!=s))

    def test_sum_tree_and_scope_refusals(self):
        x=np.ones((1,256),np.float32)
        x[0,:16]=[65504.,-65504.,1.,1.,1e-4,1e-4,1e-4,1e-4,2.,3.,4.,5.,-2.,-3.,-4.,-5.]
        q,d,s=probe.q8_ds4(x)
        wanted=scalar(x[0,:32])
        self.assertEqual(s[0,0].tobytes(),wanted[2])
        for bad in [np.zeros((1,256),np.float32),np.ones((1,32),np.float32),np.ones((1,256),np.float64),np.full((1,256),np.nan,np.float32)]:
            with self.assertRaises(ValueError):probe.q8_ds4(bad)


if __name__=='__main__':unittest.main()
