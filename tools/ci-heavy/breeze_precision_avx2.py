#!/usr/bin/env python3
"""Run the unchanged strict Breeze precision probe with explicit AVX2 dispatch."""
import os
from pathlib import Path
import runpy
import sys

os.environ.update(ATEN_CPU_CAPABILITY='avx2', ONEDNN_MAX_CPU_ISA='AVX2',
                  MKL_ENABLE_INSTRUCTIONS='AVX2',
                  BREEZE_REFERENCE_INITIALIZATION_SEED='42')
probe = Path(__file__).with_name('breeze_depth_probe.py')
sys.path.insert(0, str(probe.parent))
sys.argv = [str(probe), '--precision-decisions']
runpy.run_path(str(probe), run_name='__main__')
