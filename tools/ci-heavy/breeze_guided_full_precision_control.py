#!/usr/bin/env python3
"""Original-only full F32 inference control; no canonical fixture replacement."""
import json
import os
from pathlib import Path
import sys

os.environ.update(ATEN_CPU_CAPABILITY='avx2', ONEDNN_MAX_CPU_ISA='AVX2',
                  MKL_ENABLE_INSTRUCTIONS='AVX2')
sys.path.insert(0, str(Path(__file__).parent))
import torch
import breeze_guided_reference

torch.manual_seed(42)
out = Path(os.environ['HEAVY_OUT'])
out.mkdir(parents=True, exist_ok=True)
(out / 'arithmetic-control-provenance.json').write_text(json.dumps(dict(
    torch_cpu_capability=torch.backends.cpu.get_cpu_capability(),
    torch_config=torch.__config__.show(), initialization_seed=42,
    canonical_reference=False, native_parity_accepted=False,
    memory_before_loading=Path('/proc/meminfo').read_text()), indent=2) + '\n')
breeze_guided_reference.main(promote_joint_arithmetic=True, promote_all_inference=True)
