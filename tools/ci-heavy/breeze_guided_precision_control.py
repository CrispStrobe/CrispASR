#!/usr/bin/env python3
"""All-eight original-only F32 arithmetic diagnostic; no canonical acceptance."""
import os
import json
from pathlib import Path
import sys

# Set dispatch before importing torch, including through the oracle helper.
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
    dispatch={key: os.environ[key] for key in
              ('ATEN_CPU_CAPABILITY', 'ONEDNN_MAX_CPU_ISA', 'MKL_ENABLE_INSTRUCTIONS')},
    canonical_reference=False, native_parity_accepted=False), indent=2) + '\n')
breeze_guided_reference.main(promote_joint_arithmetic=True)
