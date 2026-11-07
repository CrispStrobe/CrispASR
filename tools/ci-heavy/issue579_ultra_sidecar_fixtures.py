#!/usr/bin/env python3
"""Create valid and invalid model-free GGUF inputs for the sidecar loader."""
import importlib.util
from pathlib import Path
import sys
import numpy as np
import gguf

spec = importlib.util.spec_from_file_location('ultra_converter', Path('models/convert-qwen3-ultra-head-to-gguf.py'))
converter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(converter)
out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('.')
out.mkdir(parents=True, exist_ok=True)
weights = {name: np.full(shape, 0.25, dtype=np.float32) for name, shape in converter.SHAPES.items()}
converter.write_gguf(weights, {'mean': [0.0]*91, 'std': [1.0]*91}, out / 'synthetic-valid.gguf')
w = gguf.GGUFWriter(str(out / 'synthetic-invalid.gguf'), arch='not_the_head')
w.add_tensor('unexpected', np.ones(1, dtype=np.float32))
w.write_header_to_file()
w.write_kv_data_to_file()
w.write_tensors_to_file()
w.close()

# Keep the correct architecture and all names, then corrupt one field at a
# time so each loader guard is exercised independently.
for kind in ('shape', 'nan', 'std'):
    w = gguf.GGUFWriter(str(out / f'synthetic-{kind}.gguf'), arch='qwen3_ultra_head')
    w.add_string('qwen3_ultra_head.source_revision', converter.REVISION)
    for name, shape in converter.SHAPES.items():
        value = weights[name]
        if kind == 'shape' and name == 'audio.0.bias':
            value = np.zeros((127,), dtype=np.float32)
        elif kind == 'nan' and name == 'audio.0.bias':
            value = np.full(shape, np.nan, dtype=np.float32)
        w.add_tensor(name, np.ascontiguousarray(value))
    w.add_tensor('normalizer.mean', np.zeros(91, dtype=np.float32))
    w.add_tensor('normalizer.std', np.zeros(91, dtype=np.float32) if kind == 'std' else np.ones(91, dtype=np.float32))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
