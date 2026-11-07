#!/usr/bin/env python3
"""Create valid and invalid model-free GGUF inputs for the sidecar loader."""
import importlib.util
from pathlib import Path
import numpy as np
import gguf

spec = importlib.util.spec_from_file_location('ultra_converter', Path('models/convert-qwen3-ultra-head-to-gguf.py'))
converter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(converter)
weights = {name: np.full(shape, 0.25, dtype=np.float32) for name, shape in converter.SHAPES.items()}
converter.write_gguf(weights, {'mean': [0.0]*91, 'std': [1.0]*91}, Path('synthetic-valid.gguf'))
w = gguf.GGUFWriter('synthetic-invalid.gguf', arch='not_the_head')
w.add_tensor('unexpected', np.ones(1, dtype=np.float32))
w.write_header_to_file()
w.write_kv_data_to_file()
w.write_tensors_to_file()
w.close()
