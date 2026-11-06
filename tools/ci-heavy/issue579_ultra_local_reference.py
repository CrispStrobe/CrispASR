#!/usr/bin/env python3
"""Capture the upstream Ultra local head's inputs, logits and final timings."""
import json
from pathlib import Path
import numpy as np
import torch
from qwen_fa_ultra import LocalBoundaryAligner

TEXT = 'And so my fellow Americans ask not what your country can do for you ask what you can do for your country.'
aligner = LocalBoundaryAligner.from_pretrained('model', device='cpu', dtype=torch.float32)
stages = {}
def capture(_module, args, output):
    assert len(args) == 4
    for name, tensor in zip(('features', 'context', 'context_mask', 'endpoints'), args):
        stages[name] = tensor.detach().cpu().numpy()
    stages['boundary_logits'] = output['boundary_logits'].detach().float().cpu().numpy()
handle = aligner._head.register_forward_hook(capture)
try:
    results = aligner.align(audio='samples/jfk.wav', text=TEXT, language='English')
finally:
    handle.remove()
words = [{'word': w.text, 'start': w.start_time, 'end': w.end_time,
          'base_start': w.unconstrained_start_time, 'base_end': w.unconstrained_end_time} for w in results[0]]
assert len(words) == 22 and all(w['start'] <= w['end'] for w in words)
assert set(stages) == {'features','context','context_mask','endpoints','boundary_logits'}
assert all(np.isfinite(v).all() for v in stages.values())
np.savez_compressed('local-head-stages.npz', **stages)
Path('local-head-reference.json').write_text(json.dumps(words, indent=2))
print('ULTRA_LOCAL_REFERENCE_PASS', json.dumps({'words':len(words), 'shapes':{k:list(v.shape) for k,v in stages.items()}}), flush=True)
