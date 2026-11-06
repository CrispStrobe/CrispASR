#!/usr/bin/env python3
"""Compare a pinned Ultra model via upstream Python and CrispASR CPU aligners."""
import json
import re
import subprocess
from pathlib import Path
import torch
from qwen_asr import Qwen3ForcedAligner

TEXT = 'And so my fellow Americans ask not what your country can do for you ask what you can do for your country.'
model = Qwen3ForcedAligner.from_pretrained('model', dtype=torch.float32, device_map='cpu')
result = model.align(audio='samples/jfk.wav', text=TEXT, language='English')
reference = [{'word': w.text, 'start': float(w.start_time), 'end': float(w.end_time)} for w in result[0]]
Path('reference.json').write_text(json.dumps(reference, indent=2))
del model
import gc
gc.collect()
cmd = ['build/bin/crispasr', '--align-only', '-am', 'qwen3-forced-aligner-ultra.gguf',
       '-f', 'samples/jfk.wav', '--ref-text', TEXT, '--align-format', 'json',
       '--align-output', 'actual.json', '-l', 'en', '-ng', '-t', '2']
subprocess.run(cmd, check=True, timeout=900)
actual = json.loads(Path('actual.json').read_text())
def norm(s):
    return re.sub(r'[^a-z0-9]', '', s.lower())
assert len(reference) == len(actual), (reference, actual)
assert [norm(w['word']) for w in reference] == [norm(w['word']) for w in actual], (reference, actual)
errors = [abs(float(a[k])-float(b[k])) for a,b in zip(reference,actual) for k in ('start','end')]
print('ULTRA_PARITY', json.dumps({'words':len(actual), 'max_seconds':max(errors), 'mean_seconds':sum(errors)/len(errors)}), flush=True)
assert max(errors) <= 0.081, (reference, actual)
