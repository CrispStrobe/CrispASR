#!/usr/bin/env python3
"""Compare the public Ultra CLI and Python/C ABI with an upstream reference."""
import json
from pathlib import Path
import subprocess
import sys
import wave
import numpy as np
import crispasr

TEXT = 'And so my fellow Americans ask not what your country can do for you ask what you can do for your country.'
base = 'qwen3-forced-aligner-ultra.gguf'
head = 'ultra-local-head.gguf'
reference = json.loads(Path('prior-reference/local-head-reference.json').read_text())
expected = np.array([[word['start'], word['end']] for word in reference])
subprocess.run(['build/bin/crispasr', '--align-only', '-am', base, '--ultra-head', head,
                '-f', 'samples/jfk.wav', '--ref-text', TEXT, '--align-format', 'json',
                '--align-output', 'ultra-cli.json', '-l', 'en', '-ng', '-t', '2'],
               check=True, timeout=900)
cli = json.loads(Path('ultra-cli.json').read_text())
with wave.open('samples/jfk.wav', 'rb') as audio:
    assert audio.getnchannels() == 1 and audio.getsampwidth() == 2 and audio.getframerate() == 16000
    pcm = np.frombuffer(audio.readframes(audio.getnframes()), dtype='<i2').astype(np.float32) / 32768.0
words = crispasr.align_words_ultra(base, head, TEXT, pcm, n_threads=2, lib_path=sys.argv[1])
for label, actual in [('cli', [[w['start'], w['end']] for w in cli]),
                      ('python_abi', [[w.start, w.end] for w in words])]:
    actual = np.asarray(actual)
    assert actual.shape == expected.shape and np.isfinite(actual).all()
    error = np.abs(actual - expected) * 1000
    print('ULTRA_INTERFACE_PARITY', label, json.dumps({'words':len(actual), 'max_ms':float(error.max()),
                                                        'mean_ms':float(error.mean())}), flush=True)
    assert float(error.max()) <= 2.0 and float(error.mean()) <= 0.5
assert len(cli) == len(words) == 22
assert all(c['word'] == w.text for c, w in zip(cli, words))
