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
invalid = subprocess.run(['build/bin/crispasr', '--align-only', '--ultra-head', head,
                          '-f', 'samples/jfk.wav', '--ref-text', 'test'], capture_output=True, text=True, timeout=60)
assert invalid.returncode != 0 and 'requires an explicit Qwen forced-aligner' in invalid.stderr
invalid = subprocess.run(['build/bin/crispasr', '--ultra-head', head, '-f', 'samples/jfk.wav'],
                         capture_output=True, text=True, timeout=60)
assert invalid.returncode != 0 and '--ultra-head requires --align-only' in invalid.stderr
subprocess.run(['build/bin/crispasr', '--align-only', '-am', base, '--ultra-head', head,
                '-f', 'samples/jfk.wav', '--ref-text', TEXT, '--align-format', 'json',
                '--align-output', 'ultra-cli.json', '-l', 'en', '-ng', '-t', '2'],
               check=True, timeout=900)
cli = json.loads(Path('ultra-cli.json').read_text())
subprocess.run(['build/bin/crispasr', '--align-only', '-am', base, '--ultra-head', head,
                '-f', 'samples/jfk.wav', '--ref-text', TEXT, '--align-format', 'json',
                '--align-granularity', 'segment', '--align-output', 'ultra-segment.json',
                '-l', 'en', '-ng', '-t', '2'], check=True, timeout=900)
segments = json.loads(Path('ultra-segment.json').read_text())
assert len(segments) == 1 and segments[0]['words'] == cli
assert abs(segments[0]['start'] - cli[0]['start']) <= 0.0005
assert abs(segments[0]['end'] - cli[-1]['end']) <= 0.0005
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
