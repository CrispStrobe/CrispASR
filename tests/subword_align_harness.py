"""Compile literal own tokenizer/DP and its immutable own reference, no encoder.

The standalone child measures only alignment time/live C++ heap; vocab/logits,
output allocation, IPC and serialization are excluded. It does not measure model
inference, process RSS, GPU memory or Japanese alignment quality.
"""
import hashlib
import os
from pathlib import Path
import shlex
import struct
import subprocess
import sys
import tempfile

import numpy as np

from test_session_utf8 import definition

ROOT = Path(__file__).resolve().parents[1]
BASELINE_SHA256 = '48e9400358ff09c4394677b8208d979212923beb47ba0d2122515312d494bd36'
FLAGS = ['-std=c++17', '-O3', '-DNDEBUG', '-fno-fast-math']


def production_code(root):
    source = (root / 'src/canary_ctc.cpp').read_text()
    code = '''
#include "canary_ctc.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>
'''
    code += definition(source, 'struct canary_ctc_vocab {') + ';\n'
    code += '''
// Exactly the context fields used by the literal postprocessor, no encoder.
struct canary_ctc_context {
    struct { struct { int blank_id = 0; int frame_dur_cs = 8; } hparams; } model;
    canary_ctc_vocab vocab;
};
'''
    for marker in ('static bool tokenise_word(', 'static int canary_ctc_align_words_baseline(',
                   'static int canary_ctc_align_words_compact(', 'extern "C" int canary_ctc_align_words('):
        code += definition(source, marker) + '\n'
    return code


class Probe:
    def __init__(self, source_root=ROOT, build_dir=None):
        if sys.byteorder != 'little':
            raise RuntimeError('The test protocol requires a little-endian host')
        self.tmp = tempfile.TemporaryDirectory(dir=build_dir)
        output = Path(self.tmp.name)
        baseline = ROOT / 'tests/fixtures/canary-align-baseline-deaf9c010.inc'
        if hashlib.sha256(baseline.read_bytes()).hexdigest() != BASELINE_SHA256:
            raise RuntimeError('Immutable CrispASR subword alignment baseline changed')
        code = production_code(Path(source_root)) + baseline.read_text()
        code += (ROOT / 'tests/subword_align_probe.cpp').read_text()
        (output / 'probe.cpp').write_text(code)
        compiler = shlex.split(os.environ.get('CXX', 'c++'))
        subprocess.run(compiler + FLAGS + ['-I', str(Path(source_root) / 'src'), '-I', str(ROOT / 'tests'),
                       str(output / 'probe.cpp'), '-o', str(output / 'probe')], check=True)
        self.proc = subprocess.Popen([str(output / 'probe')], stdin=subprocess.PIPE, stdout=subprocess.PIPE)

    def close(self):
        if getattr(self, 'proc', None):
            self.proc.stdin.close()
            try:
                code = self.proc.wait(timeout=10)
            finally:
                self.proc.stdout.close()
            if code:
                raise RuntimeError(f'Subword probe exited with {code}')
            self.proc = None
        self.tmp.cleanup()

    def _read(self, n):
        chunks = []
        while n:
            chunk = self.proc.stdout.read(n)
            if not chunk:
                raise RuntimeError(f'Subword probe stopped (exit {self.proc.poll()})')
            chunks.append(chunk)
            n -= len(chunk)
        return b''.join(chunks)

    def run(self, logits, words, vocab, *, blank=0, duration=8, reference=False, gate=1):
        a = np.ascontiguousarray(logits, dtype='<f4')
        t, v = a.shape
        packet = bytearray(struct.pack('<8i', t, v, len(words), len(vocab), blank, duration, int(reference), gate))
        for text in vocab + words:
            encoded = text.encode('utf-8')
            packet += struct.pack('<I', len(encoded)) + encoded
        packet += a.tobytes()
        self.proc.stdin.write(packet)
        self.proc.stdin.flush()
        peak, allocs, ns = struct.unpack('<3Q', self._read(24))
        payload = self._read(4 + len(words) * 80)
        rc = struct.unpack_from('<i', payload)[0]
        stamps = []
        for i in range(len(words)):
            offset = 4 + i * 80
            text = payload[offset:offset + 64].split(b'\0', 1)[0]
            stamps.append((text, *struct.unpack_from('<2q', payload, offset + 64)))
        return {'bytes': payload, 'rc': rc, 'stamps': stamps, 'peak_heap_bytes': peak,
                'allocations': allocs, 'kernel_ns': ns}
