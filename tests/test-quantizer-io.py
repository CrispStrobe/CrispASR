#!/usr/bin/env python3
"""Check storage failures through the real CLI using disposable GGUFs only."""
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

QUANT = Path(sys.argv.pop(1)).resolve()
FAULTS = Path(sys.argv.pop(1)).resolve() if len(sys.argv) > 1 and not sys.argv[1].startswith('-') else None


def string(value):
    data = value.encode()
    return struct.pack('<Q', len(data)) + data


def fixture(path, kind=0):
    # Two F32 rows: Q8 output has 68 bytes and requires alignment padding.
    header = b'GGUF' + struct.pack('<IQQ', 3, 1, 1)
    header += string('general.architecture') + struct.pack('<I', 8) + string('test')
    header += string('linear.weight') + struct.pack('<IQQIQ', 2, 32, 2, kind, 0)
    path.write_bytes(header + bytes(-len(header) % 32) + struct.pack('<64' + ('f' if kind == 0 else 'e'), *range(64)))


class QuantizerIO(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / 'input.gguf'
        fixture(self.source)
        self.original = self.source.read_bytes()

    def run_quant(self, output, fault=None, extra=()):
        env = dict(os.environ)
        if fault is not None:
            if FAULTS is None:
                self.skipTest('Linux stdio fault library unavailable')
            env.update(LD_PRELOAD=str(FAULTS), QUANTIZER_IO_OUTPUT=str(output),
                       QUANTIZER_IO_FAULT=fault, QUANTIZER_IO_INPUT=str(self.source))
        run = subprocess.run([str(QUANT), str(self.source), str(output), 'q8_0', *extra],
                             capture_output=True, text=True, env=env)
        self.assertEqual(self.source.read_bytes(), self.original)
        return run

    def assert_failed(self, run):
        self.assertNotEqual(run.returncode, 0, run.stdout + run.stderr)
        self.assertIn('failed to quantize model', run.stderr)
        self.assertNotIn('metadata rewrite:', run.stdout)
        self.assertNotIn('output tensors use quantized storage', run.stdout)

    @unittest.skipUnless(Path('/dev/full').exists(), '/dev/full unavailable')
    def test_full_device(self):
        self.assert_failed(self.run_quant(Path('/dev/full')))

    def test_short_writes_and_finalization_failures(self):
        for stage in ('placeholder', 'tensor', 'padding', 'data-flush',
                      'metadata-seek', 'metadata', 'metadata-flush', 'close'):
            with self.subTest(stage=stage):
                run = self.run_quant(self.root / 'failed.gguf', stage)
                self.assertIn(f'injected {stage} failure', run.stderr)
                self.assert_failed(run)
                self.assertIn('output', run.stderr)

    def test_failure_after_successful_chunk(self):
        # Three writes after metadata: two full chunks and one partial tail.
        name = 'linear.weight'
        header = b'GGUF' + struct.pack('<IQQ', 3, 1, 1)
        header += string('general.architecture') + struct.pack('<I', 8) + string('test')
        header += string(name) + struct.pack('<IQQIQ', 2, 256, 8193, 0, 0)
        with self.source.open('wb') as output:
            output.write(header + bytes(-len(header) % 32))
            row = struct.pack('<256f', *range(256))
            for _ in range(8193):
                output.write(row)
        self.original = self.source.read_bytes()
        for target in ('q8_0', 'f16', 'f32'):
            with self.subTest(target=target):
                extra = () if target == 'q8_0' else ('--tensor-type', '.*=' + target)
                run = self.run_quant(self.root / 'late.gguf', 'tensor-after-chunk', extra)
                self.assertIn('injected tensor-after-chunk failure', run.stderr)
                self.assert_failed(run)

    def test_input_seek_failure(self):
        run = self.run_quant(self.root / 'failed.gguf', 'input-seek')
        self.assertIn('injected input-seek failure', run.stderr)
        self.assertIn('failed to seek input tensor', run.stderr)
        self.assert_failed(run)

    def test_cast_and_copy_short_writes(self):
        for source_kind, target in ((0, 'f16'), (0, 'f32'), (1, 'f32')):
            with self.subTest(source_kind=source_kind, target=target):
                fixture(self.source, source_kind)
                self.original = self.source.read_bytes()
                run = self.run_quant(self.root / 'failed.gguf', 'tensor',
                                     ('--tensor-type', '.*=' + target))
                self.assertIn('injected tensor failure', run.stderr)
                self.assert_failed(run)

    def test_successful_interposer_control(self):
        normal, observed = self.root / 'normal.gguf', self.root / 'observed.gguf'
        for output, fault in ((normal, None), (observed, 'none')):
            run = self.run_quant(output, fault)
            self.assertEqual(run.returncode, 0, run.stderr)
            self.assertIn('metadata rewrite:', run.stdout)
        self.assertEqual(normal.read_bytes(), observed.read_bytes())
        self.assertEqual(normal.read_bytes()[:4], b'GGUF')


if __name__ == '__main__':
    unittest.main()
