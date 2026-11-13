#!/usr/bin/env python3
"""Exercise real quantizer output and diagnostics with tiny model-free GGUFs."""
import math
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

QUANT = Path(sys.argv.pop(1)).resolve()


def string(s):
    b = s.encode()
    return struct.pack('<Q', len(b)) + b


def fixture(path, arch='nemotron_h_moe', widths=(2688, 1856, 256)):
    tensors = []
    for i, width in enumerate(widths):
        # Same values, including distinct expert rows, in a 3-D bank and 2-D control.
        values = [math.sin(j * 0.17) + (j // width) * 0.03 for j in range(width * 4)]
        data = struct.pack('<%df' % len(values), *values)
        tensors.extend([(f'blk.{i}.ffn_up_exps.weight', (width, 2, 2), data),
                        (f'blk.{i}.ffn_up.weight', (width, 4), data)])
    tensors.append(('output_norm.weight', (256,), struct.pack('<256f', *([1.] * 256))))
    header = b'GGUF' + struct.pack('<IQQ', 3, len(tensors), 1)
    header += string('general.architecture') + struct.pack('<I', 8) + string(arch)
    offset = 0
    for name, shape, data in tensors:
        header += string(name) + struct.pack('<I', len(shape))
        header += struct.pack('<%dQ' % len(shape), *shape) + struct.pack('<IQ', 0, offset)
        offset += (len(data) + 31) // 32 * 32
    with path.open('wb') as f:
        f.write(header + b'\0' * (-len(header) % 32))
        for _, _, data in tensors:
            f.write(data + b'\0' * (-len(data) % 32))


def read(path):
    data = path.read_bytes()
    pos = 4

    def unpack(fmt):
        nonlocal pos
        result = struct.unpack_from('<' + fmt, data, pos)
        pos += struct.calcsize('<' + fmt)
        return result[0] if len(result) == 1 else result

    def text():
        nonlocal pos
        size = unpack('Q')
        result = data[pos:pos + size].decode()
        pos += size
        return result

    version, count, nkeys = unpack('IQQ')
    assert version == 3
    for _ in range(nkeys):
        text()
        kind = unpack('I')
        if kind == 8:
            text()
        elif kind == 4:
            unpack('I')
        else:
            raise AssertionError(f'unexpected metadata kind {kind}')
    descriptors = []
    for _ in range(count):
        name, nd = text(), unpack('I')
        shape = tuple(unpack('Q') for _ in range(nd))
        kind, offset = unpack('IQ')
        descriptors.append((name, shape, kind, offset))
    base = (pos + 31) // 32 * 32
    sizes = {0: (1, 4), 1: (1, 2), 2: (32, 18), 8: (32, 34), 11: (256, 110), 12: (256, 144)}
    result = {}
    for name, shape, kind, offset in descriptors:
        block, size = sizes[kind]
        length = math.prod(shape) // block * size
        payload = data[base + offset:base + offset + length]
        assert len(payload) == length
        result[name] = (shape, kind, payload)
    return result


class QuantizerReporting(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / 'in.gguf'
        fixture(self.source)
        self.serial = 0

    def run_quant(self, mode, source=None, *extra):
        self.serial += 1
        out = self.root / f'out{self.serial}.gguf'
        env = {k: v for k, v in os.environ.items() if not k.startswith('CRISPASR_QUANT_')}
        run = subprocess.run([str(QUANT), str(source or self.source), str(out), mode, *extra],
                             capture_output=True, text=True, env=env, check=True)
        return out, read(out), run.stdout, run.stderr

    def assert_banks(self, tensors, fallback):
        for i, width in enumerate((2688, 1856, 256)):
            bank = tensors[f'blk.{i}.ffn_up_exps.weight']
            control = tensors[f'blk.{i}.ffn_up.weight']
            self.assertEqual(bank[0], (width, 2, 2))
            self.assertEqual(bank[1], fallback if i < 2 else control[1])
            self.assertEqual(bank[1:], control[1:])
        self.assertEqual(tensors['output_norm.weight'], read(self.source)['output_norm.weight'])

    def test_experts_and_row_fallbacks(self):
        for mode in ('q3_k', 'q4_k'):
            with self.subTest(mode=mode):
                _, tensors, _, err = self.run_quant(mode)
                self.assert_banks(tensors, 2)
                self.assertIn('row-width fallback: 4 tensors (36352 elements) use q4_0', err)
                self.assertIn('instead of requested ' + mode, err.lower())
                self.assertNotIn('3-D expert banks', err)

    def test_quantized_source_and_idempotency(self):
        source, tensors, _, err = self.run_quant('q8_0')
        self.assert_banks(tensors, 8)
        self.assertNotIn('WARNING', err)
        _, same, stdout, err = self.run_quant('q8_0', source)
        self.assertEqual(tensors, same)
        self.assertIn('0 tensor types changed; 6 output tensors use quantized storage', stdout)
        self.assertNotIn('WARNING', err)
        _, tensors, _, err = self.run_quant('q4_k', source)
        self.assert_banks(tensors, 2)
        self.assertIn('row-width fallback: 4 tensors', err)

    def test_floating_overrides(self):
        _, tensors, stdout, err = self.run_quant('q4_k', None, '--tensor-type', '.*=f16')
        self.assertTrue(all(v[1] == (0 if len(v[0]) == 1 else 1) for v in tensors.values()))
        self.assertIn('6 tensor types changed; 0 output tensors use quantized storage', stdout)
        self.assertNotIn('row-width fallback', err)
        self.assertIn('no output tensors use quantized storage', err)
        self.assertNotIn('same size', err)

    def test_explicit_quantized_expert_override(self):
        _, tensors, _, err = self.run_quant('q4_k', None, '--tensor-type', '_exps.weight=q8_0')
        for i in range(3):
            self.assertEqual(tensors[f'blk.{i}.ffn_up_exps.weight'][1], 8)
        self.assertIn('row-width fallback: 2 tensors', err)
        self.assertNotIn('3-D expert banks', err)
        # An already-Q8 bank on an unrecognized architecture needs no warning
        # when the caller requests Q8 again.
        fixture(self.source, 'unknown')
        source, _, _, _ = self.run_quant('q8_0', None, '--tensor-type', '_exps.weight=q8_0')
        _, _, _, err = self.run_quant('q8_0', source)
        self.assertNotIn('WARNING', err)

    def test_override_row_fallback_is_distinct(self):
        _, tensors, _, err = self.run_quant('q8_0', None, '--tensor-type', '_exps.weight=q3_k')
        self.assertEqual(tensors['blk.0.ffn_up_exps.weight'][1], 2)
        self.assertEqual(tensors['blk.1.ffn_up_exps.weight'][1], 2)
        self.assertEqual(tensors['blk.2.ffn_up_exps.weight'][1], 11)
        self.assertIn('tensor-type row-width fallback: 2 tensors (18176 elements) use q4_0', err)
        self.assertIn('instead of override q3_k', err.lower())
        self.assertNotIn('instead of requested', err)

    def test_unsupported_expert_rows(self):
        fixture(self.source, widths=(7,))
        _, tensors, _, err = self.run_quant('q4_k')
        self.assertEqual(tensors, read(self.source))
        self.assertIn('copied 1 3-D expert banks (28 elements)', err)
        self.assertIn('unsupported dimensions or row widths', err)
        _, _, _, err = self.run_quant('q4_k', None, '--tensor-type', '_exps.weight=f16')
        self.assertNotIn('3-D expert banks', err)

    def test_other_architecture_still_copies_experts(self):
        fixture(self.source, 'unknown')
        _, tensors, _, err = self.run_quant('q4_k')
        original = read(self.source)
        for i in range(3):
            name = f'blk.{i}.ffn_up_exps.weight'
            self.assertEqual(tensors[name], original[name])
        self.assertIn('copied 3 3-D expert banks (19200 elements)', err)
        self.assertIn('row-width fallback: 2 tensors', err)
        _, _, _, err = self.run_quant('q4_k', None, '--tensor-type', '_exps.weight=f16')
        self.assertNotIn('3-D expert banks', err)


if __name__ == '__main__':
    unittest.main()
