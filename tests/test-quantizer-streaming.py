#!/usr/bin/env python3
"""Production quantizer storage equivalence and bounded-memory execution controls."""
import hashlib
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import time
import unittest

QUANT = Path(sys.argv.pop(1)).resolve()
BASELINE = Path(sys.argv.pop(1)).resolve() if len(sys.argv) > 1 else None


def string(value):
    encoded = value.encode()
    return struct.pack('<Q', len(encoded)) + encoded


def fixture(path, rows, *, arch='nemotron_h_moe', name='blk.0.ffn_up_exps.weight', shape=None, kind=0, start_row=0):
    # Chunk boundary is 4096 rows. Distinct columns and odd row count cover
    # nonzero quantization and the final partial chunk without large Python RAM.
    header = b'GGUF' + struct.pack('<IQQ', 3, 1, 1)
    header += string('general.architecture') + struct.pack('<I', 8) + string(arch)
    shape = shape or (256, rows, 2)
    header += string(name) + struct.pack('<I', len(shape))
    header += struct.pack('<%dQIQ' % len(shape), *shape, kind, 0)
    patterns = [struct.pack('<256' + ('f' if kind == 0 else 'e'),
                            *[(i - 123 + k) / 128 for i in range(256)]) for k in range(7)]
    with path.open('wb') as output:
        output.write(header + b'\0' * (-len(header) % 32))
        for row in range(rows * 2):
            output.write(patterns[(start_row + row) % 7])


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(1 << 20), b''):
            result.update(chunk)
    return result.hexdigest()


def payload(path):
    with path.open('rb') as source:
        def unpack(fmt):
            return struct.unpack('<' + fmt, source.read(struct.calcsize('<' + fmt)))
        def text():
            return source.read(unpack('Q')[0])
        assert source.read(4) == b'GGUF'
        version, nt, nk = unpack('IQQ')
        assert version == 3 and nt == 1
        for _ in range(nk):
            text()
            kind = unpack('I')[0]
            if kind == 8:
                text()
            else:
                source.read({4: 4, 10: 8}[kind])
        text()
        dimensions = unpack('I')[0]
        shape = unpack('Q' * dimensions)
        kind, offset = unpack('IQ')
        assert offset == 0
        source.seek((-source.tell()) % 32, 1)
        return shape, kind, source.read()


class Streaming(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / 'source.gguf'
        fixture(self.source, 4097)

    def run_quant(self, executable, source, output, mode, *extra, limit=False):
        preexec = None
        if limit:
            try:
                import resource
            except ImportError:
                self.skipTest('RLIMIT_AS unavailable')
            def preexec():
                resource.setrlimit(resource.RLIMIT_AS, (self.address_limit,) * 2)
                resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        env = {k: v for k, v in os.environ.items() if not k.startswith('CRISPASR_QUANT_')}
        if mode in ('f16', 'f32'):
            extra = ('--tensor-type', '.*=' + mode, *extra)
            mode = 'q8_0'
        return subprocess.run([str(executable), str(source), str(output), mode, *extra],
                              capture_output=True, env=env, preexec_fn=preexec)

    def test_chunk_concatenation(self):
        # Independently quantized row ranges must concatenate exactly, including
        # the two-row tail. This runs without a pre-change executable in CTest.
        left, right = self.root / 'left.gguf', self.root / 'right.gguf'
        fixture(left, 4096)
        fixture(right, 1, start_row=8192)
        for mode in ('q4_k', 'q8_0', 'f16'):
            with self.subTest(mode=mode):
                outputs = []
                for i, source in enumerate((self.source, left, right)):
                    output = self.root / f'part{i}.gguf'
                    result = self.run_quant(QUANT, source, output, mode)
                    self.assertEqual(result.returncode, 0, result.stderr.decode())
                    outputs.append(payload(output))
                self.assertEqual(outputs[0][0], (256, 4097, 2))
                self.assertEqual(outputs[0][1], outputs[1][1])
                self.assertEqual(outputs[0][2], outputs[1][2] + outputs[2][2])

    def test_zero_element_layout_rejected(self):
        fixture(self.source, 0)
        original = digest(self.source)
        result = self.run_quant(QUANT, self.source, self.root / 'zero.gguf', 'q8_0')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(b'invalid tensor row layout', result.stderr)
        self.assertNotIn(b'output tensors use quantized storage', result.stdout)
        self.assertEqual(digest(self.source), original)

    def test_storage_equivalence(self):
        if BASELINE is None:
            self.skipTest('optional pre-change executable was not supplied')
        for mode in ('q8_0', 'q4_0', 'q3_k', 'q4_k', 'iq4_nl', 'f16', 'f32'):
            with self.subTest(mode=mode):
                old = self.root / 'old.gguf'
                new = self.root / 'new.gguf'
                for binary, output in ((BASELINE, old), (QUANT, new)):
                    result = self.run_quant(binary, self.source, output, mode)
                    self.assertEqual(result.returncode, 0, result.stderr.decode())
                self.assertEqual(digest(old), digest(new))
        # Quantized sources: both requantization and F16/F32 casts.
        quantized = self.root / 'q8.gguf'
        self.assertEqual(self.run_quant(BASELINE, self.source, quantized, 'q8_0').returncode, 0)
        for mode in ('q4_k', 'q8_0', 'f16', 'f32'):
            with self.subTest(source='q8_0', mode=mode):
                old, new = self.root / 'old.gguf', self.root / 'new.gguf'
                for binary, output in ((BASELINE, old), (QUANT, new)):
                    result = self.run_quant(binary, quantized, output, mode)
                    self.assertEqual(result.returncode, 0, result.stderr.decode())
                self.assertEqual(digest(old), digest(new))

    def assert_equivalent(self, source, mode, *extra):
        old, new = self.root / 'old.gguf', self.root / 'new.gguf'
        for binary, output in ((BASELINE, old), (QUANT, new)):
            result = self.run_quant(binary, source, output, mode, *extra)
            self.assertEqual(result.returncode, 0, result.stderr.decode())
        self.assertEqual(digest(old), digest(new))

    def test_pointwise_f16_and_imatrix_equivalence(self):
        if BASELINE is None:
            self.skipTest('optional pre-change executable was not supplied')
        for arch, name, shape, kind in (
                ('nemotron_h_moe', 'blk.0.ffn_up_exps.weight', (256, 4097, 2), 1),
                ('parakeet', 'encoder.layers.0.conv.pw1.weight', (1, 256, 8194), 0)):
            fixture(self.source, 4097, arch=arch, name=name, shape=shape, kind=kind)
            for mode in ('q4_k', 'q8_0', 'f16', 'f32'):
                with self.subTest(arch=arch, mode=mode):
                    self.assert_equivalent(self.source, mode)
        # Real imatrix GGUF consumer, with nonuniform per-column importance.
        fixture(self.source, 4097)
        name = 'blk.0.ffn_up_exps.weight'
        header = b'GGUF' + struct.pack('<IQQ', 3, 1, 1)
        header += string('count.' + name) + struct.pack('<IQ', 10, 2)
        header += string(name) + struct.pack('<IQIQ', 1, 256, 0, 0)
        imatrix = self.root / 'imatrix.gguf'
        imatrix.write_bytes(header + b'\0' * (-len(header) % 32)
                            + struct.pack('<256f', *[2 + i / 128 for i in range(256)]))
        for mode in ('q3_k', 'q4_k', 'iq4_nl'):
            with self.subTest(imatrix=mode):
                self.assert_equivalent(self.source, mode, '--imatrix', str(imatrix))

    def startup_address_space(self, executable):
        # Block at the first input fopen, after the dynamic loader and library
        # constructors, to allow full CMake builds their own mapping headroom.
        # FIFO metadata occupies no model storage; /dev/shm supports FIFOs even
        # when the fixture directory resides on a network filesystem.
        if not Path('/dev/shm').is_dir() or not Path('/proc/self/status').is_file():
            self.skipTest('FIFO memory observer requires /dev/shm and procfs')
        try:
            local = tempfile.TemporaryDirectory(dir='/dev/shm')
            fifo = Path(local.name) / 'input.gguf'
            os.mkfifo(fifo)
        except OSError as error:
            self.skipTest(f'FIFO memory observer unavailable: {error}')
        with local:
            process = subprocess.Popen([str(executable), str(fifo), str(self.root / 'probe.gguf'), 'q8_0'],
                                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            try:
                for _ in range(300):
                    if process.poll() is not None:
                        self.fail('quantizer exited before address-space observation')
                    try:
                        wait = Path(f'/proc/{process.pid}/wchan').read_text()
                        # Kernel versions expose different names for FIFO opens.
                        if any(name in wait for name in ('wait_for_partner', 'fifo', 'pipe_wait')):
                            status = Path(f'/proc/{process.pid}/status').read_text()
                            return int(status.split('VmSize:')[1].split()[0]) * 1024
                    except (OSError, IndexError, ValueError) as error:
                        self.skipTest(f'procfs memory observer unavailable: {error}')
                    time.sleep(0.01)
                self.skipTest('kernel FIFO wait symbol unavailable')
            finally:
                process.kill()
                process.communicate()

    @unittest.skipUnless(sys.platform.startswith('linux'), 'Linux address-space control')
    def test_bounded_memory(self):
        startup = max(self.startup_address_space(binary) for binary in (QUANT, BASELINE) if binary is not None)
        if startup > 1024 * 1024 * 1024:
            self.skipTest('large reserved address space (e.g. sanitizer); use ordinary Linux build')
        self.address_limit = startup + 32 * 1024 * 1024
        fixture(self.source, 32769)  # 64 MiB + 2 rows; old writer buffers the entire bank.
        original = digest(self.source)
        for mode in ('q4_k', 'f16', 'f32'):
            with self.subTest(mode=mode):
                result = self.run_quant(QUANT, self.source, self.root / 'bounded.gguf', mode, limit=True)
                self.assertEqual(result.returncode, 0, result.stderr.decode())
                if BASELINE is not None:
                    old = self.run_quant(BASELINE, self.source, self.root / 'old.gguf', mode, limit=True)
                    self.assertNotEqual(old.returncode, 0, 'memory control did not reject the old writer')
        self.assertEqual(digest(self.source), original)


if __name__ == '__main__':
    unittest.main()
