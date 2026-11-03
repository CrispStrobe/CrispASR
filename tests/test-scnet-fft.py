#!/usr/bin/env python3
"""Independent numerical and exact-dispatch controls for SCNet's native FFT.

NumPy supplies the independent transform oracle. Scalar/SIMD byte comparisons
check the declared unchanged arithmetic order, including tails, signed zeros,
scratch reuse and sharing an immutable plan between workers. No checkpoint or
timing measurement is involved.
"""
import ctypes
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
POINTER = ctypes.POINTER(ctypes.c_float)
WRAPPER = r'''
#include "scnet_fft.h"
#include <limits>
#include <thread>

extern "C" {
const char* fft_kernel(int scalar) {
    static const scnet_fft::Plan automatic(64);
    static const scnet_fft::Plan fallback(64, scnet_fft::Implementation::Scalar);
    return scalar ? fallback.kernel_name() : automatic.kernel_name();
}
int fft_expected_implementation() {
#if SCNET_FFT_X86
    if (__builtin_cpu_supports("avx2")) return 1;
#endif
#if SCNET_FFT_NEON
    return 2;
#else
    return 0;
#endif
}
int fft_transform(float* re, float* im, int n, int rows, int scalar, int poison, int workers) {
    try {
        const scnet_fft::Plan plan(n, scalar ? scnet_fft::Implementation::Scalar
                                            : scnet_fft::Implementation::Automatic);
        auto apply = [&](int worker) {
            scnet_fft::Plan::Scratch scratch;
            for (int row = worker; row < rows; row += workers) {
                if (poison) {
                    // Oversized scratch must be shrunk and every used element
                    // initialized correctly, including reused convolution tails.
                    const float value = std::numeric_limits<float>::quiet_NaN();
                    scratch.ar.assign(4*n, value);
                    scratch.ai.assign(4*n, value);
                    scratch.br.assign(4*n, value);
                    scratch.bi.assign(4*n, value);
                }
                plan.transform(re + row*n, im + row*n, scratch);
            }
        };
        if (workers == 1) apply(0);
        else {
            std::vector<std::thread> pool;
            for (int worker = 0; worker < workers; ++worker) pool.emplace_back(apply, worker);
            for (auto& thread : pool) thread.join();
        }
        return 0;
    } catch (...) { return -1; }
}
void fft_pointwise(const float* re, const float* im, const float* cr, const float* ci,
                   float* out_r, float* out_i, int n, int operation, int scalar) {
    const auto& kernels = scalar ? scnet_fft::detail::scalar_kernels() : scnet_fft::detail::fastest_kernels();
    if (operation == 0) kernels.input(re, im, cr, ci, out_r, out_i, n);
    if (operation == 1) {
        std::copy_n(re, n, out_r);
        std::copy_n(im, n, out_i);
        kernels.product(out_r, out_i, cr, ci, n);
    }
    if (operation == 2) kernels.output(re, im, cr, ci, out_r, out_i, n, 1.0f/2048);
}
}
'''


class NativeFFTTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix='scnet-fft-')
        directory = Path(cls.temporary.name)
        source, library = directory/'wrapper.cpp', directory/'wrapper.so'
        source.write_text(WRAPPER)
        cls.command = [os.environ.get('CXX', 'c++'), '-std=c++17', '-O3', '-Wall', '-Wextra', '-Werror',
                       '-shared', '-fPIC', '-pthread', '-I', str(ROOT/'src'), str(source), '-o', str(library)]
        subprocess.run(cls.command, check=True)
        cls.lib = ctypes.CDLL(str(library.resolve()))
        cls.lib.fft_kernel.argtypes, cls.lib.fft_kernel.restype = [ctypes.c_int], ctypes.c_char_p
        cls.lib.fft_expected_implementation.argtypes, cls.lib.fft_expected_implementation.restype = [], ctypes.c_int
        cls.lib.fft_transform.argtypes = [POINTER, POINTER] + [ctypes.c_int]*5
        cls.lib.fft_transform.restype = ctypes.c_int
        cls.lib.fft_pointwise.argtypes = [POINTER]*6 + [ctypes.c_int]*3
        cls.lib.fft_pointwise.restype = None
        print('SCNet FFT selected kernel:', cls.lib.fft_kernel(0).decode(), flush=True)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def transformed(self, values, scalar=0, poison=0, workers=1):
        values = np.asarray(values)
        if values.ndim == 1:
            values = values[None, :]
        rows, n = values.shape
        # Deliberately unaligned pointers with guards on both ends.
        arrays = [np.full(rows*n+2, 123456.0, dtype=np.float32) for _ in range(2)]
        arrays[0][1:-1], arrays[1][1:-1] = values.real.reshape(-1), values.imag.reshape(-1)
        rc = self.lib.fft_transform(*[a[1:-1].ctypes.data_as(POINTER) for a in arrays],
                                    n, rows, scalar, poison, workers)
        self.assertEqual(rc, 0)
        for array in arrays:
            np.testing.assert_array_equal(array[[0, -1]], [123456.0, 123456.0])
        return tuple(a[1:-1].reshape(rows, n).copy() for a in arrays)

    def test_dispatch_and_invalid_lengths(self):
        expected = (b'scalar', b'avx2', b'neon')[self.lib.fft_expected_implementation()]
        self.assertEqual(self.lib.fft_kernel(0), expected)
        self.assertEqual(self.lib.fft_kernel(1), b'scalar')
        for n in (-1, 0, (1 << 20)+1):
            self.assertEqual(self.lib.fft_transform(None, None, n, 1, 0, 0, 1), -1)

    def test_independent_numpy_transform(self):
        lengths = tuple(range(1, 66)) + (97, 127, 128, 129, 193, 194, 195, 257, 307,
                                         512, 614, 872, 1024, 1536, 2048, 4096)
        rng = np.random.default_rng(614872)
        worst_absolute, worst_relative = (0., 0), (0., 0)
        for n in lengths:
            values = (rng.normal(0, .5, (3, n)) + 1j*rng.normal(0, .5, (3, n))).astype(np.complex64)
            values[1] = 0
            values[1, n//3] = 1-0.5j
            # A unit-energy tone, like the impulse, keeps the fixed absolute
            # oracle tolerance meaningful as the transform length increases.
            values[2] = (np.exp(2j*np.pi*3*np.arange(n)/n)/np.sqrt(n)).astype(np.complex64)
            expected = np.fft.fft(values.astype(np.complex128), axis=1)
            for scalar in (0, 1):
                real, imag = self.transformed(values, scalar=scalar)
                actual = real.astype(np.float64) + 1j*imag.astype(np.float64)
                np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=8e-5)
                absolute = float(np.max(np.abs(actual-expected)))
                relative = float(np.linalg.norm(actual-expected)/np.linalg.norm(expected))
                worst_absolute = max(worst_absolute, (absolute, n))
                worst_relative = max(worst_relative, (relative, n))
        print(f'SCNet FFT NumPy oracle: {len(lengths)} lengths, both dispatches; '
              f'max_abs={worst_absolute[0]:.12g} at n={worst_absolute[1]}, '
              f'max_relative_l2={worst_relative[0]:.12g} at n={worst_relative[1]}', flush=True)

    def test_scalar_simd_byte_identity_and_dirty_scratch(self):
        lengths = (1, 2, 3, 4, 7, 8, 9, 12, 16, 17, 31, 32, 33, 64, 65, 97,
                   127, 128, 194, 257, 512, 614, 872, 1024, 1536, 2048, 4096)
        rng = np.random.default_rng(914)
        compared = 0
        for n in lengths:
            real = rng.normal(0, 2, (7, n)).astype(np.float32)
            imag = rng.normal(0, 2, (7, n)).astype(np.float32)
            real[1:3] *= np.float32(2.0**-120)
            imag[1:3] *= np.float32(2.0**-120)
            real[3:5] = 0
            imag[3:5] = 0
            real[4] = -0.0
            imag[3] = -0.0
            values = np.empty((7, n), dtype=np.complex64)
            values.real, values.imag = real, imag
            scalar = self.transformed(values, scalar=1)
            for poison in (0, 1):
                automatic = self.transformed(values, poison=poison)
                for actual, expected in zip(automatic, scalar):
                    np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))
                compared += len(values)
        print(f'SCNet FFT scalar/SIMD bit identity: {compared} complex transforms, '
              'including poisoned scratch reuse and signed/subnormal values', flush=True)

    def test_pointwise_tails_and_guards(self):
        rng = np.random.default_rng(831)
        for n in (1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 65, 194, 614, 872):
            inputs = [rng.normal(0, 2, n).astype(np.float32) for _ in range(4)]
            inputs[0][0], inputs[1][-1] = -0.0, -0.0
            for operation in range(3):
                results = []
                for scalar in (0, 1):
                    outputs = [np.full(n+2, 123456., dtype=np.float32) for _ in range(2)]
                    self.lib.fft_pointwise(*[a.ctypes.data_as(POINTER) for a in inputs],
                                           *[a[1:-1].ctypes.data_as(POINTER) for a in outputs],
                                           n, operation, scalar)
                    for a in outputs:
                        np.testing.assert_array_equal(a[[0, -1]], [123456., 123456.])
                    results.append(outputs)
                for actual, expected in zip(results[0], results[1]):
                    np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))

    def test_immutable_plan_parallel_workers(self):
        rng = np.random.default_rng(2048)
        for n in (194, 614, 872, 4096):
            values = (rng.normal(0, 1, (11, n)) + 1j*rng.normal(0, 1, (11, n))).astype(np.complex64)
            expected = self.transformed(values, workers=1)
            for workers in (2, 4):
                actual = self.transformed(values, workers=workers)
                for a, b in zip(actual, expected):
                    np.testing.assert_array_equal(a.view(np.uint32), b.view(np.uint32))


if __name__ == '__main__':
    unittest.main()
