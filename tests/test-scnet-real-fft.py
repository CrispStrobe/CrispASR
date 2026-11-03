#!/usr/bin/env python3
"""Check production two-real feature FFTs against independent NumPy transforms.

The original normalized rfft/irfft algebra, feature-plane ordering, learned
endpoint semantics and worker ownership are checked without a checkpoint.
These controls do not establish model fidelity or throughput.
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
SENTINEL = np.float32(123456.)
WRAPPER = r'''
#include "scnet_real_fft.h"
#include <limits>
#include <thread>

extern "C" {
const char* real_fft_kernel() {
    static const scnet_fft::Plan plan(64);
    return plan.kernel_name();
}
int real_fft_rows(const float* input, float* output, int n, int frequencies, int channels,
                  int inverse, int workers, int partial_worker, int poison, int scalar) {
    try {
        const scnet_fft::Plan plan(n, scalar ? scnet_fft::Implementation::Scalar
                                            : scnet_fft::Implementation::Automatic);
        auto apply = [&](int worker) {
            std::vector<float> real(n), imag(n);
            scnet_fft::Plan::Scratch scratch;
            if (poison) {
                const float value = std::numeric_limits<float>::quiet_NaN();
                std::fill(real.begin(), real.end(), value);
                std::fill(imag.begin(), imag.end(), value);
                scratch.ar.assign(4*n, value);
                scratch.ai.assign(4*n, value);
                scratch.br.assign(4*n, value);
                scratch.bi.assign(4*n, value);
            }
            scnet_fft::transform_real_rows(plan, inverse != 0, input, output, frequencies, channels,
                                           worker, workers, real.data(), imag.data(), scratch);
        };
        if (partial_worker >= 0) apply(partial_worker);
        else if (workers == 1) apply(0);
        else {
            std::vector<std::thread> pool;
            for (int worker = 0; worker < workers; ++worker) pool.emplace_back(apply, worker);
            for (auto& thread : pool) thread.join();
        }
        return 0;
    } catch (...) { return -1; }
}
}
'''


class RealFFTTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix='scnet-real-fft-')
        directory = Path(cls.temporary.name)
        source, library = directory/'wrapper.cpp', directory/'wrapper.so'
        source.write_text(WRAPPER)
        subprocess.run([os.environ.get('CXX', 'c++'), '-std=c++17', '-O3', '-Wall', '-Wextra', '-Werror',
                        '-shared', '-fPIC', '-pthread', '-I', str(ROOT/'src'), str(source), '-o', str(library)],
                       check=True)
        cls.lib = ctypes.CDLL(str(library.resolve()))
        cls.lib.real_fft_kernel.argtypes, cls.lib.real_fft_kernel.restype = [], ctypes.c_char_p
        cls.lib.real_fft_rows.argtypes = [POINTER, POINTER] + [ctypes.c_int]*8
        cls.lib.real_fft_rows.restype = ctypes.c_int
        print('SCNet real-pair FFT selected kernel:', cls.lib.real_fft_kernel().decode(), flush=True)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def apply_rows(self, values, n, inverse=False, workers=1, partial_worker=-1, poison=True, scalar=False):
        values = np.asarray(values)
        channels, frequencies, _ = values.shape
        if inverse:
            data = np.stack((values.real, values.imag)).astype(np.float32).reshape(-1)
            shape = (channels, frequencies, n)
        else:
            data = np.asarray(values, dtype=np.float32).reshape(-1)
            shape = (2, channels, frequencies, n//2+1)
        # Exercise unaligned pointers and retain source/target guard values.
        source = np.full(data.size+2, SENTINEL, np.float32)
        source[1:-1] = data
        before = source.copy()
        target = np.full(int(np.prod(shape))+2, SENTINEL, np.float32)
        rc = self.lib.real_fft_rows(source[1:-1].ctypes.data_as(POINTER), target[1:-1].ctypes.data_as(POINTER),
                                   n, frequencies, channels, int(inverse), workers, partial_worker,
                                   int(poison), int(scalar))
        self.assertEqual(rc, 0)
        np.testing.assert_array_equal(source.view(np.uint32), before.view(np.uint32))
        np.testing.assert_array_equal(target[[0, -1]], [SENTINEL, SENTINEL])
        return target[1:-1].reshape(shape).copy()

    def check_oracle(self, actual, expected):
        np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=8e-5)
        delta = actual-expected
        absolute = float(np.max(np.abs(delta)))
        relative = float(np.linalg.norm(delta)/max(np.linalg.norm(expected), 1e-30))
        self.assertLess(relative, 2e-6)
        return absolute, relative

    def test_forward_numpy_rfft_and_feature_planes(self):
        rng = np.random.default_rng(842)
        maximum_absolute = maximum_relative = 0.
        cases = 0
        for n in (1, 2, 3, 4, 5, 8, 12, 17, 31, 64, 97, 194, 307, 614, 872):
            for channels, frequencies in ((1, 1), (1, 2), (2, 3), (3, 5)):
                values = rng.normal(0, .5, (channels, frequencies, n)).astype(np.float32)
                expected = np.fft.rfft(values.astype(np.float64), n=n, axis=-1, norm='ortho')
                planes = self.apply_rows(values, n)
                actual = planes[0].astype(np.float64)+1j*planes[1].astype(np.float64)
                absolute, relative = self.check_oracle(actual, expected)
                maximum_absolute, maximum_relative = max(maximum_absolute, absolute), max(maximum_relative, relative)
                np.testing.assert_array_equal(planes[1, ..., 0], 0.)
                if n % 2 == 0:
                    np.testing.assert_array_equal(planes[1, ..., -1], 0.)
                cases += 1
        print(f'SCNet paired rfft: {cases} geometries, max_abs={maximum_absolute:.12g}, '
              f'max_global_relative_l2={maximum_relative:.12g}', flush=True)

    def test_inverse_numpy_irfft_from_learned_spectra(self):
        rng = np.random.default_rng(614)
        maximum_absolute = maximum_relative = 0.
        cases = 0
        for n in (1, 2, 3, 4, 5, 8, 12, 17, 31, 64, 97, 194, 307, 614, 872):
            for channels, frequencies in ((1, 1), (1, 2), (2, 3), (3, 5)):
                shape = (channels, frequencies, n//2+1)
                values = (rng.normal(0, .5, shape)+1j*rng.normal(0, .5, shape)).astype(np.complex64)
                # Learned imaginary endpoint values are not necessarily zero.
                values.imag[..., 0] = 12345.
                if n % 2 == 0:
                    values.imag[..., -1] = -6789.
                expected = np.fft.irfft(values.astype(np.complex128), n=n, axis=-1, norm='ortho')
                actual = self.apply_rows(values, n, inverse=True).astype(np.float64)
                absolute, relative = self.check_oracle(actual, expected)
                maximum_absolute, maximum_relative = max(maximum_absolute, absolute), max(maximum_relative, relative)
                cases += 1
        print(f'SCNet paired irfft: {cases} geometries, max_abs={maximum_absolute:.12g}, '
              f'max_global_relative_l2={maximum_relative:.12g}', flush=True)

    def test_endpoint_imaginary_values_are_ignored_and_odd_high_bin_is_retained(self):
        rng = np.random.default_rng(81)
        for n in (1, 2, 3, 12, 17, 194, 614, 872):
            shape = (3, 5, n//2+1)
            values = (rng.normal(0, .5, shape)+1j*rng.normal(0, .5, shape)).astype(np.complex64)
            changed = values.copy()
            changed.imag[..., 0] = -98765.
            if n % 2 == 0:
                changed.imag[..., -1] = 54321.
            a = self.apply_rows(values, n, inverse=True, workers=4)
            b = self.apply_rows(changed, n, inverse=True, workers=4)
            np.testing.assert_array_equal(a.view(np.uint32), b.view(np.uint32))
            if n > 1 and n % 2:
                changed.imag[..., -1] += 10.
                c = self.apply_rows(changed, n, inverse=True)
                self.assertGreater(float(np.max(np.abs(c-b))), 1.)
                self.check_oracle(c.astype(np.float64),
                                  np.fft.irfft(changed.astype(np.complex128), n=n, norm='ortho'))

    def test_canonical_roundtrips_and_odd_frequency_tails(self):
        rng = np.random.default_rng(872)
        maximum_absolute = maximum_relative = 0.
        for n in (194, 614, 872):
            for frequencies in (1, 2, 3, 57, 85):
                values = rng.normal(0, .5, (2, frequencies, n)).astype(np.float32)
                planes = self.apply_rows(values, n, workers=2)
                spectrum = planes[0].astype(np.complex64)
                spectrum.imag = planes[1]
                actual = self.apply_rows(spectrum, n, inverse=True, workers=2).astype(np.float64)
                absolute, relative = self.check_oracle(actual, values.astype(np.float64))
                maximum_absolute, maximum_relative = max(maximum_absolute, absolute), max(maximum_relative, relative)
        print(f'SCNet real-pair canonical roundtrip: max_abs={maximum_absolute:.12g}, '
              f'max_global_relative_l2={maximum_relative:.12g}', flush=True)

    def test_widely_unequal_pair_amplitudes(self):
        rng = np.random.default_rng(57)
        amplitudes = ((1., 0.), (0., 1.), (1., 2.**-10), (2.**-10, 1.),
                      (1., 2.**-20), (2.**-20, 1.))
        for inverse in (False, True):
            maximum_absolute = maximum_relative = maximum_quiet_relative = maximum_zero_rms = 0.
            for n in (194, 614, 872):
                width = n//2+1 if inverse else n
                shape = (len(amplitudes), 3, width)
                values = rng.normal(0, .5, shape).astype(np.float32)
                if inverse:
                    values = values.astype(np.complex64)
                    values.imag = rng.normal(0, .5, shape)
                for channel, pair in enumerate(amplitudes):
                    values[channel, 0] *= pair[0]
                    values[channel, 1] *= pair[1]
                    values[channel, 2] *= 2.**-12
                raw = self.apply_rows(values, n, inverse=inverse, workers=2)
                if inverse:
                    actual = raw.astype(np.float64)
                    expected = np.fft.irfft(values.astype(np.complex128), n=n, norm='ortho')
                else:
                    actual = raw[0].astype(np.float64)+1j*raw[1].astype(np.float64)
                    expected = np.fft.rfft(values.astype(np.float64), n=n, norm='ortho')
                absolute, relative = self.check_oracle(actual, expected)
                maximum_absolute, maximum_relative = max(maximum_absolute, absolute), max(maximum_relative, relative)
                for channel, pair in enumerate(amplitudes):
                    for frequency, amplitude in enumerate(pair):
                        error = actual[channel, frequency]-expected[channel, frequency]
                        if amplitude == 0:
                            maximum_zero_rms = max(maximum_zero_rms, float(np.sqrt(np.mean(np.abs(error)**2))))
                        elif amplitude < 1:
                            quiet_relative = float(np.linalg.norm(error)/np.linalg.norm(expected[channel, frequency]))
                            maximum_quiet_relative = max(maximum_quiet_relative, quiet_relative)
            print(f'SCNet unequal-pair {"irfft" if inverse else "rfft"}: max_abs={maximum_absolute:.12g}, '
                  f'max_global_relative_l2={maximum_relative:.12g}, '
                  f'max_nonzero_quiet_row_relative_l2={maximum_quiet_relative:.12g}, '
                  f'max_zero_row_rms={maximum_zero_rms:.12g}', flush=True)

    def test_worker_ownership_and_parallel_identity(self):
        rng = np.random.default_rng(7)
        ownership_checks = 0
        for n, channels, frequencies in ((17, 3, 1), (194, 2, 3), (614, 3, 5), (872, 2, 9)):
            for inverse in (False, True):
                width = n//2+1 if inverse else n
                values = rng.normal(0, .5, (channels, frequencies, width)).astype(np.float32)
                if inverse:
                    values = values.astype(np.complex64)
                    values.imag = rng.normal(0, .5, values.shape)
                expected = self.apply_rows(values, n, inverse=inverse)
                for workers in (2, 7):
                    actual = self.apply_rows(values, n, inverse=inverse, workers=workers)
                    np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))
                    owners = np.zeros(expected.shape, np.uint8)
                    for worker in range(workers):
                        partial = self.apply_rows(values, n, inverse=inverse, workers=workers, partial_worker=worker)
                        mask = partial != SENTINEL
                        self.assertTrue(np.all(owners[mask] == 0))
                        np.testing.assert_array_equal(partial[mask].view(np.uint32), expected[mask].view(np.uint32))
                        owners[mask] += 1
                        ownership_checks += 1
                    np.testing.assert_array_equal(owners, 1)
        print(f'SCNet real-pair exclusive worker ownership: {ownership_checks} partitions', flush=True)

    def test_scalar_and_selected_fft_implementation(self):
        self.assertIn(self.lib.real_fft_kernel(), (b'scalar', b'avx2', b'neon'))
        rng = np.random.default_rng(95)
        for n in (12, 194, 614, 872):
            for inverse in (False, True):
                width = n//2+1 if inverse else n
                values = rng.normal(0, .5, (3, 5, width)).astype(np.float32)
                if inverse:
                    values = values.astype(np.complex64)
                    values.imag = rng.normal(0, .5, values.shape)
                automatic = self.apply_rows(values, n, inverse=inverse)
                scalar = self.apply_rows(values, n, inverse=inverse, scalar=True)
                np.testing.assert_array_equal(automatic.view(np.uint32), scalar.view(np.uint32))


if __name__ == '__main__':
    unittest.main()
