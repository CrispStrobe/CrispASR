#!/usr/bin/env python3
"""Exercise SCNet's actual native FFT, Python RNG and recurrent kernels.

These controls use NumPy's independent FFT/matrix implementation and CPython's
random module. They require no checkpoint and do not establish model fidelity.
"""

import ctypes
import pathlib
import random
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np


ROOT = pathlib.Path(__file__).resolve().parents[1]
POINTER = ctypes.POINTER(ctypes.c_float)

WRAPPER = r"""
#include "scnet_fft.h"
#include "scnet_shift.h"
#include "scnet_lstm.h"
extern "C" {
int scnet_math_fft(float* real, float* imag, int n) {
    try {
        scnet_fft::Plan plan(n);
        scnet_fft::Plan::Scratch scratch;
        plan.transform(real, imag, scratch);
        return 0;
    } catch (...) { return -1; }
}
void scnet_math_rng(uint64_t seed, int upper, int n, int* result) {
    scnet_detail::PythonRandom random(seed);
    for (int i = 0; i < n; ++i) result[i] = random.randint(upper);
}
void scnet_math_lstm(const float* pf, const float* pr, const float* wf, const float* wr,
                     const float* bf, const float* br, float* output,
                     int hidden, int batch, int steps, int threads) {
    for (int worker = 0; worker < threads; ++worker) {
        const int begin = int(int64_t(batch) * worker / threads);
        const int end = int(int64_t(batch) * (worker + 1) / threads);
        if (begin == end) continue;
        scnet_detail::LstmScratch scratch(end - begin, hidden);
        scnet_detail::lstm_direction(pf, wf, bf, output, hidden, batch, steps,
                                    false, begin, end, false, scratch);
        scnet_detail::lstm_direction(pr, wr, br, output, hidden, batch, steps,
                                    true, begin, end, false, scratch);
    }
}
}
"""

DISPATCH_WRAPPER = r"""
#include "scnet.cpp"
#include <cassert>

std::vector<float> exercise(bool blas, int requested, int actual) {
    constexpr int h = 3, b = 7, t = 5;
    std::array<std::vector<float>, 6> inputs;
    for (int i = 0; i < 6; ++i) {
        const int n = i < 2 ? t*b*4*h : i < 4 ? 4*h*h : 4*h;
        inputs[i].resize(n);
        for (int j = 0; j < n; ++j) inputs[i][j] = float((j*7+i*11)%23-11)/50;
    }
    std::array<ggml_tensor, 6> source{};
    ggml_tensor output{};
    std::vector<float> values(t*b*2*h, std::numeric_limits<float>::quiet_NaN());
    output.ne[0] = 2*h;
    output.ne[1] = b;
    output.ne[2] = t;
    output.ne[3] = 1;
    output.data = values.data();
    for (int i = 0; i < 6; ++i) {
        source[i].data = inputs[i].data();
        output.src[i] = &source[i];
    }
    RecurrentNode node(b, h, requested, blas);
    for (int ith = 0; ith < actual; ++ith) recurrent_compute(&output, ith, actual, &node);
    for (float v : values) assert(std::isfinite(v));
    return values;
}

int main() {
    const auto expected = exercise(false, 1, 1);
    // The production CBLAS dispatch branch is tested without a BLAS dependency:
    // the header retains portable arithmetic, while the single scratch slot and
    // actual production callback follow the accelerated worker ownership path.
    assert(exercise(true, 4, 1) == expected);
    assert(exercise(true, 4, 4) == expected);
    // Emscripten/WASI can reduce a requested worker budget to one worker.
    assert(exercise(false, 8, 1) == expected);
    assert(exercise(false, 8, 2) == expected);
    assert(exercise(false, 8, 8) == expected);
}
"""


class SCNetMath(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        compiler = shutil.which("c++") or shutil.which("g++") or shutil.which("clang++")
        if not compiler:
            raise RuntimeError("SCNet math acceptance requires a C++ compiler")
        cls.directory = tempfile.TemporaryDirectory(prefix=".scnet-math-", dir=ROOT)
        folder = pathlib.Path(cls.directory.name)
        source = folder / "probe.cpp"
        source.write_text(WRAPPER)
        library = folder / ("probe.dylib" if __import__("sys").platform == "darwin" else "probe.so")
        subprocess.run(
            [compiler, "-std=c++17", "-O2", "-fPIC", "-shared", "-I", str(ROOT / "src"),
             str(source), "-o", str(library)],
            check=True, capture_output=True, text=True,
        )
        cls.lib = ctypes.CDLL(str(library))
        cls.lib.scnet_math_fft.argtypes = [POINTER, POINTER, ctypes.c_int]
        cls.lib.scnet_math_fft.restype = ctypes.c_int
        cls.lib.scnet_math_rng.argtypes = [ctypes.c_uint64, ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
        cls.lib.scnet_math_lstm.argtypes = [POINTER] * 7 + [ctypes.c_int] * 4
        dispatch_source = folder / "dispatch.cpp"
        dispatch_source.write_text(DISPATCH_WRAPPER)
        cls.dispatch = folder / "dispatch"
        dead_strip = "-Wl,-dead_strip" if sys.platform == "darwin" else "-Wl,--gc-sections"
        subprocess.run(
            [compiler, "-std=c++17", "-O1", "-ffunction-sections", "-fdata-sections",
             "-I", str(ROOT / "src"), "-I", str(ROOT / "ggml" / "include"),
             str(dispatch_source), dead_strip, "-o", str(cls.dispatch)],
            check=True, capture_output=True, text=True,
        )

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def test_exact_length_fft_against_numpy(self):
        rng = np.random.default_rng(741)
        # Include pure radix-2, mixed radices, primes, and the prime factors
        # produced by real 20-second SCNet shifted/chunked frame lengths.
        for n in (1, 2, 3, 4, 6, 10, 16, 18, 30, 42, 64, 90, 97, 100, 194, 886, 914, 1024, 4096):
            with self.subTest(n=n):
                real = rng.normal(size=n).astype(np.float32)
                imag = rng.normal(size=n).astype(np.float32)
                expected = np.fft.fft(real.astype(np.float64) + 1j * imag)
                code = self.lib.scnet_math_fft(real.ctypes.data_as(POINTER), imag.ctypes.data_as(POINTER), n)
                self.assertEqual(code, 0)
                actual = real.astype(np.float64) + 1j * imag
                self.assertLess(np.linalg.norm(actual - expected) / np.linalg.norm(expected), 2e-6)
                np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=8e-5)

    def test_python_integer_seed_and_randint_sequence(self):
        for seed in (0, 1, 42, 2**32 - 1, 2**32, 2**63 + 17, 2**64 - 1):
            for upper in (0, 1, 17, 22050, 65535):
                with self.subTest(seed=seed, upper=upper):
                    actual = (ctypes.c_int * 1000)()
                    self.lib.scnet_math_rng(seed, upper, len(actual), actual)
                    generator = random.Random(seed)
                    expected = [generator.randint(0, upper) for _ in actual]
                    self.assertEqual(list(actual), expected)

    def test_lstm_gate_order_reverse_and_partition(self):
        rng = np.random.default_rng(395)
        for hidden, batch, steps in ((3, 1, 1), (7, 5, 9), (16, 7, 13)):
            with self.subTest(hidden=hidden, batch=batch, steps=steps):
                projections = [rng.normal(0, 0.3, (steps, batch, 4 * hidden)).astype(np.float32) for _ in range(2)]
                weights = [rng.normal(0, 0.1, (4 * hidden, hidden)).astype(np.float32) for _ in range(2)]
                biases = [rng.normal(0, 0.1, 4 * hidden).astype(np.float32) for _ in range(2)]
                expected = np.zeros((steps, batch, 2 * hidden), np.float32)
                for direction in range(2):
                    h = np.zeros((batch, hidden), np.float32)
                    c = np.zeros_like(h)
                    order = range(steps - 1, -1, -1) if direction else range(steps)
                    for t in order:
                        gates = (h @ weights[direction].T + biases[direction]) + projections[direction][t]
                        i, f, g, o = np.split(gates, 4, axis=1)
                        i = 1 / (1 + np.exp(-i))
                        f = 1 / (1 + np.exp(-f))
                        o = 1 / (1 + np.exp(-o))
                        c = f * c + i * np.tanh(g)
                        h = o * np.tanh(c)
                        expected[t, :, direction * hidden:(direction + 1) * hidden] = h
                reference_output = None
                arrays = projections + weights + biases
                for threads in (1, 2, 4, 11):
                    actual = np.full(expected.shape, np.nan, np.float32)
                    self.lib.scnet_math_lstm(
                        *[array.ctypes.data_as(POINTER) for array in arrays], actual.ctypes.data_as(POINTER),
                        hidden, batch, steps, threads,
                    )
                    np.testing.assert_allclose(actual, expected, rtol=3e-6, atol=1e-7)
                    if reference_output is not None:
                        np.testing.assert_array_equal(actual, reference_output)
                    reference_output = actual

    def test_actual_callback_worker_ownership_and_effective_thread_count(self):
        subprocess.run([str(self.dispatch)], check=True, capture_output=True, text=True)


if __name__ == "__main__":
    unittest.main()
