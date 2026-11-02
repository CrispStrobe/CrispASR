#!/usr/bin/env python3
"""Exercise SCNet's actual native FFT, Python RNG and recurrent kernels.

These controls use NumPy's independent FFT/matrix implementation and CPython's
random module. They require no checkpoint and do not establish model fidelity.
"""

import ctypes
import os
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
#include <limits>

#if SCNET_ACT_X86
SCNET_ACT_AVX2 void inspect_avx2(const float* x, float* sigmoid, float* tanh, int n) {
    int k = 0;
    for (; k + 7 < n; k += 8) {
        const auto value = _mm256_loadu_ps(x + k);
        _mm256_storeu_ps(sigmoid + k, scnet_activation::sigmoid_avx2(value));
        _mm256_storeu_ps(tanh + k, scnet_activation::tanh_avx2(value));
    }
    for (; k < n; ++k) {
        sigmoid[k] = 1.0f / (1.0f + std::exp(-x[k]));
        tanh[k] = std::tanh(x[k]);
    }
}

SCNET_ACT_AVX512 void inspect_avx512(const float* x, float* sigmoid, float* tanh, int n) {
    int k = 0;
    for (; k + 15 < n; k += 16) {
        const auto value = _mm512_loadu_ps(x + k);
        _mm512_storeu_ps(sigmoid + k, scnet_activation::sigmoid_avx512(value));
        _mm512_storeu_ps(tanh + k, scnet_activation::tanh_avx512(value));
    }
    // Preserve the existing eight-lane and scalar treatment of every tail.
    inspect_avx2(x + k, sigmoid + k, tanh + k, n - k);
}
SCNET_ACT_AVX2 void inspect_exp_avx2(const float* x, float* output, int n) {
    int k = 0;
    for (; k + 7 < n; k += 8)
        _mm256_storeu_ps(output + k, scnet_activation::exp_avx2(_mm256_loadu_ps(x + k)));
    for (; k < n; ++k) output[k] = std::exp(x[k]);
}
SCNET_ACT_AVX512 void inspect_exp_avx512(const float* x, float* output, int n) {
    int k = 0;
    for (; k + 15 < n; k += 16)
        _mm512_storeu_ps(output + k, scnet_activation::exp_avx512(_mm512_loadu_ps(x + k)));
    inspect_exp_avx2(x + k, output + k, n - k);
}
#endif
#if SCNET_ACT_NEON
void inspect_neon(const float* x, float* sigmoid, float* tanh, int n) {
    int k = 0;
    for (; k + 3 < n; k += 4) {
        const auto value = vld1q_f32(x + k);
        vst1q_f32(sigmoid + k, scnet_activation::sigmoid_neon(value));
        vst1q_f32(tanh + k, scnet_activation::tanh_neon(value));
    }
    for (; k < n; ++k) {
        sigmoid[k] = 1.0f / (1.0f + std::exp(-x[k]));
        tanh[k] = std::tanh(x[k]);
    }
}
#endif
extern "C" {

int scnet_math_x86_supported(int lanes) {
#if SCNET_ACT_X86
    if (lanes == 16) return scnet_activation::avx512_supported();
    if (lanes == 8) return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
#endif
    return 0;
}
int scnet_math_x86_functions(const float* x, float* exponential, float* sigmoid,
                              float* tanh, int n, int lanes) {
    // Never enter a target-attributed function on an unsupported CPU/OS.
    if (!scnet_math_x86_supported(lanes)) return -1;
#if SCNET_ACT_X86
    if (lanes == 16) {
        inspect_exp_avx512(x, exponential, n);
        inspect_avx512(x, sigmoid, tanh, n);
    } else {
        inspect_exp_avx2(x, exponential, n);
        inspect_avx2(x, sigmoid, tanh, n);
    }
    return 0;
#else
    return -1;
#endif
}
int scnet_math_x86_update(const float* gates, float* cell, float* hidden,
                           float* output, int width, int lanes) {
    if (!scnet_math_x86_supported(lanes)) return -1;
#if SCNET_ACT_X86
    if (lanes == 16) scnet_activation::avx512_update(gates, cell, hidden, output, width);
    else scnet_activation::avx2_update(gates, cell, hidden, output, width);
    return 0;
#else
    return -1;
#endif
}
const char* scnet_math_gate_kernel() {
    return scnet_activation::selected_kernel().name;
}
void scnet_math_activations(const float* x, float* sigmoid, float* tanh,
                           float* scalar_sigmoid, float* scalar_tanh, int n) {
    for (int k = 0; k < n; ++k) {
        scalar_sigmoid[k] = 1.0f / (1.0f + std::exp(-x[k]));
        scalar_tanh[k] = std::tanh(x[k]);
    }
    const auto kernel = scnet_activation::fastest_kernel();
#if SCNET_ACT_X86
    if (std::strcmp(kernel.name, "avx512_fma") == 0) {
        inspect_avx512(x, sigmoid, tanh, n);
        return;
    }
    if (std::strcmp(kernel.name, "avx2_fma") == 0) {
        inspect_avx2(x, sigmoid, tanh, n);
        return;
    }
#endif
#if SCNET_ACT_NEON
    if (std::strcmp(kernel.name, "neon") == 0) {
        inspect_neon(x, sigmoid, tanh, n);
        return;
    }
#endif
    std::copy_n(scalar_sigmoid, n, sigmoid);
    std::copy_n(scalar_tanh, n, tanh);
}
void scnet_math_gate_update(const float* gates, float* cell, float* hidden,
                            float* output, int width, int scalar) {
    const auto apply = scalar ? scnet_activation::scalar_update : scnet_activation::fastest_kernel().update;
    apply(gates, cell, hidden, output, width);
}
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
void scnet_math_lstm_bias(const float* pf, const float* pr, const float* wf, const float* wr,
                          const float* bf, const float* br, const float* input_bf, const float* input_br,
                          float* output, int hidden, int batch, int steps, int threads, int profile) {
    for (int worker = 0; worker < threads; ++worker) {
        const int begin = int(int64_t(batch) * worker / threads);
        const int end = int(int64_t(batch) * (worker + 1) / threads);
        if (begin == end) continue;
        scnet_detail::LstmScratch scratch(end - begin, hidden);
        scnet_detail::LstmTiming timing;
        scnet_detail::lstm_direction(pf, wf, bf, output, hidden, batch, steps,
                                    false, begin, end, false, scratch, profile ? &timing : nullptr, input_bf);
        scnet_detail::lstm_direction(pr, wr, br, output, hidden, batch, steps,
                                    true, begin, end, false, scratch, profile ? &timing : nullptr, input_br);
    }
}
void scnet_math_lstm_strided(const float* pf, const float* pr, const float* wf, const float* wr,
                             const float* bf, const float* br, const float* input_bf, const float* input_br,
                             float* output, int hidden, int batch, int steps, int threads, int profile,
                             size_t projection_row_stride) {
    for (int worker = 0; worker < threads; ++worker) {
        const int begin = int(int64_t(batch) * worker / threads);
        const int end = int(int64_t(batch) * (worker + 1) / threads);
        if (begin == end) continue;
        scnet_detail::LstmScratch scratch(end - begin, hidden);
        scnet_detail::LstmTiming timing;
        scnet_detail::lstm_direction(pf, wf, bf, output, hidden, batch, steps,
                                    false, begin, end, false, scratch, profile ? &timing : nullptr,
                                    input_bf, projection_row_stride);
        scnet_detail::lstm_direction(pr, wr, br, output, hidden, batch, steps,
                                    true, begin, end, false, scratch, profile ? &timing : nullptr,
                                    input_br, projection_row_stride);
    }
}
}
"""

DISPATCH_WRAPPER = r"""
#include "scnet.cpp"
#include <cassert>

std::vector<float> exercise(bool blas, int requested, int actual, int projection_padding = -1) {
    constexpr int h = 3, b = 7, t = 5;
    std::array<std::vector<float>, 8> inputs;
    for (int i = 0; i < 8; ++i) {
        const int n = i < 2 ? t*b*4*h : i < 4 ? 4*h*h : 4*h;
        inputs[i].resize(n);
        for (int j = 0; j < n; ++j) inputs[i][j] = i >= 6 ? 0.0f : float((j*7+i*11)%23-11)/50;
    }
    std::array<ggml_tensor, 8> source{};
    ggml_tensor output{};
    std::vector<float> values(t*b*2*h, std::numeric_limits<float>::quiet_NaN());
    output.ne[0] = 2*h;
    output.ne[1] = b;
    output.ne[2] = t;
    output.ne[3] = 1;
    output.data = values.data();
    for (int i = 0; i < 8; ++i) {
        source[i].data = inputs[i].data();
        output.src[i] = &source[i];
    }
    for (int i = 0; i < 2; ++i) {
        source[i].nb[0] = sizeof(float);
        source[i].nb[1] = 4*h*sizeof(float);
        source[i].nb[2] = b*source[i].nb[1];
    }
    std::vector<float> joined;
    if (projection_padding >= 0) {
        const int stride = 8*h + projection_padding;
        joined.resize(t*b*stride, std::numeric_limits<float>::quiet_NaN());
        for (int row = 0; row < t*b; ++row)
            for (int direction = 0; direction < 2; ++direction)
                std::copy_n(inputs[direction].data() + row*4*h, 4*h,
                            joined.data() + row*stride + direction*4*h);
        for (int direction = 0; direction < 2; ++direction) {
            source[direction].data = joined.data() + direction*4*h;
            source[direction].nb[1] = stride*sizeof(float);
            source[direction].nb[2] = b*source[direction].nb[1];
        }
    }
    RecurrentNode node(b, h, requested, blas ? RecurrentMode::cblas : RecurrentMode::portable);
    for (int ith = 0; ith < actual; ++ith) recurrent_compute(&output, ith, actual, &node);
    for (float v : values) assert(std::isfinite(v));
    return values;
}

int main() {
    HParams small, large;
    small.dims = {4,32,64,128};
    large.dims = {4,64,128,256};
    // Exact grids from the two author configurations and pinned SDlayer math.
    // This calls the same production helper used by every encoder layer.
    const std::array<std::array<int,4>,3> small_borders = {{{0,359,1162,2049},
                                                         {0,108,350,616}, {0,33,106,186}}};
    const std::array<std::array<int,4>,3> large_borders = {{{0,462,1224,2049},
                                                         {0,159,421,705}, {0,55,146,243}}};
    for (const auto& boundaries : small_borders)
        assert(small.band_boundaries(boundaries[3]) == boundaries);
    for (const auto& boundaries : large_borders)
        assert(large.band_boundaries(boundaries[3]) == boundaries);
    // Promoting rounded GGUF F32 fractions would incorrectly produce 598.
    assert(large.band_boundaries(1000)[2] == 597);
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
    // The same production callback must consume offset directional views,
    // including extra poison padding, through their actual row strides.
    for (int padding : {0, 3}) {
        assert(exercise(true, 4, 1, padding) == expected);
        assert(exercise(true, 4, 4, padding) == expected);
        assert(exercise(false, 8, 1, padding) == expected);
        assert(exercise(false, 8, 2, padding) == expected);
        assert(exercise(false, 8, 8, padding) == expected);
    }
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
        cls.lib.scnet_math_lstm_bias.argtypes = [POINTER] * 9 + [ctypes.c_int] * 5
        cls.lib.scnet_math_lstm_strided.argtypes = [POINTER] * 9 + [ctypes.c_int] * 5 + [ctypes.c_size_t]
        cls.lib.scnet_math_gate_kernel.restype = ctypes.c_char_p
        cls.lib.scnet_math_activations.argtypes = [POINTER] * 5 + [ctypes.c_int]
        cls.lib.scnet_math_gate_update.argtypes = [POINTER] * 4 + [ctypes.c_int] * 2
        cls.lib.scnet_math_x86_supported.argtypes = [ctypes.c_int]
        cls.lib.scnet_math_x86_supported.restype = ctypes.c_int
        cls.lib.scnet_math_x86_functions.argtypes = [POINTER] * 4 + [ctypes.c_int] * 2
        cls.lib.scnet_math_x86_functions.restype = ctypes.c_int
        cls.lib.scnet_math_x86_update.argtypes = [POINTER] * 4 + [ctypes.c_int] * 2
        cls.lib.scnet_math_x86_update.restype = ctypes.c_int
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

    def test_activation_accuracy_zero_threshold_saturation_and_nonfinite(self):
        rng = np.random.default_rng(519)
        # Dense operating ranges plus arbitrary F32 bit patterns cover tiny
        # values, both Taylor boundaries, saturation, extremes and NaNs.
        bits = rng.integers(0, 2**32, 200_000, dtype=np.uint32).view(np.float32)
        powers = np.exp2(np.arange(-149, 128, dtype=np.float64)).astype(np.float32)
        values = np.concatenate([
            np.linspace(-16, 16, 131_073, dtype=np.float32),
            np.linspace(-.501, -.499, 16_385, dtype=np.float32),
            np.linspace(.499, .501, 16_385, dtype=np.float32),
            np.linspace(-90, -80, 16_385, dtype=np.float32),
            powers, -powers, bits,
            np.array([0., -0., np.inf, -np.inf, np.nan], dtype=np.float32),
        ])
        outputs = [np.empty_like(values) for _ in range(4)]
        self.lib.scnet_math_activations(
            values.ctypes.data_as(POINTER), *[v.ctypes.data_as(POINTER) for v in outputs], len(values),
        )
        for name, actual, expected in (('sigmoid', outputs[0], outputs[2]),
                                       ('tanh', outputs[1], outputs[3])):
            np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
            finite = np.isfinite(expected)
            self.assertTrue(np.isfinite(actual[finite]).all())
            a, b = actual[finite], expected[finite]
            np.testing.assert_array_equal(np.signbit(a), np.signbit(b))
            # Signed F32 integer distances measure ulps for equal-sign values.
            ulps = np.abs(a.view(np.int32).astype(np.int64) - b.view(np.int32).astype(np.int64))
            self.assertLessEqual(int(ulps.max(initial=0)), 4, name)
            self.assertLessEqual(float(np.max(np.abs(a.astype(np.float64)-b))), 2.5e-7, name)
            print(f'SCNet {self.lib.scnet_math_gate_kernel().decode()} {name}: '
                  f'{len(values)} inputs, max {int(ulps.max(initial=0))} ulps', flush=True)


    def require_avx512(self):
        if not self.lib.scnet_math_x86_supported(16):
            self.skipTest('CPU/OS lacks AVX512F, AVX512DQ, AVX2 or FMA; not execution-covered')
        self.assertEqual(self.lib.scnet_math_x86_supported(8), 1)

    def assert_x86_functions_equal(self, values):
        values = np.ascontiguousarray(values, dtype=np.float32)
        before = values.view(np.uint32).copy()
        results = []
        for lanes in (8, 16):
            arrays = [np.full(len(values) + 2, 12345.0, np.float32) for _ in range(3)]
            code = self.lib.scnet_math_x86_functions(
                values.ctypes.data_as(POINTER),
                *[a[1:-1].ctypes.data_as(POINTER) for a in arrays], len(values), lanes,
            )
            self.assertEqual(code, 0)
            for array in arrays:
                np.testing.assert_array_equal(array[[0, -1]], [12345.0, 12345.0])
            results.append([a[1:-1] for a in arrays])
        np.testing.assert_array_equal(values.view(np.uint32), before)
        for name, actual, expected in zip(('exp', 'sigmoid', 'tanh'), results[1], results[0]):
            with self.subTest(function=name, size=len(values)):
                np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
                # NaN payload/sign and floating exception flags are not the API
                # contract. Every other output bit, including signed zero and
                # infinity, must equal the existing eight-lane kernel.
                non_nan = ~np.isnan(expected)
                np.testing.assert_array_equal(actual[non_nan].view(np.uint32),
                                              expected[non_nan].view(np.uint32))

    def test_avx512_functions_match_avx2_bits(self):
        self.require_avx512()
        rng = np.random.default_rng(513)
        powers = np.exp2(np.arange(-149, 128, dtype=np.float64)).astype(np.float32)
        bits = rng.integers(0, 2**32, 200_000, dtype=np.uint32).view(np.float32)
        values = np.concatenate([
            np.linspace(-150, 150, 131_073, dtype=np.float32),
            np.linspace(-.501, -.499, 16_385, dtype=np.float32),
            np.linspace(.499, .501, 16_385, dtype=np.float32),
            np.linspace(-90, -80, 16_385, dtype=np.float32),
            powers, -powers, bits,
            np.array([0., -0., np.inf, -np.inf, np.nan], np.float32),
        ])
        self.assert_x86_functions_equal(values)
        for width in range(34):
            self.assert_x86_functions_equal(values[:width])
        # One extreme in either AVX2 half changes participation in exp's
        # exceptional-lane branch after widening. All other lanes must still
        # retain their exact results; test every possible extreme position.
        for special in (150., -150., np.inf, -np.inf, np.nan):
            for lane in range(16):
                mixed = np.linspace(-.6, .6, 16, dtype=np.float32)
                mixed[lane] = special
                self.assert_x86_functions_equal(mixed)
        print('SCNet AVX512 versus AVX2: activation/exp bits, tails and mixed lanes', flush=True)

    def test_avx512_gate_updates_match_avx2_bits_and_guards(self):
        self.require_avx512()
        rng = np.random.default_rng(514)
        for width in (0, 1, 7, 8, 9, 15, 16, 17, 23, 24, 31, 32, 33, 64, 128, 256, 512):
            with self.subTest(width=width):
                gates = rng.uniform(-16, 16, (4, width)).astype(np.float32)
                for plane in gates:
                    for k, value in enumerate((0., -0., .5, -.5, 90., -90.)):
                        if k < width:
                            plane[k] = value
                before = gates.view(np.uint32).copy()
                initial = rng.uniform(-8, 8, width).astype(np.float32)
                results = []
                for lanes in (8, 16):
                    arrays = [np.full(width + 2, 12345.0, np.float32) for _ in range(3)]
                    arrays[0][1:-1] = initial
                    code = self.lib.scnet_math_x86_update(
                        gates.ctypes.data_as(POINTER),
                        *[a[1:-1].ctypes.data_as(POINTER) for a in arrays], width, lanes,
                    )
                    self.assertEqual(code, 0)
                    for array in arrays:
                        np.testing.assert_array_equal(array[[0, -1]], [12345.0, 12345.0])
                    np.testing.assert_array_equal(arrays[1].view(np.uint32), arrays[2].view(np.uint32))
                    results.append(arrays)
                np.testing.assert_array_equal(gates.view(np.uint32), before)
                for actual, expected in zip(results[1], results[0]):
                    np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))

    def test_avx512_feedback_sequence_matches_avx2_bits(self):
        self.require_avx512()
        rng = np.random.default_rng(515)
        width, steps = 64, 872
        inputs = rng.normal(0, .3, (steps, 4, width)).astype(np.float32)
        initial = rng.normal(0, .3, width).astype(np.float32)
        results = []
        for lanes in (8, 16):
            cell = initial.copy()
            hidden = np.zeros(width, np.float32)
            output = np.empty_like(hidden)
            history = np.empty((steps, 2, width), np.float32)
            for step in range(steps):
                feedback = np.multiply(hidden, np.float32(.25), dtype=np.float32)
                gates = np.add(inputs[step], feedback, dtype=np.float32)
                code = self.lib.scnet_math_x86_update(
                    gates.ctypes.data_as(POINTER), cell.ctypes.data_as(POINTER),
                    hidden.ctypes.data_as(POINTER), output.ctypes.data_as(POINTER), width, lanes,
                )
                self.assertEqual(code, 0)
                np.testing.assert_array_equal(hidden.view(np.uint32), output.view(np.uint32))
                history[step, 0], history[step, 1] = cell, hidden
            results.append(history)
        np.testing.assert_array_equal(results[1].view(np.uint32), results[0].view(np.uint32))
        print('SCNet AVX512 versus AVX2: all cell/hidden bits across 872 feedback steps', flush=True)

    def test_gate_state_update_tails_guards_and_scalar_override(self):
        rng = np.random.default_rng(902)
        for width in (1, 3, 7, 8, 9, 16, 31, 32, 128, 256):
            gates = rng.uniform(-12, 12, (4, width)).astype(np.float32)
            initial = rng.uniform(-8, 8, width).astype(np.float32)
            results = []
            for scalar in (1, 0):
                arrays = [np.full(width+2, 12345.0, np.float32) for _ in range(3)]
                arrays[0][1:-1] = initial
                self.lib.scnet_math_gate_update(
                    gates.ctypes.data_as(POINTER),
                    *[v[1:-1].ctypes.data_as(POINTER) for v in arrays], width, scalar,
                )
                for v in arrays:
                    np.testing.assert_array_equal(v[[0, -1]], [12345.0, 12345.0])
                np.testing.assert_array_equal(arrays[1], arrays[2])
                results.append(arrays)
            for actual, expected in zip(results[1], results[0]):
                np.testing.assert_allclose(actual, expected, rtol=4e-6, atol=1e-6)
        previous = os.environ.get('CRISPASR_SCNET_SCALAR_GATES')
        try:
            os.environ['CRISPASR_SCNET_SCALAR_GATES'] = '0'
            fastest = self.lib.scnet_math_gate_kernel()
            self.assertIn(fastest, (b'avx512_fma', b'avx2_fma', b'neon', b'scalar'))
            os.environ['CRISPASR_SCNET_SCALAR_GATES'] = '1'
            self.assertEqual(self.lib.scnet_math_gate_kernel(), b'scalar')
        finally:
            if previous is None:
                os.environ.pop('CRISPASR_SCNET_SCALAR_GATES', None)
            else:
                os.environ['CRISPASR_SCNET_SCALAR_GATES'] = previous

    def test_long_recurrence_simd_against_scalar_gates(self):
        rng = np.random.default_rng(202)
        hidden, batch, steps = 16, 5, 872
        projections = [rng.normal(0, .3, (steps, batch, 4*hidden)).astype(np.float32) for _ in range(2)]
        weights = [rng.normal(0, .04, (4*hidden, hidden)).astype(np.float32) for _ in range(2)]
        biases = [rng.normal(0, .07, 4*hidden).astype(np.float32) for _ in range(2)]
        results = []
        previous = os.environ.get('CRISPASR_SCNET_SCALAR_GATES')
        try:
            for override in ('1', '0'):
                os.environ['CRISPASR_SCNET_SCALAR_GATES'] = override
                output = np.empty((steps, batch, 2*hidden), dtype=np.float32)
                self.lib.scnet_math_lstm(
                    *[v.ctypes.data_as(POINTER) for v in projections+weights+biases],
                    output.ctypes.data_as(POINTER), hidden, batch, steps, 2,
                )
                results.append(output)
        finally:
            if previous is None:
                os.environ.pop('CRISPASR_SCNET_SCALAR_GATES', None)
            else:
                os.environ['CRISPASR_SCNET_SCALAR_GATES'] = previous
        error = results[1].astype(np.float64)-results[0]
        self.assertLess(float(np.linalg.norm(error)/np.linalg.norm(results[0].astype(np.float64))), 2e-6)
        self.assertLess(float(np.max(np.abs(error))), 2e-6)

    def test_input_bias_fusion_is_bit_exact_for_both_directions(self):
        rng = np.random.default_rng(821)
        previous = os.environ.get('CRISPASR_SCNET_SCALAR_GATES')
        try:
            for hidden in (7, 17, 128, 256, 512):
                batch, steps = 3, 4
                projections = [rng.normal(0, .3, (steps, batch, 4*hidden)).astype(np.float32)
                               for _ in range(2)]
                weights = [rng.normal(0, .025, (4*hidden, hidden)).astype(np.float32)
                           for _ in range(2)]
                biases = [rng.normal(0, .07, 4*hidden).astype(np.float32) for _ in range(2)]
                input_biases = [rng.normal(0, .09, 4*hidden).astype(np.float32) for _ in range(2)]
                for direction in range(2):
                    # This cancellation would lose recurrent_bias if the two
                    # biases were pre-summed or the three adds reassociated.
                    projections[direction][:, :, 2*hidden] = np.float32(2**26)
                    input_biases[direction][2*hidden] = np.float32(-2**26)
                    biases[direction][2*hidden] = np.float32(.125)
                preadded = [np.add(p, b, dtype=np.float32)
                            for p, b in zip(projections, input_biases)]
                for override in ('1', '0'):
                    os.environ['CRISPASR_SCNET_SCALAR_GATES'] = override
                    expected = np.full((steps, batch, 2*hidden), np.nan, np.float32)
                    self.lib.scnet_math_lstm(
                        *[a.ctypes.data_as(POINTER) for a in preadded+weights+biases],
                        expected.ctypes.data_as(POINTER), hidden, batch, steps, 1,
                    )
                    self.assertTrue(np.isfinite(expected).all())
                    for threads, profile in ((1, 0), (5, 0), (5, 1)):
                        with self.subTest(hidden=hidden, scalar=override, threads=threads, profile=profile):
                            guarded = np.full(expected.size+2, 12345.0, np.float32)
                            actual = guarded[1:-1].reshape(expected.shape)
                            self.lib.scnet_math_lstm_bias(
                                *[a.ctypes.data_as(POINTER)
                                  for a in projections+weights+biases+input_biases],
                                actual.ctypes.data_as(POINTER), hidden, batch, steps, threads, profile,
                            )
                            np.testing.assert_array_equal(guarded[[0, -1]], [12345.0, 12345.0])
                            np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))
        finally:
            if previous is None:
                os.environ.pop('CRISPASR_SCNET_SCALAR_GATES', None)
            else:
                os.environ['CRISPASR_SCNET_SCALAR_GATES'] = previous

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

    def test_joined_projection_views_are_bit_exact(self):
        rng = np.random.default_rng(823)
        previous = os.environ.get('CRISPASR_SCNET_SCALAR_GATES')
        try:
            for hidden in (7, 17, 128, 256, 512):
                batch, steps = 3, 3
                projections = [rng.normal(0, .3, (steps, batch, 4*hidden)).astype(np.float32)
                               for _ in range(2)]
                weights = [rng.normal(0, .025, (4*hidden, hidden)).astype(np.float32)
                           for _ in range(2)]
                biases = [rng.normal(0, .07, 4*hidden).astype(np.float32) for _ in range(2)]
                input_biases = [rng.normal(0, .09, 4*hidden).astype(np.float32) for _ in range(2)]
                for override in ('1', '0'):
                    os.environ['CRISPASR_SCNET_SCALAR_GATES'] = override
                    for fused_bias in (False, True):
                        expected = np.full((steps, batch, 2*hidden), np.nan, np.float32)
                        if fused_bias:
                            self.lib.scnet_math_lstm_bias(
                                *[a.ctypes.data_as(POINTER)
                                  for a in projections+weights+biases+input_biases],
                                expected.ctypes.data_as(POINTER), hidden, batch, steps, 1, 0,
                            )
                        else:
                            self.lib.scnet_math_lstm(
                                *[a.ctypes.data_as(POINTER) for a in projections+weights+biases],
                                expected.ctypes.data_as(POINTER), hidden, batch, steps, 1,
                            )
                        self.assertTrue(np.isfinite(expected).all())
                        for padding in (0, 3):
                            # [T,B,8H] is the single projected tensor. Each
                            # [T,B,4H] view retains its parent's row stride.
                            # Extra NaNs make unintended contiguous reads fail.
                            stride = 8*hidden + padding
                            storage = np.full(steps*batch*stride+2, np.nan, np.float32)
                            storage[[0, -1]] = 12345.0
                            joined = storage[1:-1].reshape(steps, batch, stride)
                            views = [joined[:, :, :4*hidden], joined[:, :, 4*hidden:8*hidden]]
                            for view, original in zip(views, projections):
                                view[:] = original
                                np.testing.assert_array_equal(view.view(np.uint32), original.view(np.uint32))
                            before = storage.view(np.uint32).copy()
                            bias_pointers = ([a.ctypes.data_as(POINTER) for a in input_biases]
                                             if fused_bias else [None, None])
                            for threads, profile in ((1, 0), (5, 0), (5, 1)):
                                with self.subTest(hidden=hidden, scalar=override, bias=fused_bias,
                                                  padding=padding, threads=threads, profile=profile):
                                    guarded = np.full(expected.size+2, 12345.0, np.float32)
                                    actual = guarded[1:-1].reshape(expected.shape)
                                    self.lib.scnet_math_lstm_strided(
                                        *[a.ctypes.data_as(POINTER) for a in views+weights+biases],
                                        *bias_pointers, actual.ctypes.data_as(POINTER),
                                        hidden, batch, steps, threads, profile, stride,
                                    )
                                    np.testing.assert_array_equal(guarded[[0, -1]], [12345.0, 12345.0])
                                    np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))
                                    np.testing.assert_array_equal(storage.view(np.uint32), before)
        finally:
            if previous is None:
                os.environ.pop('CRISPASR_SCNET_SCALAR_GATES', None)
            else:
                os.environ['CRISPASR_SCNET_SCALAR_GATES'] = previous

    def test_actual_callback_worker_ownership_and_effective_thread_count(self):
        subprocess.run([str(self.dispatch)], check=True, capture_output=True, text=True)


if __name__ == "__main__":
    unittest.main()
