#!/usr/bin/env python3
"""Check packed F32 recurrence arithmetic and ownership without a checkpoint.

The independent matrix oracle accumulates in NumPy float64. These are component
controls, not a model-quality or performance qualification.
"""

import ctypes
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np


ROOT = pathlib.Path(__file__).resolve().parents[1]
PTR = ctypes.POINTER(ctypes.c_float)

WRAPPER = r"""
#include "scnet_lstm.h"
#include <thread>

scnet_recurrent::Kernel inspect_kernel(int lanes) {
#if SCNET_RECURRENT_X86
    if (lanes == 16 && scnet_recurrent::avx512_supported())
        return {scnet_recurrent::avx512_product, "packed_avx512_fma"};
    if (lanes == 8 && scnet_recurrent::avx2_supported())
        return {scnet_recurrent::avx2_product, "packed_avx2_fma"};
#endif
    return {};
}
extern "C" {
int recurrent_supported(int lanes) { return inspect_kernel(lanes).product != nullptr; }
const char* recurrent_selected() { return scnet_recurrent::selected_kernel().name; }
int recurrent_pack(const float* source, int hidden, float* target) {
    try {
        const scnet_recurrent::PackedWeight packed(source, hidden);
        if (target) std::memcpy(target, packed.values.data(), packed.values.size()*sizeof(float));
        return int(packed.values.size());
    } catch (const std::exception&) { return -1; }
}
int recurrent_product(const float* input, const float* weights, float* output,
                      int hidden, int rows, int lanes) {
    const auto kernel = inspect_kernel(lanes);
    if (!kernel.product) return -1;
    const scnet_recurrent::PackedWeight packed(weights, hidden);
    const auto unchanged = packed.values;
    kernel.product(input, packed, output, rows);
    return std::memcmp(unchanged.data(), packed.values.data(), unchanged.size()*sizeof(float)) == 0 ? 0 : -2;
}
int recurrent_feedback(const float* projected, const float* weights, const float* bias_h,
                       const float* bias_i, float* output, int hidden, int batch,
                       int steps, int stride, int lanes, int workers) {
    const auto kernel = inspect_kernel(lanes);
    if (!kernel.product) return -1;
    const scnet_recurrent::PackedWeight forward(weights, hidden);
    const scnet_recurrent::PackedWeight reverse(weights + size_t(4)*hidden*hidden, hidden);
    auto run = [&](int worker) {
        const int begin = batch*worker/workers, end = batch*(worker+1)/workers;
        if (begin == end) return;
        scnet_detail::LstmScratch scratch(end-begin, hidden);
        for (int direction = 0; direction < 2; ++direction) {
            const auto& packed = direction ? reverse : forward;
            scnet_detail::lstm_direction(projected + direction*4*hidden,
                weights + size_t(direction)*4*hidden*hidden,
                bias_h + direction*4*hidden, output, hidden, batch, steps, direction != 0,
                begin, end, false, scratch, nullptr, bias_i + direction*4*hidden,
                stride, &packed, kernel.product);
        }
    };
    if (workers == 1) run(0);
    else {
        std::thread second([&] { run(1); });
        run(0);
        second.join();
    }
    return 0;
}
}
"""

CALLBACK = r"""
#include "scnet.cpp"
#include <atomic>
#include <cassert>
#include <thread>

std::vector<float> exercise(int hidden, int batch, int steps, int requested, int workers) {
    const int stride = 8*hidden+3;
    std::vector<float> projected(size_t(steps)*batch*stride, std::numeric_limits<float>::quiet_NaN());
    for (int row = 0; row < steps*batch; ++row)
        for (int gate = 0; gate < 8*hidden; ++gate)
            projected[size_t(row)*stride+gate] = float((row*7+gate*3)%29-14)/64;
    const auto original_projection = projected;
    std::array<std::vector<float>, 6> inputs;
    for (int i = 0; i < 6; ++i) {
        inputs[i].resize(i < 2 ? size_t(4)*hidden*hidden : 4*hidden);
        for (size_t j = 0; j < inputs[i].size(); ++j)
            inputs[i][j] = float(int((j*7+i*3)%19)-9)/(i < 2 ? 512 : 128);
    }
    const auto original_inputs = inputs;
    scnet_recurrent::PackedWeight forward(inputs[0].data(), hidden), reverse(inputs[1].data(), hidden);
    std::array<ggml_tensor, 8> sources{};
    ggml_tensor output{};
    const size_t count = size_t(steps)*batch*2*hidden;
    std::vector<float> values(count+2, 12345.0f);
    std::fill_n(values.data()+1, count, std::numeric_limits<float>::quiet_NaN());
    output.ne[0] = 2*hidden; output.ne[1] = batch; output.ne[2] = steps; output.ne[3] = 1;
    output.data = values.data()+1;
    for (int i = 0; i < 8; ++i) {
        output.src[i] = &sources[i];
        if (i < 2) {
            sources[i].data = projected.data()+i*4*hidden;
            sources[i].nb[0] = sizeof(float);
            sources[i].nb[1] = stride*sizeof(float);
            sources[i].nb[2] = batch*sources[i].nb[1];
        } else sources[i].data = inputs[i-2].data();
    }
    RecurrentNode node(batch, hidden, requested, RecurrentMode::packed);
    node.kernel = scnet_recurrent::selected_kernel();
    node.packed = {&forward, &reverse};
    assert(node.kernel.product);
    if (workers == 1) recurrent_compute(&output, 0, 1, &node);
    else {
        std::atomic<int> ready{0};
        std::atomic<bool> start{false};
        auto work = [&](int ith) {
            ready.fetch_add(1);
            while (!start.load()) std::this_thread::yield();
            recurrent_compute(&output, ith, 2, &node);
        };
        std::thread first(work, 0), second(work, 1);
        while (ready.load() != 2) std::this_thread::yield();
        start.store(true);
        first.join(); second.join();
    }
    assert(values.front() == 12345.0f && values.back() == 12345.0f);
    for (size_t i = 1; i <= count; ++i) assert(std::isfinite(values[i]));
    assert(std::memcmp(projected.data(), original_projection.data(), projected.size()*sizeof(float)) == 0);
    for (int i = 0; i < 6; ++i)
        assert(std::memcmp(inputs[i].data(), original_inputs[i].data(), inputs[i].size()*sizeof(float)) == 0);
    return values;
}
int main() {
    if (!scnet_recurrent::selected_kernel().product) return 77;
    int checked = 0;
    for (const auto shape : {std::array<int,3>{128,57,5}, {256,85,3}, {512,3,3}, {17,1,7}}) {
        const auto expected = exercise(shape[0], shape[1], shape[2], 1, 1);
        for (const auto workers : {std::array<int,2>{2,2}, {8,2}, {8,1}}) {
            const auto actual = exercise(shape[0], shape[1], shape[2], workers[0], workers[1]);
            assert(expected.size() == actual.size());
            assert(std::memcmp(expected.data(), actual.data(), expected.size()*sizeof(float)) == 0);
            ++checked;
        }
    }
    std::printf("PASS: %d production callback partition comparisons; concurrent two-worker and clamped-worker paths; input/output guards\n", checked);
}
"""


def pointer(array):
    return array.ctypes.data_as(PTR)


class RecurrentGemm(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        compiler = shutil.which("c++") or shutil.which("g++") or shutil.which("clang++")
        if not compiler:
            raise RuntimeError("SCNet recurrent controls require a C++ compiler")
        cls.temporary = tempfile.TemporaryDirectory(prefix="scnet-recurrent-gemm-")
        folder = pathlib.Path(cls.temporary.name)
        source = folder / "wrapper.cpp"
        source.write_text(WRAPPER)
        library = folder / ("wrapper.dylib" if sys.platform == "darwin" else "wrapper.so")
        subprocess.run([compiler, "-std=c++17", "-O3", "-fPIC", "-shared", "-pthread",
                        "-I", str(ROOT / "src"), str(source), "-o", str(library)],
                       check=True, capture_output=True, text=True)
        cls.lib = ctypes.CDLL(str(library))
        cls.lib.recurrent_supported.argtypes = [ctypes.c_int]
        cls.lib.recurrent_selected.restype = ctypes.c_char_p
        cls.lib.recurrent_pack.argtypes = [PTR, ctypes.c_int, PTR]
        cls.lib.recurrent_product.argtypes = [PTR, PTR, PTR] + [ctypes.c_int] * 3
        cls.lib.recurrent_feedback.argtypes = [PTR] * 5 + [ctypes.c_int] * 6
        cls.lanes = [lane for lane in (8, 16) if cls.lib.recurrent_supported(lane)]
        callback_source = folder / "callback.cpp"
        callback_source.write_text(CALLBACK)
        cls.callback = folder / "callback"
        strip = "-Wl,-dead_strip" if sys.platform == "darwin" else "-Wl,--gc-sections"
        subprocess.run([compiler, "-std=c++17", "-O1", "-ffunction-sections", "-fdata-sections", "-pthread",
                        "-I", str(ROOT / "src"), "-I", str(ROOT / "ggml/include"), str(callback_source),
                        strip, "-o", str(cls.callback)], check=True, capture_output=True, text=True)
        print("recurrent kernel:", cls.lib.recurrent_selected().decode(), "supported lanes:", cls.lanes)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def need_simd(self):
        if not self.lanes:
            self.skipTest("No supported packed x86 kernel; production keeps its existing fallback")

    def test_pack_preserves_coefficient_bits_and_rejects_invalid_shapes(self):
        for hidden in (17, 128, 256, 512):
            source = np.arange(4 * hidden * hidden, dtype=np.uint32).reshape(4 * hidden, hidden)
            source = ((source * np.uint32(2654435761)) & np.uint32(0xFFFFFFFF)).view(np.float32)
            before = source.view(np.uint32).copy()
            size = ((4 * hidden + 15) // 16) * hidden * 16
            guarded = np.full(size + 2, 12345.0, np.float32)
            self.assertEqual(self.lib.recurrent_pack(pointer(source), hidden, pointer(guarded[1:-1])), size)
            expected = np.zeros(((4 * hidden + 15) // 16, hidden, 16), np.float32)
            for gate in range(4 * hidden):
                expected[gate // 16, :, gate % 16] = source[gate]
            np.testing.assert_array_equal(guarded[1:-1].view(np.uint32), expected.ravel().view(np.uint32))
            np.testing.assert_array_equal(source.view(np.uint32), before)
            np.testing.assert_array_equal(guarded[[0, -1]], [12345.0, 12345.0])
        value = np.zeros(1, np.float32)
        for hidden in (0, -1, 2**31 - 1):
            self.assertEqual(self.lib.recurrent_pack(pointer(value), hidden, None), -1)
        self.assertEqual(self.lib.recurrent_pack(None, 128, None), -1)

    def matrix(self, hidden, rows, rng):
        x = rng.uniform(-0.5, 0.5, (rows, hidden)).astype(np.float32)
        w = rng.uniform(-0.5, 0.5, (4 * hidden, hidden)).astype(np.float32)
        x_before, w_before = x.copy(), w.copy()
        oracle = x.astype(np.float64) @ w.astype(np.float64).T
        absolute_sum = np.abs(x.astype(np.float64)) @ np.abs(w.astype(np.float64)).T
        gamma = hidden * 2**-24 / (1 - hidden * 2**-24)
        bound = gamma * absolute_sum + np.finfo(np.float32).tiny
        results = []
        for lane in self.lanes:
            guarded = np.full(rows * 4 * hidden + 2, 12345.0, np.float32)
            actual = guarded[1:-1].reshape(rows, 4 * hidden)
            self.assertEqual(self.lib.recurrent_product(pointer(x), pointer(w), pointer(actual),
                                                        hidden, rows, lane), 0)
            self.assertTrue(np.isfinite(actual).all())
            self.assertTrue(np.all(np.abs(actual.astype(np.float64) - oracle) <= bound))
            np.testing.assert_array_equal(guarded[[0, -1]], [12345.0, 12345.0])
            results.append(actual.copy())
        for result in results[1:]:
            np.testing.assert_array_equal(result.view(np.uint32), results[0].view(np.uint32))
        np.testing.assert_array_equal(x.view(np.uint32), x_before.view(np.uint32))
        np.testing.assert_array_equal(w.view(np.uint32), w_before.view(np.uint32))

    def test_model_widths_split_batches_and_f64_error_bounds(self):
        self.need_simd()
        rng = np.random.default_rng(6123)
        for hidden, rows in ((128, 1), (128, 28), (128, 29), (256, 42), (256, 43), (256, 57), (512, 85)):
            with self.subTest(hidden=hidden, rows=rows):
                self.matrix(hidden, rows, rng)

    def test_every_row_tail_partial_gate_block_and_empty_product(self):
        self.need_simd()
        rng = np.random.default_rng(82)
        for rows in range(10):
            with self.subTest(rows=rows):
                self.matrix(17, rows, rng)

    def test_nonfinite_products_remain_visible(self):
        self.need_simd()
        hidden, rows = 17, 9
        x = np.ones((rows, hidden), np.float32)
        w = np.ones((4 * hidden, hidden), np.float32)
        w[2, 3] = np.inf
        w[3, 7] = np.nan
        for lane in self.lanes:
            actual = np.empty((rows, 4 * hidden), np.float32)
            self.assertEqual(self.lib.recurrent_product(pointer(x), pointer(w), pointer(actual),
                                                        hidden, rows, lane), 0)
            self.assertTrue(np.isposinf(actual[:, 2]).all())
            self.assertTrue(np.isnan(actual[:, 3]).all())
            self.assertTrue(np.isfinite(np.delete(actual, (2, 3), axis=1)).all())

    def test_bidirectional_feedback_joined_stride_and_parallel_ownership(self):
        self.need_simd()
        rng = np.random.default_rng(481)
        for hidden, batch, steps in ((128, 57, 7), (256, 85, 5), (512, 3, 3), (17, 9, 13), (128, 3, 872)):
            with self.subTest(hidden=hidden, batch=batch, steps=steps):
                stride = 8 * hidden + 3
                p = np.full((steps, batch, stride), np.nan, np.float32)
                p[:, :, :8 * hidden] = rng.uniform(-0.8, 0.8, (steps, batch, 8 * hidden))
                weights = (rng.normal(size=(2, 4 * hidden, hidden)) * (0.2 / np.sqrt(hidden))).astype(np.float32)
                bh = rng.uniform(-0.2, 0.2, (2, 4 * hidden)).astype(np.float32)
                bi = rng.uniform(-0.2, 0.2, (2, 4 * hidden)).astype(np.float32)
                untouched = [a.view(np.uint32).copy() for a in (p, weights, bh, bi)]
                oracle = np.empty((steps, batch, 2 * hidden), np.float64)
                for direction in range(2):
                    state = np.zeros((batch, hidden), np.float64)
                    cell = np.zeros_like(state)
                    for time in (range(steps - 1, -1, -1) if direction else range(steps)):
                        recurrent = state @ weights[direction].astype(np.float64).T
                        gates = recurrent + bh[direction] + p[time, :, direction*4*hidden:(direction+1)*4*hidden] + bi[direction]
                        i, f, g, o = np.split(gates, 4, axis=1)
                        cell = cell / (1 + np.exp(-f)) + np.tanh(g) / (1 + np.exp(-i))
                        state = np.tanh(cell) / (1 + np.exp(-o))
                        oracle[time, :, direction*hidden:(direction+1)*hidden] = state
                results = []
                for lane in self.lanes:
                    for workers in (1, 2):
                        guarded = np.full(oracle.size + 2, 12345.0, np.float32)
                        actual = guarded[1:-1].reshape(oracle.shape)
                        self.assertEqual(self.lib.recurrent_feedback(
                            *[pointer(a) for a in (p, weights, bh, bi, actual)],
                            hidden, batch, steps, stride, lane, workers), 0)
                        self.assertTrue(np.isfinite(actual).all())
                        np.testing.assert_allclose(actual, oracle, atol=2e-6, rtol=2e-5)
                        np.testing.assert_array_equal(guarded[[0, -1]], [12345.0, 12345.0])
                        results.append(actual.copy())
                for result in results[1:]:
                    np.testing.assert_array_equal(result.view(np.uint32), results[0].view(np.uint32))
                for value, before in zip((p, weights, bh, bi), untouched):
                    np.testing.assert_array_equal(value.view(np.uint32), before)

    def test_actual_production_callback_concurrent_and_clamped_workers(self):
        self.need_simd()
        result = subprocess.run([str(self.callback)], check=True, capture_output=True, text=True,
                                env={**os.environ, "CRISPASR_SCNET_BENCH": "0"})
        self.assertIn("12 production callback partition comparisons", result.stdout)
        print(result.stdout.strip())


if __name__ == "__main__":
    unittest.main()
