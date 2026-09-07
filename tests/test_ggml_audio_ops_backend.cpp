// Regression coverage for CrispASR's carried ggml audio-op backend patches.
//
// This intentionally runs GGML_OP_CONV_TRANSPOSE_1D and GGML_OP_COL2IM_1D
// directly on one selected GPU backend, with no CPU fallback scheduler. If a
// ggml bump drops the backend kernels or dtype support, the test fails at graph
// compute time instead of silently passing through CPU.

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

// The production callsite under test: every TTS decoder's ConvTranspose1d goes
// through core_convt::convt1d_decomp (20 callsites, and convt1d_decomp_tf just
// delegates to it), so this header is what actually has to stay correct across a
// ggml bump — not ggml_col2im_1d in isolation.
#include "../src/core/conv.h"

#if defined(CRISPASR_TEST_BACKEND_METAL)
#include "ggml-metal.h"
#elif defined(CRISPASR_TEST_BACKEND_VULKAN)
#include "ggml-vulkan.h"
#elif defined(CRISPASR_TEST_BACKEND_CUDA)
#include "ggml-cuda.h"
#endif

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

static float value_at(int i, int salt) {
    int v = (i * 37 + salt * 53) % 211;
    return ((float)v - 105.0f) / 37.0f;
}

static ggml_backend_t init_test_backend() {
#if defined(CRISPASR_TEST_BACKEND_METAL)
    return ggml_backend_metal_init();
#elif defined(CRISPASR_TEST_BACKEND_VULKAN)
    if (ggml_backend_vk_get_device_count() <= 0)
        return nullptr;
    return ggml_backend_vk_init(0);
#elif defined(CRISPASR_TEST_BACKEND_CUDA)
    if (ggml_backend_cuda_get_device_count() <= 0)
        return nullptr;
    return ggml_backend_cuda_init(0);
#elif defined(CRISPASR_TEST_BACKEND_CPU)
    return ggml_backend_cpu_init();
#else
    return nullptr;
#endif
}

static const char* backend_name() {
#if defined(CRISPASR_TEST_BACKEND_METAL)
    return "metal";
#elif defined(CRISPASR_TEST_BACKEND_VULKAN)
    return "vulkan";
#elif defined(CRISPASR_TEST_BACKEND_CUDA)
    return "cuda";
#elif defined(CRISPASR_TEST_BACKEND_CPU)
    return "cpu";
#else
    return "unknown";
#endif
}

static bool close_enough(const std::vector<float>& a, const std::vector<float>& b, float tol, const char* tag) {
    if (a.size() != b.size()) {
        std::fprintf(stderr, "%s: size mismatch %zu vs %zu\n", tag, a.size(), b.size());
        return false;
    }
    float max_abs = 0.0f;
    double mse = 0.0;
    for (size_t i = 0; i < a.size(); i++) {
        float d = std::fabs(a[i] - b[i]);
        max_abs = std::max(max_abs, d);
        mse += (double)d * d;
    }
    double rmse = std::sqrt(mse / std::max<size_t>(a.size(), 1));
    std::printf("%s: max_abs=%g rmse=%g n=%zu\n", tag, max_abs, rmse, a.size());
    return max_abs <= tol;
}

static bool run_conv_transpose_1d(ggml_backend_t backend, std::vector<float>& out) {
    const int K = 5;
    const int Cout = 4;
    const int Cin = 3;
    const int T = 9;
    const int stride = 3;

    std::vector<ggml_fp16_t> w((size_t)K * Cout * Cin);
    std::vector<float> x((size_t)T * Cin);
    for (size_t i = 0; i < w.size(); i++)
        w[i] = ggml_fp32_to_fp16(value_at((int)i, 1));
    for (size_t i = 0; i < x.size(); i++)
        x[i] = value_at((int)i, 2);

    ggml_init_params ip{};
    ip.mem_size = ggml_tensor_overhead() * 8 + ggml_graph_overhead();
    ip.no_alloc = true;
    ggml_context* ctx = ggml_init(ip);
    ggml_tensor* wt = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, K, Cout, Cin);
    ggml_tensor* xt = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, T, Cin);
    ggml_set_name(wt, "w");
    ggml_set_name(xt, "x");
    ggml_set_input(wt);
    ggml_set_input(xt);

    ggml_tensor* y = ggml_conv_transpose_1d(ctx, wt, xt, stride, 0, 1);
    ggml_set_name(y, "y");
    ggml_set_output(y);
    ggml_cgraph* gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, y);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buf) {
        std::fprintf(stderr, "%s conv_transpose_1d: alloc_ctx_tensors failed\n", backend_name());
        ggml_free(ctx);
        return false;
    }

    ggml_backend_tensor_set(wt, w.data(), 0, w.size() * sizeof(ggml_fp16_t));
    ggml_backend_tensor_set(xt, x.data(), 0, x.size() * sizeof(float));
    enum ggml_status st = ggml_backend_graph_compute(backend, gf);
    if (st != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "%s conv_transpose_1d: compute failed (%d)\n", backend_name(), (int)st);
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
        return false;
    }

    out.assign((size_t)ggml_nelements(y), 0.0f);
    ggml_backend_tensor_get(y, out.data(), 0, out.size() * sizeof(float));
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return true;
}

static std::vector<float> col2im_reference(const std::vector<float>& col, int T_in, int T_out, int OC, int K, int s0,
                                           int p0) {
    const int K_OC = K * OC;
    std::vector<float> out((size_t)T_out * OC, 0.0f);
    for (int idx = 0; idx < T_out * OC; idx++) {
        const int t_out = idx % T_out;
        const int oc = idx / T_out;
        const int t_abs = t_out + p0;
        int t_in_min = (t_abs - K + s0) / s0;
        if (t_in_min < 0)
            t_in_min = 0;
        int t_in_max = t_abs / s0;
        if (t_in_max >= T_in)
            t_in_max = T_in - 1;
        float sum = 0.0f;
        for (int t_in = t_in_min; t_in <= t_in_max; t_in++) {
            const int k = t_abs - t_in * s0;
            sum += col[(size_t)(oc * K + k) + (size_t)t_in * K_OC];
        }
        out[idx] = sum;
    }
    return out;
}

static bool run_col2im_1d(ggml_backend_t backend, ggml_type type, std::vector<float>& out,
                          std::vector<float>& expected) {
    const int K = 4;
    const int OC = 3;
    const int T_in = 8;
    const int stride = 2;
    const int crop_left = 1;
    // upstream's ggml_col2im_1d crops p0 from BOTH sides while its kernel treats
    // p0 as a left offset (t_abs = t_out + p0) — col2im_reference below matches
    // that kernel exactly, so only the LENGTH rule differs from our old op.
    const int T_out = (T_in - 1) * stride + K - 2 * crop_left;
    const int K_OC = K * OC;

    std::vector<float> col_f32((size_t)K_OC * T_in);
    for (size_t i = 0; i < col_f32.size(); i++)
        col_f32[i] = value_at((int)i, 3);
    expected = col2im_reference(col_f32, T_in, T_out, OC, K, stride, crop_left);

    std::vector<ggml_fp16_t> col_f16;
    if (type == GGML_TYPE_F16) {
        col_f16.resize(col_f32.size());
        for (size_t i = 0; i < col_f32.size(); i++) {
            col_f16[i] = ggml_fp32_to_fp16(col_f32[i]);
            col_f32[i] = ggml_fp16_to_fp32(col_f16[i]);
        }
        expected = col2im_reference(col_f32, T_in, T_out, OC, K, stride, crop_left);
    }

    ggml_init_params ip{};
    ip.mem_size = ggml_tensor_overhead() * 8 + ggml_graph_overhead();
    ip.no_alloc = true;
    ggml_context* ctx = ggml_init(ip);
    ggml_tensor* col = ggml_new_tensor_2d(ctx, type, K_OC, T_in);
    ggml_set_name(col, "col");
    ggml_set_input(col);
    ggml_tensor* y = ggml_col2im_1d(ctx, col, stride, OC, crop_left);
    ggml_set_name(y, "y");
    ggml_set_output(y);
    ggml_cgraph* gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, y);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buf) {
        std::fprintf(stderr, "%s col2im_1d: alloc_ctx_tensors failed\n", backend_name());
        ggml_free(ctx);
        return false;
    }

    if (type == GGML_TYPE_F16) {
        ggml_backend_tensor_set(col, col_f16.data(), 0, col_f16.size() * sizeof(ggml_fp16_t));
    } else {
        ggml_backend_tensor_set(col, col_f32.data(), 0, col_f32.size() * sizeof(float));
    }
    enum ggml_status st = ggml_backend_graph_compute(backend, gf);
    if (st != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "%s col2im_1d: compute failed (%d)\n", backend_name(), (int)st);
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
        return false;
    }

    // upstream's ggml_col2im_1d is TYPE-PRESERVING (dst type == src type); the op
    // we carried before the v0.17 sync always produced F32 regardless of input.
    // Read back in the tensor's own type, and round the reference the same way so
    // the F16 comparison measures the kernel, not the store.
    out.assign((size_t)ggml_nelements(y), 0.0f);
    if (y->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> tmp(out.size());
        ggml_backend_tensor_get(y, tmp.data(), 0, tmp.size() * sizeof(ggml_fp16_t));
        for (size_t i = 0; i < out.size(); i++) {
            out[i] = ggml_fp16_to_fp32(tmp[i]);
        }
        for (size_t i = 0; i < expected.size(); i++) {
            expected[i] = ggml_fp16_to_fp32(ggml_fp32_to_fp16(expected[i]));
        }
    } else {
        ggml_backend_tensor_get(y, out.data(), 0, out.size() * sizeof(float));
    }
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return true;
}

// End-to-end check of core_convt::convt1d_decomp against a plain host
// ConvTranspose1d, including the weight permutation and the asymmetric crop.
//
// This is the gate that matters for the v0.17 sync. Upstream's ggml_col2im_1d
// crops p0 from BOTH sides (T_out = (T_in-1)*s + K - 2*p0) whereas the op we
// carried treated p0 as a LEFT offset. convt1d_decomp was rewritten to pass
// p0 = 0 and do both crops with a view, which is the only way to express the
// asymmetric causal crop (crop_left = 0, crop_right = K - stride). Swapping the
// op semantics without that rewrite still compiles and links — it just changes
// decoder output LENGTH, which is exactly the failure this catches.
static bool run_convt1d_decomp(ggml_backend_t backend, int crop_left, int crop_right, const char* label) {
    const int IC = 3;
    const int OC = 2;
    const int K = 4;
    const int stride = 2;
    const int T_in = 5;

    const int T_full = (T_in - 1) * stride + K;
    const int T_out = T_full - crop_left - crop_right;

    // Weight in ggml ConvTranspose1d layout [K, OC, IC]: w[ic][oc][k].
    std::vector<float> w_src((size_t)K * OC * IC);
    for (size_t i = 0; i < w_src.size(); i++)
        w_src[i] = value_at((int)i, 11);

    // Input (IC, T_in) channel-major: ne0 = IC, so x[t][ic].
    std::vector<float> x_src((size_t)IC * T_in);
    for (size_t i = 0; i < x_src.size(); i++)
        x_src[i] = value_at((int)i, 17);

    std::vector<float> b_src((size_t)OC);
    for (size_t i = 0; i < b_src.size(); i++)
        b_src[i] = value_at((int)i, 23);

    // Host reference: full ConvTranspose1d, then crop, laid out ne0 = OC.
    std::vector<float> expected((size_t)T_out * OC, 0.0f);
    for (int t_in = 0; t_in < T_in; t_in++) {
        for (int k = 0; k < K; k++) {
            const int t = t_in * stride + k;
            if (t < crop_left || t >= T_full - crop_right)
                continue;
            for (int oc = 0; oc < OC; oc++) {
                float acc = 0.0f;
                for (int ic = 0; ic < IC; ic++) {
                    acc += w_src[(size_t)ic * OC * K + (size_t)oc * K + k] * x_src[(size_t)t_in * IC + ic];
                }
                expected[(size_t)(t - crop_left) * OC + oc] += acc;
            }
        }
    }
    for (int t = 0; t < T_out; t++)
        for (int oc = 0; oc < OC; oc++)
            expected[(size_t)t * OC + oc] += b_src[oc];

    // Stage the source weight on the backend so permute_convt1d_weight is
    // exercised through ggml_backend_tensor_get, exactly as loaders use it.
    ggml_init_params wp{};
    wp.mem_size = ggml_tensor_overhead() * 4;
    wp.no_alloc = true;
    ggml_context* ctx_w = ggml_init(wp);
    ggml_tensor* w_raw = ggml_new_tensor_3d(ctx_w, GGML_TYPE_F32, K, OC, IC);
    ggml_backend_buffer_t buf_w = ggml_backend_alloc_ctx_tensors(ctx_w, backend);
    if (!buf_w) {
        std::fprintf(stderr, "%s %s: weight alloc failed\n", backend_name(), label);
        ggml_free(ctx_w);
        return false;
    }
    ggml_backend_tensor_set(w_raw, w_src.data(), 0, w_src.size() * sizeof(float));
    std::unique_ptr<float[]> w_perm_host = core_convt::permute_convt1d_weight(w_raw);

    ggml_init_params ip{};
    ip.mem_size = ggml_tensor_overhead() * 32 + ggml_graph_overhead();
    ip.no_alloc = true;
    ggml_context* ctx = ggml_init(ip);

    ggml_tensor* w_perm = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, IC, K * OC);
    ggml_tensor* x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, IC, T_in);
    ggml_tensor* b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, OC);
    ggml_set_input(w_perm);
    ggml_set_input(x);
    ggml_set_input(b);

    ggml_tensor* y = core_convt::convt1d_decomp(ctx, x, w_perm, b, stride, K, crop_left, crop_right);
    ggml_set_output(y);

    if (y->ne[0] != OC || y->ne[1] != T_out) {
        std::fprintf(stderr, "%s %s: shape mismatch — got [%lld,%lld], want [%d,%d]\n", backend_name(), label,
                     (long long)y->ne[0], (long long)y->ne[1], OC, T_out);
        ggml_backend_buffer_free(buf_w);
        ggml_free(ctx_w);
        ggml_free(ctx);
        return false;
    }

    ggml_cgraph* gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, y);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buf) {
        std::fprintf(stderr, "%s %s: alloc_ctx_tensors failed\n", backend_name(), label);
        ggml_backend_buffer_free(buf_w);
        ggml_free(ctx_w);
        ggml_free(ctx);
        return false;
    }
    ggml_backend_tensor_set(w_perm, w_perm_host.get(), 0, (size_t)IC * K * OC * sizeof(float));
    ggml_backend_tensor_set(x, x_src.data(), 0, x_src.size() * sizeof(float));
    ggml_backend_tensor_set(b, b_src.data(), 0, b_src.size() * sizeof(float));

    const enum ggml_status st = ggml_backend_graph_compute(backend, gf);
    bool ok = (st == GGML_STATUS_SUCCESS);
    if (!ok) {
        std::fprintf(stderr, "%s %s: compute failed (%d)\n", backend_name(), label, (int)st);
    } else {
        std::vector<float> out((size_t)ggml_nelements(y), 0.0f);
        ggml_backend_tensor_get(y, out.data(), 0, out.size() * sizeof(float));
        ok = close_enough(expected, out, 1e-4f, label);
    }

    ggml_backend_buffer_free(buf);
    ggml_backend_buffer_free(buf_w);
    ggml_free(ctx);
    ggml_free(ctx_w);
    return ok;
}

int main() {
    ggml_backend_t cpu = ggml_backend_cpu_init();
    ggml_backend_t gpu = init_test_backend();
    if (!gpu) {
        std::fprintf(stderr, "%s backend unavailable at runtime; skipping\n", backend_name());
        ggml_backend_free(cpu);
        return 4;
    }

    std::vector<float> conv_cpu;
    std::vector<float> conv_gpu;
    bool ok = run_conv_transpose_1d(cpu, conv_cpu) && run_conv_transpose_1d(gpu, conv_gpu) &&
              close_enough(conv_cpu, conv_gpu, 5e-3f, "conv_transpose_1d_f16_weight");

    std::vector<float> col_gpu;
    std::vector<float> col_ref;
    ok = ok && run_col2im_1d(gpu, GGML_TYPE_F32, col_gpu, col_ref) &&
         close_enough(col_ref, col_gpu, 1e-5f, "col2im_1d_f32");

#if defined(CRISPASR_TEST_COL2IM_F16)
    ok = ok && run_col2im_1d(gpu, GGML_TYPE_F16, col_gpu, col_ref) &&
         close_enough(col_ref, col_gpu, 2e-3f, "col2im_1d_f16");
#endif

    // The production ConvTranspose1d path, in both crop patterns that appear
    // across the decoder ports: causal (crop_left=0, crop_right=K-stride, used by
    // pocket_tts / csm / kugelaudio / vibevoice) and symmetric (stride/2 each
    // side, used by kokoro / tada / snac / dac / parler).
    ok = ok && run_convt1d_decomp(gpu, /*crop_left=*/0, /*crop_right=*/2, "convt1d_decomp_causal");
    ok = ok && run_convt1d_decomp(gpu, /*crop_left=*/1, /*crop_right=*/1, "convt1d_decomp_symmetric");
    ok = ok && run_convt1d_decomp(cpu, /*crop_left=*/0, /*crop_right=*/2, "convt1d_decomp_causal_cpu");

    ggml_backend_free(gpu);
    ggml_backend_free(cpu);
    return ok ? 0 : 1;
}
