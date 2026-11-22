// Exercise the production CUDA capability predicate through the real scheduler.
// The default backend emulates CUDA capability on CPU; --cuda requires hardware.
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-backend-impl.h"
#include "ggml-cpu.h"
#include "ggml-cuda/ssm-conv-support.h"
#ifdef CRISPASR_TEST_REAL_CUDA
#include "ggml-cuda.h"
#endif

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

static bool legacy_support = false;
static bool emulated_support(ggml_backend_dev_t, const ggml_tensor* op) {
    if (op->op != GGML_OP_SSM_CONV)
        return true;
    return legacy_support ? op->src[0]->ne[1] % 128 == 0 : ggml_cuda_ssm_conv_supported(op);
}

static bool expected_support(int k, int channels) {
    return channels % 128 == 0 && (k == 2 || k == 3 || k == 4 || k == 5 || k == 7 || k == 9 || k == 15);
}

static bool run_case(ggml_backend_t preferred, ggml_backend_t cpu, int k, int channels, int tokens) {
    const int sequences = 2;
    const int tin = tokens + k - 1;
    ggml_init_params ip{ggml_tensor_overhead() * 16 + ggml_graph_overhead(), nullptr, true};
    auto* inputs = ggml_init(ip);
    auto* ctx = ggml_init(ip);
    auto* x = ggml_new_tensor_3d(inputs, GGML_TYPE_F32, tin, channels, sequences);
    auto* w = ggml_new_tensor_2d(inputs, GGML_TYPE_F32, k, channels);
    ggml_set_name(x, "xasr_conv_input");
    ggml_set_name(w, "xasr_conv_weight");
    auto* y = ggml_ssm_conv(ctx, x, w);
    ggml_set_name(y, "xasr_ssm_conv");
    ggml_set_output(y);
    auto* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, y);
    auto* weights = ggml_backend_alloc_ctx_tensors(inputs, preferred);
    if (!weights)
        return false;
    ggml_backend_buffer_set_usage(weights, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    ggml_backend_t backends[] = {preferred, cpu};
    auto* sched = ggml_backend_sched_new(backends, nullptr, 2, GGML_DEFAULT_GRAPH_SIZE, false, false);
    bool ok = ggml_backend_supports_op(preferred, y) == expected_support(k, channels);
    ok = ok && ggml_backend_sched_alloc_graph(sched, graph);
    if (ok) {
        auto* placed = ggml_backend_sched_get_tensor_backend(sched, y);
        ok = placed == (expected_support(k, channels) ? preferred : cpu);
    }
    if (!ok) {
        std::fprintf(stderr, "FAIL capability/placement K=%d C=%d T=%d\n", k, channels, tokens);
    } else {
        std::vector<float> xv(tin * channels * sequences), wv(k * channels), actual(channels * tokens * sequences);
        for (size_t i = 0; i < xv.size(); ++i)
            xv[i] = (int(i % 29) - 14) / 32.0f;
        for (size_t i = 0; i < wv.size(); ++i)
            wv[i] = (int(i % 13) - 6) / 16.0f;
        ggml_backend_tensor_set(x, xv.data(), 0, xv.size() * sizeof(float));
        ggml_backend_tensor_set(w, wv.data(), 0, wv.size() * sizeof(float));
        ok = ggml_backend_sched_graph_compute(sched, graph) == GGML_STATUS_SUCCESS;
        if (ok) {
            ggml_backend_tensor_get(y, actual.data(), 0, actual.size() * sizeof(float));
            for (int s = 0; s < sequences; ++s)
                for (int t = 0; t < tokens; ++t)
                    for (int c = 0; c < channels; ++c) {
                        float expected = 0;
                        for (int j = 0; j < k; ++j)
                            expected += xv[(s * channels + c) * tin + t + j] * wv[c * k + j];
                        float got = actual[(s * tokens + t) * channels + c];
                        if (!std::isfinite(got) || std::fabs(got - expected) > 1e-5f)
                            ok = false;
                    }
        }
        std::printf("%s K=%d C=%d T=%d backend=%s (scalar correlation oracle)\n", ok ? "PASS" : "FAIL", k, channels,
                    tokens, expected_support(k, channels) ? "preferred" : "CPU fallback");
    }
    ggml_backend_sched_free(sched);
    ggml_backend_buffer_free(weights);
    ggml_free(ctx);
    ggml_free(inputs);
    return ok;
}

int main(int argc, char** argv) {
    legacy_support = argc == 2 && std::strcmp(argv[1], "--legacy-support") == 0;
    auto* cpu = ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(cpu, 2);
    ggml_backend_t preferred = nullptr;
    ggml_backend_device emulated_device{};
#ifdef CRISPASR_TEST_REAL_CUDA
    if (ggml_backend_cuda_get_device_count() == 0) {
        std::puts("SKIP: no CUDA device; no CUDA runtime proof");
        ggml_backend_free(cpu);
        return 4;
    }
    preferred = ggml_backend_cuda_init(0);
#else
    preferred = ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(preferred, 2);
    // Preserve CPU execution/buffer methods, overriding only capability. This
    // is an explicit emulation, not evidence that a CUDA kernel ran.
    emulated_device = *ggml_backend_get_device(preferred);
    emulated_device.iface.supports_op = emulated_support;
    preferred->device = &emulated_device;
    std::puts("CPU emulation of CUDA capability; real ggml scheduler and CPU compute");
#endif
    if (!preferred || !cpu)
        return 1;
    bool ok = true;
    int cases = 0;
    for (int k : {2, 3, 4, 5, 7, 8, 9, 15, 16, 31, 32})
        for (int channels : {127, 128, 256})
            for (int tokens : {1, 32, 33, 65}) {
                ok = run_case(preferred, cpu, k, channels, tokens) && ok;
                ++cases;
                if (legacy_support && !ok) {
                    ggml_backend_free(preferred);
                    ggml_backend_free(cpu);
                    return 1;
                }
            }
    ggml_backend_free(preferred);
    ggml_backend_free(cpu);
    std::printf("%s %d scheduling/numerical cases\n", ok ? "PASS" : "FAIL", cases);
    return ok ? 0 : 1;
}
