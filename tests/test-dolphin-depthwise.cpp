#include "dolphin_depthwise.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

struct ConvCase {
    int C, T, K;
    bool causal, csgu;
    std::vector<float> x, w, b, lnw, lnb, y;
};

#include "fixtures/dolphin-depthwise-reference.h"

int main() {
    int failures = 0;
    for (const auto& test : cases) {
        auto* ctx = ggml_init({16 * 1024 * 1024, nullptr, true});
        auto* backend = ggml_backend_cpu_init();
        if (!ctx || !backend)
            return 2;
        ggml_backend_cpu_set_n_threads(backend, 2);
        auto* x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, test.C, test.T);
        auto* w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, test.K, test.C);
        auto* b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, test.C);
        auto* lnw = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, test.C);
        auto* lnb = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, test.C);
        auto* y = dolphin_detail::depthwise_conv(ctx, x, w, b, test.K, test.causal, [&](ggml_tensor* input) {
            return test.csgu ? ggml_add(ctx, ggml_mul(ctx, ggml_norm(ctx, input, 1e-5f), lnw), lnb) : input;
        });
        auto* graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, y);
        auto* buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
        if (!buffer)
            return 2;
        ggml_backend_tensor_set(x, test.x.data(), 0, test.x.size() * sizeof(float));
        ggml_backend_tensor_set(w, test.w.data(), 0, test.w.size() * sizeof(float));
        ggml_backend_tensor_set(b, test.b.data(), 0, test.b.size() * sizeof(float));
        if (test.csgu) {
            ggml_backend_tensor_set(lnw, test.lnw.data(), 0, test.lnw.size() * sizeof(float));
            ggml_backend_tensor_set(lnb, test.lnb.data(), 0, test.lnb.size() * sizeof(float));
        }
        if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS)
            return 2;
        std::vector<float> actual(test.y.size());
        ggml_backend_tensor_get(y, actual.data(), 0, actual.size() * sizeof(float));
        float error = 0;
        bool finite = true;
        for (size_t i = 0; i < actual.size(); ++i) {
            finite = finite && std::isfinite(actual[i]);
            error = std::max(error, std::abs(actual[i] - test.y[i]));
        }
        const bool pass = finite && y->ne[0] == test.C && y->ne[1] == test.T && error < 2e-5f;
        std::printf("%s %s T=%d K=%d max_error=%.9g %s\n", test.csgu ? "CSGU" : "merge",
                    test.causal ? "causal" : "noncausal", test.T, test.K, error, pass ? "PASS" : "FAIL");
        failures += !pass;
        ggml_backend_buffer_free(buffer);
        ggml_backend_free(backend);
        ggml_free(ctx);
    }
    return failures ? 1 : 0;
}
