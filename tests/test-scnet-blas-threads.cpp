// Verify the actual OpenBLAS thread budget when OUT_PROD is the first graph op.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-blas.h"

#include <cblas.h>

#include <cstdio>
#include <memory>
#include <stdexcept>
#include <vector>

namespace {

void require(bool condition, const char* message) {
    if (!condition)
        throw std::runtime_error(message);
}

struct GraphResources {
    ggml_context* context = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    ~GraphResources() {
        if (buffer)
            ggml_backend_buffer_free(buffer);
        if (context)
            ggml_free(context);
    }
};

// Provider setters have process-wide semantics. This standalone control restores
// the previous value; it does not claim isolation for concurrent BLAS contexts.
struct RestoreThreads {
    int previous = openblas_get_num_threads();
    ~RestoreThreads() { openblas_set_num_threads(previous); }
};

void exercise(ggml_backend_t backend, bool multiply, bool transposed, int requested, int previous) {
    constexpr int positions = 97, channels = 64, outputs = 47;
    GraphResources resources;
    resources.context = ggml_init({64 * ggml_tensor_overhead() + ggml_graph_overhead_custom(16, false), nullptr, true});
    require(resources.context != nullptr, "graph context allocation failed");
    auto* ctx = resources.context;
    auto* input =
        ggml_new_tensor_2d(ctx, GGML_TYPE_F32, multiply ? channels : positions, multiply ? positions : channels);
    auto* weights = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, multiply || transposed ? channels : outputs,
                                       multiply || transposed ? outputs : channels);
    auto* result = multiply ? ggml_mul_mat(ctx, weights, input)
                            : ggml_out_prod(ctx, input, transposed ? ggml_transpose(ctx, weights) : weights);
    require(ggml_backend_supports_op(backend, result), "the BLAS backend does not support the test operation");
    auto* graph = ggml_new_graph_custom(ctx, 16, false);
    ggml_build_forward_expand(graph, result);
    resources.buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    require(resources.buffer != nullptr, "BLAS graph allocation failed");

    std::vector<float> x(positions * channels), w(outputs * channels), expected(positions * outputs);
    for (size_t i = 0; i < x.size(); ++i)
        x[i] = static_cast<float>(static_cast<int>((i * 17) % 31) - 15) / 16;
    for (size_t i = 0; i < w.size(); ++i)
        w[i] = static_cast<float>(static_cast<int>((i * 23) % 29) - 14) / 32;
    for (int row = 0; row < outputs; ++row)
        for (int column = 0; column < positions; ++column) {
            float sum = 0;
            for (int k = 0; k < channels; ++k) {
                const float a = x[multiply ? column * channels + k : k * positions + column];
                const float b = w[multiply || transposed ? row * channels + k : k * outputs + row];
                sum += a * b;
            }
            expected[multiply ? column * outputs + row : row * positions + column] = sum;
        }
    ggml_backend_tensor_set(input, x.data(), 0, x.size() * sizeof(float));
    ggml_backend_tensor_set(weights, w.data(), 0, w.size() * sizeof(float));
    ggml_backend_blas_set_n_threads(backend, requested);
    openblas_set_num_threads(previous);
    require(openblas_get_num_threads() == previous, "test could not establish the distinct prior thread budget");
    require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "BLAS graph execution failed");
    require(openblas_get_num_threads() == requested, "BLAS operation did not apply its context thread budget");
    std::vector<float> actual(expected.size());
    ggml_backend_tensor_get(result, actual.data(), 0, actual.size() * sizeof(float));
    // These binary fractions and short reductions are exactly representable.
    require(actual == expected, "BLAS operation differs from the independent scalar product");
    std::printf("%s%s threads=%d prior=%d: exact numeric output and requested provider budget\n",
                multiply ? "MUL_MAT" : "OUT_PROD", transposed ? " transposed-rhs" : "", requested, previous);
}

} // namespace

int main() {
    try {
        RestoreThreads restore;
        std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> backend(ggml_backend_blas_init(),
                                                                            ggml_backend_free);
        require(backend != nullptr, "OpenBLAS backend initialization failed");
        // No preceding MUL_MAT may silently initialize the provider budget.
        exercise(backend.get(), false, false, 2, 1);
        exercise(backend.get(), false, true, 1, 2);
        exercise(backend.get(), false, true, 2, 1);
        exercise(backend.get(), false, false, 1, 2);
        exercise(backend.get(), true, false, 2, 1);
        exercise(backend.get(), true, false, 1, 2);
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "BLAS thread-budget regression failed: %s\n", error.what());
        return 1;
    }
}
