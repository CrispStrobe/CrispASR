// Hermetic production-graph controls: no model or reference download.
#include "core/attention_mask.h"
#include "ggml-cpu.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <vector>

static void require(bool ok, const char* message) {
    if (!ok)
        throw std::runtime_error(message);
}

static std::vector<float> center(const std::vector<float>& values, bool strided) {
    ggml_init_params ip = {4 * 1024 * 1024, nullptr, false};
    auto* ctx = ggml_init(ip);
    auto* input = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, strided ? 2 : 4, strided ? 4 : 2, 2, 2);
    if (strided) {
        // Transpose creates noncontiguous key rows, with the same logical data.
        auto* data = static_cast<float*>(input->data);
        for (int plane = 0; plane < 4; ++plane)
            for (int q = 0; q < 2; ++q)
                for (int k = 0; k < 4; ++k)
                    data[plane * 8 + k * 2 + q] = values[plane * 8 + q * 4 + k];
        input = ggml_transpose(ctx, input);
    } else {
        std::copy(values.begin(), values.end(), static_cast<float*>(input->data));
    }
    auto* result = core_attention::centered_flash_mask(ctx, input);
    require(result->ne[0] == 4 && result->ne[1] == 2 && result->ne[2] == 2 && result->ne[3] == 2,
            "centering changed key/query/head/batch layout");
    auto* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, result);
    require(ggml_graph_compute_with_ctx(ctx, graph, 1) == GGML_STATUS_SUCCESS, "mask graph failed");
    std::vector<float> out(values.size());
    auto* data = static_cast<ggml_fp16_t*>(result->data);
    for (size_t i = 0; i < out.size(); ++i)
        out[i] = ggml_fp16_to_fp32(data[i]);
    ggml_free(ctx);
    return out;
}

int main() {
    try {
        const float inf = std::numeric_limits<float>::infinity();
        const float nan = std::numeric_limits<float>::quiet_NaN();
        const std::vector<float> input = {
            2390.f,   2390.25f, 2390.5f, 2390.75f, -2400.f,   -2400.25f,  -2400.5f,  -2400.75f,
            100000.f, 99999.5f, 99999.f, -inf,     -100000.f, -100000.5f, -100001.f, -inf,
            -inf,     -inf,     -inf,    -inf,     nan,       2.f,        3.f,       4.f,
            inf,      2.f,      3.f,     4.f,      -7.f,      -3.f,       -5.f,      -inf,
        };
        for (bool strided : {false, true}) {
            const auto out = center(input, strided);
            for (int row : {0, 1, 2, 3, 7}) {
                const auto begin = input.begin() + row * 4;
                const float maximum = *std::max_element(begin, begin + 4);
                double old_sum = 0, new_sum = 0;
                for (int k = 0; k < 4; ++k) {
                    const int i = row * 4 + k;
                    require(out[i] == input[i] - maximum, "lost finite differences or masking");
                    old_sum += std::exp(double(input[i]) - maximum);
                    new_sum += std::exp(double(out[i]));
                }
                for (int k = 0; k < 4; ++k) {
                    const int i = row * 4 + k;
                    require(std::abs(std::exp(double(input[i]) - maximum) / old_sum -
                                     std::exp(double(out[i])) / new_sum) < 1e-12,
                            "centering changed softmax probabilities");
                }
            }
            for (int i = 16; i < 20; ++i)
                require(std::isinf(out[i]) && out[i] < 0, "all-masked row gained NaNs or finite keys");
            require(std::isnan(out[20]), "NaN input was sanitized");
            require(std::isinf(out[24]) && out[24] > 0, "+inf input was sanitized");
        }
        std::puts("PASS: finite/negative/large-offset/masked/nonfinite masks, independent 4-D rows and strided input");
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
