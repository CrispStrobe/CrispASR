#include "../src/core/qwen3_ultra_head.h"
#ifdef ULTRA_TEST_SIDECAR
#include "../src/core/qwen3_ultra_sidecar.h"
#include "../src/core/ggml_cpu_backend.h"
#include "qwen3_ultra_normalizer_fixture.h"
#endif
#include <array>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <vector>

using namespace qwen3_ultra_head;

static bool read(std::ifstream& in, std::vector<float>& data, size_t n) {
    data.resize(n);
    return static_cast<bool>(in.read(reinterpret_cast<char*>(data.data()), n * sizeof(float)));
}
static bool read_attention(std::ifstream& in, AttentionWeights& w) {
    return read(in, w.in_weight, 3 * dim * dim) && read(in, w.in_bias, 3 * dim) && read(in, w.out_weight, dim * dim) &&
           read(in, w.out_bias, dim);
}
int main(int argc, char** argv) {
#ifdef ULTRA_TEST_SIDECAR
    if (argc != 3)
        return 1;
#else
    if (argc != 2)
        return 1;
#endif
    std::ifstream in(argv[1], std::ios::binary);
    Weights w;
    if (!read(in, w.audio0_weight, dim * feature_dim * 5) || !read(in, w.audio0_bias, dim) ||
        !read(in, w.audio2_weight, dim * dim * 5) || !read(in, w.audio2_bias, dim) ||
        !read(in, w.context_weight, dim * qwen_dim) || !read(in, w.context_bias, dim) ||
        !read(in, w.role_embedding, roles * dim) || !read(in, w.endpoint_embedding, 2 * dim) ||
        !read_attention(in, w.context_attention) || !read_attention(in, w.audio_attention) ||
        !read(in, w.query_norm_weight, dim) || !read(in, w.query_norm_bias, dim) ||
        !read(in, w.audio_norm_weight, dim) || !read(in, w.audio_norm_bias, dim) || !read(in, w.activity_weight, dim) ||
        !read(in, w.activity_bias, 1) || !read(in, w.boundary_weight, dim) || !read(in, w.boundary_bias, 1) ||
        !read(in, w.derivative_scale, 1) || !read(in, w.prior, offsets))
        return 2;
#ifdef ULTRA_TEST_SIDECAR
    auto* backend = core_cpu_backend::init();
    if (!backend)
        return 7;
    qwen3_ultra_sidecar::Package package;
    const bool loaded = qwen3_ultra_sidecar::load(argv[2], backend, package);
    ggml_backend_free(backend);
    if (!loaded || package.mean != k_mean || package.stddev != k_std)
        return 8;
    // The two paths come from independent safetensors and GGUF reads. Compare
    // every stored value, not just the output logits.
    const auto& p = package.weights;
    auto same_attention = [](const AttentionWeights& a, const AttentionWeights& b) {
        return a.in_weight == b.in_weight && a.in_bias == b.in_bias && a.out_weight == b.out_weight &&
               a.out_bias == b.out_bias;
    };
    if (w.audio0_weight != p.audio0_weight || w.audio0_bias != p.audio0_bias || w.audio2_weight != p.audio2_weight ||
        w.audio2_bias != p.audio2_bias || w.context_weight != p.context_weight || w.context_bias != p.context_bias ||
        w.role_embedding != p.role_embedding || w.endpoint_embedding != p.endpoint_embedding ||
        !same_attention(w.context_attention, p.context_attention) ||
        !same_attention(w.audio_attention, p.audio_attention) || w.query_norm_weight != p.query_norm_weight ||
        w.query_norm_bias != p.query_norm_bias || w.audio_norm_weight != p.audio_norm_weight ||
        w.audio_norm_bias != p.audio_norm_bias || w.activity_weight != p.activity_weight ||
        w.activity_bias != p.activity_bias || w.boundary_weight != p.boundary_weight ||
        w.boundary_bias != p.boundary_bias || w.derivative_scale != p.derivative_scale || w.prior != p.prior)
        return 9;
    w = std::move(package.weights);
#endif
    float max_error = 0.0f;
    for (int i = 0; i < 44; ++i) {
        std::vector<float> features, context, mask, endpoint, expected;
        if (!read(in, features, frames * feature_dim) || !read(in, context, roles * qwen_dim) ||
            !read(in, mask, roles) || !read(in, endpoint, 1) || !read(in, expected, offsets))
            return 3;
        std::array<float, offsets> actual{};
        if (!run(w, features.data(), context.data(), mask.data(), static_cast<int>(endpoint[0]), actual))
            return 4;
        for (int j = 0; j < offsets; ++j)
            max_error = std::max(max_error, std::abs(actual[j] - expected[j]));
    }
    if (in.peek() != std::char_traits<char>::eof())
        return 5;
    std::printf("Ultra head original-model logits: max absolute error %.9g\n", max_error);
    return max_error <= 0.002f ? 0 : 6;
}
