// Model-free checks of the actual runtime, following the internal-context
// pattern used by test-qwen3tts-voice-pack.
#include "../src/meanvc2.cpp"
#include "../ggml/src/ggml-backend-impl.h"

#include <catch2/catch_test_macros.hpp>
#include <limits>

TEST_CASE("MeanVC2 interpolation aligns BNF endpoints", "[unit][meanvc2][interpolation]") {
    // Nonconstant channels catch a time/channel layout transposition as well
    // as the old t/4 rule, which repeated the last input at the tail.
    const float bns[] = {1.0f, 20.0f, 8.0f, -1.0f};
    const auto actual = interpolate_bnf_4x(bns, 2, 2);
    REQUIRE(actual.size() == 16);
    for (int t = 0; t < 8; ++t) {
        CAPTURE(t);
        REQUIRE(std::isfinite(actual[(size_t)t * 2]));
        REQUIRE(std::isfinite(actual[(size_t)t * 2 + 1]));
        REQUIRE(std::abs(actual[(size_t)t * 2] - (1.0f + t)) < 2e-6f);
        REQUIRE(std::abs(actual[(size_t)t * 2 + 1] - (20.0f - 3.0f * t)) < 2e-6f);
    }
    REQUIRE(actual.front() == bns[0]);
    REQUIRE(actual[14] == bns[2]);
    REQUIRE(actual.back() == bns[3]);

    const float triangle[] = {0.0f, 11.0f, 0.0f};
    const auto interior = interpolate_bnf_4x(triangle, 3, 1);
    REQUIRE(interior.size() == 12);
    for (int t = 0; t < 12; ++t) {
        const float expected = (float)(t <= 5 ? 2 * t : 22 - 2 * t);
        CAPTURE(t);
        REQUIRE(std::abs(interior[t] - expected) < 2e-6f);
    }

    const float one_frame[] = {3.0f, -7.0f};
    const auto repeated = interpolate_bnf_4x(one_frame, 1, 2);
    const std::vector<float> expected = {3.0f, -7.0f, 3.0f, -7.0f, 3.0f, -7.0f, 3.0f, -7.0f};
    REQUIRE(repeated == expected);
    REQUIRE(interpolate_bnf_4x(nullptr, 2, 2).empty());
    REQUIRE(interpolate_bnf_4x(bns, 0, 2).empty());
    REQUIRE(interpolate_bnf_4x(bns, 2, 0).empty());
}

TEST_CASE("MeanVC2 reference audio reports unsupported without changing the speaker", "[unit][meanvc2][conditioning]") {
    meanvc2_context ctx;
    ctx.params = meanvc2_context_default_params();
    ctx.params.verbosity = 0;
    const float audio[] = {0.0f, 0.5f, -0.25f};
    REQUIRE(meanvc2_set_speaker_audio(&ctx, audio, 3) == -2);
    REQUIRE(ctx.speaker_emb.empty());

    std::vector<float> embedding(256);
    for (size_t i = 0; i < embedding.size(); ++i)
        embedding[i] = (float)i / 256.0f;
    REQUIRE(meanvc2_set_speaker_embedding(&ctx, embedding.data()) == 0);
    REQUIRE(meanvc2_set_speaker_audio(&ctx, audio, 3) == -2);
    REQUIRE(ctx.speaker_emb == embedding);
    REQUIRE(meanvc2_set_speaker_audio(&ctx, nullptr, 3) == -1);
    REQUIRE(meanvc2_set_speaker_audio(&ctx, audio, 0) == -1);
    REQUIRE(ctx.speaker_emb == embedding);

    auto invalid = embedding;
    invalid[5] = std::numeric_limits<float>::quiet_NaN();
    REQUIRE(meanvc2_set_speaker_embedding(&ctx, invalid.data()) == -1);
    REQUIRE(ctx.speaker_emb == embedding);
    invalid[5] = std::numeric_limits<float>::infinity();
    REQUIRE(meanvc2_set_speaker_embedding(&ctx, invalid.data()) == -1);
    REQUIRE(ctx.speaker_emb == embedding);
}

TEST_CASE("MeanVC2 conversion rejects missing conditioning before neural compute", "[unit][meanvc2][conditioning]") {
    meanvc2_context ctx;
    ctx.params = meanvc2_context_default_params();
    ctx.params.verbosity = 0;
    const float audio[] = {0.0f, 0.5f, -0.25f};
    int n_samples = 123;
    REQUIRE(meanvc2_convert(&ctx, audio, 3, &n_samples) == nullptr);
    REQUIRE(n_samples == 0);
    REQUIRE(ctx.speaker_emb.empty());
    n_samples = 123;
    REQUIRE(meanvc2_convert(nullptr, audio, 3, &n_samples) == nullptr);
    REQUIRE(n_samples == 0);
}

TEST_CASE("MeanVC2 destruction releases both weight partitions", "[unit][meanvc2][lifecycle]") {
    auto* ctx = new meanvc2_context();
    int freed = 0;
    // Count releases using ggml's buffer interface, without a model or GPU.
    ggml_backend_buffer_i iface = {};
    iface.free_buffer = [](ggml_backend_buffer_t buffer) { ++*static_cast<int*>(buffer->context); };
    ctx->backend_cpu = core_cpu_backend::init();
    ctx->backend = ctx->backend_cpu;
    REQUIRE(ctx->backend != nullptr);
    const auto buft = ggml_backend_get_default_buffer_type(ctx->backend);
    ctx->wl.buf = ggml_backend_buffer_init(buft, iface, &freed, 0);
    ctx->wl.buf_cpu = ggml_backend_buffer_init(buft, iface, &freed, 0);
    ctx->wl.ctx = ggml_init({64 * 1024, nullptr, true});
    REQUIRE(ctx->wl.ctx != nullptr);
    meanvc2_free(ctx);
    REQUIRE(freed == 2);
    meanvc2_free(nullptr);
}
