// Sidon live integration test — model load + 16 kHz speech restoration.
//
// Requires CRISPASR_MODEL_SIDON to point to a Sidon GGUF. Skips cleanly
// when the model is not available.

#include <catch2/catch_test_macros.hpp>

#include "sidon.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

static std::vector<float> load_wav_16k(const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f)
        return {};

    fseek(f, 0, SEEK_END);
    const long size = ftell(f) - 44;
    fseek(f, 44, SEEK_SET);
    if (size <= 0 || size % (long)sizeof(int16_t) != 0) {
        fclose(f);
        return {};
    }

    std::vector<int16_t> raw((size_t)size / sizeof(int16_t));
    const size_t read = fread(raw.data(), sizeof(int16_t), raw.size(), f);
    fclose(f);
    if (read != raw.size())
        return {};

    std::vector<float> pcm(raw.size());
    for (size_t i = 0; i < raw.size(); ++i)
        pcm[i] = raw[i] / 32768.0f;
    return pcm;
}

TEST_CASE("sidon speech restoration", "[integration][sidon]") {
    const char* model_path = std::getenv("CRISPASR_MODEL_SIDON");
    if (!model_path || !*model_path)
        SKIP("CRISPASR_MODEL_SIDON not set");

    auto params = sidon_context_default_params();
    params.verbosity = 0;
    auto* ctx = sidon_init_from_file(model_path, params);
    REQUIRE(ctx != nullptr);

    const auto input = load_wav_16k("samples/jfk.wav");
    REQUIRE(!input.empty());

    const auto output = sidon_restore(ctx, input.data(), (int)input.size());
    REQUIRE(output.size() > input.size() * 2);
    REQUIRE(std::all_of(output.begin(), output.end(), [](float sample) { return std::isfinite(sample); }));

    float peak = 0.0f;
    for (float sample : output)
        peak = std::max(peak, std::fabs(sample));
    CHECK(peak > 0.01f);

    sidon_free(ctx);
}
