// Diagnostic adapter: use the actual production sampler inside original Dia.
// No probability/filter implementation is duplicated here.
#include "../../src/dia_sampling.h"
#include <new>
#include <cstdlib>
#include <cstring>

extern "C" {
void* dia_sampler_create(uint32_t seed) {
    return new (std::nothrow) std::mt19937(seed);
}

void dia_sampler_free(void* handle) {
    delete static_cast<std::mt19937*>(handle);
}

int dia_sampler_rows(void* handle, const float* logits, int rows, int vocab, float temperature, float top_p, int top_k,
                     int64_t* output) {
    if (!handle || !logits || !output || rows <= 0 || vocab <= 0 || !std::isfinite(temperature) || temperature < 0 ||
        !std::isfinite(top_p) || top_p <= 0 || top_p > 1 || top_k < 0 || top_k > vocab)
        return 1;
    // Validate every row before advancing the state, including masked -inf.
    for (int row = 0; row < rows; ++row) {
        bool finite = false;
        for (int col = 0; col < vocab; ++col) {
            const float value = logits[(size_t)row * vocab + col];
            if (std::isnan(value) || value == INFINITY)
                return 2;
            finite = finite || std::isfinite(value);
        }
        if (!finite)
            return 2;
    }
    try {
        auto& rng = *static_cast<std::mt19937*>(handle);
        std::vector<float> noise;
        const char* candidate = std::getenv("CRISPASR_DIA_PORTABLE_RNG");
        if (candidate && std::strcmp(candidate, "1") == 0 && temperature > 0.0f) {
            noise.resize((size_t)rows * vocab);
            dia_rng::exponential_batch(noise.data(), noise.size(), rng);
        }
        for (int row = 0; row < rows; ++row)
            output[row] = dia_sample_token(logits + (size_t)row * vocab, (uint32_t)vocab, temperature, top_p, top_k,
                                           rng, noise.empty() ? nullptr : noise.data() + (size_t)row * vocab);
    } catch (...) {
        return 3;
    }
    return 0;
}
}
