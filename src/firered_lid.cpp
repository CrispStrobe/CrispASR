// firered_lid.cpp — FireRedLID runtime (thin wrapper around firered_asr).
//
// The LID model reuses the same Conformer encoder + Transformer decoder
// as FireRedASR2-AED, just with a 120-class language vocabulary instead
// of 8667 BPE tokens. We load it via firered_asr_init_from_file and
// call firered_asr_transcribe — the first decoded token is the language.

#include "firered_lid.h"
#include "firered_asr.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

struct firered_lid_context {
    firered_asr_context* asr_ctx = nullptr;
    std::string last_result;
};

extern "C" struct firered_lid_context* firered_lid_init(const char* model_path, int n_threads) {
    firered_asr_context_params params = firered_asr_context_default_params();
    params.n_threads = n_threads > 0 ? n_threads : 4;
    params.verbosity = 0;

    firered_asr_context* asr = firered_asr_init_from_file(model_path, params);
    if (!asr)
        return nullptr;

    auto* ctx = new firered_lid_context();
    ctx->asr_ctx = asr;
    return ctx;
}

extern "C" void firered_lid_free(struct firered_lid_context* ctx) {
    if (!ctx)
        return;
    if (ctx->asr_ctx)
        firered_asr_free(ctx->asr_ctx);
    delete ctx;
}

extern "C" const char* firered_lid_detect(struct firered_lid_context* ctx, const float* samples, int n_samples,
                                          float* confidence) {
    if (!ctx || !ctx->asr_ctx || !samples || n_samples <= 0)
        return nullptr;

    // LID doesn't need long audio — cap at 5 seconds for speed.
    // The encoder is O(T²) per layer, so halving T gives 4x speedup.
    constexpr int kMaxLidSamples = 16000 * 5; // 5 seconds at 16kHz
    int n_use = (n_samples > kMaxLidSamples) ? kMaxLidSamples : n_samples;

    char* result = firered_asr_transcribe(ctx->asr_ctx, samples, n_use);
    if (!result)
        return nullptr;

    ctx->last_result = result;
    free(result);

    if (confidence)
        *confidence = 1.0f; // beam search returns argmax, no explicit probability

    return ctx->last_result.c_str();
}
