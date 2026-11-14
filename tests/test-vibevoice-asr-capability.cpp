// #589: tokenizer encoders also exist in base TTS checkpoints; they do not
// establish that the language model was trained to transcribe speech.
#include "vibevoice.h"
#include "ggml.h"
#include "gguf.h"

#include <catch2/catch_test_macros.hpp>
#include <chrono>
#include <filesystem>
#include <string>

namespace {
struct Fixture {
    std::filesystem::path path;
    vibevoice_context* ctx = nullptr;

    Fixture(const char* arch, int supports_asr, bool prediction_head, bool acoustic = true, bool semantic = true) {
        path = std::filesystem::temp_directory_path() /
               ("vibevoice-capability-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) +
                ".gguf");
        ggml_init_params ip = {4096, nullptr, false};
        auto* tensors = ggml_init(ip);
        auto* file = gguf_init_empty();
        gguf_set_val_str(file, "general.architecture", arch);
        if (supports_asr >= 0)
            gguf_set_val_u32(file, "vibevoice.supports_asr", supports_asr);
        for (const char* name :
             {acoustic ? "at_enc.fixture" : "unused.acoustic", semantic ? "st_enc.fixture" : "unused.semantic",
              prediction_head ? "pred.fixture" : "unused.prediction"}) {
            auto* t = ggml_new_tensor_1d(tensors, GGML_TYPE_F32, 1);
            ggml_set_name(t, name);
            *static_cast<float*>(t->data) = 0.0f;
            gguf_add_tensor(file, t);
        }
        const bool written = gguf_write_to_file(file, path.string().c_str(), false);
        gguf_free(file);
        ggml_free(tensors);
        REQUIRE(written);
        auto params = vibevoice_context_default_params();
        params.use_gpu = false;
        params.verbosity = 0;
        ctx = vibevoice_init_from_file(path.string().c_str(), params);
        REQUIRE(ctx != nullptr);
    }
    ~Fixture() {
        vibevoice_free(ctx);
        std::filesystem::remove(path);
    }
};
} // namespace

TEST_CASE("VibeVoice base TTS encoders do not grant ASR capability", "[unit][vibevoice]") {
    // Legacy converter mislabeled base 1.5B/7B as vibevoice-asr; the retained
    // diffusion prediction tensors distinguish these from the ASR checkpoint.
    Fixture legacy("vibevoice-asr", -1, true);
    REQUIRE_FALSE(vibevoice_has_asr(legacy.ctx));
    const float sample = 0.0f;
    REQUIRE(vibevoice_transcribe(legacy.ctx, &sample, 1) == nullptr);
    REQUIRE(vibevoice_transcribe_with_probs(legacy.ctx, &sample, 1) == nullptr);
    // This is the exact entry point called by the session C ABI dispatch.
    REQUIRE(vibevoice_transcribe_with_probs_and_context(legacy.ctx, &sample, 1, "hotword") == nullptr);
}

TEST_CASE("VibeVoice explicit task metadata rejects TTS even without synthesis tensors", "[unit][vibevoice]") {
    Fixture declared("vibevoice-asr", 0, false);
    CHECK_FALSE(vibevoice_has_asr(declared.ctx));
    Fixture architecture("vibevoice-tts", -1, false);
    CHECK_FALSE(vibevoice_has_asr(architecture.ctx));
}

TEST_CASE("VibeVoice ASR task metadata preserves legacy and streaming checkpoints", "[unit][vibevoice]") {
    Fixture legacy("vibevoice-asr", -1, false);
    CHECK(vibevoice_has_asr(legacy.ctx));
    Fixture declared("vibevoice-asr", 1, false);
    CHECK(vibevoice_has_asr(declared.ctx));
    Fixture conflicting("vibevoice-asr", 1, true);
    CHECK_FALSE(vibevoice_has_asr(conflicting.ctx));
    Fixture missing_encoder("vibevoice-asr", 1, false, false);
    CHECK_FALSE(vibevoice_has_asr(missing_encoder.ctx));
    CHECK_FALSE(vibevoice_has_asr(nullptr));
}
