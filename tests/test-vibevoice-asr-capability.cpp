// #589: tokenizer encoders also exist in base TTS checkpoints; they do not
// establish that the language model was trained to transcribe speech.
#include "vibevoice.h"
#include "ggml.h"
#include "gguf.h"

#include <catch2/catch_test_macros.hpp>
#include <chrono>
#include <cstdio>
#if defined(_WIN32)
#include <io.h>
#define dup _dup
#define dup2 _dup2
#define close _close
#else
#include <unistd.h>
#endif
#include <filesystem>
#include <string>

namespace {
struct Fixture {
    std::filesystem::path path;
    vibevoice_context* ctx = nullptr;

    Fixture(const char* arch, int supports_asr, bool prediction_head, bool acoustic = true, bool semantic = true,
            bool streaming = false, bool decoder = false) {
        path = std::filesystem::temp_directory_path() /
               ("vibevoice-capability-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) +
                ".gguf");
        ggml_init_params ip = {4096, nullptr, false};
        auto* tensors = ggml_init(ip);
        auto* file = gguf_init_empty();
        gguf_set_val_str(file, "general.architecture", arch);
        gguf_set_val_u32(file, "vibevoice.asr_streaming", streaming);
        gguf_set_val_u32(file, "vibevoice.has_decoder", decoder);
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
struct StderrCapture {
    FILE* file = std::tmpfile();
    int saved = -1;
    StderrCapture() {
        REQUIRE(file != nullptr);
        std::fflush(stderr);
        saved = dup(fileno(stderr));
        REQUIRE(saved >= 0);
        REQUIRE(dup2(fileno(file), fileno(stderr)) >= 0);
    }
    ~StderrCapture() {
        std::fflush(stderr);
        dup2(saved, fileno(stderr));
        close(saved);
        std::fclose(file);
    }
    std::string text() {
        std::fflush(stderr);
        std::rewind(file);
        std::string result;
        char buffer[256];
        while (const auto count = std::fread(buffer, 1, sizeof(buffer), file))
            result.append(buffer, count);
        return result;
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
    Fixture streaming("vibevoice-asr", 1, false, true, true, true);
    CHECK(vibevoice_has_asr(streaming.ctx));
    CHECK(vibevoice_is_asr_streaming(streaming.ctx));
    Fixture conflicting("vibevoice-asr", 1, true);
    CHECK_FALSE(vibevoice_has_asr(conflicting.ctx));
    Fixture missing_encoder("vibevoice-asr", 1, false, false);
    CHECK_FALSE(vibevoice_has_asr(missing_encoder.ctx));
    CHECK_FALSE(vibevoice_has_asr(nullptr));
}

TEST_CASE("VibeVoice ASR checkpoints refuse synthesis with an explicit task error", "[unit][vibevoice]") {
    for (bool streaming : {false, true}) {
        // Even misleading decoder metadata must not send an ASR task into TTS.
        Fixture asr("vibevoice-asr", 1, false, true, true, streaming, true);
        StderrCapture capture;
        int samples = 0;
        CHECK(vibevoice_synthesize(asr.ctx, "Hello", &samples) == nullptr);
        CHECK(capture.text().find("ASR checkpoint and does not support TTS") != std::string::npos);
        CHECK(capture.text().find("--include-decoder") == std::string::npos);
    }
}

TEST_CASE("VibeVoice TTS task cannot open an ASR streaming session", "[unit][vibevoice]") {
    Fixture tts("vibevoice-tts", 0, true, true, true, true);
    CHECK(vibevoice_stream_open(tts.ctx, nullptr) == nullptr);
}

TEST_CASE("VibeVoice synthesis checkpoint encoders do not trigger the ASR synthesis refusal", "[unit][vibevoice]") {
    Fixture legacy_tts("vibevoice-asr", -1, true);
    StderrCapture capture;
    int samples = 0;
    CHECK(vibevoice_synthesize(legacy_tts.ctx, "Hello", &samples) == nullptr);
    // This minimal fixture has no audio decoder; the task guard must allow it
    // through to the ordinary missing-decoder validation rather than call it ASR.
    CHECK(capture.text().find("model lacks decoder tensors") != std::string::npos);
    CHECK(capture.text().find("ASR checkpoint and does not support TTS") == std::string::npos);
}
