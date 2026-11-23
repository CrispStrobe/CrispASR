// test-vad-thresholds.cpp — verify VAD behavior across different thresholds.

#include "crispasr.h"
#include "common-crispasr.h"
#include <vector>
#include <iostream>
#include <cassert>
#include <cstdio>
#include <cmath>
#include <cstdlib>

static const char* env_or(const char* name, const char* fallback) {
    const char* v = std::getenv(name);
    return (v && *v) ? v : fallback;
}


// Exit code ctest treats as SKIPPED (SKIP_RETURN_CODE 77): these tests need a
// real whisper model that is not part of the repository.
static bool file_exists(const char* path) {
    FILE* f = std::fopen(path, "rb");
    if (!f)
        return false;
    std::fclose(f);
    return true;
}

#define VAD_CHECK(cond)                                                                                                \
    do {                                                                                                               \
        if (!(cond)) {                                                                                                 \
            std::fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #cond);                              \
            return 1;                                                                                                  \
        }                                                                                                              \
    } while (0)

int main() {
    std::string sample_path = env_or("CRISPASR_AUDIO_EN", SAMPLE_PATH);
    std::string vad_model_path = env_or("CRISPASR_VAD_MODEL", VAD_MODEL_PATH);
    const std::string whisper_model_path = env_or("CRISPASR_MODEL_WHISPER", CRISPASR_MODEL_PATH);
    for (const std::string& path : {whisper_model_path, vad_model_path}) {
        if (!file_exists(path.c_str())) {
            std::fprintf(stderr, "SKIP: model not found: %s (set CRISPASR_MODEL_WHISPER / CRISPASR_VAD_MODEL)\n",
                         path.c_str());
            return 77;
        }
    }

    std::vector<float> pcmf32;
    std::vector<std::vector<float>> pcmf32s;
    if (!read_audio_data(sample_path.c_str(), pcmf32, pcmf32s, false)) {
        std::cerr << "Failed to read audio: " << sample_path << std::endl;
        return 1;
    }

    struct whisper_context_params cparams = whisper_context_default_params();
    struct whisper_context* wctx = whisper_init_from_file_with_params(whisper_model_path.c_str(), cparams);
    VAD_CHECK(wctx != nullptr);

    float thresholds[] = {0.1f, 0.5f, 0.9f};
    for (float thold : thresholds) {
        struct whisper_full_params wparams = whisper_full_default_params(CRISPASR_SAMPLING_GREEDY);
        wparams.vad = true;
        wparams.vad_model_path = vad_model_path.c_str();
        wparams.vad_params.threshold = thold;

        std::cout << "Testing VAD threshold: " << thold << std::endl;
        int ret = whisper_full_parallel(wctx, wparams, pcmf32.data(), pcmf32.size(), 1);
        VAD_CHECK(ret == 0);

        int n_segments = whisper_full_n_segments(wctx);
        std::cout << "  n_segments: " << n_segments << std::endl;
        // Higher threshold should generally result in fewer segments or fewer tokens
    }

    whisper_free(wctx);
    return 0;
}
