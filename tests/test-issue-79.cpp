// test-issue-79.cpp — regression test for VibeVoice ASR KV cache reallocation.
//
// Reproduces #79: VibeVoice ASR crashes when transcribing a short buffer
// followed by a longer buffer because the KV cache wasn't reallocated.
#include "vibevoice.h"
#include <vector>
#include <iostream>
#include <cmath>
#include <cassert>
#include <cstdlib>

void generate_sine(std::vector<float>& pcm, float freq, int sr, float duration_s) {
    int n = (int)(sr * duration_s);
    pcm.resize(n);
    for (int i = 0; i < n; i++) {
        pcm[i] = std::sin(2.0f * M_PI * freq * i / sr);
    }
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <path_to_vibevoice_asr_gguf>" << std::endl;
        return 1;
    }

    const char* model_path = argv[1];
    struct vibevoice_context_params params = vibevoice_context_default_params();
    params.verbosity = 1;
    params.use_gpu = false; // CPU is enough for crash reproduction
    params.max_new_tokens = 1; // Speed up test: we only care about KV allocation

    std::cout << "Initializing VibeVoice context with: " << model_path << std::endl;
    struct vibevoice_context* ctx = vibevoice_init_from_file(model_path, params);
    if (!ctx) {
        std::cerr << "Failed to load model" << std::endl;
        return 1;
    }

    const int sr = 24000;
    std::vector<float> pcm_short;
    std::vector<float> pcm_long;

    // Short buffer: 0.5 seconds
    generate_sine(pcm_short, 440.0f, sr, 0.5f);
    // Long buffer: 2.0 seconds (enough to trigger reallocation if first was small)
    generate_sine(pcm_long, 440.0f, sr, 2.0f);

    std::cout << "Step 1: Transcribing short buffer (" << pcm_short.size() << " samples)..." << std::endl;
    char* text1 = vibevoice_transcribe(ctx, pcm_short.data(), (int)pcm_short.size());
    if (text1) {
        std::cout << "Result 1: " << text1 << std::endl;
        free(text1);
    }

    std::cout << "Step 2: Transcribing long buffer (" << pcm_long.size() << " samples)..." << std::endl;
    // BEFORE FIX: This would crash with GGML_ASSERT in ggml_view_3d
    char* text2 = vibevoice_transcribe(ctx, pcm_long.data(), (int)pcm_long.size());
    if (text2) {
        std::cout << "Result 2: " << text2 << std::endl;
        free(text2);
    }

    std::cout << "Step 3: Transcribing even longer buffer (4.0 seconds)..." << std::endl;
    std::vector<float> pcm_vlong;
    generate_sine(pcm_vlong, 440.0f, sr, 4.0f);
    char* text3 = vibevoice_transcribe(ctx, pcm_vlong.data(), (int)pcm_vlong.size());
    if (text3) {
        std::cout << "Result 3: " << text3 << std::endl;
        free(text3);
    }

    vibevoice_free(ctx);
    std::cout << "Test passed successfully (no crash)." << std::endl;

    return 0;
}
