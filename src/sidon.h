// Sidon speech-restoration runtime (w2v-BERT predictor + continuous DAC).
#pragma once

#include <vector>

struct sidon_context;

struct sidon_context_params {
    int n_threads;
    int verbosity;
    bool use_gpu;
};

sidon_context_params sidon_context_default_params();
sidon_context* sidon_init_from_file(const char* path, sidon_context_params params);
void sidon_free(sidon_context* ctx);

// Input is 16 kHz mono float PCM; output is 48 kHz mono float PCM.
std::vector<float> sidon_restore(sidon_context* ctx, const float* samples, int n_samples);
