// frcrn_se.h — ClearerVoice-Studio FRCRN_SE_16K speech enhancement (denoise).
//
// Native ggml port of modelscope/ClearerVoice-Studio's FRCRN (DCCRN-style
// ConvSTFT + two cascaded complex UNets with FSMN memories and complex SE
// layers, complex ratio mask, ConviSTFT). frcrn_se_enhance() reproduces
// upstream's whole inference path: DataReader `audio_norm` input
// normalisation, `decode_one_audio_frcrn_se_16k` padding / segmentation
// (> 120 s: 1 s windows, 0.75 s stride) and the final de-normalisation, and
// returns exactly as many samples as it was given.
//
// Convert with models/convert-frcrn-se-to-gguf.py (arch "frcrn-se").
// See docs/frcrn-se.md.

#pragma once

#include <vector>

struct CrispasrEnhanceModelOps;
struct frcrn_se_context;

struct frcrn_se_params {
    int n_threads = 4;
    bool use_gpu = false; // also enabled by CRISPASR_FRCRN_SE_GPU=1
    bool verbose = false;
};

frcrn_se_context* frcrn_se_init_from_file(const char* path, frcrn_se_params params);

/// Native sample rate (16000).
int frcrn_se_sample_rate(const frcrn_se_context* ctx);

/// Enhance mono 16 kHz PCM. `out` receives exactly `n` samples.
bool frcrn_se_enhance(frcrn_se_context* ctx, const float* in, int n, std::vector<float>& out);

void frcrn_se_free(frcrn_se_context* ctx);

/// Registration entry for src/crispasr_enhancer.cpp.
const CrispasrEnhanceModelOps* frcrn_se_enhance_ops();
