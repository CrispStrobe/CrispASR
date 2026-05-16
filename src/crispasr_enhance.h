// crispasr_enhance.h — audio-enhancement helpers.
//
// Single backend so far:
//
//   * `CrispasrEnhanceMethod::Rnnoise` — RNNoise (xiph/rnnoise v0.1,
//     classic GRU model, ~425 KB weights embedded in libcrispasr).
//     16 kHz mono float32 input/output in [-1, 1]. Internally
//     upsamples to 48 kHz, runs RNNoise's 480-sample / 10 ms frames,
//     and downsamples back. State is allocated and freed per call —
//     callers can use this freely from worker isolates.
//
// Shared by the CLI shim and by the C-ABI
// `crispasr_enhance_audio_rnnoise` in crispasr_c_api.cpp. Same
// "consume PCM → produce PCM" layering as crispasr_lid.{h,cpp}.

#pragma once

#include <stdint.h>

enum class CrispasrEnhanceMethod {
    Rnnoise = 0,
};

struct CrispasrEnhanceOptions {
    CrispasrEnhanceMethod method = CrispasrEnhanceMethod::Rnnoise;
    bool verbose = false;
};

/// Apply enhancement to a 16 kHz mono f32 buffer. Writes exactly
/// `n_samples` floats into `out` (same length, [-1, 1]). Returns
/// true on success; on failure the reason is printed to stderr
/// when `opts.verbose` is true, and `out` is left untouched.
bool crispasr_enhance_audio(const float* in_samples, int n_samples, float* out_samples,
                            const CrispasrEnhanceOptions& opts);
