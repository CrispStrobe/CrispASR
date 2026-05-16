// crispasr_enhance.cpp — RNNoise-based audio enhancement for the
// transcribe pre-step. Same shape as crispasr_lid.cpp: consume PCM
// → produce PCM. State is allocated and freed per call so multiple
// worker isolates can run enhancement concurrently without coordination.

#include "crispasr_enhance.h"

extern "C" {
#include "rnnoise/rnnoise.h"
}

// miniaudio's IMPLEMENTATION symbols live in crispasr_audio.cpp;
// here we just want the header declarations for ma_resampler_*.
#include "miniaudio.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

constexpr int   kRnnoiseFrameSize  = 480;     // samples per RNNoise frame @ 48 kHz
constexpr int   kInputSampleRate   = 16000;   // ASR-canonical rate
constexpr int   kRnnoiseSampleRate = 48000;   // RNNoise internal rate
constexpr float kRnnoiseScale      = 32768.0f; // classic rnnoise wants short-range floats

bool enhance_rnnoise(const float* in_samples, int n_samples, float* out_samples, bool verbose) {
    if (in_samples == nullptr || out_samples == nullptr || n_samples <= 0) {
        if (verbose) std::fprintf(stderr, "[enhance] invalid args\n");
        return false;
    }

    // ---- 16 kHz -> 48 kHz upsample ----
    ma_resampler_config up_cfg = ma_resampler_config_init(
        ma_format_f32, /*channels=*/1, kInputSampleRate, kRnnoiseSampleRate,
        ma_resample_algorithm_linear);
    ma_resampler up;
    if (ma_resampler_init(&up_cfg, NULL, &up) != MA_SUCCESS) {
        if (verbose) std::fprintf(stderr, "[enhance] upsampler init failed\n");
        return false;
    }

    // Worst-case output count for 3x upsample plus headroom for the
    // resampler's internal filter tail.
    const size_t up_cap = static_cast<size_t>(n_samples) * 3 + kRnnoiseFrameSize;
    std::vector<float> up_buf(up_cap, 0.0f);

    ma_uint64 frames_in  = static_cast<ma_uint64>(n_samples);
    ma_uint64 frames_out = up_cap;
    ma_result rc = ma_resampler_process_pcm_frames(
        &up, in_samples, &frames_in, up_buf.data(), &frames_out);
    ma_resampler_uninit(&up, NULL);
    if (rc != MA_SUCCESS) {
        if (verbose) std::fprintf(stderr, "[enhance] upsample failed (rc=%d)\n", (int)rc);
        return false;
    }

    int up_count = static_cast<int>(frames_out);
    // Pad to a multiple of RNNoise's frame size so the frame loop
    // doesn't lose the tail. The padding is zero so it doesn't
    // affect downsampled output content.
    int padded = ((up_count + kRnnoiseFrameSize - 1) / kRnnoiseFrameSize) * kRnnoiseFrameSize;
    if (padded > static_cast<int>(up_buf.size())) up_buf.resize(padded, 0.0f);

    // ---- scale [-1, 1] -> [-32768, 32767] ----
    for (int i = 0; i < up_count; ++i) up_buf[i] *= kRnnoiseScale;

    // ---- RNNoise frame loop ----
    DenoiseState* st = rnnoise_create(NULL);
    if (st == nullptr) {
        if (verbose) std::fprintf(stderr, "[enhance] rnnoise_create failed\n");
        return false;
    }
    float frame[kRnnoiseFrameSize];
    for (int off = 0; off < padded; off += kRnnoiseFrameSize) {
        std::memcpy(frame, up_buf.data() + off, kRnnoiseFrameSize * sizeof(float));
        rnnoise_process_frame(st, frame, frame);
        std::memcpy(up_buf.data() + off, frame, kRnnoiseFrameSize * sizeof(float));
    }
    rnnoise_destroy(st);

    // ---- scale back to [-1, 1] ----
    for (int i = 0; i < padded; ++i) up_buf[i] /= kRnnoiseScale;

    // ---- 48 kHz -> 16 kHz downsample ----
    ma_resampler_config down_cfg = ma_resampler_config_init(
        ma_format_f32, /*channels=*/1, kRnnoiseSampleRate, kInputSampleRate,
        ma_resample_algorithm_linear);
    ma_resampler down;
    if (ma_resampler_init(&down_cfg, NULL, &down) != MA_SUCCESS) {
        if (verbose) std::fprintf(stderr, "[enhance] downsampler init failed\n");
        return false;
    }

    std::vector<float> down_buf(static_cast<size_t>(n_samples) + 64, 0.0f);
    ma_uint64 dn_in  = static_cast<ma_uint64>(padded);
    ma_uint64 dn_out = down_buf.size();
    rc = ma_resampler_process_pcm_frames(
        &down, up_buf.data(), &dn_in, down_buf.data(), &dn_out);
    ma_resampler_uninit(&down, NULL);
    if (rc != MA_SUCCESS) {
        if (verbose) std::fprintf(stderr, "[enhance] downsample failed (rc=%d)\n", (int)rc);
        return false;
    }

    // Caller expects exactly n_samples written. Pad with zero if
    // resampler under-produced (it can drop a couple of frames as
    // internal filter delay).
    int produced = static_cast<int>(dn_out);
    int copy = std::min(n_samples, produced);
    std::memcpy(out_samples, down_buf.data(), copy * sizeof(float));
    if (copy < n_samples) {
        std::memset(out_samples + copy, 0, (n_samples - copy) * sizeof(float));
    }
    return true;
}

}  // anonymous

bool crispasr_enhance_audio(const float* in_samples, int n_samples, float* out_samples,
                            const CrispasrEnhanceOptions& opts) {
    switch (opts.method) {
        case CrispasrEnhanceMethod::Rnnoise:
            return enhance_rnnoise(in_samples, n_samples, out_samples, opts.verbose);
    }
    return false;
}
