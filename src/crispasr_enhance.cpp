// crispasr_enhance.cpp — RNNoise-based audio enhancement for the
// transcribe pre-step. Same shape as crispasr_lid.cpp: consume PCM
// → produce PCM. State is allocated and freed per call so multiple
// worker isolates can run enhancement concurrently without coordination.

#include "crispasr_enhance.h"

extern "C" {
#include "rnnoise/rnnoise.h"
}

#include "core/audio_resample.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

constexpr int kRnnoiseFrameSize = 480;    // samples per RNNoise frame @ 48 kHz
constexpr int kInputSampleRate = 16000;   // ASR-canonical rate
constexpr int kRnnoiseSampleRate = 48000; // RNNoise internal rate
constexpr float kRnnoiseScale = 32768.0f; // classic rnnoise wants short-range floats

// RNNoise on 48 kHz mono PCM in [-1, 1]. Output is time-aligned with the
// input: RNNoise emits each frame one frame (10 ms) late, so one extra zero
// frame is processed and the first output frame dropped — the same
// compensation xiph's rnnoise_demo applies by not writing its first frame.
bool rnnoise_48k(const float* in, int n, float* out, bool verbose) {
    const int padded = ((n + kRnnoiseFrameSize + kRnnoiseFrameSize - 1) / kRnnoiseFrameSize) * kRnnoiseFrameSize;
    std::vector<float> buf(padded, 0.0f);
    for (int i = 0; i < n; ++i)
        buf[i] = in[i] * kRnnoiseScale;
    DenoiseState* st = rnnoise_create(NULL);
    if (st == nullptr) {
        if (verbose)
            std::fprintf(stderr, "[enhance] rnnoise_create failed\n");
        return false;
    }
    float frame[kRnnoiseFrameSize];
    for (int off = 0; off < padded; off += kRnnoiseFrameSize) {
        std::memcpy(frame, buf.data() + off, kRnnoiseFrameSize * sizeof(float));
        rnnoise_process_frame(st, frame, frame);
        std::memcpy(buf.data() + off, frame, kRnnoiseFrameSize * sizeof(float));
    }
    rnnoise_destroy(st);
    for (int i = 0; i < n; ++i)
        out[i] = buf[kRnnoiseFrameSize + i] / kRnnoiseScale;
    return true;
}

bool enhance_rnnoise(const float* in_samples, int n_samples, float* out_samples, bool verbose) {
    if (in_samples == nullptr || out_samples == nullptr || n_samples <= 0) {
        if (verbose)
            std::fprintf(stderr, "[enhance] invalid args\n");
        return false;
    }
    // 16 kHz -> 48 kHz -> RNNoise -> 16 kHz with the shared zero-phase
    // Kaiser-windowed polyphase resampler. (miniaudio's linear resampler used
    // before attenuated 4-8 kHz by up to 6 dB and cost 0.6 dB SNR against
    // RNNoise run on properly resampled 48 kHz audio.)
    std::vector<float> up = core_audio::resample_polyphase(in_samples, n_samples, kInputSampleRate, kRnnoiseSampleRate);
    if (up.empty()) {
        if (verbose)
            std::fprintf(stderr, "[enhance] upsample failed\n");
        return false;
    }
    std::vector<float> den(up.size());
    if (!rnnoise_48k(up.data(), (int)up.size(), den.data(), verbose))
        return false;
    std::vector<float> down =
        core_audio::resample_polyphase(den.data(), (int)den.size(), kRnnoiseSampleRate, kInputSampleRate);
    const int copy = std::min(n_samples, (int)down.size());
    std::memcpy(out_samples, down.data(), copy * sizeof(float));
    if (copy < n_samples)
        std::memset(out_samples + copy, 0, (n_samples - copy) * sizeof(float));
    return true;
}

} // namespace

bool crispasr_enhance_rnnoise_48k(const float* in_samples, int n_samples, float* out_samples, bool verbose) {
    if (in_samples == nullptr || out_samples == nullptr || n_samples <= 0)
        return false;
    return rnnoise_48k(in_samples, n_samples, out_samples, verbose);
}

bool crispasr_enhance_audio(const float* in_samples, int n_samples, float* out_samples,
                            const CrispasrEnhanceOptions& opts) {
    switch (opts.method) {
    case CrispasrEnhanceMethod::Rnnoise:
        return enhance_rnnoise(in_samples, n_samples, out_samples, opts.verbose);
    }
    return false;
}
