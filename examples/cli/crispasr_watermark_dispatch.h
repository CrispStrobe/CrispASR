// crispasr_watermark_dispatch.h — unified watermark dispatch.
//
// Routes watermark embed/detect to either:
//   1. AudioSeal neural watermark (when --watermark-model is set and GGUF loaded)
//   2. Built-in spread-spectrum watermark (fallback)
//
// The dispatcher is initialized once at startup. Thread-safe after init.

#pragma once

#include "audioseal.h"
#include "crispasr_watermark.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace crispasr_wm_dispatch {

// Global AudioSeal context (null = not loaded → use spread-spectrum)
inline audioseal_ctx*& get_ctx() {
    static audioseal_ctx* ctx = nullptr;
    return ctx;
}

// Initialize AudioSeal from GGUF path. Call once at startup.
// Returns true if loaded, false if not (falls back to spread-spectrum).
inline bool init(const std::string& model_path) {
    if (model_path.empty())
        return false;

    auto params = audioseal_default_params();
    params.verbosity = 1;
    get_ctx() = audioseal_init_from_file(model_path.c_str(), params);
    if (!get_ctx()) {
        fprintf(stderr, "crispasr: warning: could not load AudioSeal model '%s'; "
                        "falling back to spread-spectrum watermark\n",
                model_path.c_str());
        return false;
    }
    return true;
}

inline void shutdown() {
    if (get_ctx()) {
        audioseal_free(get_ctx());
        get_ctx() = nullptr;
    }
}

// Embed watermark into float32 PCM. Modifies in-place.
// If AudioSeal is loaded, resamples to 16kHz if needed, embeds, and
// resamples back. Otherwise uses spread-spectrum.
inline void embed(float* pcm, int n_samples, int sample_rate = 24000) {
    if (get_ctx()) {
        // AudioSeal operates at 16 kHz. If audio is at a different rate,
        // resample down, embed, resample back.
        if (sample_rate == 16000) {
            // Direct: embed in-place
            float* watermarked = audioseal_embed(get_ctx(), pcm, n_samples, nullptr);
            if (watermarked) {
                std::memcpy(pcm, watermarked, (size_t)n_samples * sizeof(float));
                std::free(watermarked);
                return;
            }
            // Fall through to spread-spectrum on failure
        } else {
            // Resample to 16 kHz
            int n_16k = (int)((int64_t)n_samples * 16000 / sample_rate);
            std::vector<float> resampled(n_16k);
            for (int i = 0; i < n_16k; i++) {
                float pos = (float)i * (float)sample_rate / 16000.0f;
                int s0 = (int)pos;
                int s1 = (s0 + 1 < n_samples) ? s0 + 1 : s0;
                float frac = pos - (float)s0;
                resampled[i] = pcm[s0] * (1.0f - frac) + pcm[s1] * frac;
            }

            // Embed at 16 kHz
            float* watermarked = audioseal_embed(get_ctx(), resampled.data(), n_16k, nullptr);
            if (watermarked) {
                // Compute watermark delta at 16 kHz
                for (int i = 0; i < n_16k; i++)
                    resampled[i] = watermarked[i] - resampled[i];
                std::free(watermarked);

                // Upsample the watermark delta back to original rate and add
                for (int i = 0; i < n_samples; i++) {
                    float pos = (float)i * 16000.0f / (float)sample_rate;
                    int s0 = (int)pos;
                    int s1 = (s0 + 1 < n_16k) ? s0 + 1 : s0;
                    float frac = pos - (float)s0;
                    float delta = resampled[s0] * (1.0f - frac) + resampled[s1] * frac;
                    pcm[i] += delta;
                }
                return;
            }
        }
    }
    // Fallback: spread-spectrum watermark
    crispasr_watermark_embed_impl(pcm, n_samples);
}

// Detect watermark in float32 PCM.
// Returns confidence in [0, 1]. If AudioSeal is loaded, resamples and
// uses neural detection. Otherwise uses spread-spectrum.
inline float detect(const float* pcm, int n_samples, int sample_rate = 24000) {
    if (get_ctx()) {
        // Resample to 16 kHz if needed
        std::vector<float> pcm_16k;
        const float* detect_pcm = pcm;
        int detect_n = n_samples;
        if (sample_rate != 16000) {
            detect_n = (int)((int64_t)n_samples * 16000 / sample_rate);
            pcm_16k.resize(detect_n);
            for (int i = 0; i < detect_n; i++) {
                float pos = (float)i * (float)sample_rate / 16000.0f;
                int s0 = (int)pos;
                int s1 = (s0 + 1 < n_samples) ? s0 + 1 : s0;
                float frac = pos - (float)s0;
                pcm_16k[i] = pcm[s0] * (1.0f - frac) + pcm[s1] * frac;
            }
            detect_pcm = pcm_16k.data();
        }
        int n_frames = 0;
        float* probs = audioseal_detect(get_ctx(), detect_pcm, detect_n, &n_frames, nullptr);
        if (probs && n_frames > 0) {
            // Average detection probability across frames
            double avg = 0.0;
            for (int i = 0; i < n_frames; i++)
                avg += probs[i];
            avg /= (double)n_frames;
            std::free(probs);
            return (float)avg;
        }
        if (probs) std::free(probs);
    }
    // Fallback: spread-spectrum
    return crispasr_watermark_detect_impl(pcm, n_samples);
}

} // namespace crispasr_wm_dispatch
