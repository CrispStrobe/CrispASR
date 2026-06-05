// src/core/istft.h — CPU inverse Short-Time Fourier Transform (header-only).
//
// Hoists the overlap-add iSTFT that three TTS backends duplicate inline:
//
//   outetts_wavtok.cpp  — WavTokenizer Vocos (n_fft=1280, hop=320)
//   kokoro.cpp          — Kokoro iSTFTNet   (n_fft=20,   hop=5)
//   cosyvoice3_tts.cpp  — CosyVoice3 HiFT  (n_fft=16,   hop=4)
//
// ---------------------------------------------------------------------------
// Per-source adoption verdict (audited 2026-05-31):
//
//   outetts_wavtok.cpp — FAITHFUL. Defaults match (trim=TRIM_SAME,
//                        clamp=1.0, win_eps=1e-8, zero_below_eps=false).
//   kokoro.cpp         — FAITHFUL-WITH-CONFIG. Pass win_eps=1e-11f and
//                        zero_below_eps=true (kokoro writes 0.0 for
//                        below-eps window-sum positions instead of the
//                        un-normalized value).
//   cosyvoice3_tts.cpp — DIVERGENT, do NOT adopt as-is. CosyVoice3 HiFT
//                        uses a different output-length contract
//                        (T_mel*480), front-only trimming, and transposed
//                        channel-major input. This header does NOT serve
//                        it without restructuring; listed above only for
//                        the n_fft/hop reference.
//
// Future consumers: Zonos, SpeechT5, any Vocos/HiFi-GAN variant.
//
// All three follow the same pattern:
//   1. For each STFT frame: reconstruct complex spectrum from
//      magnitude + phase, then compute the real-valued IRFFT using
//      Hermitian symmetry of the half-spectrum (N/2+1 bins → N samples).
//   2. Overlap-add with a Hann window.
//   3. Normalize by the squared-window sum (COLA reconstruction).
//   4. Trim padding (center=True strips n_fft/2 from each end, or
//      same-padding strips (n_fft - hop)/2 from each end).
//
// The unified API accepts configurable n_fft, hop, and trim mode so all
// three backends can share a single implementation.

#pragma once

#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace core_istft {

// Trim mode after overlap-add reconstruction.
enum TrimMode {
    // center=True: trim n_fft/2 samples from each end (torch.istft default).
    TRIM_CENTER,
    // same-padding: trim (n_fft - hop)/2 from each end (WavTokenizer style).
    TRIM_SAME,
    // No trimming — return the full OLA buffer.
    TRIM_NONE,
};

// Half-spectrum inverse RFFT using Hermitian symmetry.
//
// Given n_freq = N/2+1 complex bins (spec_re, spec_im), computes the
// N-point real IFFT:
//   x[n] = (1/N) * ( X[0].re
//                   + X[N/2].re * (-1)^n    [if N even]
//                   + 2 * sum_{k=1}^{N/2-1} Re(X[k] * exp(j*2*pi*k*n/N)) )
//
// O(N^2) but N is small (16-1280) so the simplicity wins over FFT.
static inline void irfft_hermitian(const float* spec_re, const float* spec_im, int N, float* out) {
    const int half = N / 2;
    const double inv_N = 1.0 / N;

    for (int n = 0; n < N; n++) {
        double val = spec_re[0]; // DC (always real)
        if (N % 2 == 0) {
            // Nyquist bin: X[N/2] is real for real signals.
            val += spec_re[half] * ((n & 1) ? -1.0 : 1.0);
        }
        double angle_base = 2.0 * M_PI * n * inv_N;
        for (int k = 1; k < half; k++) {
            double angle = angle_base * k;
            val += 2.0 * (spec_re[k] * std::cos(angle) - spec_im[k] * std::sin(angle));
        }
        out[n] = (float)(val * inv_N);
    }
}

// Build a periodic Hann window of length N:
//   w[n] = 0.5 * (1 - cos(2*pi*n / N))
// (periodic = divide by N, not N-1; matches torch.hann_window(N, periodic=True)).
static inline void hann_periodic(int N, float* out) {
    for (int i = 0; i < N; i++) {
        out[i] = 0.5f * (1.0f - std::cos(2.0f * (float)M_PI * (float)i / (float)N));
    }
}

// Core iSTFT: magnitude + phase (half-spectrum) → time-domain PCM.
//
// Parameters:
//   mag, phase  — (T_frames, n_freq) row-major, where n_freq = n_fft/2 + 1.
//                 Element (t, f) is at offset t * n_freq + f.
//   n_fft       — FFT size (window length).
//   hop         — hop length in samples.
//   T_frames    — number of STFT frames.
//   window      — optional window of length n_fft. If nullptr, a periodic
//                 Hann window is synthesized.
//   trim        — how to trim the output (see TrimMode).
//   out_len     — if non-null and trim == TRIM_NONE, receives the raw
//                 OLA buffer length. For TRIM_CENTER and TRIM_SAME, receives
//                 the trimmed length.
//   clamp       — if > 0, clamp output samples to [-clamp, +clamp].
//                 Pass 0 or negative to skip clamping.
//   win_eps     — window-sum threshold below which a sample is NOT
//                 normalized (avoids divide-by-tiny). Default 1e-8f
//                 (faithful to outetts_wavtok). Kokoro uses 1e-11f.
//   zero_below_eps — if true, write 0.0 for positions whose window-sum is
//                    <= win_eps (kokoro behavior). If false (default,
//                    outetts), leave the un-normalized accumulated value.
//
// Returns a vector of PCM samples.
static inline std::vector<float> istft(const float* mag, const float* phase, int n_fft, int hop, int T_frames,
                                       const float* window = nullptr, TrimMode trim = TRIM_CENTER, float clamp = 0.0f,
                                       float win_eps = 1e-8f, bool zero_below_eps = false) {
    const int n_freq = n_fft / 2 + 1;
    const int ola_len = (T_frames - 1) * hop + n_fft;

    // Build window if not provided.
    std::vector<float> hann_buf;
    if (!window) {
        hann_buf.resize(n_fft);
        hann_periodic(n_fft, hann_buf.data());
        window = hann_buf.data();
    }

    // Overlap-add buffers.
    std::vector<float> output(ola_len, 0.0f);
    std::vector<float> win_sum(ola_len, 0.0f);

    // Per-frame scratch.
    std::vector<float> frame_re(n_freq);
    std::vector<float> frame_im(n_freq);
    std::vector<float> frame_out(n_fft);

    for (int t = 0; t < T_frames; t++) {
        // Build complex spectrum from magnitude + phase.
        const float* m_row = mag + (size_t)t * n_freq;
        const float* p_row = phase + (size_t)t * n_freq;
        for (int f = 0; f < n_freq; f++) {
            frame_re[f] = m_row[f] * std::cos(p_row[f]);
            frame_im[f] = m_row[f] * std::sin(p_row[f]);
        }

        // Inverse RFFT.
        irfft_hermitian(frame_re.data(), frame_im.data(), n_fft, frame_out.data());

        // Overlap-add with window.
        int offset = t * hop;
        for (int i = 0; i < n_fft && (offset + i) < ola_len; i++) {
            float w = window[i];
            output[offset + i] += frame_out[i] * w;
            win_sum[offset + i] += w * w;
        }
    }

    // Normalize by squared-window sum (COLA reconstruction).
    for (int i = 0; i < ola_len; i++) {
        if (win_sum[i] > win_eps) {
            output[i] /= win_sum[i];
        } else if (zero_below_eps) {
            // Kokoro writes 0.0 for below-eps positions instead of
            // leaving the un-normalized accumulated value.
            output[i] = 0.0f;
        }
    }

    // Trim based on mode.
    int trim_left = 0;
    int trim_right = 0;
    switch (trim) {
    case TRIM_CENTER:
        trim_left = n_fft / 2;
        trim_right = n_fft / 2;
        break;
    case TRIM_SAME:
        trim_left = (n_fft - hop) / 2;
        trim_right = (n_fft - hop) / 2;
        break;
    case TRIM_NONE:
        break;
    }

    int final_len = ola_len - trim_left - trim_right;
    if (final_len <= 0) {
        return {};
    }

    std::vector<float> result(final_len);
    for (int i = 0; i < final_len; i++) {
        result[i] = output[trim_left + i];
    }

    // Optional clamping.
    if (clamp > 0.0f) {
        for (auto& s : result) {
            if (s > clamp) {
                s = clamp;
            } else if (s < -clamp) {
                s = -clamp;
            }
        }
    }

    return result;
}

// Convenience overload returning a malloc'd buffer (C ABI friendly).
// The caller must free() the returned pointer.
// Returns nullptr on failure; *out_n is set to the number of samples.
static inline float* istft_alloc(const float* mag, const float* phase, int n_fft, int hop, int T_frames,
                                 const float* window, TrimMode trim, float clamp_val, int* out_n, float win_eps = 1e-8f,
                                 bool zero_below_eps = false) {
    std::vector<float> pcm = istft(mag, phase, n_fft, hop, T_frames, window, trim, clamp_val, win_eps, zero_below_eps);
    if (pcm.empty()) {
        if (out_n) {
            *out_n = 0;
        }
        return nullptr;
    }
    float* buf = (float*)std::malloc(pcm.size() * sizeof(float));
    if (!buf) {
        if (out_n) {
            *out_n = 0;
        }
        return nullptr;
    }
    std::memcpy(buf, pcm.data(), pcm.size() * sizeof(float));
    if (out_n) {
        *out_n = (int)pcm.size();
    }
    return buf;
}

} // namespace core_istft
