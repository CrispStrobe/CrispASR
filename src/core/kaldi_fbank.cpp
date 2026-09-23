// src/core/kaldi_fbank.cpp — see kaldi_fbank.h for the contract.

#include "kaldi_fbank.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace core_kaldi {

namespace {

// Round a window length up to the next power of two. Matches kaldi's
// `round_to_power_of_two=True` default — the FFT operates on this padded
// length so the energy at the original window length still contributes
// to the right bins.
static int next_pow2(int x) {
    int n = 1;
    while (n < x)
        n <<= 1;
    return n;
}

// Iterative radix-2 Cooley-Tukey FFT. Same loop as the firered_asr.cpp
// helper (kept inline so this TU stays self-contained — moving it onto
// the shared core_fft would mean exposing the in-place mutating variant,
// which doesn't currently exist there).
static void fft_radix2(float* re, float* im, int n) {
    for (int i = 1, j = 0; i < n; i++) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1)
            j ^= bit;
        j ^= bit;
        if (i < j) {
            std::swap(re[i], re[j]);
            std::swap(im[i], im[j]);
        }
    }
    for (int len = 2; len <= n; len <<= 1) {
        const float ang = -2.0f * (float)M_PI / (float)len;
        const float wre = std::cos(ang);
        const float wim = std::sin(ang);
        for (int i = 0; i < n; i += len) {
            float cr = 1.0f, ci = 0.0f;
            for (int j = 0; j < len / 2; j++) {
                const float tr = re[i + j + len / 2] * cr - im[i + j + len / 2] * ci;
                const float ti = re[i + j + len / 2] * ci + im[i + j + len / 2] * cr;
                re[i + j + len / 2] = re[i + j] - tr;
                im[i + j + len / 2] = im[i + j] - ti;
                re[i + j] += tr;
                im[i + j] += ti;
                const float nr = cr * wre - ci * wim;
                ci = cr * wim + ci * wre;
                cr = nr;
            }
        }
    }
}

// Build the HTK-style mel filterbank that kaldi uses. Triangular filters
// in HTK mel space (`mel = 1127 * log(1 + hz/700)`). Returns a flat
// `n_mels * (n_fft/2+1)` row-major buffer.
//
// Differs from a standard librosa-Slaney basis in two ways:
//   1. HTK mel scale (single log curve), not the librosa piecewise.
//   2. NO Slaney area normalization — kaldi uses bare triangles. The
//      narrower lower-frequency bins get less weight than the wider
//      upper bins, but kaldi expects this and the trained models match.
static std::vector<float> build_kaldi_mel_fb(int sr, int n_fft, int n_mels, float low_freq, float high_freq,
                                             bool mel_domain) {
    const int n_bins = n_fft / 2 + 1;
    std::vector<float> fb((size_t)n_mels * (size_t)n_bins, 0.0f);

    auto hz2mel = [](float hz) { return 1127.0f * std::log(1.0f + hz / 700.0f); };
    auto mel2hz = [](float m) { return 700.0f * (std::exp(m / 1127.0f) - 1.0f); };

    if (high_freq <= 0.0f)
        high_freq += (float)sr / 2.0f; // 0 = Nyquist, negative = Nyquist + high_freq (Kaldi)
    const float mel_lo = hz2mel(low_freq);
    const float mel_hi = hz2mel(high_freq);

    std::vector<float> centers((size_t)n_mels + 2);
    for (int i = 0; i < n_mels + 2; i++) {
        centers[(size_t)i] = mel2hz(mel_lo + (float)i * (mel_hi - mel_lo) / (float)(n_mels + 1));
    }

    if (mel_domain) {
        // kaldi MelBanks / torchaudio get_mel_banks: triangles linear in mel,
        // over the n_fft/2 bins below Nyquist (the Nyquist bin gets weight 0).
        const float delta = (mel_hi - mel_lo) / (float)(n_mels + 1);
        for (int m = 0; m < n_mels; m++) {
            const float left = mel_lo + (float)m * delta, center = left + delta, right = center + delta;
            for (int k = 0; k < n_fft / 2; k++) {
                const float mel = hz2mel((float)k * (float)sr / (float)n_fft);
                if (mel > left && mel < right) {
                    fb[(size_t)m * (size_t)n_bins + (size_t)k] =
                        mel <= center ? (mel - left) / (center - left) : (right - mel) / (right - center);
                }
            }
        }
        return fb;
    }

    for (int m = 0; m < n_mels; m++) {
        for (int k = 0; k < n_bins; k++) {
            const float freq = (float)k * (float)sr / (float)n_fft;
            const float lo = centers[(size_t)m];
            const float mid = centers[(size_t)m + 1];
            const float hi = centers[(size_t)m + 2];
            if (freq > lo && freq <= mid && mid > lo) {
                fb[(size_t)m * (size_t)n_bins + (size_t)k] = (freq - lo) / (mid - lo);
            } else if (freq > mid && freq < hi && hi > mid) {
                fb[(size_t)m * (size_t)n_bins + (size_t)k] = (hi - freq) / (hi - mid);
            }
        }
    }
    return fb;
}

// Povey window: hann(i, periodic=False)^0.85. Matches kaldi's
// "povey" choice (the default `window_type` in torchaudio's kaldi
// fbank). The hann here is the *symmetric* form (i / (N-1)) — kaldi's
// FeatureWindowFunction uses (i / (N-1)) for povey.
static std::vector<float> make_povey_window(int N) {
    std::vector<float> w((size_t)N);
    for (int i = 0; i < N; i++) {
        const float hann = 0.5f - 0.5f * std::cos(2.0f * (float)M_PI * (float)i / (float)(N - 1));
        w[(size_t)i] = std::pow(hann, 0.85f);
    }
    return w;
}

// Hamming window: 0.54 - 0.46 * cos(2π i / (N-1)). Matches kaldi's
// FeatureWindowFunction "hamming" choice — the one FunASR's WavFrontend
// uses when constructed with window="hamming".
static std::vector<float> make_hamming_window(int N) {
    std::vector<float> w((size_t)N);
    for (int i = 0; i < N; i++) {
        w[(size_t)i] = 0.54f - 0.46f * std::cos(2.0f * (float)M_PI * (float)i / (float)(N - 1));
    }
    return w;
}

} // namespace

std::vector<float> compute_fbank(const float* pcm, int n_samples, const FbankParams& p, int& T_frames_out) {
    T_frames_out = 0;
    if (!pcm || n_samples <= 0)
        return {};

    const int win = (int)((int64_t)p.frame_length_ms * p.sample_rate / 1000); // 25 * 16000 / 1000 = 400
    const int hop = (int)((int64_t)p.frame_shift_ms * p.sample_rate / 1000);  // 10 * 16000 / 1000 = 160
    const int n_fft = next_pow2(win);                                         // 400 → 512
    const int n_bins = n_fft / 2 + 1;
    const int n_mels = p.n_mels;
    const float scale = p.int16_scale ? 32768.0f : 1.0f;

    int T = 0;
    if (p.snip_edges) {
        if (n_samples < win)
            return {};
        T = (n_samples - win) / hop + 1; // drop trailing partial frames
    } else {
        T = (n_samples + hop / 2) / hop; // kaldi NumFrames(..., flush=true)
    }
    if (T <= 0) {
        return {};
    }

    static thread_local std::vector<float> mel_fb;
    // Everything the bank depends on: a thread that alternates configurations
    // must never be handed the previous one's filters.
    static thread_local std::vector<float> mel_fb_key;
    const std::vector<float> key = {(float)p.sample_rate, (float)n_fft, (float)n_mels,
                                    p.low_freq,           p.high_freq,  (float)p.mel_domain_triangles};
    if (mel_fb_key != key || mel_fb.empty()) {
        mel_fb = build_kaldi_mel_fb(p.sample_rate, n_fft, n_mels, p.low_freq, p.high_freq, p.mel_domain_triangles);
        mel_fb_key = key;
    }

    static thread_local std::vector<float> window;
    static thread_local int window_sig = 0;
    const int win_sig = win * 4 + (int)p.window_type;
    if (window_sig != win_sig || window.empty()) {
        switch (p.window_type) {
        case WindowType::Hamming:
            window = make_hamming_window(win);
            break;
        case WindowType::Povey:
        default:
            window = make_povey_window(win);
            break;
        }
        window_sig = win_sig;
    }

    std::vector<float> features((size_t)T * (size_t)n_mels);
    // Thread-local scratch avoids per-call heap allocation for the FFT
    // buffers and windowed frame (§176f).
    static thread_local std::vector<float> fft_re, fft_im, frame;
    if ((int)fft_re.size() < n_fft) {
        fft_re.resize((size_t)n_fft);
        fft_im.resize((size_t)n_fft);
    }
    if ((int)frame.size() < win)
        frame.resize((size_t)win);

    for (int t = 0; t < T; t++) {
        const int offset = p.snip_edges ? t * hop : t * hop + hop / 2 - win / 2;

        // Pull frame and (optionally) scale to int16-magnitude. Without
        // snip_edges, samples outside [0, n) mirror back in (kaldi ExtractWindow).
        float dc = 0.0f;
        for (int i = 0; i < win; i++) {
            int j = offset + i;
            float s = 0.0f;
            if (p.snip_edges) {
                s = j < n_samples ? pcm[j] : 0.0f;
            } else {
                while (j < 0 || j >= n_samples)
                    j = j < 0 ? -j - 1 : 2 * n_samples - 1 - j;
                s = pcm[j];
            }
            frame[(size_t)i] = s * scale;
            dc += frame[(size_t)i];
        }
        if (p.remove_dc_offset) {
            dc /= (float)win;
            for (int i = 0; i < win; i++)
                frame[(size_t)i] -= dc;
        }

        // Pre-emphasis: s[i] -= preemph * s[i-1]. Kaldi uses s[-1] = s[0]
        // for the boundary, so the i=0 step becomes s[0] -= preemph * s[0]
        // → s[0] *= (1 - preemph). We must walk the array in REVERSE so
        // each step reads the unmodified s[i-1].
        if (p.preemph > 0.0f) {
            for (int i = win - 1; i > 0; i--) {
                frame[(size_t)i] -= p.preemph * frame[(size_t)(i - 1)];
            }
            frame[0] -= p.preemph * frame[0];
        }

        // Apply window + zero-pad to n_fft.
        std::fill(fft_re.begin(), fft_re.end(), 0.0f);
        std::fill(fft_im.begin(), fft_im.end(), 0.0f);
        for (int i = 0; i < win; i++) {
            fft_re[(size_t)i] = frame[(size_t)i] * window[(size_t)i];
        }

        fft_radix2(fft_re.data(), fft_im.data(), n_fft);

        // Power → mel projection → log(max(x, eps)).
        for (int m = 0; m < n_mels; m++) {
            const float* row = mel_fb.data() + (size_t)m * (size_t)n_bins;
            float sum = 0.0f;
            for (int k = 0; k < n_bins; k++) {
                const float power = fft_re[(size_t)k] * fft_re[(size_t)k] + fft_im[(size_t)k] * fft_im[(size_t)k];
                sum += power * row[(size_t)k];
            }
            features[(size_t)t * (size_t)n_mels + (size_t)m] = std::log(std::max(sum, p.log_floor));
        }
    }

    T_frames_out = T;
    return features;
}

} // namespace core_kaldi
