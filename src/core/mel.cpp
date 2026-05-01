// src/core/mel.cpp — implementation of compute_log_mel().
// See src/core/mel.h for the interface contract.

#include "mel.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

namespace core_mel {

std::vector<float> compute(const float* samples, int n_samples, const float* window_in, int win_length,
                           const float* mel_fb, int n_freqs, FftR2C fft, const Params& p, int& T_out) {
    const int n_fft = p.n_fft;
    const int hop = p.hop_length;
    const int nmels = p.n_mels;

    // -----------------------------------------------------------------
    // 1a. Optional pre-emphasis: y[0] = x[0]; y[i] = x[i] - α*x[i-1].
    //     NeMo applies this to the raw input before center-padding,
    //     so the high-pass filter sees the true first sample (not the
    //     zero pad) — see NeMo FilterbankFeatures.forward.
    // -----------------------------------------------------------------
    std::vector<float> preemph_in;
    const float* base_ptr = samples;
    if (p.preemph != 0.0f && n_samples > 0) {
        preemph_in.resize((size_t)n_samples);
        preemph_in[0] = samples[0];
        const float a = p.preemph;
        for (int i = 1; i < n_samples; i++)
            preemph_in[i] = samples[i] - a * samples[i - 1];
        base_ptr = preemph_in.data();
    }

    // -----------------------------------------------------------------
    // 1b. Optional center-pad of the input by n_fft/2 on each side.
    // -----------------------------------------------------------------
    std::vector<float> padded_in;
    const float* in_ptr;
    int in_len;
    if (p.center_pad) {
        const int pad = n_fft / 2;
        padded_in.assign((size_t)(pad + n_samples + pad), 0.0f);
        std::memcpy(padded_in.data() + pad, base_ptr, (size_t)n_samples * sizeof(float));
        in_ptr = padded_in.data();
        in_len = (int)padded_in.size();
    } else {
        in_ptr = base_ptr;
        in_len = n_samples;
    }

    // -----------------------------------------------------------------
    // 2. Build the STFT window, center-padded to n_fft if win_length is
    //    shorter. NeMo cluster stores a win_length-sized window; the
    //    HF cluster stores it already padded to n_fft. Both work: the
    //    center-pad is a no-op when win_length == n_fft.
    // -----------------------------------------------------------------
    std::vector<float> window(n_fft, 0.0f);
    {
        const int lpad = (n_fft - win_length) / 2;
        const int wn = std::min(win_length, n_fft);
        for (int i = 0; i < wn; i++)
            window[lpad + i] = window_in[i];
    }

    // -----------------------------------------------------------------
    // 3. STFT → power spectrum (stored [T, n_freqs] row-major).
    // -----------------------------------------------------------------
    int T = in_len >= n_fft ? (in_len - n_fft) / hop + 1 : 0;
    if (p.drop_last_frame && T > 0)
        T -= 1;
    // Optional even-T guarantee: when the remaining frame count is odd,
    // shift the window start by one hop to produce an even T. This is
    // what voxtral4b needs (it feeds the mel into a stride-2 conv).
    int t_start = 0;
    if (p.drop_first_frame_if_odd && T > 0 && (T % 2 != 0)) {
        T -= 1;
        t_start = 1;
    }
    if (T <= 0) {
        T_out = 0;
        return {};
    }

    std::vector<float> power((size_t)T * n_freqs, 0.0f);
    {
        std::vector<float> fft_in((size_t)n_fft);
        std::vector<float> fft_out((size_t)n_fft * 2);
        const bool use_magnitude = (p.spec_kind == SpecKind::Magnitude);
        for (int t = 0; t < T; t++) {
            const float* frame = in_ptr + (size_t)(t + t_start) * hop;
            for (int n = 0; n < n_fft; n++)
                fft_in[n] = frame[n] * window[n];
            fft(fft_in.data(), n_fft, fft_out.data());
            for (int k = 0; k < n_freqs; k++) {
                const float re = fft_out[2 * k];
                const float im = fft_out[2 * k + 1];
                const float pw = re * re + im * im;
                power[(size_t)t * n_freqs + k] = use_magnitude ? std::sqrt(pw) : pw;
            }
        }
    }

    // -----------------------------------------------------------------
    // 4. Mel projection: mel[t, m] = sum_k power[t, k] * fb[m, k]
    //    (or fb[k, m] if fb_layout == FreqsMels).
    //    Two accumulator precisions.
    // -----------------------------------------------------------------
    std::vector<float> mel_tn((size_t)T * nmels, 0.0f);

    auto do_matmul = [&](auto acc_zero) {
        using Acc = decltype(acc_zero);
        for (int t = 0; t < T; t++) {
            const float* pp = power.data() + (size_t)t * n_freqs;
            float* mp = mel_tn.data() + (size_t)t * nmels;
            for (int m = 0; m < nmels; m++) {
                Acc s = 0;
                if (p.fb_layout == FbLayout::MelsFreqs) {
                    const float* fb = mel_fb + (size_t)m * n_freqs;
                    for (int k = 0; k < n_freqs; k++) {
                        s += static_cast<Acc>(pp[k]) * static_cast<Acc>(fb[k]);
                    }
                } else {
                    // FreqsMels: fb[k * nmels + m]
                    for (int k = 0; k < n_freqs; k++) {
                        s += static_cast<Acc>(pp[k]) * static_cast<Acc>(mel_fb[(size_t)k * nmels + m]);
                    }
                }
                mp[m] = (float)s;
            }
        }
    };
    if (p.matmul == MatmulPrecision::Double)
        do_matmul(double{0});
    else
        do_matmul(float{0.0f});

    // -----------------------------------------------------------------
    // 5. log with guard
    // -----------------------------------------------------------------
    auto apply_log = [&](float v) -> float {
        if (p.log_guard == LogGuard::MaxClip) {
            if (v < p.log_eps)
                v = p.log_eps;
            return (p.log_base == LogBase::Log10) ? std::log10(v) : std::log(v);
        } else { // AddEpsilon
            const float vv = v + p.log_eps;
            return (p.log_base == LogBase::Log10) ? std::log10(vv) : std::log(vv);
        }
    };
    for (size_t i = 0; i < mel_tn.size(); i++) {
        mel_tn[i] = apply_log(mel_tn[i]);
    }

    // -----------------------------------------------------------------
    // 6. Optional right-pad in LOG space with log_guard(0).
    //    Padding here (before normalization) lets the padded frames
    //    participate correctly in the GlobalClipMax mean/max step.
    //    Voxtral relies on this.
    // -----------------------------------------------------------------
    const int T_final = (p.pad_to_T > 0 && p.pad_to_T > T) ? p.pad_to_T : T;

    if (T_final > T) {
        const float pad_val = apply_log(0.0f);
        std::vector<float> ext((size_t)T_final * nmels, 0.0f);
        // Copy existing [T, n_mels] into leading [T, n_mels] slot.
        std::memcpy(ext.data(), mel_tn.data(), (size_t)T * nmels * sizeof(float));
        // Fill trailing frames with pad_val.
        for (int t = T; t < T_final; t++) {
            for (int m = 0; m < nmels; m++)
                ext[(size_t)t * nmels + m] = pad_val;
        }
        mel_tn = std::move(ext);
    }
    T_out = T_final;

    // -----------------------------------------------------------------
    // 7. Normalization
    // -----------------------------------------------------------------
    switch (p.norm) {
    case Normalization::PerFeatureZ: {
        // Per-mel band z-score across time. Matches NeMo
        // FilterbankFeatures.normalize_batch("per_feature"):
        //   var = sum_sq / (T - 1)            # Bessel-corrected sample variance
        //   std = sqrt(var); std = 0 if NaN   # NaN guard for T == 1
        //   std += 1e-5                       # eps on std, OUTSIDE the sqrt
        //   y = (x - mean) / std
        // The placement of eps matters on low-variance mel bands
        // (silence / high-freq during quiet speech). Adding it inside the
        // sqrt under-amplifies those bands relative to NeMo, which can
        // shift downstream encoder activations enough to cause TDT token
        // deletions on conversational JA audio (issue #37).
        const int denom = (T_final > 1) ? (T_final - 1) : 1;
        for (int m = 0; m < nmels; m++) {
            double sum = 0.0, sq = 0.0;
            for (int t = 0; t < T_final; t++)
                sum += mel_tn[(size_t)t * nmels + m];
            const double mean = sum / T_final;
            for (int t = 0; t < T_final; t++) {
                const double d = mel_tn[(size_t)t * nmels + m] - mean;
                sq += d * d;
            }
            float std_val = std::sqrt((float)(sq / denom));
            if (!(std_val == std_val))
                std_val = 0.0f;
            std_val += 1e-5f;
            const float inv_std = 1.0f / std_val;
            for (int t = 0; t < T_final; t++) {
                mel_tn[(size_t)t * nmels + m] = (float)(mel_tn[(size_t)t * nmels + m] - mean) * inv_std;
            }
        }
        break;
    }
    case Normalization::GlobalClipMax: {
        float mx = -1e30f;
        for (size_t i = 0; i < mel_tn.size(); i++)
            if (mel_tn[i] > mx)
                mx = mel_tn[i];
        const float floor_v = mx - 8.0f;
        for (size_t i = 0; i < mel_tn.size(); i++) {
            float v = mel_tn[i];
            if (v < floor_v)
                v = floor_v;
            mel_tn[i] = (v + 4.0f) / 4.0f;
        }
        break;
    }
    case Normalization::GlobalClipFixed: {
        const float floor_v = p.fixed_max - 8.0f;
        for (size_t i = 0; i < mel_tn.size(); i++) {
            float v = mel_tn[i];
            if (v < floor_v)
                v = floor_v;
            mel_tn[i] = (v + 4.0f) / 4.0f;
        }
        break;
    }
    case Normalization::None:
        // Raw log-mel — no post-log normalization. Gemma4 uses this.
        break;
    }

    // -----------------------------------------------------------------
    // 8. Optional frame stacking (TimeMels only).
    //    Collapses `stacked_frames` consecutive rows into one wider row.
    //    Because mel_tn is already row-major [T_final, n_mels] and the
    //    output layout keeps rows contiguous, stacking is a pure memory
    //    reinterpret: drop any trailing frames that don't fill a full
    //    group, then pretend the buffer is [T_final / s, n_mels * s].
    // -----------------------------------------------------------------
    if (p.stacked_frames > 1 && p.layout == Layout::TimeMels) {
        const int s = p.stacked_frames;
        const int T_stacked = T_final / s;
        if (T_stacked > 0) {
            mel_tn.resize((size_t)T_stacked * s * nmels);
        } else {
            mel_tn.clear();
        }
        T_out = T_stacked;
    }

    // -----------------------------------------------------------------
    // 9. Output layout
    // -----------------------------------------------------------------
    if (p.layout == Layout::TimeMels) {
        return mel_tn;
    }

    // Transpose to (n_mels, T_final) row-major.
    std::vector<float> out((size_t)nmels * T_final, 0.0f);
    for (int t = 0; t < T_final; t++) {
        for (int m = 0; m < nmels; m++) {
            out[(size_t)m * T_final + t] = mel_tn[(size_t)t * nmels + m];
        }
    }
    return out;
}

std::vector<float> build_htk_fb(int sr, int n_fft, int n_mels, float fmin, float fmax, FbLayout layout) {
    const int n_freqs = n_fft / 2 + 1;
    if (fmax <= 0.0f)
        fmax = (float)sr * 0.5f;
    auto hz2mel = [](float hz) { return 2595.0f * std::log10(1.0f + hz / 700.0f); };
    auto mel2hz = [](float m) { return 700.0f * (std::pow(10.0f, m / 2595.0f) - 1.0f); };
    const float ml = hz2mel(fmin);
    const float mh = hz2mel(fmax);
    std::vector<float> centers((size_t)n_mels + 2);
    for (int i = 0; i < n_mels + 2; i++)
        centers[i] = mel2hz(ml + (mh - ml) * (float)i / (float)(n_mels + 1));

    std::vector<float> fb((size_t)n_mels * n_freqs, 0.0f);
    for (int m = 0; m < n_mels; m++) {
        const float lo = centers[m];
        const float md = centers[m + 1];
        const float hi = centers[m + 2];
        for (int k = 0; k < n_freqs; k++) {
            const float f = (float)k * (float)sr / (float)n_fft;
            float w = 0.0f;
            if (f >= lo && f <= md && md > lo)
                w = (f - lo) / (md - lo);
            else if (f >= md && f <= hi && hi > md)
                w = (hi - f) / (hi - md);
            if (w < 0.0f)
                w = 0.0f;
            if (layout == FbLayout::MelsFreqs)
                fb[(size_t)m * n_freqs + k] = w;
            else
                fb[(size_t)k * n_mels + m] = w;
        }
    }
    return fb;
}

} // namespace core_mel
