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

#include "core/parallel_for.h"

#if defined(HAVE_ACCELERATE)
#include <Accelerate/Accelerate.h>
#elif defined(CRISPASR_HAVE_BLAS)
#include <cblas.h>
#endif

#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <mutex>
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
//                   + 2 * sum_{k=1}^{ceil(N/2)-1} Re(X[k] * exp(j*2*pi*k*n/N)) )
//
// O(N^2) but N is small (16-1280) so the simplicity wins over FFT.
static inline void irfft_hermitian(const float* spec_re, const float* spec_im, int N, float* out) {
    const int half = N / 2;
    const int positive_end = half + N % 2;
    const double inv_N = 1.0 / N;

    for (int n = 0; n < N; n++) {
        double val = spec_re[0]; // DC (always real)
        if (N % 2 == 0) {
            // Nyquist bin: X[N/2] is real for real signals.
            val += spec_re[half] * ((n & 1) ? -1.0 : 1.0);
        }
        double angle_base = 2.0 * M_PI * n * inv_N;
        for (int k = 1; k < positive_end; k++) {
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
//   n_threads   — budget for this helper's workers (0 uses the runtime default).
//                 A linked BLAS library controls its own internal thread pool.
//
// Frame scratch is bounded independently of T_frames. The serial scalar path
// keeps just one frame; parallel frame-result tiles and BLAS tiles use at most
// 256 KiB (or one frame when it exceeds that limit). Scalar workers additionally
// own O(n_fft) spectrum scratch each.
//
// Returns a vector of PCM samples.
static inline std::vector<float> istft(const float* mag, const float* phase, int n_fft, int hop, int T_frames,
                                       const float* window = nullptr, TrimMode trim = TRIM_CENTER, float clamp = 0.0f,
                                       float win_eps = 1e-8f, bool zero_below_eps = false, int n_threads = 0) {
    if (!mag || !phase || n_fft <= 0 || hop <= 0 || T_frames <= 0)
        return {};
    if (trim == TRIM_SAME && hop > n_fft)
        return {}; // SAME padding cannot have a negative trim width.
    const int n_freq = n_fft / 2 + 1;
    const int64_t ola_len_wide = (int64_t)(T_frames - 1) * hop + n_fft;
    if (ola_len_wide > std::numeric_limits<int>::max())
        return {}; // offsets and the public C buffer length are int-sized.
    const int ola_len = (int)ola_len_wide;

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

    const char* env_serial = std::getenv("CRISPASR_ISTFT_SERIAL");
    const bool force_serial = (env_serial && env_serial[0] == '1');
    const size_t scratch_bytes = 256 * 1024;
    auto overlap_add = [&](int t, const float* frame) {
        const int offset = t * hop;
        for (int i = 0; i < n_fft; i++) {
            const float w = window[i];
            output[offset + i] += frame[i] * w;
            win_sum[offset + i] += w * w;
        }
    };

#if defined(HAVE_ACCELERATE) || defined(CRISPASR_HAVE_BLAS)
    if (n_freq >= 3) {
        // Odd transforms have no Nyquist bin: their highest positive bin
        // also has a conjugate partner and must participate in the GEMM.
        const int K = n_freq - 2 + n_fft % 2;
        const double inv_N = 1.0 / n_fft;

        struct FourierBasis {
            std::vector<float> M_cos;
            std::vector<float> M_sin;
        };
        static std::map<int, FourierBasis> s_basis_cache;
        static std::mutex s_basis_mtx;

        const FourierBasis* basis_ptr = nullptr;
        {
            std::lock_guard<std::mutex> lock(s_basis_mtx);
            auto it = s_basis_cache.find(n_fft);
            if (it == s_basis_cache.end()) {
                FourierBasis fb;
                fb.M_cos.resize((size_t)n_fft * K);
                fb.M_sin.resize((size_t)n_fft * K);
                for (int n = 0; n < n_fft; n++) {
                    double angle_base = 2.0 * M_PI * n * inv_N;
                    for (int k = 1; k <= K; k++) {
                        double angle = angle_base * k;
                        fb.M_cos[(size_t)n * K + (k - 1)] = (float)(2.0 * std::cos(angle));
                        fb.M_sin[(size_t)n * K + (k - 1)] = (float)(-2.0 * std::sin(angle));
                    }
                }
                it = s_basis_cache.emplace(n_fft, std::move(fb)).first;
            }
            basis_ptr = &it->second;
        }

        const auto& M_cos = basis_ptr->M_cos;
        const auto& M_sin = basis_ptr->M_sin;
        const size_t bytes_per_frame = ((size_t)2 * K + n_fft) * sizeof(float);
        const int tile_frames =
            force_serial ? 1 : (int)std::min((size_t)T_frames, std::max((size_t)1, scratch_bytes / bytes_per_frame));
        std::vector<float> spec_re((size_t)tile_frames * K);
        std::vector<float> spec_im((size_t)tile_frames * K);
        std::vector<float> frames((size_t)tile_frames * n_fft);

        for (int first = 0; first < T_frames;) {
            const int count = std::min(tile_frames, T_frames - first);
            // Phase preparation is O(N), much cheaper than the O(N^2)
            // scalar inverse transform. Do not spawn a team for tiny tiles.
            core_parallel::for_each_chunk_work(count, force_serial ? 1 : n_threads, (uint64_t)K, 65536,
                                               [&](int begin, int end) {
                                                   for (int local = begin; local < end; local++) {
                                                       const int t = first + local;
                                                       const float* m_row = mag + (size_t)t * n_freq;
                                                       const float* p_row = phase + (size_t)t * n_freq;
                                                       float* re_row = spec_re.data() + (size_t)local * K;
                                                       float* im_row = spec_im.data() + (size_t)local * K;
                                                       for (int k = 1; k <= K; k++) {
                                                           float s, c;
#if defined(__APPLE__)
                                                           __sincosf(p_row[k], &s, &c);
#elif defined(_GNU_SOURCE)
                        sincosf(p_row[k], &s, &c);
#else
                        s = std::sin(p_row[k]);
                        c = std::cos(p_row[k]);
#endif
                                                           re_row[k - 1] = m_row[k] * c;
                                                           im_row[k - 1] = m_row[k] * s;
                                                       }
                                                   }
                                               });
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, count, n_fft, K, 1.0f, spec_re.data(), K, M_cos.data(),
                        K, 0.0f, frames.data(), n_fft);
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, count, n_fft, K, 1.0f, spec_im.data(), K, M_sin.data(),
                        K, 1.0f, frames.data(), n_fft);

            for (int local = 0; local < count; local++) {
                const int t = first + local;
                const float* m_row = mag + (size_t)t * n_freq;
                const float* p_row = phase + (size_t)t * n_freq;
                const float dc = m_row[0] * std::cos(p_row[0]);
                const float nyq = (n_fft % 2 == 0) ? m_row[n_fft / 2] * std::cos(p_row[n_fft / 2]) : 0.0f;
                float* frame = frames.data() + (size_t)local * n_fft;
                for (int i = 0; i < n_fft; i++)
                    frame[i] = (frame[i] + dc + nyq * ((i & 1) ? -1.0f : 1.0f)) * (float)inv_N;
                overlap_add(t, frame);
            }
            first += count;
        }
    } else
#endif
    {
        // A transform's cost depends on N^2, not merely on the number of
        // frames. In particular Kokoro's N=20 frames should stay serial for
        // short clips. Resolve before allocating to preserve one-frame memory.
        const uint64_t work_per_frame = (uint64_t)n_fft * (n_fft / 2);
        const int nt =
            force_serial ? 1 : core_parallel::resolve_threads_for_work(T_frames, n_threads, work_per_frame, 65536);
        if (nt == 1) {
            std::vector<float> frame_re(n_freq);
            std::vector<float> frame_im(n_freq);
            std::vector<float> frame(n_fft);
            for (int t = 0; t < T_frames; t++) {
                const float* m_row = mag + (size_t)t * n_freq;
                const float* p_row = phase + (size_t)t * n_freq;
                for (int f = 0; f < n_freq; f++) {
                    frame_re[f] = m_row[f] * std::cos(p_row[f]);
                    frame_im[f] = m_row[f] * std::sin(p_row[f]);
                }
                irfft_hermitian(frame_re.data(), frame_im.data(), n_fft, frame.data());
                overlap_add(t, frame.data());
            }
        } else {
            const int tile_frames =
                (int)std::min((size_t)T_frames, std::max((size_t)1, scratch_bytes / ((size_t)n_fft * sizeof(float))));
            std::vector<float> frames((size_t)tile_frames * n_fft);
            for (int first = 0; first < T_frames;) {
                const int count = std::min(tile_frames, T_frames - first);
                core_parallel::for_each_chunk_work(count, nt, work_per_frame, 65536, [&](int begin, int end) {
                    std::vector<float> frame_re(n_freq);
                    std::vector<float> frame_im(n_freq);
                    for (int local = begin; local < end; local++) {
                        const int t = first + local;
                        const float* m_row = mag + (size_t)t * n_freq;
                        const float* p_row = phase + (size_t)t * n_freq;
                        for (int f = 0; f < n_freq; f++) {
                            frame_re[f] = m_row[f] * std::cos(p_row[f]);
                            frame_im[f] = m_row[f] * std::sin(p_row[f]);
                        }
                        irfft_hermitian(frame_re.data(), frame_im.data(), n_fft, frames.data() + (size_t)local * n_fft);
                    }
                });
                // Only independent transforms run concurrently. Keep the
                // original frame order for every overlapping accumulation.
                for (int local = 0; local < count; local++)
                    overlap_add(first + local, frames.data() + (size_t)local * n_fft);
                first += count;
            }
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
                                 bool zero_below_eps = false, int n_threads = 0) {
    std::vector<float> pcm =
        istft(mag, phase, n_fft, hop, T_frames, window, trim, clamp_val, win_eps, zero_below_eps, n_threads);
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
