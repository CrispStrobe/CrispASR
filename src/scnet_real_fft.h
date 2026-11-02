// Real feature transforms using SCNet's unchanged exact-length complex plan.
// Two adjacent frequency rows share one complex FFT. Pairing stays within a
// channel; an odd final frequency uses one ordinary real transform.
#pragma once

#include "scnet_fft.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace scnet_fft {
namespace real_detail {

SCNET_FFT_NO_CONTRACT inline void forward_pair(const Plan& plan, const float* a, const float* b, float* ar, float* ai,
                                               float* br, float* bi, float scale, float* work_r, float* work_i,
                                               Plan::Scratch& scratch) {
    SCNET_FFT_PRECISE
    const int n = plan.size();
    std::copy_n(a, n, work_r);
    std::copy_n(b, n, work_i);
    plan.transform(work_r, work_i, scratch);

    // For Z=FFT(a+i*b), A=(Z+conj(Z[-k]))/2 and
    // B=(Z-conj(Z[-k]))/(2i). Real-input DC/Nyquist are real exactly.
    ar[0] = work_r[0] * scale;
    br[0] = work_i[0] * scale;
    ai[0] = bi[0] = 0.0f;
    for (int k = 1; k < (n + 1) / 2; ++k) {
        const int mirror = n - k;
        ar[k] = (0.5f * (work_r[k] + work_r[mirror])) * scale;
        ai[k] = (0.5f * (work_i[k] - work_i[mirror])) * scale;
        br[k] = (0.5f * (work_i[k] + work_i[mirror])) * scale;
        bi[k] = (0.5f * (work_r[mirror] - work_r[k])) * scale;
    }
    if (n % 2 == 0) {
        ar[n / 2] = work_r[n / 2] * scale;
        br[n / 2] = work_i[n / 2] * scale;
        ai[n / 2] = bi[n / 2] = 0.0f;
    }
}

SCNET_FFT_NO_CONTRACT inline void inverse_pair(const Plan& plan, const float* ar, const float* ai, const float* br,
                                               const float* bi, float* a, float* b, float scale, float* work_r,
                                               float* work_i, Plan::Scratch& scratch) {
    SCNET_FFT_PRECISE
    const int n = plan.size();
    // Construct conj(A+i*B) with Hermitian A/B. A forward transform then
    // produces sqrt(n)*(a-i*b) under the existing orthonormal scaling.
    // The imaginary endpoint values may be learned and must be ignored.
    work_r[0] = ar[0];
    work_i[0] = -br[0];
    for (int k = 1; k < (n + 1) / 2; ++k) {
        work_r[k] = ar[k] - bi[k];
        work_i[k] = -(ai[k] + br[k]);
        work_r[n - k] = ar[k] + bi[k];
        work_i[n - k] = ai[k] - br[k];
    }
    if (n % 2 == 0) {
        work_r[n / 2] = ar[n / 2];
        work_i[n / 2] = -br[n / 2];
    }
    plan.transform(work_r, work_i, scratch);
    for (int k = 0; k < n; ++k) {
        a[k] = work_r[k] * scale;
        b[k] = -work_i[k] * scale;
    }
}

SCNET_FFT_NO_CONTRACT inline void forward_single(const Plan& plan, const float* input, float* real, float* imag,
                                                 float scale, float* work_r, float* work_i, Plan::Scratch& scratch) {
    SCNET_FFT_PRECISE
    const int n = plan.size(), bins = n / 2 + 1;
    std::copy_n(input, n, work_r);
    std::fill_n(work_i, n, 0.0f);
    plan.transform(work_r, work_i, scratch);
    for (int k = 0; k < bins; ++k) {
        real[k] = work_r[k] * scale;
        imag[k] = k == 0 || (n % 2 == 0 && k == n / 2) ? 0.0f : work_i[k] * scale;
    }
}

SCNET_FFT_NO_CONTRACT inline void inverse_single(const Plan& plan, const float* real, const float* imag, float* output,
                                                 float scale, float* work_r, float* work_i, Plan::Scratch& scratch) {
    SCNET_FFT_PRECISE
    const int n = plan.size(), bins = n / 2 + 1;
    for (int k = 0; k < bins; ++k) {
        work_r[k] = real[k];
        work_i[k] = k == 0 || (n % 2 == 0 && k == n / 2) ? 0.0f : -imag[k];
    }
    for (int k = bins; k < n; ++k) {
        work_r[k] = work_r[n - k];
        work_i[k] = -work_i[n - k];
    }
    plan.transform(work_r, work_i, scratch);
    for (int k = 0; k < n; ++k)
        output[k] = work_r[k] * scale;
}

} // namespace real_detail

// Contiguous feature layout: real input/output [N,F,C]; complex features have
// the real [bins,F,C] plane followed by its imaginary plane. Each work item
// owns one adjacent frequency pair (or an odd tail), so workers never overlap.
inline void transform_real_rows(const Plan& plan, bool inverse, const float* input, float* output, int frequencies,
                                int channels, int worker, int workers, float* work_r, float* work_i,
                                Plan::Scratch& scratch) {
    const int n = plan.size(), bins = n / 2 + 1;
    const int pairs_per_channel = (frequencies + 1) / 2;
    const int64_t tasks = int64_t(channels) * pairs_per_channel;
    const size_t plane = size_t(channels) * frequencies * bins;
    const float scale = 1.0f / std::sqrt(static_cast<float>(n));
    for (int64_t task = worker; task < tasks; task += workers) {
        const int channel = static_cast<int>(task / pairs_per_channel);
        const int frequency = 2 * static_cast<int>(task % pairs_per_channel);
        const size_t row = size_t(channel) * frequencies + frequency;
        const bool paired = frequency + 1 < frequencies;
        if (inverse) {
            const float* ar = input + row * bins;
            const float* ai = ar + plane;
            float* a = output + row * n;
            if (paired)
                real_detail::inverse_pair(plan, ar, ai, ar + bins, ai + bins, a, a + n, scale, work_r, work_i, scratch);
            else
                real_detail::inverse_single(plan, ar, ai, a, scale, work_r, work_i, scratch);
        } else {
            const float* a = input + row * n;
            float* ar = output + row * bins;
            float* ai = ar + plane;
            if (paired)
                real_detail::forward_pair(plan, a, a + n, ar, ai, ar + bins, ai + bins, scale, work_r, work_i, scratch);
            else
                real_detail::forward_single(plan, a, ar, ai, scale, work_r, work_i, scratch);
        }
    }
}

} // namespace scnet_fft
