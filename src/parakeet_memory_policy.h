// Parakeet encoder memory policy shared by routing and the encoder boundary.
#pragma once

#include <algorithm>

inline double parakeet_est_singlepass_peak_mb(int T_enc, int n_heads, double coeff) {
    if (T_enc <= 0 || n_heads <= 0)
        return 0.0;
    const double T = (double)T_enc;
    return coeff * T * T * (double)n_heads * 4.0 / (1024.0 * 1024.0);
}

// A non-positive policy budget means routing policy disabled. This preserves
// the documented tuning escape hatch; the encoder boundary still applies the
// non-disableable physical-memory guard below.
inline bool parakeet_singlepass_fits_budget(int T_enc, int n_heads, double budget_mb, double coeff) {
    if (budget_mb <= 0.0 || coeff <= 0.0)
        return true;
    return parakeet_est_singlepass_peak_mb(T_enc, n_heads, coeff) <= budget_mb;
}

// Last line of defence at the allocation site. Keep some memory for weights,
// PCM, mel features, the decoder and the host process. Unlike the routing
// policy this cannot be disabled: a forced strategy may be slow, but it must
// not ask the kernel for an allocation larger than the process can obtain.
inline bool parakeet_encoder_fits_available_memory(int T_enc, int n_heads, double available_mb, double coeff,
                                                   double usable_fraction = 0.75) {
    if (available_mb <= 0.0 || coeff <= 0.0)
        return true; // unknown platform memory; checked allocator remains the backstop
    usable_fraction = std::max(0.05, std::min(0.95, usable_fraction));
    return parakeet_est_singlepass_peak_mb(T_enc, n_heads, coeff) <= available_mb * usable_fraction;
}
