// sidon_rpe_gates.h — relative-position-bias formulation selection for Sidon,
// pure and unit-testable. Mirrors the htdemucs_gates.h / mel_band_gates.h
// pattern: one resolve() from the environment plus the facts of this call,
// decided in one place and locked by a decision-table unit test.
//
// The three formulations (sidon_rpe_mode) are algebraically identical and
// differ only in what the graph materialises per layer:
//
//   expand         rel_idx I32 [T*T]        4*T^2
//                  rpe     F32 [hd, T, T]   4*hd*T^2      <- dominates
//   bucket         rel_idx I32 [T, T]       4*T^2, plus an in-graph REPEAT to
//                                           [T,T,H] (no Metal I32 kernel, so it
//                                           lands on the CPU backend)
//   bucket_direct  rel_idx I32 [T, T, H]    4*H*T^2
//
// Both remaining paths then build bias [T, T, H], so that part is common. The
// honest comparison is therefore expand's EXTRA transient footprint over
// bucket_direct:
//
//   extra = 4*T^2*(hd + 1 - H)
//
// which for the shipped model (hd=64, H=16) is 196*T^2 bytes — 58 MiB at
// T=557, 196 MiB at T=1024, 1.64 GiB at the 3000-frame cap
// (CRISPASR_SIDON_MAX_FRAMES). That growth is why bucket_direct is the safe
// default and expand cannot simply become it.
//
// WHY AUTO-SELECT AT ALL — the measurement, from the #416 reporter's logs
// (GTX 1660 SUPER, Vulkan, int dot: 1, T=557, same file and same binary, so the
// RPE mode is the only variable):
//
//   bucket          predictor 661.22 ms    total 1952.72 ms
//   bucket_direct   predictor 575.25 ms    total 1875.69 ms   (default)
//   expand          predictor 213.80 ms    total 1508.13 ms   <- 2.7x faster
//
// So on a real GPU, at a length where expand's extra allocation is affordable,
// the default was leaving a 2.7x predictor speedup on the table. AUTO takes it
// exactly when the extra footprint fits the budget, and falls back to
// bucket_direct — the previous behaviour — everywhere else.
//
// SCOPE OF THE EVIDENCE, stated plainly: the 2.7x above is ONE device, ONE
// length. See sidon_rpe_auto_gpu_only below for how that limit is encoded
// rather than generalised.
//
// Env semantics:
//   CRISPASR_SIDON_RPE          expand | bucket | bucket-direct — an explicit
//                               value is honoured EXACTLY and never
//                               auto-overridden (it is the #416 bisection
//                               handle; an AUTO that second-guesses it would
//                               destroy its diagnostic value).
//   CRISPASR_SIDON_RPE_BUDGET_MB  AUTO's extra-footprint budget in MiB
//                               (default sidon_rpe_default_budget_mb). 0
//                               disables AUTO entirely -> always bucket_direct.
#pragma once

#include <cstdint>
#include <cstdlib>
#include <cstring>

namespace sidon_rpe_gates {

enum class mode { expand, bucket, bucket_direct };

// AUTO's default budget for expand's EXTRA transient bytes over bucket_direct.
// 256 MiB admits expand up to T=1170 at hd=64,H=16 (196*T^2 bytes, so ~23 s of
// audio at 50 frames/s) and declines beyond it. Chosen to cover the short-clip
// case the speedup was measured in (T=557) with headroom, while staying well
// inside a small discrete GPU — the reporter's card has 6 GB.
inline constexpr int sidon_rpe_default_budget_mb = 256;

// AUTO only upgrades to expand on a GPU backend. The 2.7x was measured on
// Vulkan; no CPU measurement supports flipping the CPU default, and the CPU
// path is what every unit/live test and every CPU-only user exercises. Keeping
// AUTO GPU-only means a device we have no data for keeps exactly its previous
// behaviour instead of inheriting a conclusion drawn from other hardware.
inline constexpr bool sidon_rpe_auto_gpu_only = true;

// Extra transient bytes expand needs over bucket_direct, per the layout above.
// Returns 0 when expand is not the heavier path (hd + 1 <= H), and saturates
// rather than overflowing on absurd T.
inline uint64_t expand_extra_bytes(int T, int head_dim, int heads) {
    if (T <= 0 || head_dim <= 0 || heads <= 0)
        return 0;
    const int64_t per = (int64_t)head_dim + 1 - (int64_t)heads;
    if (per <= 0)
        return 0;
    const uint64_t t2 = (uint64_t)T * (uint64_t)T;
    if (t2 > UINT64_MAX / (4ull * (uint64_t)per))
        return UINT64_MAX;
    return 4ull * (uint64_t)per * t2;
}

struct Resolved {
    mode m = mode::bucket_direct;
    bool from_env = false;    // an explicit CRISPASR_SIDON_RPE picked this
    bool auto_expand = false; // AUTO upgraded to expand
};

// env_mode  — CRISPASR_SIDON_RPE (nullptr/empty = AUTO)
// env_budget— CRISPASR_SIDON_RPE_BUDGET_MB (nullptr/empty = default)
// T         — feature frames this graph is being built for
// is_gpu    — the predictor runs on a non-CPU backend
inline Resolved resolve(const char* env_mode, const char* env_budget, int T, int head_dim, int heads, bool is_gpu) {
    Resolved r;

    if (env_mode && *env_mode) {
        r.from_env = true;
        if (std::strcmp(env_mode, "expand") == 0)
            r.m = mode::expand;
        else if (std::strcmp(env_mode, "bucket") == 0)
            r.m = mode::bucket;
        else
            r.m = mode::bucket_direct; // includes "bucket-direct" and anything unrecognised
        return r;
    }

    // AUTO. Default stays bucket_direct; upgrade to expand only when the extra
    // footprint fits and we have evidence for the device class.
    if (sidon_rpe_auto_gpu_only && !is_gpu)
        return r;

    int budget_mb = sidon_rpe_default_budget_mb;
    if (env_budget && *env_budget) {
        const int v = atoi(env_budget);
        budget_mb = v < 0 ? 0 : v;
    }
    if (budget_mb <= 0)
        return r;

    const uint64_t extra = expand_extra_bytes(T, head_dim, heads);
    if (extra <= (uint64_t)budget_mb * 1024ull * 1024ull) {
        r.m = mode::expand;
        r.auto_expand = true;
    }
    return r;
}

// True when an unrecognised CRISPASR_SIDON_RPE value was supplied, so the
// caller can warn once (resolve() itself stays silent and pure).
inline bool env_mode_is_unknown(const char* env_mode) {
    return env_mode && *env_mode && std::strcmp(env_mode, "expand") != 0 && std::strcmp(env_mode, "bucket") != 0 &&
           std::strcmp(env_mode, "bucket-direct") != 0;
}

} // namespace sidon_rpe_gates
