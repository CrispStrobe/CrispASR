// core/basic_pitch_conv.h — the six 2-D convolutions of the Basic Pitch
// network, and the SIMD/threaded fast path for them.
//
// Extracted from src/basic_pitch.cpp so the two paths can be diffed against
// each other by a hermetic, weight-free unit test (tests/test-basic-pitch-conv.cpp)
// rather than only through a model load — the same reason src/btc_chord_vocab.h
// exists. Nothing here touches ggml or the GGUF; it is plain arrays in, plain
// arrays out.
#pragma once

#include "core/cpu_packed_conv1d.h"
#include "core/crispasr_env.h"
#include "core/env_gate.h"
#include "core/parallel_for.h"

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <string>
#include <vector>

namespace core_basic_pitch {

struct bp_conv {
    std::vector<float> w; // [OC][IC][KH][KW]
    std::vector<float> b; // [OC]
    int oc = 0, ic = 0, kh = 0, kw = 0;
};

// 2-D correlation. Input is channel-major [IC][H][W]; output [OC][H][W_out].
// Time stride is always 1 (upstream never strides time); frequency stride is
// `stride_w`. Padding is symmetric, matching the TF "same" pads the ONNX
// export made explicit (see the table in src/basic_pitch.h's port notes).
inline std::vector<float> bp_conv2d_ref(const std::vector<float>& in, int IC, int H, int W, const bp_conv& c,
                                        int stride_w, int pad_h, int pad_w, int& W_out) {
    W_out = (W + 2 * pad_w - c.kw) / stride_w + 1;
    std::vector<float> out((size_t)c.oc * (size_t)H * (size_t)W_out);
    const int KH = c.kh, KW = c.kw, OC = c.oc;
    for (int oc = 0; oc < OC; oc++) {
        const float bias = c.b[(size_t)oc];
        for (int h = 0; h < H; h++) {
            float* orow = out.data() + ((size_t)oc * (size_t)H + (size_t)h) * (size_t)W_out;
            for (int wo = 0; wo < W_out; wo++)
                orow[wo] = bias;
            for (int ic = 0; ic < IC; ic++) {
                const float* wbase = c.w.data() + ((size_t)oc * (size_t)IC + (size_t)ic) * (size_t)KH * (size_t)KW;
                for (int kh = 0; kh < KH; kh++) {
                    const int ih = h + kh - pad_h;
                    if (ih < 0 || ih >= H)
                        continue;
                    const float* irow = in.data() + ((size_t)ic * (size_t)H + (size_t)ih) * (size_t)W;
                    const float* krow = wbase + (size_t)kh * (size_t)KW;
                    for (int kw = 0; kw < KW; kw++) {
                        const float kv = krow[kw];
                        if (kv == 0.0f)
                            continue;
                        // iw = wo*stride - pad + kw; keep wo inside [0, W_out)
                        // and iw inside [0, W).
                        int wo0 = 0;
                        const int num = pad_w - kw;
                        if (num > 0)
                            wo0 = (num + stride_w - 1) / stride_w;
                        int wo1 = W_out;
                        const int lim = W - 1 + pad_w - kw;
                        if (lim < 0)
                            continue;
                        wo1 = std::min(W_out, lim / stride_w + 1);
                        for (int wo = wo0; wo < wo1; wo++)
                            orow[wo] += kv * irow[wo * stride_w - pad_w + kw];
                    }
                }
            }
        }
    }
    return out;
}

// ─── Fast convolution path (CRISPASR_BASIC_PITCH_FASTCONV) ──────────────────
//
// Two things the reference loop above leaves on the table:
//
//   1. CrispASR's own sources compile at baseline x86-64 (CMAKE_CXX_FLAGS is
//      empty; this TU's ninja rule carries no -march, and the resulting .o has
//      1460 %xmm references and zero %ymm). Note the vendored ggml is NOT in
//      the same position — GGML_NATIVE=ON gives libggml-cpu the host ISA — so
//      the gap is specific to hand-written kernels under src/. GCC
//      auto-vectorised the contiguous stride_w==1 inner loop to
//      4-wide SSE and nothing wider. Per-function __attribute__((target(...)))
//      plus a runtime __builtin_cpu_supports check widens that to AVX2/AVX-512
//      without making the shipped binary unportable — the same Isa/best_isa
//      machinery core/cpu_packed_conv1d.h uses.
//   2. The loops were single-threaded. params.n_threads only ever reached
//      core_cpu_backend::set_n_threads, which in this backend governs GGUF
//      loading and nothing else, which is why 4 threads measured no faster than
//      2. The (oc, h) output index space is embarrassingly parallel: each task
//      owns one disjoint `orow`, satisfying core_parallel's determinism
//      contract without a serial fixup pass.
//
// BIT-IDENTITY. This is not an approximation. Per output element the
// accumulation order is unchanged — ic, then kh, then kw ascending, same
// zero-weight skip, same in-bounds predicate — and vectorising across `wo`
// cannot reorder anything, because distinct wo are distinct accumulators. The
// AVX2 kernel therefore uses separate mul + add and is byte-for-byte equal to
// the reference on every layer (tests/test-basic-pitch-conv.cpp asserts it).
//
// CRISPASR_BASIC_PITCH_CONV_ISA=scalar|avx2|avx2fma|avx512 overrides dispatch
// for A/B. avx2fma and avx512 are NOT bit-identical and are never selected
// automatically: an FMA rounds once where mul+add rounds twice, and GCC
// contracts into FMA inside an `avx512f` target clone because AVX-512F carries
// FMA of its own. They exist so the cost of that extra rounding is a measured
// number rather than a guess.

// core/cpu_packed_conv1d.h #undef's its own CRISPASR_CPU_PACKED_CONV1D_X86 at
// the end of the header, so it cannot be reused here.
#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>
#define BP_CONV_X86 1
#else
#define BP_CONV_X86 0
#endif

namespace bp_conv_fast {

using Isa = core_cpu_conv1d::Isa;

enum class Kernel { scalar, avx2, avx2fma, avx512 };

inline Kernel pick_kernel() {
    static const Kernel k = [] {
        const char* e = std::getenv("CRISPASR_BASIC_PITCH_CONV_ISA");
        const std::string want = e ? e : "";
#if BP_CONV_X86
        if (want == "avx2fma" && core_cpu_conv1d::isa_available(Isa::avx2) && __builtin_cpu_supports("fma"))
            return Kernel::avx2fma;
        if (want == "avx512" && core_cpu_conv1d::isa_available(Isa::avx512f))
            return Kernel::avx512;
        if (want.empty() || want == "avx2") {
            if (core_cpu_conv1d::isa_available(Isa::avx2))
                return Kernel::avx2;
        }
#endif
        return Kernel::scalar;
    }();
    return k;
}

inline const char* kernel_name(Kernel k) {
    switch (k) {
    case Kernel::avx2:
        return "avx2";
    case Kernel::avx2fma:
        return "avx2fma";
    case Kernel::avx512:
        return "avx512f";
    default:
        return "scalar";
    }
}

// Contraction control. core/cpu_packed_conv1d.h uses
// __attribute__((optimize("fp-contract=off"))) on GCC; do NOT copy that here.
// GCC's `optimize` attribute REPLACES the function's optimisation options
// rather than adding to them, and measured on this kernel it cost 3.7x. The
// bit-identical kernels instead stay contraction-free by construction: the TU
// is built without -mfma and `avx2` has no FMA of its own.
#if defined(__clang__)
#define BP_CONV_CLANG_NO_CONTRACT _Pragma("clang fp contract(off)")
#else
#define BP_CONV_CLANG_NO_CONTRACT
#endif

// One (ic, kh) tap row over the output sub-range [wo_begin, wo_end), portable.
//
// This is the reference loop's own shape — kw OUTER, wo inner and contiguous —
// restricted to a range, and that shape is the point. The first version of this
// kernel put kw on the INSIDE (one output element at a time, accumulating in a
// register) and measured 3.7x SLOWER than the loop it replaced: a branchy
// reduction into a scalar cannot auto-vectorise, while kw-outer gives GCC unit
// stride on both sides and it emits SSE even at baseline x86-64. Order per
// output element is kw ascending either way, so this stays bit-identical.
inline void tap_ref(float* orow, const float* irow, const float* krow, int W, int KW, int stride_w, int pad_w,
                    int wo_begin, int wo_end) {
    for (int kw = 0; kw < KW; kw++) {
        const float kv = krow[kw];
        if (kv == 0.0f)
            continue;
        int wo0 = wo_begin;
        const int num = pad_w - kw;
        if (num > 0)
            wo0 = std::max(wo0, (num + stride_w - 1) / stride_w);
        const int lim = W - 1 + pad_w - kw;
        if (lim < 0)
            continue;
        const int wo1 = std::min(wo_end, lim / stride_w + 1);
        for (int wo = wo0; wo < wo1; wo++)
            orow[wo] += kv * irow[wo * stride_w - pad_w + kw];
    }
}

#if BP_CONV_X86

#define BP_MULADD_AVX2(k, x, a) _mm256_add_ps(a, _mm256_mul_ps(k, x))
#define BP_MULADD_AVX2FMA(k, x, a) _mm256_fmadd_ps(k, x, a)
#define BP_MULADD_AVX512(k, x, a) _mm512_add_ps(a, _mm512_mul_ps(k, x))

// Interior of a stride_w == 1 tap row: every kw lands inside [0, W) for every
// wo in [wo_begin, wo_end), so the bounds test drops out, the load is
// contiguous, and the running orow value stays in a register across the whole
// kw loop instead of being round-tripped through L1 once per tap. Four
// independent accumulators hide the vector-add latency.
#define BP_CONV_DEFINE_AVX(NAME, TARGET, VEC, LANES, LOAD, STORE, SET1, MULADD)                                        \
    __attribute__((target(TARGET))) inline int NAME(float* orow, const float* irow, const float* krow, int KW,         \
                                                    int pad_w, int wo_begin, int wo_end) {                             \
        BP_CONV_CLANG_NO_CONTRACT                                                                                      \
        int wo = wo_begin;                                                                                             \
        for (; wo + 4 * (LANES) <= wo_end; wo += 4 * (LANES)) {                                                        \
            VEC a0 = LOAD(orow + wo);                                                                                  \
            VEC a1 = LOAD(orow + wo + (LANES));                                                                        \
            VEC a2 = LOAD(orow + wo + 2 * (LANES));                                                                    \
            VEC a3 = LOAD(orow + wo + 3 * (LANES));                                                                    \
            const float* base = irow + wo - pad_w;                                                                     \
            for (int kw = 0; kw < KW; kw++) {                                                                          \
                const float kv = krow[kw];                                                                             \
                if (kv == 0.0f)                                                                                        \
                    continue;                                                                                          \
                const VEC k = SET1(kv);                                                                                \
                const float* p = base + kw;                                                                            \
                a0 = MULADD(k, LOAD(p), a0);                                                                           \
                a1 = MULADD(k, LOAD(p + (LANES)), a1);                                                                 \
                a2 = MULADD(k, LOAD(p + 2 * (LANES)), a2);                                                             \
                a3 = MULADD(k, LOAD(p + 3 * (LANES)), a3);                                                             \
            }                                                                                                          \
            STORE(orow + wo, a0);                                                                                      \
            STORE(orow + wo + (LANES), a1);                                                                            \
            STORE(orow + wo + 2 * (LANES), a2);                                                                        \
            STORE(orow + wo + 3 * (LANES), a3);                                                                        \
        }                                                                                                              \
        for (; wo + (LANES) <= wo_end; wo += (LANES)) {                                                                \
            VEC a0 = LOAD(orow + wo);                                                                                  \
            const float* base = irow + wo - pad_w;                                                                     \
            for (int kw = 0; kw < KW; kw++) {                                                                          \
                const float kv = krow[kw];                                                                             \
                if (kv == 0.0f)                                                                                        \
                    continue;                                                                                          \
                a0 = MULADD(SET1(kv), LOAD(base + kw), a0);                                                            \
            }                                                                                                          \
            STORE(orow + wo, a0);                                                                                      \
        }                                                                                                              \
        return wo;                                                                                                     \
    }

BP_CONV_DEFINE_AVX(tap_avx2, "avx2", __m256, 8, _mm256_loadu_ps, _mm256_storeu_ps, _mm256_set1_ps, BP_MULADD_AVX2)
BP_CONV_DEFINE_AVX(tap_avx2fma, "avx2,fma", __m256, 8, _mm256_loadu_ps, _mm256_storeu_ps, _mm256_set1_ps,
                   BP_MULADD_AVX2FMA)
BP_CONV_DEFINE_AVX(tap_avx512, "avx512f", __m512, 16, _mm512_loadu_ps, _mm512_storeu_ps, _mm512_set1_ps,
                   BP_MULADD_AVX512)

#undef BP_CONV_DEFINE_AVX
#endif // BP_CONV_X86

// One (oc, h) output row.
inline void conv_row(const float* in, float* out, int IC, int H, int W, const bp_conv& c, int stride_w, int pad_h,
                     int pad_w, int W_out, int oc, int h, Kernel kern) {
    const int KH = c.kh, KW = c.kw;
    float* orow = out + ((size_t)oc * (size_t)H + (size_t)h) * (size_t)W_out;
    const float bias = c.b[(size_t)oc];
    for (int wo = 0; wo < W_out; wo++)
        orow[wo] = bias;

    // Interior band where every kw lands inside [0, W). Only stride 1 gives a
    // contiguous load, so the strided layers (note_conv, onset_conv) stay on
    // the portable kernel and get whatever the compiler manages there.
    int lo = W_out, hi = W_out;
    if (stride_w == 1 && kern != Kernel::scalar) {
        lo = std::max(0, pad_w);
        hi = std::min(W_out, W - KW + pad_w + 1);
        if (hi < lo)
            lo = hi = W_out;
    }

    for (int ic = 0; ic < IC; ic++) {
        const float* wbase = c.w.data() + ((size_t)oc * (size_t)IC + (size_t)ic) * (size_t)KH * (size_t)KW;
        for (int kh = 0; kh < KH; kh++) {
            const int ih = h + kh - pad_h;
            if (ih < 0 || ih >= H)
                continue;
            const float* irow = in + ((size_t)ic * (size_t)H + (size_t)ih) * (size_t)W;
            const float* krow = wbase + (size_t)kh * (size_t)KW;
            if (lo >= hi) {
                tap_ref(orow, irow, krow, W, KW, stride_w, pad_w, 0, W_out);
                continue;
            }
            tap_ref(orow, irow, krow, W, KW, stride_w, pad_w, 0, lo);
            int done = lo;
#if BP_CONV_X86
            switch (kern) {
            case Kernel::avx512:
                done = tap_avx512(orow, irow, krow, KW, pad_w, lo, hi);
                break;
            case Kernel::avx2fma:
                done = tap_avx2fma(orow, irow, krow, KW, pad_w, lo, hi);
                break;
            case Kernel::avx2:
                done = tap_avx2(orow, irow, krow, KW, pad_w, lo, hi);
                break;
            default:
                break;
            }
#endif
            tap_ref(orow, irow, krow, W, KW, stride_w, pad_w, done, W_out);
        }
    }
}

} // namespace bp_conv_fast

// Gate. DEFAULT ON since the CI A/B closed the verdict;
// CRISPASR_BASIC_PITCH_FASTCONV=0 is the way back to the reference loop, which
// is kept verbatim as bp_conv2d_ref and is never removed.
//
// It defaults on because it wins on speed AND quality, which is the bar. On
// quality it is not merely non-regressed but BYTE-IDENTICAL, on both runners,
// at 1 and 4 threads (tests/test-basic-pitch-conv.cpp, and the raw heads plus
// all note events end to end). On speed, measured one-arm-per-process, cold
// run discarded, median of 3, on clean CI runners rather than the
// oversubscribed dev VPS (run 35471451173):
//
//   ubuntu-24.04, 4 cores, avx2+fma   reference 130.9 ms -> 72.1 (1.82x) t1
//                                                        -> 36.5 (3.58x) t2
//                                                        -> 34.4 (3.81x) t4
//   macos-14, M1 3 cores, NEON        reference  85.3 ms -> 80.4 (1.06x) t1
//                                                        -> 35.6 (2.39x) t4
//
// Note what those two rows say together: on aarch64 the SIMD half is worth
// almost nothing (NEON is already 4-wide at baseline, so t1 gains 6% from the
// register blocking alone) and the entire arm64 win is threading, while on
// x86-64 both halves pay. Neither platform regresses.
inline bool bp_fastconv_on() {
    static const bool on = !core_env::explicitly_off("CRISPASR_BASIC_PITCH_FASTCONV");
    return on;
}

inline bool bp_timing_on() {
    static const bool on = crispasr_env::truthy("CRISPASR_BASIC_PITCH_TIMING");
    return on;
}

inline std::vector<float> bp_conv2d(const std::vector<float>& in, int IC, int H, int W, const bp_conv& c, int stride_w,
                                    int pad_h, int pad_w, int& W_out, int n_threads) {
    if (!bp_fastconv_on())
        return bp_conv2d_ref(in, IC, H, W, c, stride_w, pad_h, pad_w, W_out);

    W_out = (W + 2 * pad_w - c.kw) / stride_w + 1;
    std::vector<float> out((size_t)c.oc * (size_t)H * (size_t)W_out);
    const bp_conv_fast::Kernel kern = bp_conv_fast::pick_kernel();
    const float* inp = in.data();
    float* outp = out.data();
    const int OC = c.oc;
    core_parallel::for_each_chunk(OC * H, n_threads, [&](int begin, int end) {
        for (int idx = begin; idx < end; idx++)
            bp_conv_fast::conv_row(inp, outp, IC, H, W, c, stride_w, pad_h, pad_w, W_out, idx / H, idx % H, kern);
    });
    return out;
}

} // namespace core_basic_pitch
