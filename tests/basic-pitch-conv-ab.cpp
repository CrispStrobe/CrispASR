// basic-pitch-conv-ab — hermetic A/B harness for the Basic Pitch convolutions.
//
// Unlike the other *-ab harnesses here it needs NO GGUF and no audio: the six
// shapes are the six real call sites in bp_forward_window, and the weights are
// a fixed PRNG. That makes it runnable on a CI runner, which is the whole
// point — the VPS this work was done on is so oversubscribed that a
// SINGLE-threaded process gets 36-54% of one core, so the threaded arm cannot
// be measured there in either direction (see
// docs/music-transcription/BASIC_PITCH_CONV_PERF.md §5).
//
//   basic-pitch-conv-ab [iters] [n_threads] [--check]
//
// Prints per-layer and total wall AND CPU time. Read the CPU column on a
// contended box and the wall column on a quiet one; with --check it instead
// byte-compares the fast path against the reference and exits nonzero on any
// difference.
//
// Select the arm with the same env vars the runtime uses:
//   CRISPASR_BASIC_PITCH_FASTCONV=1
//   CRISPASR_BASIC_PITCH_CONV_ISA=scalar|avx2|avx2fma|avx512
//
// Run each arm as a SEPARATE PROCESS (the gates are read once into
// function-local statics, and a shared process warms caches across arms).

#include "core/basic_pitch_conv.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <string>
#include <vector>

using core_basic_pitch::bp_conv;

namespace {

struct Shape {
    const char* name;
    int IC, H, W, OC, KH, KW, stride_w, pad_h, pad_w;
};

// The six call sites in bp_forward_window, at the production window (T = 172).
const Shape SHAPES[] = {
    {"contour_conv", 8, 172, 264, 8, 3, 39, 1, 1, 19}, {"contour_out", 8, 172, 264, 1, 5, 5, 1, 2, 2},
    {"note_conv", 1, 172, 264, 32, 7, 7, 3, 3, 2},     {"note_out", 32, 172, 88, 1, 7, 3, 1, 3, 1},
    {"onset_conv", 8, 172, 264, 32, 5, 5, 3, 2, 1},    {"onset_out", 33, 172, 88, 1, 3, 3, 1, 1, 1},
};

double cpu_ms() {
#if defined(CLOCK_PROCESS_CPUTIME_ID)
    timespec t{};
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &t);
    return (double)t.tv_sec * 1e3 + (double)t.tv_nsec / 1e6;
#else
    return (double)clock() * 1e3 / (double)CLOCKS_PER_SEC;
#endif
}

struct Rng {
    unsigned s = 12345u;
    float next() {
        s = s * 1664525u + 1013904223u;
        return ((float)(s >> 8) / (float)(1 << 24)) * 2.0f - 1.0f;
    }
};

} // namespace

int main(int argc, char** argv) {
    int iters = 8, threads = 1;
    bool check = false;
    for (int i = 1; i < argc; i++) {
        if (std::strcmp(argv[i], "--check") == 0)
            check = true;
        else if (iters == 8 && i == 1)
            iters = std::atoi(argv[i]);
        else
            threads = std::atoi(argv[i]);
    }

    const core_basic_pitch::bp_conv_fast::Kernel kern = core_basic_pitch::bp_conv_fast::pick_kernel();
    const char* arm =
        core_basic_pitch::bp_fastconv_on() ? core_basic_pitch::bp_conv_fast::kernel_name(kern) : "reference";

    double tot_wall = 0, tot_cpu = 0, tot_mmac = 0;
    int failures = 0;

    for (const Shape& sh : SHAPES) {
        Rng rng;
        bp_conv c;
        c.oc = sh.OC;
        c.ic = sh.IC;
        c.kh = sh.KH;
        c.kw = sh.KW;
        c.w.resize((size_t)sh.OC * sh.IC * sh.KH * sh.KW);
        for (float& v : c.w)
            v = rng.next() * 0.1f;
        c.b.resize((size_t)sh.OC);
        for (float& v : c.b)
            v = rng.next() * 0.01f;
        c.w[c.w.size() / 3] = 0.0f; // exercise the shared kv == 0 skip
        std::vector<float> in((size_t)sh.IC * sh.H * sh.W);
        for (float& v : in)
            v = rng.next();

        if (check) {
            int w_ref = 0, w_new = 0;
            const std::vector<float> ref =
                core_basic_pitch::bp_conv2d_ref(in, sh.IC, sh.H, sh.W, c, sh.stride_w, sh.pad_h, sh.pad_w, w_ref);
            const std::vector<float> got =
                core_basic_pitch::bp_conv2d(in, sh.IC, sh.H, sh.W, c, sh.stride_w, sh.pad_h, sh.pad_w, w_new, threads);
            const bool same = w_ref == w_new && ref.size() == got.size() &&
                              std::memcmp(ref.data(), got.data(), ref.size() * sizeof(float)) == 0;
            double maxabs = 0;
            for (size_t i = 0; i < ref.size() && i < got.size(); i++)
                maxabs = std::max(maxabs, (double)std::fabs(ref[i] - got[i]));
            std::printf("%-13s n=%zu max_abs=%.3g  %s\n", sh.name, ref.size(), maxabs,
                        same ? "BIT-IDENTICAL" : "DIFFERS");
            if (!same)
                failures++;
            continue;
        }

        const int W_out = (sh.W + 2 * sh.pad_w - sh.KW) / sh.stride_w + 1;
        const double mmac = (double)sh.OC * sh.IC * sh.KH * sh.KW * sh.H * W_out / 1e6;
        int wo = 0;
        volatile float sink = 0;
        bp_conv2d(in, sh.IC, sh.H, sh.W, c, sh.stride_w, sh.pad_h, sh.pad_w, wo, threads); // warm
        double best_wall = 1e18, best_cpu = 1e18;
        for (int r = 0; r < iters; r++) {
            const double c0 = cpu_ms();
            const auto t0 = std::chrono::steady_clock::now();
            const std::vector<float> o =
                bp_conv2d(in, sh.IC, sh.H, sh.W, c, sh.stride_w, sh.pad_h, sh.pad_w, wo, threads);
            const auto t1 = std::chrono::steady_clock::now();
            const double c1 = cpu_ms();
            sink += o[o.size() / 2];
            best_wall = std::min(best_wall, std::chrono::duration<double, std::milli>(t1 - t0).count());
            best_cpu = std::min(best_cpu, c1 - c0);
        }
        (void)sink;
        std::printf("%-13s wall %8.2f ms  cpu %8.2f ms  %7.1f MMAC  %6.2f GMAC/s\n", sh.name, best_wall, best_cpu, mmac,
                    mmac / 1e3 / (best_wall / 1e3));
        tot_wall += best_wall;
        tot_cpu += best_cpu;
        tot_mmac += mmac;
    }

    if (check) {
        std::printf("arm=%s threads=%d : %s\n", arm, threads, failures ? "FAIL" : "all layers bit-identical");
        return failures ? 1 : 0;
    }
    std::printf("TOTAL         wall %8.2f ms  cpu %8.2f ms  %7.1f MMAC  wall %6.2f GMAC/s  cpu %6.2f GMAC/s"
                "  [arm=%s threads=%d]\n",
                tot_wall, tot_cpu, tot_mmac, tot_mmac / 1e3 / (tot_wall / 1e3), tot_mmac / 1e3 / (tot_cpu / 1e3), arm,
                threads);
    // Achieved parallelism. Below ~1.0 at threads>1 means the box had no spare
    // cores and the wall column says nothing about the threading arm.
    std::printf("achieved_parallelism %.2f\n", tot_cpu / tot_wall);
    return 0;
}
