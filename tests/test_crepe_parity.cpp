// test_crepe_parity.cpp — run the CREPE ggml graph on deterministic frames and
// dump the 360-bin activations, so tools/crepe_numpy_parity.py (validated at
// cos=1.0 vs torchcrepe) can score the C++ port against the same input.
//
//   ./test-crepe-parity <crepe.gguf> <out.bin>
//
// Writes: int32 n_frames, int32 n_bins, then n_frames*n_bins float32.
// The frames are generated here with the same PRNG recipe the python side
// uses, so neither end needs an audio fixture.

#include "crepe.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <crepe.gguf> <out.bin> [seconds]\n", argv[0]);
        return 2;
    }

    // A deterministic 16 kHz test signal: a 3-tone sweep well inside CREPE's
    // range, so the decoded F0 is checkable by hand as well as by cosine.
    const int sr = CREPE_SAMPLE_RATE;
    const double secs = (argc > 3) ? atof(argv[3]) : 1.0;
    const int n = (int)(sr * secs);
    std::vector<float> pcm((size_t)n);
    for (int i = 0; i < n; i++) {
        const double t = (double)i / sr;
        const double f = (t < 0.33) ? 220.0 : (t < 0.66 ? 440.0 : 880.0);
        pcm[i] = (float)(0.5 * std::sin(2.0 * M_PI * f * t));
    }

    crepe_context* ctx = crepe_init(argv[1], 4);
    if (!ctx) {
        fprintf(stderr, "crepe_init failed for %s\n", argv[1]);
        return 1;
    }

    const float hop_ms = 10.0f;
    const int max_frames = crepe_n_frames(ctx, n, hop_ms);
    std::vector<float> act((size_t)max_frames * CREPE_PITCH_BINS);
    const int got = crepe_compute_activation(ctx, pcm.data(), n, hop_ms, act.data(), max_frames);
    if (got <= 0) {
        fprintf(stderr, "crepe_compute_activation failed\n");
        crepe_free(ctx);
        return 1;
    }

    std::vector<crepe_frame> frames((size_t)max_frames);
    const int nf = crepe_compute_f0(ctx, pcm.data(), n, hop_ms, frames.data(), max_frames);
    printf("crepe: capacity=%s frames=%d\n", crepe_capacity(ctx), got);
    for (int i = 0; i < nf; i += std::max(1, nf / 6))
        printf("  t=%7.1f ms  f0=%8.2f Hz  voiced=%.3f\n", frames[i].time_ms, frames[i].f0_hz, frames[i].voiced_prob);

    FILE* f = fopen(argv[2], "wb");
    if (!f) {
        fprintf(stderr, "cannot write %s\n", argv[2]);
        crepe_free(ctx);
        return 1;
    }
    const int32_t hdr[2] = {(int32_t)got, (int32_t)CREPE_PITCH_BINS};
    fwrite(hdr, sizeof(int32_t), 2, f);
    fwrite(act.data(), sizeof(float), (size_t)got * CREPE_PITCH_BINS, f);
    fclose(f);

    crepe_free(ctx);
    return 0;
}
