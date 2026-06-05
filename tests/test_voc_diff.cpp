// test_voc_diff.cpp — per-stage vocoder diff against Python reference.
// Uses the OLD voc-ref.gguf (which has C as fast axis in GGUF, requiring transpose).

#include "chatterbox_s3gen.h"
#include "crispasr_diff.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static std::vector<float> transpose_2d(const float* data, int slow_dim, int fast_dim) {
    std::vector<float> out(slow_dim * fast_dim);
    for (int s = 0; s < slow_dim; s++)
        for (int f = 0; f < fast_dim; f++)
            out[f * slow_dim + s] = data[s * fast_dim + f];
    return out;
}

int main(int argc, char** argv) {
    const char* model_path = "/mnt/storage/chatterbox/chatterbox-s3gen-f16.gguf";
    const char* ref_path = "/mnt/storage/chatterbox/voc-ref.gguf";
    const char* mel_path = "/mnt/storage/chatterbox/ref_mel_80x62.bin";

    if (argc >= 2)
        model_path = argv[1];
    if (argc >= 3)
        ref_path = argv[2];
    if (argc >= 4)
        mel_path = argv[3];

    crispasr_diff::Ref ref;
    if (!ref.load(ref_path)) {
        fprintf(stderr, "ERROR: cannot load ref\n");
        return 1;
    }

    int T_mel = 62;
    std::vector<float> mel(T_mel * 80);
    FILE* fmel = fopen(mel_path, "rb");
    if (!fmel || fread(mel.data(), 4, mel.size(), fmel) != mel.size()) {
        fprintf(stderr, "ERROR: cannot load mel\n");
        return 1;
    }
    fclose(fmel);

    auto* ctx = chatterbox_s3gen_init_from_file(model_path, 4, 2, false);
    if (!ctx) {
        fprintf(stderr, "ERROR: cannot load model\n");
        return 1;
    }

    const char* stage_names[] = {"voc_conv_pre", "voc_ups_0", "voc_rb_0", "voc_ups_1",
                                 "voc_rb_1",     "voc_ups_2", "voc_rb_2", "voc_conv_post"};
    constexpr int N = 8;
    float* stage_data[N] = {};
    int stage_sizes[N] = {};

    // Shapes: (C, T) for each stage
    int stage_C[] = {512, 256, 256, 128, 128, 64, 64, 18};
    int stage_T[] = {62, 496, 496, 2480, 2480, 7440, 7440, 7440};

    int n_samples = 0;
    float* pcm =
        chatterbox_s3gen_vocode_dump(ctx, mel.data(), T_mel, &n_samples, stage_names, stage_data, stage_sizes, N);
    fprintf(stderr, "\nVocoder: %d samples\n\n", n_samples);

    fprintf(stderr, "=== Per-stage comparison ===\n");
    const char* first_fail = nullptr;
    for (int i = 0; i < N; i++) {
        if (!stage_data[i]) {
            fprintf(stderr, "  %-16s  MISSING\n", stage_names[i]);
            continue;
        }
        // C++ data: ne[0]=T(fast), ne[1]=C(slow) → data[c*T+t]
        // Ref data: ne[0]=C(fast), ne[1]=T(slow) → data[t*C+c]
        // Transpose C++ to match ref layout
        auto cpp_t = transpose_2d(stage_data[i], stage_C[i], stage_T[i]);
        auto r = ref.compare(stage_names[i], cpp_t.data(), cpp_t.size());
        if (!r.found) {
            fprintf(stderr, "  %-16s  NOT IN REF\n", stage_names[i]);
            continue;
        }
        bool pass = r.is_pass(0.999f);
        fprintf(stderr, "  %-16s  cos_min=%.6f  cos_mean=%.6f  max_abs=%.2e  rms=%.2e  %s\n", stage_names[i], r.cos_min,
                r.cos_mean, r.max_abs, r.rms, pass ? "PASS" : "**FAIL**");
        if (!pass && !first_fail)
            first_fail = stage_names[i];
    }

    fprintf(stderr, "\n");
    if (first_fail)
        fprintf(stderr, ">>> FIRST DIVERGENT STAGE: %s\n", first_fail);
    else
        fprintf(stderr, ">>> ALL STAGES PASS\n");

    for (int i = 0; i < N; i++)
        free(stage_data[i]);
    if (pcm)
        free(pcm);
    chatterbox_s3gen_free(ctx);
    return first_fail ? 1 : 0;
}
