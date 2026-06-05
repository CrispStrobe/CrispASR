// qwen3-asr-test-conv — two-stage differential test for the Qwen3-ASR
// conv front-end + full audio encoder.
//
// Stage 1: run qwen3_asr_run_conv() on the reference mel and compare
//          against conv_out.
// Stage 2: run qwen3_asr_run_encoder() on the reference mel and compare
//          against proj2_out (per-row cosine similarity).
//
// Both stages used to inline ~50 lines of NPY parsing + metric
// computation; they now go through the shared crispasr_diff::Ref
// harness defined in examples/cli/crispasr_diff.{h,cpp}.
//
// Reference archive is produced by:
//
//   python tools/dump_reference.py --backend qwen3 \
//       --model-dir /path/to/hf/qwen3-asr-0.6b \
//       --audio samples/jfk.wav \
//       --output /tmp/qwen3-ref.gguf
//
// Usage:
//   qwen3-asr-test-conv qwen3-asr-0.6b.gguf /tmp/qwen3-ref.gguf

#include "crispasr_diff.h"
#include "qwen3_asr.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

int main(int argc, char ** argv) {
    if (argc < 3) {
        fprintf(stderr,
                "usage: %s qwen3-asr-0.6b.gguf reference.gguf\n"
                "\n"
                "  reference.gguf  archive from tools/dump_reference.py --backend qwen3\n",
                argv[0]);
        return 1;
    }
    const char * model_path = argv[1];
    const char * ref_path   = argv[2];

    crispasr_diff::Ref ref;
    if (!ref.load(ref_path)) return 2;

    // ---- Load reference mel ----
    auto [mel_ptr, mel_n] = ref.get_f32("mel_spectrogram");
    if (!mel_ptr) {
        fprintf(stderr, "reference archive is missing 'mel_spectrogram'\n");
        return 3;
    }
    auto mel_shape = ref.shape("mel_spectrogram");
    // Dumper writes (1, 128, T) → squeezed to (128, T). Ref::shape
    // returns the ordering we find by matching the known n_mels=128.
    int n_mels = 128, T_mel = 0;
    for (auto d : mel_shape) {
        if (d == 128) { /* mel axis */ }
        else if (T_mel == 0) T_mel = (int)d;
    }
    if (T_mel == 0) {
        fprintf(stderr, "unexpected mel_spectrogram shape\n");
        return 3;
    }
    fprintf(stderr, "reference mel: %d mels x %d frames\n", n_mels, T_mel);

    // ---- Init model ----
    auto cp = qwen3_asr_context_default_params();
    cp.n_threads = 4;
    cp.verbosity = 1;
    auto * ctx = qwen3_asr_init_from_file(model_path, cp);
    if (!ctx) { fprintf(stderr, "init failed\n"); return 5; }

    // ==============================================
    // Stage 1: conv front-end
    // ==============================================
    fprintf(stderr, "\n=== Stage 1: conv front-end ===\n");
    int n_chunks = 0, T_out = 0, d = 0;
    float * conv_out = qwen3_asr_run_conv(ctx, mel_ptr, n_mels, T_mel,
                                          &n_chunks, &T_out, &d);
    if (!conv_out) {
        fprintf(stderr, "run_conv failed\n");
        qwen3_asr_free(ctx);
        return 6;
    }
    fprintf(stderr, "C++ conv_out: num_chunks=%d T_chunk_out=%d d=%d\n",
            n_chunks, T_out, d);

    int stage1 = 0;
    {
        auto rep = ref.compare("conv_out", conv_out,
                               (size_t)n_chunks * T_out * d);
        if (!rep.found) {
            fprintf(stderr, "  (conv_out missing from reference — skipping)\n");
        } else {
            fprintf(stderr,
                    "  max_abs=%.4e mean_abs=%.4e rms=%.4e cos_min=%.6f\n",
                    rep.max_abs, rep.mean_abs, rep.rms, rep.cos_min);
            // Stage 1 has historically been judged on max_abs < 1e-2.
            const bool pass1 = rep.max_abs < 1e-2f;
            fprintf(stderr, "  verdict: %s (max_abs<1e-2)\n",
                    pass1 ? "PASS" : "FAIL");
            stage1 = pass1 ? 0 : 1;
        }
    }
    free(conv_out);

    // ==============================================
    // Stage 2: full encoder + projector
    // ==============================================
    fprintf(stderr, "\n=== Stage 2: full encoder + projector ===\n");
    int N = 0, pdim = 0;
    float * enc = qwen3_asr_run_encoder(ctx, mel_ptr, n_mels, T_mel, &N, &pdim);
    if (!enc) {
        fprintf(stderr, "run_encoder failed\n");
        qwen3_asr_free(ctx);
        return 8;
    }
    fprintf(stderr, "C++ encoder output: N=%d pdim=%d\n", N, pdim);

    int stage2 = 0;
    {
        // The reference name is proj2_out in the qwen3 reference backend.
        // Newer archives may also expose the final encoder_output tensor.
        const char * name =
            ref.has("proj2_out") ? "proj2_out" : "encoder_output";
        auto rep = ref.compare(name, enc, (size_t)N * pdim);
        if (!rep.found) {
            fprintf(stderr,
                    "  (neither proj2_out nor encoder_output in reference)\n");
        } else {
            fprintf(stderr,
                    "  max_abs=%.4e mean_abs=%.4e rms=%.4e "
                    "cos_mean=%.6f cos_min=%.6f\n",
                    rep.max_abs, rep.mean_abs, rep.rms,
                    rep.cos_mean, rep.cos_min);
            const bool pass2 = rep.is_pass(0.999f);
            fprintf(stderr, "  verdict: %s (cos>0.999)\n",
                    pass2 ? "PASS" : "FAIL");
            stage2 = pass2 ? 0 : 1;
        }
    }
    free(enc);
    qwen3_asr_free(ctx);

    return stage1 | (stage2 << 1);
}
