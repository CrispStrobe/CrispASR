// voxtral-test-encoder — differential test for the Voxtral audio
// encoder + 4-frame-stack projector.
//
// Runs voxtral_run_encoder() on a reference mel spectrogram captured
// from the HF PyTorch forward pass, and compares the result against
// the PyTorch reference's projector output. Uses the shared
// crispasr_diff::Ref harness so the inline NPY parser + per-row
// cosine-similarity loop that this file used to carry
// (examples/cli/crispasr_diff.{h,cpp}) lives in one place now.
//
// Reference archive is produced by:
//
//   python tools/dump_reference.py --backend voxtral \
//       --model-dir /path/to/hf/voxtral-mini-3b-2507 \
//       --audio samples/jfk.wav \
//       --output /tmp/voxtral-ref.gguf
//
// Usage:
//   voxtral-test-encoder model.gguf /tmp/voxtral-ref.gguf

#include "crispasr_diff.h"
#include "voxtral.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

int main(int argc, char ** argv) {
    if (argc < 3) {
        fprintf(stderr,
                "usage: %s model.gguf reference.gguf\n"
                "\n"
                "  model.gguf      voxtral-mini-3b-2507 weights\n"
                "  reference.gguf  archive from tools/dump_reference.py\n",
                argv[0]);
        return 1;
    }

    crispasr_diff::Ref ref;
    if (!ref.load(argv[2])) return 2;

    // Pull the reference mel spectrogram (shape (n_mels, T) in the
    // archive — the voxtral3b reference backend writes it straight from
    // feat["input_features"][0] which is already 2D).
    auto [mel_ptr, mel_n] = ref.get_f32("mel_spectrogram");
    if (!mel_ptr) {
        fprintf(stderr, "reference archive is missing 'mel_spectrogram'\n");
        return 3;
    }
    auto mel_shape = ref.shape("mel_spectrogram");
    if (mel_shape.size() < 2) {
        fprintf(stderr, "mel_spectrogram shape has <2 dims\n");
        return 3;
    }
    // ref.shape() returns the dims from fastest to slowest, so the
    // last entry is the outermost axis for the underlying ndarray.
    // For (n_mels, T) stored row-major, that's shape = [T, n_mels].
    const int T_mel  = (int)mel_shape[0];
    const int n_mels = (int)mel_shape[1];
    fprintf(stderr, "reference mel: %d mels x %d frames  (%zu floats)\n",
            n_mels, T_mel, mel_n);

    // Init the voxtral context.
    auto cp = voxtral_context_default_params();
    cp.n_threads = 4;
    cp.verbosity = 0;
    auto * ctx = voxtral_init_from_file(argv[1], cp);
    if (!ctx) { fprintf(stderr, "failed to load model\n"); return 4; }

    // Run the encoder + projector on the reference mel.
    int N = 0, pdim = 0;
    fprintf(stderr, "running voxtral_run_encoder ...\n");
    float * out = voxtral_run_encoder(ctx, mel_ptr, n_mels, T_mel, &N, &pdim);
    if (!out) {
        fprintf(stderr, "voxtral_run_encoder returned null\n");
        voxtral_free(ctx);
        return 5;
    }
    fprintf(stderr, "C++ encoder output: %d x %d\n", N, pdim);

    // Compare against the reference projector_output. The archive name
    // is 'projector_output' in the voxtral reference backend; older
    // archives used 'proj2_out' from the legacy dump script, so try
    // both.
    crispasr_diff::Report rep;
    if (ref.has("projector_output")) {
        rep = ref.compare("projector_output", out, (size_t)N * pdim);
    } else if (ref.has("proj2_out")) {
        rep = ref.compare("proj2_out", out, (size_t)N * pdim);
    } else {
        fprintf(stderr,
                "reference archive has neither 'projector_output' nor "
                "'proj2_out' — dump with tools/dump_reference.py --backend "
                "voxtral\n");
        free(out);
        voxtral_free(ctx);
        return 6;
    }

    const float COS_THRESHOLD = 0.99f;
    fprintf(stderr, "\nDIFF vs projector output:\n");
    fprintf(stderr, "  max abs diff:       %.4e\n", rep.max_abs);
    fprintf(stderr, "  mean abs diff:      %.4e\n", rep.mean_abs);
    fprintf(stderr, "  RMS error:          %.4e\n", rep.rms);
    fprintf(stderr, "  per-row cos min:    %.6f\n", rep.cos_min);
    fprintf(stderr, "  per-row cos mean:   %.6f\n", rep.cos_mean);
    const bool pass = rep.is_pass(COS_THRESHOLD);
    fprintf(stderr, "  verdict: %s (cos threshold %.2f)\n",
            pass ? "PASS" : "FAIL", COS_THRESHOLD);

    free(out);
    voxtral_free(ctx);
    return pass ? 0 : 1;
}
