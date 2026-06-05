// voxtral-test-llm — differential test for the Voxtral (3B) text-only
// LLM forward path. Runs voxtral_run_llm() on a fixed prompt and
// compares the last-token logits + argmax against PyTorch.
//
// Reference archive is produced by:
//
//   python tools/dump_reference.py --backend voxtral \
//       --model-dir /path/to/hf/voxtral-mini-3b-2507 \
//       --audio samples/jfk.wav \
//       --stages mel_spectrogram,llm_input_ids,llm_logits \
//       --output /tmp/voxtral-ref.gguf
//
// Note: the current tools/reference_backends/voxtral.py doesn't emit
// `llm_input_ids` / `llm_logits` by default (it focuses on the audio
// encoder + projector). When those stages are missing the driver
// reports [SKIP] and returns 0 — not a failure. Extending the
// reference backend to emit them is tracked in TODO.md.
//
// Usage:
//   voxtral-test-llm voxtral-mini-3b-2507.gguf /tmp/voxtral-ref.gguf

#include "crispasr_diff.h"
#include "voxtral.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

int main(int argc, char ** argv) {
    if (argc < 3) {
        fprintf(stderr,
                "usage: %s voxtral-mini-3b-2507.gguf reference.gguf\n",
                argv[0]);
        return 1;
    }
    const char * model_path = argv[1];
    const char * ref_path   = argv[2];

    crispasr_diff::Ref ref;
    if (!ref.load(ref_path)) return 2;

    if (!ref.has("llm_input_ids") || !ref.has("llm_logits")) {
        fprintf(stderr,
                "[SKIP] reference archive missing 'llm_input_ids' and/or "
                "'llm_logits'. The current voxtral reference backend does "
                "not emit these stages; extend "
                "tools/reference_backends/voxtral.py if you need this test.\n");
        return 0;
    }

    // ---- Load input tokens (stored as F32 in the archive) ----
    auto [ids_f32, ids_n] = ref.get_f32("llm_input_ids");
    std::vector<int32_t> ids(ids_n);
    for (size_t i = 0; i < ids_n; i++) ids[i] = (int32_t)ids_f32[i];
    const int T = (int)ids.size();
    fprintf(stderr, "input_ids: %d tokens\n", T);

    auto ref_shape = ref.shape("llm_logits");
    const int ref_T     = ref_shape.size() >= 2 ? (int)ref_shape[1] : 0;
    const int ref_vocab = ref_shape.size() >= 1 ? (int)ref_shape[0] : 0;
    fprintf(stderr, "ref logits: T=%d vocab=%d\n", ref_T, ref_vocab);

    auto cp = voxtral_context_default_params();
    cp.n_threads = 4;
    auto * ctx = voxtral_init_from_file(model_path, cp);
    if (!ctx) { fprintf(stderr, "init failed\n"); return 4; }

    int n_t = 0, vocab = 0;
    float * logits = voxtral_run_llm(ctx, ids.data(), T, &n_t, &vocab);
    if (!logits) {
        fprintf(stderr, "voxtral_run_llm failed\n");
        voxtral_free(ctx);
        return 5;
    }
    fprintf(stderr, "C++ logits: %d x %d (last-token-only)\n", n_t, vocab);

    if (vocab != ref_vocab) {
        fprintf(stderr, "vocab mismatch: cpp=%d ref=%d\n", vocab, ref_vocab);
        free(logits); voxtral_free(ctx); return 6;
    }

    // The C++ side returns only the last position's logits; the
    // reference has the full (T, vocab). Compare against ref[T-1].
    auto [ref_all, ref_all_n] = ref.get_f32("llm_logits");
    const float * ref_last = ref_all + (size_t)(ref_T - 1) * ref_vocab;

    int cpp_argmax = 0, ref_argmax = 0;
    float cpp_max = -1e30f, ref_max = -1e30f;
    double dot = 0.0, na = 0.0, nb = 0.0;
    float max_abs = 0.0f;
    for (int k = 0; k < vocab; k++) {
        if (logits[k]   > cpp_max) { cpp_max = logits[k]; cpp_argmax = k; }
        if (ref_last[k] > ref_max) { ref_max = ref_last[k]; ref_argmax = k; }
        const double a = logits[k], b = ref_last[k];
        dot += a * b; na += a * a; nb += b * b;
        const float ad = std::fabs(logits[k] - ref_last[k]);
        if (ad > max_abs) max_abs = ad;
    }
    const double cs = dot / (std::sqrt(na) * std::sqrt(nb) + 1e-12);
    fprintf(stderr, "\nLAST-TOKEN LOGIT DIFF:\n");
    fprintf(stderr, "  cpp argmax: %d (logit %.4f)\n", cpp_argmax, cpp_max);
    fprintf(stderr, "  ref argmax: %d (logit %.4f)\n", ref_argmax, ref_max);
    fprintf(stderr, "  cosine sim: %.6f\n", cs);
    fprintf(stderr, "  max abs:    %.4e\n", max_abs);
    fprintf(stderr, "  match: %s\n", cpp_argmax == ref_argmax ? "PASS" : "FAIL");

    const bool pass = (cpp_argmax == ref_argmax) && (cs > 0.99);
    fprintf(stderr, "\nverdict: %s\n", pass ? "PASS" : "FAIL");

    free(logits);
    voxtral_free(ctx);
    return pass ? 0 : 1;
}
