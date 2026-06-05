// qwen3-asr-test-llm — differential test for the Qwen3 text-only LLM
// forward path. Runs qwen3_asr_run_llm() on a fixed prompt and compares
// the output logits against the PyTorch reference, per position,
// including top-1 argmax agreement.
//
// Reference archive is produced by:
//
//   python tools/dump_reference.py --backend qwen3 \
//       --model-dir /path/to/hf/qwen3-asr-0.6b \
//       --audio samples/jfk.wav \
//       --output /tmp/qwen3-ref.gguf
//
// Usage:
//   qwen3-asr-test-llm qwen3-asr-0.6b.gguf /tmp/qwen3-ref.gguf

#include "crispasr_diff.h"
#include "qwen3_asr.h"

#include <cstdio>
#include <cstdlib>
#include <vector>

int main(int argc, char ** argv) {
    if (argc < 3) {
        fprintf(stderr,
                "usage: %s qwen3-asr-0.6b.gguf reference.gguf\n"
                "\n"
                "  reference.gguf  archive from tools/dump_reference.py --backend qwen3\n"
                "                  must include 'llm_input_ids' and 'llm_logits'\n",
                argv[0]);
        return 1;
    }

    crispasr_diff::Ref ref;
    if (!ref.load(argv[2])) return 2;

    // ---- Load input token IDs ----
    // Note: the new reference backend doesn't dump input_ids by default.
    // Older legacy archives stored them as 'llm_input_ids'. If that's
    // missing the tool can't run — return early with a clear message.
    if (!ref.has("llm_input_ids")) {
        fprintf(stderr,
                "reference archive is missing 'llm_input_ids'.\n"
                "Add the stage to tools/reference_backends/qwen3.py's "
                "DEFAULT_STAGES and re-dump.\n");
        return 3;
    }
    auto [ids_ptr, ids_n] = ref.get_f32("llm_input_ids");
    // ids are stored as F32 in the archive (core_diff only handles F32
    // for now). The Python side should convert ids -> int32.
    if (!ids_ptr || ids_n == 0) {
        fprintf(stderr, "llm_input_ids is empty\n");
        return 3;
    }
    std::vector<int32_t> ids(ids_n);
    for (size_t i = 0; i < ids_n; i++) ids[i] = (int32_t)ids_ptr[i];
    fprintf(stderr, "input_ids: %zu tokens\n", ids.size());

    if (!ref.has("llm_logits")) {
        fprintf(stderr, "reference archive is missing 'llm_logits'\n");
        return 4;
    }
    auto ref_shape = ref.shape("llm_logits");
    if (ref_shape.size() < 2) {
        fprintf(stderr, "llm_logits has fewer than 2 dims\n");
        return 4;
    }
    // ne-order: [vocab, T] so vocab is dim 0, T is dim 1.
    const int ref_vocab = (int)ref_shape[0];
    const int ref_T     = (int)ref_shape[1];
    fprintf(stderr, "ref logits: T=%d vocab=%d\n", ref_T, ref_vocab);

    // ---- Init model + run LLM ----
    auto cp = qwen3_asr_context_default_params();
    cp.n_threads = 4;
    auto * ctx = qwen3_asr_init_from_file(argv[1], cp);
    if (!ctx) { fprintf(stderr, "init failed\n"); return 5; }

    int n_t = 0, vocab = 0;
    float * logits = qwen3_asr_run_llm(ctx, ids.data(), (int)ids.size(),
                                       &n_t, &vocab);
    if (!logits) {
        fprintf(stderr, "run_llm failed\n");
        qwen3_asr_free(ctx);
        return 6;
    }
    fprintf(stderr, "C++ logits: T=%d vocab=%d\n", n_t, vocab);

    if (n_t != ref_T || vocab != ref_vocab) {
        fprintf(stderr, "shape mismatch: cpp(%d,%d) vs ref(%d,%d)\n",
                n_t, vocab, ref_T, ref_vocab);
        free(logits);
        qwen3_asr_free(ctx);
        return 7;
    }

    // ---- Diff ----
    const size_t n_elem = (size_t)n_t * vocab;
    auto elem = ref.compare("llm_logits", logits, n_elem);
    auto top1 = ref.compare_argmax("llm_logits", logits, n_elem);

    fprintf(stderr, "\nLOGIT DIFF:\n");
    fprintf(stderr, "  max_abs=%.4e mean_abs=%.4e rms=%.4e\n",
            elem.max_abs, elem.mean_abs, elem.rms);
    fprintf(stderr, "  cos_mean=%.6f cos_min=%.6f\n",
            elem.cos_mean, elem.cos_min);
    fprintf(stderr, "  top-1 match: %d / %d positions\n",
            top1.top1_match, top1.top1_total);

    const bool pass = elem.cos_min > 0.999f &&
                      top1.top1_match == top1.top1_total;
    fprintf(stderr, "  verdict: %s\n", pass ? "PASS" : "FAIL");

    free(logits);
    qwen3_asr_free(ctx);
    return pass ? 0 : 1;
}
