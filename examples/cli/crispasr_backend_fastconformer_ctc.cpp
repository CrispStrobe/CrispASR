// crispasr_backend_fastconformer_ctc.cpp — adapter for standalone
// NeMo FastConformer-CTC models (e.g. nvidia/stt_en_fastconformer_ctc_large).
//
// The underlying runtime lives in src/canary_ctc.cpp — it was originally
// written to host the auxiliary CTC aligner that ships inside the Canary
// 1B v2 .nemo tarball, but the FastConformer-CTC encoder it implements is
// the exact same architecture NeMo uses for its standalone ASR releases:
// ``stt_{lang}_fastconformer_ctc_{size}``. The only structural difference
// is that the standalone models carry biases on every Q/K/V/output/FFN
// linear (and on the conv module's pointwise convolutions), while the
// Canary aligner is bias-less. src/canary_ctc.cpp's loader now makes all
// of those bias slots optional, so a single runtime hosts both variants.
//
// This adapter exposes that runtime under the user-facing ``--backend
// fastconformer-ctc`` flag and produces a single `crispasr_segment`
// covering the whole clip, with the greedy CTC transcript as text. Native
// NeMo FastConformer-CTC models emit no segment boundaries and no
// per-word timestamps on their own — that comes via the separate
// canary-ctc forced aligner path (-am) when word timestamps are needed.

#include "crispasr_backend.h"
#include "whisper_params.h"

#include "canary_ctc.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace {

class FastConformerCtcBackend : public CrispasrBackend {
public:
    FastConformerCtcBackend() = default;
    ~FastConformerCtcBackend() override { FastConformerCtcBackend::shutdown(); }

    const char * name() const override { return "fastconformer-ctc"; }

    uint32_t capabilities() const override {
        // The NeMo FastConformer-CTC releases we target (stt_en_*_large and
        // friends) are English-only greedy-CTC models with no native
        // timestamp or punctuation control. Word-level timestamps via the
        // CTC aligner second pass (-am) work when requested.
        return CAP_TIMESTAMPS_CTC | CAP_PARALLEL_PROCESSORS;
    }

    bool init(const whisper_params & p) override {
        canary_ctc_context_params cp = canary_ctc_context_default_params();
        cp.n_threads = p.n_threads;
        cp.verbosity = p.no_prints ? 0 : 1;
        ctx_ = canary_ctc_init_from_file(p.model.c_str(), cp);
        if (!ctx_) {
            fprintf(stderr, "crispasr[fastconformer-ctc]: failed to load model '%s'\n",
                    p.model.c_str());
            return false;
        }
        return true;
    }

    std::vector<crispasr_segment> transcribe(
        const float * samples, int n_samples,
        int64_t t_offset_cs,
        const whisper_params & /*params*/) override
    {
        std::vector<crispasr_segment> out;
        if (!ctx_) return out;

        // Stage 1: encoder + CTC head → raw frame logits.
        float * logits = nullptr;
        int T_enc = 0, V = 0;
        int rc = canary_ctc_compute_logits(ctx_, samples, n_samples,
                                            &logits, &T_enc, &V);
        if (rc != 0 || !logits) {
            fprintf(stderr, "crispasr[fastconformer-ctc]: compute_logits failed (%d)\n", rc);
            return out;
        }

        // Stage 2: greedy CTC collapse → SentencePiece-detokenized text.
        char * text = canary_ctc_greedy_decode(ctx_, logits, T_enc, V);
        std::free(logits);
        if (!text) {
            fprintf(stderr, "crispasr[fastconformer-ctc]: greedy_decode failed\n");
            return out;
        }

        crispasr_segment seg;
        seg.t0 = t_offset_cs;
        seg.t1 = t_offset_cs + (int64_t)((double)n_samples / 16000.0 * 100.0);
        seg.text = text;
        std::free(text);
        out.push_back(std::move(seg));
        return out;
    }

    void shutdown() override {
        if (ctx_) {
            canary_ctc_free(ctx_);
            ctx_ = nullptr;
        }
    }

private:
    canary_ctc_context * ctx_ = nullptr;
};

} // namespace

std::unique_ptr<CrispasrBackend> crispasr_make_fastconformer_ctc_backend() {
    return std::unique_ptr<CrispasrBackend>(new FastConformerCtcBackend());
}
