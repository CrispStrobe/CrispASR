// crispasr_backend_canary_qwen.cpp — adapter for nvidia/canary-qwen-2.5b.
//
// SALM: FastConformer encoder + linear projection + Qwen3-1.7B LLM decoder.
// English ASR only. Wraps canary_qwen_init_from_file + canary_qwen_transcribe_ex.

#include "crispasr_backend.h"
#include "crispasr_backend_utils.h"
#include "whisper_params.h"

#include "canary_qwen.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

class CanaryQwenBackend : public CrispasrBackend {
public:
    CanaryQwenBackend() = default;
    ~CanaryQwenBackend() override { CanaryQwenBackend::shutdown(); }

    const char* name() const override { return "canary-qwen"; }

    uint32_t capabilities() const override {
        return CAP_TOKEN_CONFIDENCE | CAP_FLASH_ATTN | CAP_TEMPERATURE | CAP_BEAM_SEARCH | CAP_AUTO_DOWNLOAD |
               CAP_UNBOUNDED_INPUT | CAP_INTERNAL_CHUNKING;
    }

    bool init(const whisper_params& p) override {
        canary_qwen_context_params cp = canary_qwen_context_default_params();
        cp.n_threads = p.n_threads;
        cp.flash_attn = p.flash_attn;
        cp.verbosity = p.no_prints ? 0 : 1;
        cp.use_gpu = crispasr_backend_should_use_gpu(p);

        ctx_ = canary_qwen_init_from_file(p.model.c_str(), cp);
        if (!ctx_) {
            fprintf(stderr, "crispasr[canary-qwen]: failed to load model '%s'\n", p.model.c_str());
            return false;
        }
        return true;
    }

    void warmup() override {
        if (!ctx_)
            return;
        std::vector<float> silence(8000, 0.0f);
        canary_qwen_result* r = canary_qwen_transcribe_ex(ctx_, silence.data(), (int)silence.size());
        if (r)
            canary_qwen_result_free(r);
    }

    std::vector<crispasr_segment> transcribe(const float* samples, int n_samples, int64_t t_offset_cs,
                                             const whisper_params& params) override {
        std::vector<crispasr_segment> out;
        if (!ctx_)
            return out;

        canary_qwen_set_temperature(ctx_, params.temperature, params.seed);
        canary_qwen_set_beam_size(ctx_, params.beam_size > 0 ? params.beam_size : 1);

        canary_qwen_result* r = canary_qwen_transcribe_ex(ctx_, samples, n_samples);
        if (!r)
            return out;

        crispasr_segment seg;
        seg.text = r->text ? r->text : "";
        crispasr_apply_ngram_loop_fix(seg.text, name(), params.no_prints);
        seg.t0 = t_offset_cs;
        // Estimate duration from audio length
        seg.t1 = t_offset_cs + (int64_t)(100.0 * n_samples / 16000.0);

        // Per-token data
        for (int i = 0; i < r->n_tokens; i++) {
            crispasr_token tok;
            tok.id = r->tokens[i].id;
            tok.text = r->tokens[i].text;
            tok.confidence = r->tokens[i].p;
            tok.t0 = seg.t0;
            tok.t1 = seg.t1;
            seg.tokens.push_back(tok);
        }

        out.push_back(std::move(seg));
        canary_qwen_result_free(r);
        return out;
    }

    void shutdown() override {
        if (ctx_) {
            canary_qwen_free(ctx_);
            ctx_ = nullptr;
        }
    }

private:
    canary_qwen_context* ctx_ = nullptr;
};

} // namespace

std::unique_ptr<CrispasrBackend> crispasr_make_canary_qwen_backend() {
    return std::make_unique<CanaryQwenBackend>();
}
