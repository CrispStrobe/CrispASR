// crispasr_backend_funasr.cpp — FunAudioLLM/Fun-ASR-Nano-2512 adapter.
//
// The library's funasr_transcribe() already implements the full pipeline
// (kaldi-fbank + LFR → 70 SANM encoder blocks → 2-block adaptor → ChatML
// prompt + audio splice → Qwen3-0.6B KV-cached AR decode). The adapter is
// just a thin wrapper that funnels params through and lifts the resulting
// UTF-8 string into a single crispasr_segment.

#include "crispasr_backend.h"
#include "crispasr_backend_utils.h"
#include "funasr.h"
#include "whisper_params.h"

#include <cstdlib>
#include <cstring>
#include <vector>

class FunAsrBackend : public CrispasrBackend {
public:
    FunAsrBackend() = default;

    const char* name() const override { return "funasr"; }

    uint32_t capabilities() const override {
        return CAP_AUTO_DOWNLOAD | CAP_FLASH_ATTN | CAP_PUNCTUATION_TOGGLE | CAP_DIARIZE | CAP_TIMESTAMPS_CTC |
               CAP_TEMPERATURE | CAP_TOKEN_CONFIDENCE | CAP_BEAM_SEARCH;
    }

    bool init(const whisper_params& p) override {
        funasr_context_params cp = funasr_context_default_params();
        cp.n_threads = p.n_threads;
        cp.verbosity = p.no_prints ? 0 : 1;
        cp.use_gpu = crispasr_backend_should_use_gpu(p);
        cp.temperature = p.temperature;
        ctx_ = funasr_init_from_file(p.model.c_str(), cp);
        if (!ctx_) {
            fprintf(stderr, "crispasr[funasr]: failed to load model '%s'\n", p.model.c_str());
            return false;
        }
        return true;
    }

    std::vector<crispasr_segment> transcribe(const float* samples, int n_samples, int64_t t_offset_cs,
                                             const whisper_params& params) override {
        std::vector<crispasr_segment> out;
        if (!ctx_)
            return out;

        funasr_set_beam_size(ctx_, params.beam_size > 0 ? params.beam_size : 1);
        funasr_result* r = funasr_transcribe_with_probs(ctx_, samples, n_samples);
        if (!r || !r->text) {
            funasr_result_free(r);
            fprintf(stderr, "crispasr[funasr]: transcribe failed\n");
            return out;
        }

        crispasr_segment seg;
        seg.t0 = t_offset_cs;
        seg.t1 = t_offset_cs + (int64_t)((double)n_samples / 16000.0 * 100.0);
        seg.text = r->text;

        for (int i = 0; i < r->n_tokens; i++) {
            crispasr_token tok;
            tok.id = r->token_ids[i];
            tok.confidence = r->token_probs[i];
            const char* piece = funasr_token_text(ctx_, r->token_ids[i]);
            if (piece)
                tok.text = piece;
            seg.tokens.push_back(std::move(tok));
        }
        funasr_result_free(r);

        while (!seg.text.empty() && (seg.text.front() == ' ' || seg.text.front() == '\n'))
            seg.text.erase(seg.text.begin());
        while (!seg.text.empty() && (seg.text.back() == ' ' || seg.text.back() == '\n'))
            seg.text.pop_back();

        if (!params.punctuation) {
            crispasr_strip_ascii_punctuation(seg.text);
            crispasr_lowercase_ascii(seg.text);
        }

        if (!seg.text.empty())
            out.push_back(std::move(seg));
        return out;
    }

    void shutdown() override {
        if (ctx_) {
            funasr_free(ctx_);
            ctx_ = nullptr;
        }
    }

    ~FunAsrBackend() override { FunAsrBackend::shutdown(); }

private:
    funasr_context* ctx_ = nullptr;
};

std::unique_ptr<CrispasrBackend> crispasr_make_funasr_backend() {
    return std::make_unique<FunAsrBackend>();
}
