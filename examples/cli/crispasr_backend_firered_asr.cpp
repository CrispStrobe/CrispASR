// crispasr_backend_firered_asr.cpp — FireRedASR2-AED backend adapter.

#include "crispasr_backend.h"
#include "crispasr_backend_utils.h"
#include "firered_asr.h"
#include "whisper_params.h"

#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

class FireredAsrBackend : public CrispasrBackend {
public:
    FireredAsrBackend() = default;

    const char* name() const override { return "firered-asr"; }

    uint32_t capabilities() const override {
        return CAP_TIMESTAMPS_CTC | CAP_AUTO_DOWNLOAD | CAP_BEAM_SEARCH | CAP_TOKEN_CONFIDENCE | CAP_FLASH_ATTN;
    }

    bool init(const whisper_params& params) override {
        firered_asr_context_params cp = firered_asr_context_default_params();
        cp.n_threads = params.n_threads;
        cp.verbosity = params.no_prints ? 0 : 1;
        if (getenv("CRISPASR_VERBOSE") || getenv("FIRERED_BENCH"))
            cp.verbosity = 2;
        cp.use_gpu = crispasr_backend_should_use_gpu(params);
        cp.beam_size = params.beam_size > 0 ? params.beam_size : 3;
        ctx_ = firered_asr_init_from_file(params.model.c_str(), cp);
        return ctx_ != nullptr;
    }

    std::vector<crispasr_segment> transcribe(const float* samples, int n_samples, int64_t t_offset_cs,
                                             const whisper_params& params) override {
        (void)params;
        std::vector<crispasr_segment> out;
        if (!ctx_)
            return out;

        firered_asr_result* r = firered_asr_transcribe_with_probs(ctx_, samples, n_samples);
        if (!r || !r->text)
            return out;

        crispasr_segment seg;
        seg.t0 = t_offset_cs;
        seg.t1 = t_offset_cs + (int64_t)((double)n_samples / 16000.0 * 100.0);
        seg.text = r->text;

        while (!seg.text.empty() && (seg.text.front() == ' ' || seg.text.front() == '\n'))
            seg.text.erase(seg.text.begin());
        while (!seg.text.empty() && (seg.text.back() == ' ' || seg.text.back() == '\n'))
            seg.text.pop_back();

        // Per-token confidence: one entry per emitted (non-special) token from
        // the winning beam. Timestamps are not available at token granularity
        // here (firered's AR decoder doesn't emit time-aligned outputs), so
        // we leave t0/t1 unset (-1).
        seg.tokens.reserve((size_t)r->n_tokens);
        for (int i = 0; i < r->n_tokens; i++) {
            crispasr_token tok;
            tok.id = r->token_ids[i];
            tok.confidence = r->token_probs[i];
            const char* piece = firered_asr_token_text(ctx_, r->token_ids[i]);
            if (piece) {
                // Convert ▁ (U+2581, 0xE2 0x96 0x81) to space — same convention as
                // firered_asr_transcribe_impl.
                std::string p = piece;
                std::string decoded;
                for (size_t ci = 0; ci < p.size(); ci++) {
                    if ((unsigned char)p[ci] == 0xE2 && ci + 2 < p.size() && (unsigned char)p[ci + 1] == 0x96 &&
                        (unsigned char)p[ci + 2] == 0x81) {
                        decoded += ' ';
                        ci += 2;
                    } else {
                        decoded += p[ci];
                    }
                }
                tok.text = std::move(decoded);
            }
            seg.tokens.push_back(std::move(tok));
        }

        firered_asr_result_free(r);

        if (!seg.text.empty())
            out.push_back(std::move(seg));
        return out;
    }

    void shutdown() override {
        if (ctx_) {
            firered_asr_free(ctx_);
            ctx_ = nullptr;
        }
    }

    ~FireredAsrBackend() override { FireredAsrBackend::shutdown(); }

private:
    firered_asr_context* ctx_ = nullptr;
};

std::unique_ptr<CrispasrBackend> crispasr_make_firered_asr_backend() {
    return std::make_unique<FireredAsrBackend>();
}
