// crispasr_backend_moonshine.cpp — Moonshine ASR backend adapter.

#include "crispasr_backend.h"
#include "crispasr_backend_utils.h"
#include "moonshine.h"
#include "whisper_params.h"

#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

class MoonshineBackend : public CrispasrBackend {
public:
    MoonshineBackend() = default;

    const char* name() const override { return "moonshine"; }

    uint32_t capabilities() const override {
        return CAP_AUTO_DOWNLOAD | CAP_TOKEN_CONFIDENCE | CAP_TEMPERATURE | CAP_PUNCTUATION_TOGGLE;
    }

    bool init(const whisper_params& params) override {
        struct moonshine_init_params mp = {};
        mp.model_path = params.model.c_str();
        mp.tokenizer_path = nullptr; // auto-detect
        mp.n_threads = params.n_threads;
        ctx_ = moonshine_init_with_params(mp);
        return ctx_ != nullptr;
    }

    std::vector<crispasr_segment> transcribe(const float* samples, int n_samples, int64_t t_offset_cs,
                                             const whisper_params& params) override {
        std::vector<crispasr_segment> out;
        if (!ctx_)
            return out;

        moonshine_set_temperature(ctx_, params.temperature);
        moonshine_result* r = moonshine_transcribe_with_probs(ctx_, samples, n_samples);
        if (!r || !r->text || !r->text[0]) {
            if (r)
                moonshine_result_free(r);
            return out;
        }

        crispasr_segment seg;
        seg.t0 = t_offset_cs;
        seg.t1 = t_offset_cs + (int64_t)((double)n_samples / 16000.0 * 100.0);
        seg.text = r->text;

        while (!seg.text.empty() && (seg.text.front() == ' ' || seg.text.front() == '\n'))
            seg.text.erase(seg.text.begin());
        while (!seg.text.empty() && (seg.text.back() == ' ' || seg.text.back() == '\n'))
            seg.text.pop_back();

        // Token-level confidence: one entry per emitted (non-EOS) decoder
        // step. No per-token timestamps (moonshine's decoder isn't time-aligned),
        // leave t0/t1 unset (-1).
        seg.tokens.reserve((size_t)r->n_tokens);
        for (int i = 0; i < r->n_tokens; i++) {
            crispasr_token tok;
            tok.id = r->token_ids[i];
            tok.confidence = r->token_probs[i];
            const char* piece = moonshine_token_text(ctx_, r->token_ids[i]);
            if (piece && piece[0])
                tok.text = piece;
            seg.tokens.push_back(std::move(tok));
        }

        moonshine_result_free(r);

        // --no-punctuation: post-strip ASCII punctuation + lowercase.
        if (!params.punctuation) {
            crispasr_strip_ascii_punctuation(seg.text);
            crispasr_lowercase_ascii(seg.text);
            for (auto& tok : seg.tokens) {
                crispasr_strip_ascii_punctuation(tok.text);
                crispasr_lowercase_ascii(tok.text);
            }
        }

        if (!seg.text.empty())
            out.push_back(std::move(seg));
        return out;
    }

    void shutdown() override {
        if (ctx_) {
            moonshine_free(ctx_);
            ctx_ = nullptr;
        }
    }

    ~MoonshineBackend() override { MoonshineBackend::shutdown(); }

private:
    moonshine_context* ctx_ = nullptr;
};

std::unique_ptr<CrispasrBackend> crispasr_make_moonshine_backend() {
    return std::make_unique<MoonshineBackend>();
}
