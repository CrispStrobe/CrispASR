// crispasr_backend_moonshine_streaming.cpp — Moonshine Streaming ASR backend adapter.

#include "crispasr_backend.h"
#include "moonshine_streaming.h"
#include "whisper_params.h"

#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

class MoonshineStreamingBackend : public CrispasrBackend {
public:
    MoonshineStreamingBackend() = default;

    const char* name() const override { return "moonshine-streaming"; }

    uint32_t capabilities() const override {
        // Verified against src/moonshine_streaming.cpp as of 2026-05-04:
        // uses ggml_flash_attn_ext (×3); produces segments → CAP_DIARIZE
        // works as the framework post-step.
        return CAP_AUTO_DOWNLOAD | CAP_TIMESTAMPS_CTC | CAP_FLASH_ATTN | CAP_DIARIZE | CAP_TEMPERATURE |
               CAP_TOKEN_CONFIDENCE;
    }

    bool init(const whisper_params& params) override {
        moonshine_streaming_context_params cp = moonshine_streaming_context_default_params();
        cp.n_threads = params.n_threads;
        cp.verbosity = params.no_prints ? 0 : 1;
        cp.temperature = params.temperature;
        if (getenv("CRISPASR_VERBOSE") || getenv("MOONSHINE_STREAMING_BENCH"))
            cp.verbosity = 2;
        ctx_ = moonshine_streaming_init_from_file(params.model.c_str(), cp);
        return ctx_ != nullptr;
    }

    std::vector<crispasr_segment> transcribe(const float* samples, int n_samples, int64_t t_offset_cs,
                                             const whisper_params& /*params*/) override {
        std::vector<crispasr_segment> out;
        if (!ctx_)
            return out;

        moonshine_streaming_result* r = moonshine_streaming_transcribe_with_probs(ctx_, samples, n_samples);
        if (!r || !r->text || !r->text[0]) {
            moonshine_streaming_result_free(r);
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
            const char* piece = moonshine_streaming_token_text(ctx_, r->token_ids[i]);
            if (piece)
                tok.text = piece;
            seg.tokens.push_back(std::move(tok));
        }
        moonshine_streaming_result_free(r);

        while (!seg.text.empty() && (seg.text.front() == ' ' || seg.text.front() == '\n'))
            seg.text.erase(seg.text.begin());
        while (!seg.text.empty() && (seg.text.back() == ' ' || seg.text.back() == '\n'))
            seg.text.pop_back();

        if (!seg.text.empty())
            out.push_back(std::move(seg));
        return out;
    }

    void shutdown() override {
        if (ctx_) {
            moonshine_streaming_free(ctx_);
            ctx_ = nullptr;
        }
    }

    ~MoonshineStreamingBackend() override { MoonshineStreamingBackend::shutdown(); }

private:
    moonshine_streaming_context* ctx_ = nullptr;
};

std::unique_ptr<CrispasrBackend> crispasr_make_moonshine_streaming_backend() {
    return std::make_unique<MoonshineStreamingBackend>();
}
