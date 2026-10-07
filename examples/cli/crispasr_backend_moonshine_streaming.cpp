// crispasr_backend_moonshine_streaming.cpp — Moonshine Streaming ASR backend adapter.

#include "crispasr_backend.h"
#include "moonshine_streaming.h"
#include "whisper_params.h"

#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include "core/crispasr_env.h"

class MoonshineStreamingBackend : public CrispasrBackend {
public:
    MoonshineStreamingBackend() = default;

    const char* name() const override { return "moonshine-streaming"; }

    // These checkpoints are language-specific; skip external LID on -l auto.
    const char* sole_language() const override { return sole_; }

    uint32_t capabilities() const override {
        // #300/#308 audit: the model emits punctuated, sentence-cased text (verified
        // on jfk: "And so, my fellow Americans, ... your country."), so the CLI's
        // automatic FireRedPunc pass must not run over it — it appended a second
        // full stop ("country.."). Its sibling `moonshine` already opts out via
        // CAP_PUNCTUATION_TOGGLE; the streaming variant was missed.
        // Verified against src/moonshine_streaming.cpp as of 2026-05-04:
        // uses ggml_flash_attn_ext (×3); produces segments → CAP_DIARIZE
        // works as the framework post-step.
        return CAP_AUTO_DOWNLOAD | CAP_TIMESTAMPS_CTC | CAP_FLASH_ATTN | CAP_DIARIZE | CAP_TEMPERATURE |
               CAP_TOKEN_CONFIDENCE | CAP_BEAM_SEARCH | CAP_PUNCTUATION_NATIVE;
    }

    bool init(const whisper_params& params) override {
        sole_ = params.model.find("-de") != std::string::npos || params.model.find("_de") != std::string::npos ? "de"
                                                                                                               : "en";
        moonshine_streaming_context_params cp = moonshine_streaming_context_default_params();
        cp.n_threads = params.n_threads;
        cp.verbosity = params.no_prints ? 0 : 1;
        cp.temperature = params.temperature;
        // use_gpu deliberately NOT forwarded by default: the streaming encoder
        // runs ~550 frame-by-frame forward passes, which are launch-bound on
        // GPU — measured 3.2x SLOWER on M1 Metal (27.4s vs 8.5s CPU, identical
        // text). Revisit once the offline batch-encoder fast-path lands
        // (PLAN §232 Fix 2). MOONSHINE_STREAMING_GPU=1 opts in for A/B (and to
        // exercise the §232 hybrid decoder-weight placement).
        if (const char* g = crispasr_env::get("CRISPASR_MOONSHINE_STREAMING_GPU"))
            cp.use_gpu = (g[0] == '1');
        if (getenv("CRISPASR_VERBOSE") || crispasr_env::get("CRISPASR_MOONSHINE_STREAMING_BENCH"))
            cp.verbosity = 2;
        ctx_ = moonshine_streaming_init_from_file(params.model.c_str(), cp);
        return ctx_ != nullptr;
    }

    std::vector<crispasr_segment> transcribe(const float* samples, int n_samples, int64_t t_offset_cs,
                                             const whisper_params& params) override {
        std::vector<crispasr_segment> out;
        if (!ctx_)
            return out;

        if (sole_ && std::string(sole_) == "en" && !params.language.empty() && params.language != "auto" &&
            params.language != "en")
            fprintf(stderr, "crispasr[moonshine-streaming]: English-only model; language='%s' ignored\n",
                    params.language.c_str());
        moonshine_streaming_set_beam_size(ctx_, params.beam_size > 0 ? params.beam_size : 1);
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

    void transcribe_streaming(const float* samples, int n_samples, int64_t /*t_offset_cs*/,
                              const whisper_params& params, crispasr_stream_callback on_text) override {
        if (!ctx_) {
            CrispasrBackend::transcribe_streaming(samples, n_samples, 0, params, on_text);
            return;
        }
        std::string accumulated;
        bool first_tok = true;
        auto cb = [&](int tok_id, float /*prob*/, void* /*ud*/) {
            const char* raw = moonshine_streaming_token_text(ctx_, tok_id);
            if (!raw || !*raw)
                return;
            std::string piece(raw);
            size_t pos = 0;
            while ((pos = piece.find("\xe2\x96\x81", pos)) != std::string::npos) {
                piece.replace(pos, 3, " ");
                pos++;
            }
            if (first_tok) {
                size_t sp = 0;
                while (sp < piece.size() && (piece[sp] == ' ' || piece[sp] == '\n'))
                    sp++;
                piece = piece.substr(sp);
                if (!piece.empty())
                    first_tok = false;
            }
            accumulated += piece;
            if (!accumulated.empty())
                on_text(accumulated.c_str(), false);
        };
        auto cb_fn = [](int id, float p, void* ud) { (*static_cast<decltype(cb)*>(ud))(id, p, nullptr); };
        moonshine_streaming_transcribe_cb(ctx_, samples, n_samples, cb_fn, &cb);
        on_text(accumulated.c_str(), true);
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
    const char* sole_ = nullptr;
};

std::unique_ptr<CrispasrBackend> crispasr_make_moonshine_streaming_backend() {
    return std::make_unique<MoonshineStreamingBackend>();
}
