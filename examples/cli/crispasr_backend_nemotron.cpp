// crispasr_backend_nemotron.cpp — adapter for nvidia/nemotron-3.5-asr-streaming-0.6b.
//
// Wraps nemotron_init_from_file + nemotron_transcribe_ex and converts the
// native nemotron_result into std::vector<crispasr_segment>.

#include "crispasr_backend.h"
#include "crispasr_backend_utils.h"
#include "whisper_params.h"

#include "nemotron.h"
#include "core/nemotron_text.h"

#include <cstdio>
#include <cstring>

namespace {

using core_nemotron_text::strip_lang_tags;

class NemotronRealtimeSession final : public CrispasrRealtimeSession {
public:
    explicit NemotronRealtimeSession(nemotron_context* ctx) : ctx_(ctx), stream_(nemotron_stream_create(ctx)) {}
    ~NemotronRealtimeSession() override { nemotron_stream_free(stream_); }

    bool append(const float* samples, int n_samples, bool flush, callback on_text) override {
        if (!stream_)
            return false;
        struct CallbackState {
            NemotronRealtimeSession* self;
            callback* fn;
        } state{this, &on_text};
        auto token_cb = [](int id, float, void* userdata) {
            auto& state = *static_cast<CallbackState*>(userdata);
            const char* raw = nemotron_token_to_str(state.self->ctx_, id);
            if (!raw || !*raw)
                return;
            std::string piece(raw);
            size_t pos = 0;
            while ((pos = piece.find("\xe2\x96\x81", pos)) != std::string::npos) {
                piece.replace(pos, 3, " ");
                pos++;
            }
            if (state.self->first_token_) {
                const size_t first = piece.find_first_not_of(" \n");
                piece = first == std::string::npos ? std::string{} : piece.substr(first);
                if (!piece.empty())
                    state.self->first_token_ = false;
            }
            state.self->text_ += piece;
            const std::string clean = strip_lang_tags(state.self->text_);
            if (!clean.empty())
                (*state.fn)(clean, false);
        };
        if (!nemotron_stream_append(stream_, samples, n_samples, flush, token_cb, &state))
            return false;
        if (flush)
            on_text(strip_lang_tags(text_), true);
        return true;
    }

    void reset() override {
        nemotron_stream_reset(stream_);
        text_.clear();
        first_token_ = true;
    }

private:
    nemotron_context* ctx_ = nullptr;
    nemotron_stream* stream_ = nullptr;
    std::string text_;
    bool first_token_ = true;
};

class NemotronBackend : public CrispasrBackend {
public:
    NemotronBackend() = default;
    ~NemotronBackend() override { NemotronBackend::shutdown(); }

    const char* name() const override { return "nemotron"; }

    uint32_t capabilities() const override {
        return CAP_TIMESTAMPS_NATIVE | CAP_WORD_TIMESTAMPS | CAP_TOKEN_CONFIDENCE | CAP_FLASH_ATTN |
               CAP_PUNCTUATION_NATIVE | CAP_TEMPERATURE | CAP_BEAM_SEARCH | CAP_DIARIZE | CAP_AUTO_DOWNLOAD |
               CAP_UNBOUNDED_INPUT | CAP_STREAMING;
    }

    bool init(const whisper_params& p) override {
        nemotron_context_params cp = nemotron_context_default_params();
        cp.n_threads = p.n_threads;
        cp.use_flash = p.flash_attn;
        cp.verbosity = p.no_prints ? 0 : 1;
        cp.use_gpu = crispasr_backend_should_use_gpu(p);

        ctx_ = nemotron_init_from_file(p.model.c_str(), cp);
        if (!ctx_) {
            fprintf(stderr, "crispasr[nemotron]: failed to load model '%s'\n", p.model.c_str());
            return false;
        }

        // Set language if specified
        if (!p.language.empty()) {
            nemotron_set_language(ctx_, p.language.c_str());
        }

        return true;
    }

    void warmup() override {
        if (!ctx_)
            return;
        std::vector<float> silence(8000, 0.0f);
        nemotron_result* r = nemotron_transcribe_ex(ctx_, silence.data(), (int)silence.size(), 0);
        if (r)
            nemotron_result_free(r);
    }

    std::vector<crispasr_segment> transcribe(const float* samples, int n_samples, int64_t t_offset_cs,
                                             const whisper_params& params) override {
        std::vector<crispasr_segment> out;
        if (!ctx_)
            return out;

        if (!params.language.empty())
            nemotron_set_language(ctx_, params.language.c_str());

        nemotron_set_temperature(ctx_, params.temperature, params.seed);
        nemotron_set_beam_size(ctx_, params.beam_size > 0 ? params.beam_size : 1);

        // MAES beam search (env: CRISPASR_NEMOTRON_MAES=1).
        // Requires beam_size > 1. Configurable via env vars.
        {
            const char* maes_env = std::getenv("CRISPASR_NEMOTRON_MAES");
            bool use_maes = (maes_env && atoi(maes_env) > 0);
            if (use_maes && params.beam_size > 1) {
                int num_steps = 2;
                float gamma = 2.3f;
                int beta = 2;
                if (const char* v = std::getenv("CRISPASR_MAES_NUM_STEPS"))
                    num_steps = atoi(v);
                if (const char* v = std::getenv("CRISPASR_MAES_GAMMA"))
                    gamma = (float)atof(v);
                if (const char* v = std::getenv("CRISPASR_MAES_BETA"))
                    beta = atoi(v);
                nemotron_set_maes(ctx_, true, num_steps, gamma, beta);
            } else {
                nemotron_set_maes(ctx_, false, 0, 0.0f, 0);
            }
        }

        nemotron_result* r = nemotron_transcribe_ex(ctx_, samples, n_samples, t_offset_cs);
        if (!r)
            return out;

        crispasr_segment seg;
        seg.t0 = t_offset_cs;
        seg.t1 = t_offset_cs;
        seg.text = strip_lang_tags(r->text ? r->text : "");

        // Words
        seg.words.reserve(r->n_words);
        for (int i = 0; i < r->n_words; i++) {
            const auto& w = r->words[i];
            crispasr_word cw;
            cw.text = strip_lang_tags(w.text);
            if (cw.text.empty())
                continue; // the tag was a "word" of its own
            cw.t0 = w.t0;
            cw.t1 = w.t1;
            seg.words.push_back(std::move(cw));
        }

        // Tokens
        seg.tokens.reserve(r->n_tokens);
        for (int i = 0; i < r->n_tokens; i++) {
            const auto& t = r->tokens[i];
            crispasr_token ct;
            ct.text = t.text;
            ct.id = t.id;
            ct.t0 = t.t0;
            ct.t1 = t.t1;
            ct.confidence = t.p;
            seg.tokens.push_back(std::move(ct));
        }

        if (!seg.words.empty()) {
            seg.t0 = seg.words.front().t0;
            seg.t1 = seg.words.back().t1;
        } else if (!seg.tokens.empty()) {
            seg.t0 = seg.tokens.front().t0;
            seg.t1 = seg.tokens.back().t1;
        }

        nemotron_result_free(r);
        out.push_back(std::move(seg));
        return out;
    }

    void transcribe_streaming(const float* samples, int n_samples, int64_t /*t_offset_cs*/,
                              const whisper_params& params, crispasr_stream_callback on_text) override {
        if (!ctx_) {
            CrispasrBackend::transcribe_streaming(samples, n_samples, 0, params, on_text);
            return;
        }
        if (!params.language.empty())
            nemotron_set_language(ctx_, params.language.c_str());
        std::string accumulated;
        bool first_tok = true;
        // nemotron_transcribe_cb fires per-emitted non-blank RNN-T token
        auto cb = [&](int tok_id, float /*prob*/, void* /*ud*/) {
            const char* raw = nemotron_token_to_str(ctx_, tok_id);
            if (!raw || !*raw)
                return;
            std::string piece(raw);
            // SentencePiece ▁ (0xE2 0x96 0x81) → space
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
            const std::string clean = strip_lang_tags(accumulated);
            if (!clean.empty())
                on_text(clean, false);
        };
        auto cb_fn = [](int id, float p, void* ud) { (*static_cast<decltype(cb)*>(ud))(id, p, nullptr); };
        nemotron_transcribe_cb(ctx_, samples, n_samples, cb_fn, &cb);
        on_text(strip_lang_tags(accumulated), true);
    }

    std::unique_ptr<CrispasrRealtimeSession> create_realtime_session(const whisper_params& params) override {
        if (!ctx_)
            return nullptr;
        if (!params.language.empty())
            nemotron_set_language(ctx_, params.language.c_str());
        return std::make_unique<NemotronRealtimeSession>(ctx_);
    }

    void shutdown() override {
        if (ctx_) {
            nemotron_free(ctx_);
            ctx_ = nullptr;
        }
    }

private:
    nemotron_context* ctx_ = nullptr;
};

} // namespace

std::unique_ptr<CrispasrBackend> crispasr_make_nemotron_backend() {
    return std::unique_ptr<CrispasrBackend>(new NemotronBackend());
}
