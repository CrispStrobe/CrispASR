// crispasr_backend_canary.cpp — adapter for nvidia/canary-1b-v2.
//
// Wraps canary_init_from_file + canary_transcribe_ex. Canary supports
// explicit source/target language pairs (for speech translation) and a
// punctuation toggle, so this backend reads params.source_lang,
// params.target_lang, and params.punctuation from whisper_params.
//
// When source_lang is empty it defaults to params.language. When target_lang
// is empty it defaults to source_lang (ASR rather than translation). When
// params.translate is true and target_lang is unset, target_lang is forced
// to "en" — matching the semantics of whisper's --translate flag.

#include "crispasr_backend.h"
#include "crispasr_backend_utils.h"
#include "whisper_params.h"

#include "canary.h"

#include <cstdio>
#include <cstring>
#include <string>

namespace {

class CanaryBackend : public CrispasrBackend {
public:
    CanaryBackend() = default;
    ~CanaryBackend() override { CanaryBackend::shutdown(); }

    const char* name() const override { return "canary"; }

    uint32_t capabilities() const override {
        // CAP_INTERNAL_CHUNKING: same FastConformer encoder as parakeet —
        // the 30 s auto-chunk causes z-norm drift and content loss.  Let
        // the backend handle full audio in a single encoder pass (safe up
        // to ~60 s; for longer audio, parakeet_transcribe_streamed-style
        // chunking can be added later if needed).  Issue #89 follow-up.
        return CAP_TIMESTAMPS_NATIVE | CAP_TIMESTAMPS_CTC | CAP_WORD_TIMESTAMPS | CAP_TOKEN_CONFIDENCE | CAP_TRANSLATE |
               CAP_SRC_TGT_LANGUAGE | CAP_PUNCTUATION_TOGGLE | CAP_FLASH_ATTN | CAP_TEMPERATURE | CAP_DIARIZE |
               CAP_PARALLEL_PROCESSORS | CAP_AUTO_DOWNLOAD | CAP_UNBOUNDED_INPUT | CAP_INTERNAL_CHUNKING;
    }

    bool init(const whisper_params& p) override {
        canary_context_params cp = canary_context_default_params();
        cp.n_threads = p.n_threads;
        cp.use_flash = p.flash_attn;
        cp.verbosity = p.no_prints ? 0 : 1;
        cp.use_gpu = crispasr_backend_should_use_gpu(p);

        ctx_ = canary_init_from_file(p.model.c_str(), cp);
        if (!ctx_) {
            fprintf(stderr, "crispasr[canary]: failed to load model '%s'\n", p.model.c_str());
            return false;
        }
        return true;
    }

    void warmup() override {
        if (!ctx_)
            return;
        std::vector<float> silence(8000, 0.0f);
        canary_result* r = canary_transcribe_ex(ctx_, silence.data(), (int)silence.size(), "en", "en", true, 0);
        if (r)
            canary_result_free(r);
    }

    std::vector<crispasr_segment> transcribe(const float* samples, int n_samples, int64_t t_offset_cs,
                                             const whisper_params& params) override {
        std::vector<crispasr_segment> out;
        if (!ctx_)
            return out;

        // Sticky decode-time sampling controls.
        canary_set_temperature(ctx_, params.temperature, params.seed);

        // Resolve src/tgt language with the fallback chain:
        //   source_lang -> language
        //   target_lang -> source_lang (ASR) or "en" (--translate)
        // Canary's prompt embeds the language token LITERALLY (e.g.
        // "<|en|>"); it has no "auto" token. If the dispatcher's LID
        // step failed and left params.language="auto", or the caller
        // never set one, fall back to "en" with a stderr note. This is
        // a defensive layer — the dispatcher should already have
        // resolved this — but we keep it so canary degrades gracefully
        // even if a future code path skips LID resolution.
        std::string src = params.source_lang.empty() ? params.language : params.source_lang;
        if (src == "auto" || src.empty()) {
            if (!params.no_prints) {
                fprintf(stderr,
                        "canary: no source language set (got '%s'); "
                        "defaulting to 'en'. Pass `-l <lang>` or "
                        "`--source-lang <lang>` to set explicitly.\n",
                        src.c_str());
            }
            src = "en";
        }
        std::string tgt = params.target_lang;
        if (tgt.empty() || tgt == "auto") {
            tgt = params.translate ? std::string("en") : src;
        }

        canary_result* r =
            canary_transcribe_ex(ctx_, samples, n_samples, src.c_str(), tgt.c_str(), params.punctuation, t_offset_cs);
        if (!r)
            return out;

        crispasr_segment seg;
        seg.t0 = t_offset_cs;
        seg.t1 = t_offset_cs;
        seg.text = r->text ? r->text : "";

        seg.words.reserve(r->n_words);
        for (int i = 0; i < r->n_words; i++) {
            const auto& w = r->words[i];
            crispasr_word cw;
            cw.text = w.text;
            cw.t0 = w.t0;
            cw.t1 = w.t1;
            seg.words.push_back(std::move(cw));
        }

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

        canary_result_free(r);
        out.push_back(std::move(seg));
        return out;
    }

    void shutdown() override {
        if (ctx_) {
            canary_free(ctx_);
            ctx_ = nullptr;
        }
    }

private:
    canary_context* ctx_ = nullptr;
};

} // namespace

std::unique_ptr<CrispasrBackend> crispasr_make_canary_backend() {
    return std::unique_ptr<CrispasrBackend>(new CanaryBackend());
}
