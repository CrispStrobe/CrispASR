// crispasr_backend_t5.cpp — adapter for T5-family encoder-decoder
// translation models (MADLAD-400 and friends).
//
// Today's covered model: google/madlad400-3b-mt — a T5-style
// encoder-decoder with bucketed relative-position bias, gated-GELU
// FFN, RMSNorm, and a 256K SentencePiece vocab over 419 languages.
// Output is canonically correct: tokens match Python SP bit-by-bit
// (see `tokenize_sp` in src/t5_translate.cpp for the Viterbi
// implementation) and translations match the HF reference on
// flan-t5-small + produce sensible German/etc. on MADLAD itself.
//
// User-facing surface mirrors the m2m100 adapter: `--text "..." -sl
// <src> -tl <tgt>` drives a single translation call. T5 has no
// separate source-language tag (the encoder is language-agnostic);
// the target language is encoded as a "<2xx>" prefix on the input
// per the MADLAD-400 convention. We synthesise that prefix here so
// the same -sl/-tl interface works across m2m100 and madlad.

#include "crispasr_backend.h"
#include "crispasr_backend_utils.h"
#include "whisper_params.h"

#include "t5_translate.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

class T5Backend : public CrispasrBackend {
public:
    T5Backend() = default;
    ~T5Backend() override { T5Backend::shutdown(); }

    const char* name() const override { return "madlad"; }

    uint32_t capabilities() const override {
        // MADLAD-400 takes a target-language prefix on the input;
        // -sl is informational (T5 encoders are language-agnostic).
        // CAP_SRC_TGT_LANGUAGE suppresses the warn_unsupported nag
        // that would otherwise complain about -sl/-tl on translate.
        return CAP_TRANSLATE | CAP_AUTO_DOWNLOAD | CAP_SRC_TGT_LANGUAGE;
    }

    std::vector<crispasr_segment> transcribe(const float* /*samples*/, int /*n_samples*/, int64_t /*t_offset_cs*/,
                                             const whisper_params& /*params*/) override {
        fprintf(stderr, "crispasr[madlad]: transcription is not supported — this is a translation backend\n");
        return {};
    }

    bool init(const whisper_params& p) override {
        t5_translate_context_params cp = t5_translate_context_default_params();
        cp.n_threads = p.n_threads;
        cp.verbosity = p.no_prints ? 0 : 1;
        cp.use_gpu = crispasr_backend_should_use_gpu(p);
        ctx_ = t5_translate_init_from_file(p.model.c_str(), cp);
        if (!ctx_) {
            fprintf(stderr, "crispasr[madlad]: failed to load model '%s'\n", p.model.c_str());
            return false;
        }
        return true;
    }

    std::vector<float> synthesize(const std::string& /*text*/, const whisper_params& /*params*/) override {
        return {}; // Not a TTS backend
    }

    std::string translate_text(const std::string& text, const std::string& /*src_lang*/, const std::string& tgt_lang,
                               const whisper_params& params) override {
        if (!ctx_ || text.empty() || tgt_lang.empty()) {
            return {};
        }
        // MADLAD-400 input convention: prefix with "<2xx>" where xx is
        // the target ISO language code (a single-piece token in MADLAD's
        // 256K vocab covering all 419 supported languages). T5's
        // encoder is language-agnostic so source-lang isn't needed.
        //
        // Other T5 family models (flan-t5, mT5, T5-base, …) DO NOT have
        // <2xx> entries in their vocab. Prepending it on those would
        // tokenize as [▁, <unk>] (= garbage) at the front of the
        // encoder input and corrupt the cross-attention context. So
        // we probe the vocab first; if the tag isn't a piece, we skip
        // the prefix and pass the user's raw text through.
        const std::string lang_tag = "<2" + tgt_lang + ">";
        std::string input;
        if (t5_has_token(ctx_, lang_tag.c_str())) {
            input = lang_tag + " " + text;
        } else {
            input = text;
            if (params.verbose && !params.no_prints) {
                fprintf(stderr,
                        "crispasr[madlad]: target-lang tag '%s' not in vocab — passing input through "
                        "unmodified (this T5 variant uses prompt-driven translation, not language tags).\n",
                        lang_tag.c_str());
            }
        }
        const int max_tokens = params.translate_max_tokens > 0 ? params.translate_max_tokens : 256;
        char* out = t5_translate(ctx_, input.c_str(), max_tokens);
        if (!out) {
            return {};
        }
        std::string result(out);
        free(out);
        return result;
    }

    void shutdown() override {
        if (ctx_) {
            t5_translate_free(ctx_);
            ctx_ = nullptr;
        }
    }

private:
    t5_translate_context* ctx_ = nullptr;
};

} // namespace

std::unique_ptr<CrispasrBackend> crispasr_make_t5_backend() {
    return std::unique_ptr<CrispasrBackend>(new T5Backend());
}
