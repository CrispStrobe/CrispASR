// crispasr_backend_hojo_asr.cpp — adapter for HojoAI/Hojo-ASR-Multi-V1 (#438).
//
// Pipeline: Whisper log-mel → Qwen3-Omni audio tower (conv stem + 32 L,
// 3000-frame chunks) → 2-block WeNet Conformer adapter + ln_speech →
// [embed(<|im_start|>)] ++ speech → Qwen3-4B beam decode. ASR only.

#include "crispasr_backend.h"
#include "crispasr_backend_utils.h"
#include "whisper_params.h"
#include "core/bpe.h"

#include "hojo_asr.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

std::string decode_token(const std::string& s) {
    return core_bpe::token_bytes_to_utf8(s);
}

class HojoAsrBackend : public CrispasrBackend {
public:
    HojoAsrBackend() = default;
    ~HojoAsrBackend() override { HojoAsrBackend::shutdown(); }

    const char* name() const override { return "hojo-asr"; }

    // CAP_PUNCTUATION_NATIVE: the Qwen3-4B decoder emits punctuated,
    // sentence-cased text, so the auto FireRedPunc pass must NOT fire (it is
    // trained on Chinese + English and would stamp full-width marks into
    // German/French/Italian output — the GigaAM-v3 failure in HARD RULE 3c).
    // CAP_LANGUAGE_DETECT: the model is natively multilingual and takes no
    // language conditioning at all (there is no prompt), so running whisper-tiny
    // LID ahead of it would cost time and decide nothing.
    uint32_t capabilities() const override {
        return CAP_AUTO_DOWNLOAD | CAP_PUNCTUATION_NATIVE | CAP_BEAM_SEARCH | CAP_LANGUAGE_DETECT;
    }

    bool init(const whisper_params& p) override {
        auto cp = hojo_asr_context_default_params();
        cp.n_threads = p.n_threads;
        cp.verbosity = p.no_prints ? 0 : 1;
        cp.use_gpu = crispasr_backend_should_use_gpu(p);
        ctx_ = hojo_asr_init_from_file(p.model.c_str(), cp);
        if (!ctx_) {
            fprintf(stderr, "crispasr[hojo-asr]: failed to load model '%s'\n", p.model.c_str());
            return false;
        }
        return true;
    }

    std::vector<crispasr_segment> transcribe(const float* samples, int n_samples, int64_t t_offset_cs,
                                             const whisper_params& params) override {
        if (!ctx_)
            return {};
        // whisper_params::beam_size defaults to -1 ("not set"); 0/-1 keeps the
        // checkpoint's own generate.num_beams = 4, which is the reference recipe.
        hojo_asr_set_beam_size(ctx_, params.beam_size);
        // #292: forward --max-new-tokens only when explicit; 0 keeps the default.
        hojo_asr_set_max_new_tokens(ctx_, params.max_new_tokens_explicit ? params.max_new_tokens : 0);

        // The model takes no language conditioning — there is no prompt to put a
        // hint in. Say so instead of silently ignoring -l.
        if (!params.language.empty() && params.language != "auto" && !warned_lang_) {
            warned_lang_ = true;
            fprintf(stderr,
                    "crispasr[hojo-asr]: this model has no language conditioning "
                    "(inputs are BOS + speech only); language='%s' is ignored\n",
                    params.language.c_str());
        }

        char* result = hojo_asr_transcribe(ctx_, samples, n_samples);
        if (!result)
            return {};
        std::string text(result);
        free(result);

        crispasr_segment seg;
        seg.text = text;
        seg.t0 = t_offset_cs;
        const int64_t dur_cs = (int64_t)((double)n_samples / 16000.0 * 100.0);
        seg.t1 = t_offset_cs + dur_cs;
        return {seg};
    }

    void transcribe_streaming(const float* samples, int n_samples, int64_t t_offset_cs, const whisper_params& params,
                              crispasr_stream_callback on_text) override {
        if (!ctx_) {
            CrispasrBackend::transcribe_streaming(samples, n_samples, t_offset_cs, params, on_text);
            return;
        }
        hojo_asr_set_beam_size(ctx_, params.beam_size);
        hojo_asr_set_max_new_tokens(ctx_, params.max_new_tokens_explicit ? params.max_new_tokens : 0);

        std::string accumulated;
        bool first_tok = true;
        auto cb = [&](int tok_id, float /*prob*/, void* /*ud*/) {
            const char* raw = hojo_asr_token_text(ctx_, tok_id);
            if (!raw)
                return;
            std::string piece = decode_token(std::string(raw));
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
        auto cb_fn = [](int tok_id, float prob, void* ud) { (*static_cast<decltype(cb)*>(ud))(tok_id, prob, nullptr); };
        hojo_asr_transcribe_cb(ctx_, samples, n_samples, cb_fn, &cb);
        on_text(accumulated.c_str(), true);
    }

    void shutdown() override {
        if (ctx_) {
            hojo_asr_free(ctx_);
            ctx_ = nullptr;
        }
    }

private:
    hojo_asr_context* ctx_ = nullptr;
    bool warned_lang_ = false;
};

} // namespace

std::unique_ptr<CrispasrBackend> crispasr_make_hojo_asr_backend() {
    return std::unique_ptr<CrispasrBackend>(new HojoAsrBackend());
}
