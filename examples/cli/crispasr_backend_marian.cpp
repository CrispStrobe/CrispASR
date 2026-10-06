// crispasr_backend_marian.cpp — adapter for MarianMT / Opus-MT translation.
//
// Text-to-text translation (not ASR): Helsinki-NLP/opus-mt-de-en and
// opus-mt-en-de, ~75M parameters each. Used standalone via `--backend marian
// --text … -sl de -tl en`, or behind a streaming recogniser via
// `--translate-model <marian.gguf>` (docs/streaming.md).
//
// The runtime is m2m100.cpp — Marian is the same encoder-decoder skeleton and
// its differences are GGUF-driven branches there. This file exists so the
// backend has its own name, its own default beam width and a load check that
// refuses a non-Marian GGUF.

#include "crispasr_backend.h"
#include "crispasr_backend_utils.h"
#include "whisper_params.h"

#include "m2m100.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {

class MarianBackend : public CrispasrBackend {
public:
    MarianBackend() = default;
    ~MarianBackend() override { MarianBackend::shutdown(); }

    const char* name() const override { return "marian"; }

    uint32_t capabilities() const override {
        // A single-pair model has its direction baked in, but -sl / -tl are
        // still how every translation backend is addressed (and what selects
        // the `>>xx<<` token of a multi-target checkpoint), so they must not
        // be reported as unsupported. No CAP_AUTO_DOWNLOAD: there is no
        // registry entry — pass the GGUF with -m.
        return CAP_TRANSLATE | CAP_SRC_TGT_LANGUAGE | CAP_BEAM_SEARCH;
    }

    std::vector<crispasr_segment> transcribe(const float* /*samples*/, int /*n_samples*/, int64_t /*t_offset_cs*/,
                                             const whisper_params& /*params*/) override {
        fprintf(stderr, "crispasr[marian]: transcription is not supported — this is a translation backend\n");
        return {};
    }

    bool init(const whisper_params& p) override {
        m2m100_context_params cp = m2m100_context_default_params();
        cp.n_threads = p.n_threads;
        cp.use_gpu = p.use_gpu;
        cp.verbosity = p.no_prints ? 0 : 1;
        ctx_ = m2m100_init_from_file(p.model.c_str(), cp);
        if (!ctx_) {
            fprintf(stderr, "crispasr[marian]: failed to load model '%s'\n", p.model.c_str());
            return false;
        }
        if (!m2m100_is_marian(ctx_)) {
            fprintf(stderr, "crispasr[marian]: '%s' is an M2M-100 GGUF — use --backend m2m100\n", p.model.c_str());
            shutdown();
            return false;
        }
        return true;
    }

    std::string translate_text(const std::string& text, const std::string& src_lang, const std::string& tgt_lang,
                               const whisper_params& params) override {
        return translate_text_prefixed(text, src_lang, tgt_lang, std::string(), params);
    }

    std::string translate_text_prefixed(const std::string& text, const std::string& src_lang,
                                        const std::string& tgt_lang, const std::string& target_prefix,
                                        const whisper_params& params) override {
        if (!ctx_ || text.empty()) {
            return {};
        }
        // Unset → the checkpoint's generation_config num_beams (4 for Opus-MT),
        // as m2m100 does (#439). `--beam-size 1` is greedy; live translation
        // passes 1 by default (--translate-beam).
        m2m100_set_beam_size(ctx_, params.beam_size > 0 ? params.beam_size : m2m100_model_beam_size(ctx_));
        // 0 = the runtime's default: the checkpoint's max_length.
        const int max_tokens = params.translate_max_tokens > 0 ? params.translate_max_tokens : 0;
        char* out = m2m100_translate_prefixed(ctx_, text.c_str(), src_lang.c_str(), tgt_lang.c_str(), max_tokens,
                                              target_prefix.c_str());
        if (!out) {
            return {};
        }
        std::string result(out);
        free(out);
        return result;
    }

    void shutdown() override {
        if (ctx_) {
            m2m100_free(ctx_);
            ctx_ = nullptr;
        }
    }

private:
    m2m100_context* ctx_ = nullptr;
};

} // namespace

std::unique_ptr<CrispasrBackend> crispasr_make_marian_backend() {
    return std::unique_ptr<CrispasrBackend>(new MarianBackend());
}
