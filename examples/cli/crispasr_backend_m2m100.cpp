// crispasr_backend_m2m100.cpp — adapter for M2M-100 multilingual translation.
//
// Text-to-text translation model (not ASR). Used as a post-processor
// via --translate-model, or standalone via --backend m2m100.

#include "crispasr_backend.h"
#include "crispasr_backend_utils.h"
#include "whisper_params.h"

#include "m2m100.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

class M2M100Backend : public CrispasrBackend {
public:
    M2M100Backend() = default;
    ~M2M100Backend() override { M2M100Backend::shutdown(); }

    const char* name() const override { return "m2m100"; }

    uint32_t capabilities() const override {
        // m2m100 takes a source AND target language (text-to-text); both
        // are required arguments to m2m100_translate(). CAP_SRC_TGT_LANGUAGE
        // suppresses the "backend does not support --source-lang" warning
        // that warn_unsupported() would otherwise raise when the user
        // passes -sl/-tl.
        return CAP_TRANSLATE | CAP_AUTO_DOWNLOAD | CAP_SRC_TGT_LANGUAGE | CAP_BEAM_SEARCH;
    }

    std::vector<crispasr_segment> transcribe(const float* /*samples*/, int /*n_samples*/, int64_t /*t_offset_cs*/,
                                             const whisper_params& /*params*/) override {
        fprintf(stderr, "crispasr[m2m100]: transcription is not supported — this is a translation backend\n");
        return {};
    }

    bool init(const whisper_params& p) override {
        m2m100_context_params cp = m2m100_context_default_params();
        cp.n_threads = p.n_threads;
        cp.verbosity = p.no_prints ? 0 : 1;
        ctx_ = m2m100_init_from_file(p.model.c_str(), cp);
        if (!ctx_) {
            fprintf(stderr, "crispasr[m2m100]: failed to load model '%s'\n", p.model.c_str());
            return false;
        }
        return true;
    }

    std::vector<float> synthesize(const std::string& /*text*/, const whisper_params& /*params*/) override {
        return {}; // Not a TTS backend
    }

    std::string translate_text(const std::string& text, const std::string& src_lang, const std::string& tgt_lang,
                               const whisper_params& params) override {
        if (!ctx_ || text.empty() || src_lang.empty() || tgt_lang.empty()) {
            return {};
        }
        m2m100_set_beam_size(ctx_, params.beam_size > 0 ? params.beam_size : 1);
        const int max_tokens = params.translate_max_tokens > 0 ? params.translate_max_tokens : 256;
        char* out = m2m100_translate(ctx_, text.c_str(), src_lang.c_str(), tgt_lang.c_str(), max_tokens);
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

    // Translation: called from crispasr_run.cpp when --translate-model is active
    m2m100_context* get_ctx() { return ctx_; }

private:
    m2m100_context* ctx_ = nullptr;
};

} // namespace

std::unique_ptr<CrispasrBackend> crispasr_make_m2m100_backend() {
    return std::unique_ptr<CrispasrBackend>(new M2M100Backend());
}
