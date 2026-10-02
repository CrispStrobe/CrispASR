#include "crispasr_backend.h"
#include "crispasr_backend_utils.h"
#include "whisper_params.h"
#include "index_echo.h"
#include <cmath>
#include <cstdio>
#include <stdexcept>

namespace {
class IndexEchoBackend : public CrispasrBackend {
public:
    ~IndexEchoBackend() override { release_context(); }
    const char* name() const override { return "index-echo"; }
    uint32_t capabilities() const override {
        return CAP_TIMESTAMPS_NATIVE | CAP_TRANSLATE | CAP_SRC_TGT_LANGUAGE | CAP_TEMPERATURE | CAP_FLASH_ATTN |
               CAP_INTERNAL_CHUNKING | CAP_PUNCTUATION_NATIVE | CAP_LANGUAGE_DETECT | CAP_AUTO_DOWNLOAD;
    }
    bool init(const whisper_params& p) override {
        auto cp = index_echo_context_default_params();
        cp.n_threads = p.n_threads;
        cp.verbosity = p.no_prints ? 0 : 1;
        cp.use_gpu = crispasr_backend_should_use_gpu(p);
        cp.flash_attn = p.flash_attn;
        ctx_ = index_echo_init_from_file(p.model.c_str(), cp);
        if (ctx_ && !p.vad_model.empty() && !index_echo_set_vad_model(ctx_, p.vad_model.c_str())) {
            shutdown();
            return false;
        }
        return ctx_ != nullptr;
    }
    std::vector<crispasr_segment> transcribe(const float* samples, int count, int64_t offset,
                                             const whisper_params& p) override {
        if (!ctx_)
            return {};
        if (!index_echo_set_target_lang(ctx_, p.target_lang.empty() ? "en" : p.target_lang.c_str()))
            throw std::runtime_error("index-echo target language must be en, ja or es");
        index_echo_set_temperature(ctx_, p.temperature, (uint32_t)p.seed);
        index_echo_set_max_new_tokens(ctx_, p.max_new_tokens_explicit ? p.max_new_tokens : 0);
        index_echo_set_glossary(ctx_, p.prompt.c_str());
        index_echo_set_ask(ctx_, p.ask.c_str());
        if (!p.language.empty() && p.language != "auto")
            fprintf(stderr, "index-echo: source-language hints are absent from the released prompt; the model infers "
                            "the source language\n");
        auto* result = index_echo_transcribe(ctx_, samples, count);
        if (!result)
            throw std::runtime_error("index-echo inference failed");
        std::vector<crispasr_segment> out;
        for (int i = 0; i < result->n_cues; ++i) {
            const auto& cue = result->cues[i];
            crispasr_segment segment;
            segment.t0 = offset + (int64_t)std::llround(cue.start_seconds * 100);
            segment.t1 = offset + (int64_t)std::llround(cue.end_seconds * 100);
            segment.text = cue.transcript;
            if (*cue.translation)
                segment.text += std::string("\n") + cue.translation;
            out.push_back(std::move(segment));
        }
        if (result->parse_warnings)
            fprintf(stderr, "index-echo: %d malformed subtitle lines\n", result->parse_warnings);
        index_echo_result_free(result);
        return out;
    }
    void shutdown() override { release_context(); }

private:
    void release_context() {
        index_echo_free(ctx_);
        ctx_ = nullptr;
    }
    index_echo_context* ctx_ = nullptr;
};
} // namespace
std::unique_ptr<CrispasrBackend> crispasr_make_index_echo_backend() {
    return std::make_unique<IndexEchoBackend>();
}
