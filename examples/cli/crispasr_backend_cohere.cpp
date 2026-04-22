// crispasr_backend_cohere.cpp — adapter for Cohere Transcribe.
//
// Wraps cohere_init_from_file + cohere_transcribe_ex. Cohere returns
// per-token confidence and linearly-interpolated timestamps but no word
// grouping, so we emit one segment per transcribe() call with tokens
// attached.
//
// Cohere's punctuation and diarization toggles are set on the context
// params at init() time, not per call, so this backend reads them from
// whisper_params once during init.

#include "crispasr_backend.h"
#include "whisper_params.h"

#include "cohere.h"

#include <cstdio>
#include <cstring>

namespace {

class CohereBackend : public CrispasrBackend {
public:
    CohereBackend() = default;
    ~CohereBackend() override { CohereBackend::shutdown(); }

    const char* name() const override { return "cohere"; }

    uint32_t capabilities() const override {
        return CAP_TIMESTAMPS_NATIVE | CAP_WORD_TIMESTAMPS | CAP_TOKEN_CONFIDENCE | CAP_DIARIZE |
               CAP_PUNCTUATION_TOGGLE | CAP_FLASH_ATTN | CAP_TEMPERATURE | CAP_PARALLEL_PROCESSORS | CAP_AUTO_DOWNLOAD;
    }

    bool init(const whisper_params& p) override {
        cohere_context_params cp = cohere_context_default_params();
        cp.n_threads = p.n_threads;
        cp.use_flash = p.flash_attn;
        cp.use_gpu = p.use_gpu && p.gpu_backend != "cpu";
        cp.no_punctuation = !p.punctuation;
        cp.diarize = p.diarize;
        cp.verbosity = p.no_prints ? 0 : 1;

        ctx_ = cohere_init_from_file(p.model.c_str(), cp);
        if (!ctx_) {
            fprintf(stderr, "crispasr[cohere]: failed to load model '%s'\n", p.model.c_str());
            return false;
        }
        return true;
    }

    std::vector<crispasr_segment> transcribe(const float* samples, int n_samples, int64_t t_offset_cs,
                                             const whisper_params& params) override {
        std::vector<crispasr_segment> out;
        if (!ctx_)
            return out;

        // Sticky decode-time sampling controls.
        cohere_set_temperature(ctx_, params.temperature, /*seed=*/0);

        cohere_result* r = cohere_transcribe_ex(ctx_, samples, n_samples, params.language.c_str(), t_offset_cs);
        if (!r)
            return out;

        crispasr_segment seg;
        seg.t0 = t_offset_cs;
        seg.t1 = t_offset_cs;
        seg.text = r->text ? r->text : "";

        seg.tokens.reserve(r->n_tokens);
        for (int i = 0; i < r->n_tokens; i++) {
            const auto& t = r->tokens[i];
            crispasr_token ct;
            ct.text = t.text;
            ct.id = t.id;
            ct.confidence = t.p;
            ct.t0 = t.t0;
            ct.t1 = t.t1;
            seg.tokens.push_back(std::move(ct));
        }

        if (!seg.tokens.empty()) {
            seg.t0 = seg.tokens.front().t0;
            seg.t1 = seg.tokens.back().t1;
        }

        // Synthesize word-level timestamps by grouping adjacent tokens on
        // leading-space boundaries. Cohere's tokenizer already converts the
        // SentencePiece '▁' marker into a literal space in token.text, so a
        // token that starts with ' ' is the first sub-word of a new word.
        {
            crispasr_word w;
            bool have = false;
            for (const auto& t : seg.tokens) {
                const bool starts_word = !t.text.empty() && t.text[0] == ' ';
                if (starts_word && have) {
                    seg.words.push_back(std::move(w));
                    w = {};
                    have = false;
                }
                if (!have) {
                    w.t0 = t.t0;
                    have = true;
                }
                w.text += t.text;
                w.t1 = t.t1;
            }
            if (have)
                seg.words.push_back(std::move(w));
            // Trim leading whitespace off each word text (cosmetic — the
            // space sat there as the word-start marker).
            for (auto& word : seg.words) {
                while (!word.text.empty() && word.text.front() == ' ')
                    word.text.erase(word.text.begin());
            }
        }

        cohere_result_free(r);
        out.push_back(std::move(seg));
        return out;
    }

    void shutdown() override {
        if (ctx_) {
            cohere_free(ctx_);
            ctx_ = nullptr;
        }
    }

private:
    cohere_context* ctx_ = nullptr;
};

} // namespace

std::unique_ptr<CrispasrBackend> crispasr_make_cohere_backend() {
    return std::unique_ptr<CrispasrBackend>(new CohereBackend());
}
