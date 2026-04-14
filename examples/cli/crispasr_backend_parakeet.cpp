// crispasr_backend_parakeet.cpp — adapter for nvidia/parakeet-tdt-0.6b-v3.
//
// Wraps parakeet_init_from_file + parakeet_transcribe_ex and converts the
// native parakeet_result into a std::vector<crispasr_segment>. One segment
// per transcribe() call, with word-level data attached (parakeet emits word
// timestamps for free via its TDT duration head).

#include "crispasr_backend.h"
#include "whisper_params.h"

#include "parakeet.h"

#include <cstdio>
#include <cstring>

namespace {

class ParakeetBackend : public CrispasrBackend {
public:
    ParakeetBackend() = default;
    ~ParakeetBackend() override { ParakeetBackend::shutdown(); }

    const char * name() const override { return "parakeet"; }

    uint32_t capabilities() const override {
        return CAP_TIMESTAMPS_NATIVE
             | CAP_WORD_TIMESTAMPS
             | CAP_TOKEN_CONFIDENCE
             | CAP_FLASH_ATTN
             | CAP_LANGUAGE_DETECT
             | CAP_PUNCTUATION_TOGGLE
             | CAP_TEMPERATURE
             | CAP_DIARIZE
             | CAP_PARALLEL_PROCESSORS
             | CAP_AUTO_DOWNLOAD;
    }

    bool init(const whisper_params & p) override {
        parakeet_context_params cp = parakeet_context_default_params();
        cp.n_threads = p.n_threads;
        cp.use_flash = p.flash_attn;
        cp.verbosity = p.no_prints ? 0 : 1;

        ctx_ = parakeet_init_from_file(p.model.c_str(), cp);
        if (!ctx_) {
            fprintf(stderr, "crispasr[parakeet]: failed to load model '%s'\n",
                    p.model.c_str());
            return false;
        }
        return true;
    }

    std::vector<crispasr_segment> transcribe(
        const float * samples, int n_samples,
        int64_t t_offset_cs,
        const whisper_params & params) override
    {
        std::vector<crispasr_segment> out;
        if (!ctx_) return out;

        // Sticky per-call sampling state. The setter just stores the
        // value on the parakeet_context, so subsequent transcribe calls
        // re-pick it up. We zero it on the first temp==0 call so a user
        // who toggles --temperature back off doesn't keep the previous
        // sampling state from a prior file.
        parakeet_set_temperature(ctx_, params.temperature, /*seed=*/0);

        parakeet_result * r = parakeet_transcribe_ex(
            ctx_, samples, n_samples, t_offset_cs);
        if (!r) return out;

        crispasr_segment seg;
        seg.t0 = t_offset_cs;
        seg.t1 = t_offset_cs;
        seg.text = r->text ? r->text : "";

        // Words
        seg.words.reserve(r->n_words);
        for (int i = 0; i < r->n_words; i++) {
            const auto & w = r->words[i];
            crispasr_word cw;
            cw.text = w.text;
            cw.t0   = w.t0;
            cw.t1   = w.t1;
            seg.words.push_back(std::move(cw));
        }

        // Tokens (sub-word pieces with their own timing + softmax confidence)
        seg.tokens.reserve(r->n_tokens);
        for (int i = 0; i < r->n_tokens; i++) {
            const auto & t = r->tokens[i];
            crispasr_token ct;
            ct.text       = t.text;
            ct.id         = t.id;
            ct.t0         = t.t0;
            ct.t1         = t.t1;
            ct.confidence = t.p;
            seg.tokens.push_back(std::move(ct));
        }

        // Segment t0/t1 bracketed by first/last word when available.
        if (!seg.words.empty()) {
            seg.t0 = seg.words.front().t0;
            seg.t1 = seg.words.back().t1;
        } else if (!seg.tokens.empty()) {
            seg.t0 = seg.tokens.front().t0;
            seg.t1 = seg.tokens.back().t1;
        }

        parakeet_result_free(r);
        out.push_back(std::move(seg));
        return out;
    }

    void shutdown() override {
        if (ctx_) {
            parakeet_free(ctx_);
            ctx_ = nullptr;
        }
    }

private:
    parakeet_context * ctx_ = nullptr;
};

} // namespace

std::unique_ptr<CrispasrBackend> crispasr_make_parakeet_backend() {
    return std::unique_ptr<CrispasrBackend>(new ParakeetBackend());
}
