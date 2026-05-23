// crispasr_backend_parakeet.cpp — adapter for nvidia/parakeet-tdt-0.6b-v3.
//
// Wraps parakeet_init_from_file + parakeet_transcribe_ex and converts the
// native parakeet_result into a std::vector<crispasr_segment>. One segment
// per transcribe() call, with word-level data attached (parakeet emits word
// timestamps for free via its TDT duration head).

#include "crispasr_backend.h"
#include "crispasr_backend_utils.h"
#include "whisper_params.h"
#include "core/asr_context_bias.h"

#include "parakeet.h"

#include <cstdio>
#include <cstring>

namespace {

class ParakeetBackend : public CrispasrBackend {
public:
    ParakeetBackend() = default;
    ~ParakeetBackend() override { ParakeetBackend::shutdown(); }

    const char* name() const override { return "parakeet"; }

    uint32_t capabilities() const override {
        // CAP_LANGUAGE_DETECT intentionally NOT declared: the parakeet
        // backend has no native LID code path. Declaring the cap would
        // disable the framework's pre-step LID gate
        // (crispasr_run.cpp:`!has_native_lid`), so users wanting LID
        // get nothing. With the cap absent, `-dl` correctly routes
        // through the whisper-tiny pre-step.
        return CAP_TIMESTAMPS_NATIVE | CAP_WORD_TIMESTAMPS | CAP_TOKEN_CONFIDENCE | CAP_FLASH_ATTN |
               CAP_PUNCTUATION_TOGGLE | CAP_TEMPERATURE | CAP_DIARIZE | CAP_PARALLEL_PROCESSORS | CAP_AUTO_DOWNLOAD |
               CAP_UNBOUNDED_INPUT | CAP_INTERNAL_CHUNKING;
    }

    bool init(const whisper_params& p) override {
        parakeet_context_params cp = parakeet_context_default_params();
        cp.n_threads = p.n_threads;
        cp.use_flash = p.flash_attn;
        cp.verbosity = p.no_prints ? 0 : 1;
        cp.use_gpu = crispasr_backend_should_use_gpu(p);

        ctx_ = parakeet_init_from_file(p.model.c_str(), cp);
        if (!ctx_) {
            fprintf(stderr, "crispasr[parakeet]: failed to load model '%s'\n", p.model.c_str());
            return false;
        }
        // CTC decode mode (hybrid TDT+CTC models).
        if (p.parakeet_decoder == "ctc") {
            if (parakeet_has_ctc(ctx_)) {
                parakeet_set_ctc_mode(ctx_, true);
                if (!p.no_prints)
                    fprintf(stderr, "crispasr[parakeet]: using CTC decoder\n");
            } else {
                fprintf(stderr, "crispasr[parakeet]: --parakeet-decoder ctc requested but model has no CTC head\n");
            }
        }
        return true;
    }

    std::vector<crispasr_segment> transcribe(const float* samples, int n_samples, int64_t t_offset_cs,
                                             const whisper_params& params) override {
        std::vector<crispasr_segment> out;
        if (!ctx_)
            return out;

        // Sticky per-call sampling state. The setter just stores the
        // value on the parakeet_context, so subsequent transcribe calls
        // re-pick it up. We zero it on the first temp==0 call so a user
        // who toggles --temperature back off doesn't keep the previous
        // sampling state from a prior file.
        parakeet_set_temperature(ctx_, params.temperature, params.seed);

        // PLAN #98: CTC-WS hotword phrase boost
        if (!params.hotwords.empty()) {
            auto hw = core_context_bias::parse_hotwords(params.hotwords);
            std::vector<const char*> ptrs;
            for (auto& s : hw)
                ptrs.push_back(s.c_str());
            parakeet_set_hotwords(ctx_, ptrs.data(), (int)ptrs.size(), params.hotwords_boost);
        }

        // Issue #89 / PLAN #104: encoding path selection.
        //
        // transcribe_ex (single-pass): best quality — bidirectional encoder
        // sees full context.  99.5% coverage on the reporter's 60 s JA clip.
        // Fails past ~60 s on some hardware (z-norm drift on Vulkan/AMD).
        //
        // transcribe_streamed (chunked encode): safe for any length — global
        // z-norm + 8 s encoder chunks + single TDT decode.  ~93% coverage
        // (encoder sees only 8 s context per chunk → slightly worse features).
        //
        // Default: single-pass up to 60 s, streamed above.  Tuneable via env:
        //   CRISPASR_PARAKEET_STREAM_THRESHOLD=0  → always streamed
        //   CRISPASR_PARAKEET_STREAM_THRESHOLD=999 → always single-pass
        //   CRISPASR_PARAKEET_STREAM_CHUNK=16      → 16 s encoder chunks
        //   CRISPASR_PARAKEET_STREAM_OVERLAP=4     → 4 s encoder overlap
        int stream_threshold_s = 60;
        int stream_chunk_s = 8;
        int stream_overlap_s = 2;
        if (const char* e = getenv("CRISPASR_PARAKEET_STREAM_THRESHOLD"))
            stream_threshold_s = std::max(0, atoi(e));
        if (const char* e = getenv("CRISPASR_PARAKEET_STREAM_CHUNK"))
            stream_chunk_s = std::max(2, atoi(e));
        if (const char* e = getenv("CRISPASR_PARAKEET_STREAM_OVERLAP"))
            stream_overlap_s = std::max(0, atoi(e));

        parakeet_result* r;
        if (stream_threshold_s > 0 && n_samples > stream_threshold_s * 16000) {
            r = parakeet_transcribe_streamed(ctx_, samples, n_samples, t_offset_cs, stream_chunk_s, stream_overlap_s);
        } else {
            r = parakeet_transcribe_ex(ctx_, samples, n_samples, t_offset_cs);
        }
        if (!r)
            return out;

        crispasr_segment seg;
        seg.t0 = t_offset_cs;
        seg.t1 = t_offset_cs;
        seg.text = r->text ? r->text : "";

        // Words
        seg.words.reserve(r->n_words);
        for (int i = 0; i < r->n_words; i++) {
            const auto& w = r->words[i];
            crispasr_word cw;
            cw.text = w.text;
            cw.t0 = w.t0;
            cw.t1 = w.t1;
            seg.words.push_back(std::move(cw));
        }

        // Tokens (sub-word pieces with their own timing + softmax confidence)
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
    parakeet_context* ctx_ = nullptr;
};

} // namespace

std::unique_ptr<CrispasrBackend> crispasr_make_parakeet_backend() {
    return std::unique_ptr<CrispasrBackend>(new ParakeetBackend());
}
