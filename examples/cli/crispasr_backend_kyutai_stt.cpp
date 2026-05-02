// crispasr_backend_kyutai_stt.cpp — Kyutai STT backend adapter.

#include "crispasr_backend.h"
#include "crispasr_backend_utils.h"
#include "kyutai_stt.h"
#include "whisper_params.h"

#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

class KyutaiSttBackend : public CrispasrBackend {
public:
    KyutaiSttBackend() = default;

    const char* name() const override { return "kyutai-stt"; }

    uint32_t capabilities() const override {
        // PLAN #61c: kyutai's "delayed-streams" architecture aligns each
        // emitted text token to its source audio frame at zero extra cost,
        // so both segment and word timestamps are native (no DTW or CTC
        // aligner needed).
        return CAP_AUTO_DOWNLOAD | CAP_TOKEN_CONFIDENCE | CAP_TEMPERATURE | CAP_TIMESTAMPS_NATIVE |
               CAP_WORD_TIMESTAMPS | CAP_PUNCTUATION_TOGGLE;
    }

    bool init(const whisper_params& params) override {
        kyutai_stt_context_params cp = kyutai_stt_context_default_params();
        cp.n_threads = params.n_threads;
        cp.verbosity = params.no_prints ? 0 : 1;
        cp.use_gpu = crispasr_backend_should_use_gpu(params);
        cp.temperature = params.temperature;
        ctx_ = kyutai_stt_init_from_file(params.model.c_str(), cp);
        return ctx_ != nullptr;
    }

    std::vector<crispasr_segment> transcribe(const float* samples, int n_samples, int64_t t_offset_cs,
                                             const whisper_params& params) override {
        (void)params;
        std::vector<crispasr_segment> out;
        if (!ctx_)
            return out;

        // PLAN #61c: use the _ex API to get per-token + word-level
        // timestamps. The kyutai LM emits one text token per Mimi frame
        // (12.5 Hz = 8 cs/frame) with the audio_delay correction baked in.
        kyutai_stt_result_ex* r = kyutai_stt_transcribe_ex(ctx_, samples, n_samples, t_offset_cs);
        if (!r || !r->text)
            return out;

        crispasr_segment seg;
        seg.text = r->text;
        while (!seg.text.empty() && (seg.text.front() == ' ' || seg.text.front() == '\n'))
            seg.text.erase(seg.text.begin());
        while (!seg.text.empty() && (seg.text.back() == ' ' || seg.text.back() == '\n'))
            seg.text.pop_back();

        // Segment span: prefer the actual word range; fall back to the
        // full audio buffer when no words emitted.
        if (r->n_words > 0) {
            seg.t0 = r->words[0].t0;
            seg.t1 = r->words[r->n_words - 1].t1;
        } else {
            seg.t0 = t_offset_cs;
            seg.t1 = t_offset_cs + (int64_t)((double)n_samples / 16000.0 * 100.0);
        }

        seg.tokens.reserve((size_t)r->n_tokens);
        for (int i = 0; i < r->n_tokens; i++) {
            const kyutai_stt_token_data& src = r->tokens[i];
            crispasr_token tok;
            tok.id = src.id;
            tok.confidence = src.p;
            tok.t0 = src.t0;
            tok.t1 = src.t1;
            tok.text = src.text;
            seg.tokens.push_back(std::move(tok));
        }

        seg.words.reserve((size_t)r->n_words);
        for (int i = 0; i < r->n_words; i++) {
            crispasr_word w;
            w.text = r->words[i].text;
            w.t0 = r->words[i].t0;
            w.t1 = r->words[i].t1;
            seg.words.push_back(std::move(w));
        }

        kyutai_stt_result_ex_free(r);

        // --no-punctuation: post-strip ASCII punctuation + lowercase. Kyutai
        // emits punctuated mixed-case English by default; the toggle gives
        // the historical CTC-style "lowercase, no punc" surface.
        if (!params.punctuation) {
            crispasr_strip_ascii_punctuation(seg.text);
            crispasr_lowercase_ascii(seg.text);
            for (auto& tok : seg.tokens) {
                crispasr_strip_ascii_punctuation(tok.text);
                crispasr_lowercase_ascii(tok.text);
            }
            for (auto& w : seg.words) {
                crispasr_strip_ascii_punctuation(w.text);
                crispasr_lowercase_ascii(w.text);
            }
        }

        if (!seg.text.empty())
            out.push_back(std::move(seg));
        return out;
    }

    void shutdown() override {
        if (ctx_) {
            kyutai_stt_free(ctx_);
            ctx_ = nullptr;
        }
    }

    ~KyutaiSttBackend() override { KyutaiSttBackend::shutdown(); }

private:
    kyutai_stt_context* ctx_ = nullptr;
};

std::unique_ptr<CrispasrBackend> crispasr_make_kyutai_stt_backend() {
    return std::make_unique<KyutaiSttBackend>();
}
