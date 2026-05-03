// crispasr_backend_moonshine.cpp — Moonshine ASR backend adapter.

#include "crispasr_backend.h"
#include "crispasr_backend_utils.h"
#include "moonshine.h"
#include "whisper_params.h"

#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

class MoonshineBackend : public CrispasrBackend {
public:
    MoonshineBackend() = default;

    const char* name() const override { return "moonshine"; }

    uint32_t capabilities() const override {
        return CAP_AUTO_DOWNLOAD | CAP_TOKEN_CONFIDENCE | CAP_TEMPERATURE | CAP_BEAM_SEARCH | CAP_PUNCTUATION_TOGGLE |
               CAP_TIMESTAMPS_CTC | CAP_FLASH_ATTN;
        // Best-of-N is implemented in transcribe() as a sequential loop over
        // _transcribe_with_probs with a sticky seed. There's no CAP_BEST_OF_N
        // bit today; the matrix tracks it via README + adapter behaviour.
    }

    bool init(const whisper_params& params) override {
        struct moonshine_init_params mp = {};
        mp.model_path = params.model.c_str();
        mp.tokenizer_path = nullptr; // auto-detect
        mp.n_threads = params.n_threads;
        ctx_ = moonshine_init_with_params(mp);
        return ctx_ != nullptr;
    }

    std::vector<crispasr_segment> transcribe(const float* samples, int n_samples, int64_t t_offset_cs,
                                             const whisper_params& params) override {
        std::vector<crispasr_segment> out;
        if (!ctx_)
            return out;

        moonshine_set_temperature(ctx_, params.temperature);
        moonshine_set_beam_size(ctx_, params.beam_size > 0 ? params.beam_size : 1);

        // Best-of-N: when temperature > 0 and best_of > 1, run N seeded
        // decodes and keep the one with highest mean per-token softmax prob.
        // Sequential — moonshine's RNG is per-context so threads are
        // independent across sessions but not across runs of the same session.
        const int n_runs = (params.temperature > 0.0f && params.best_of > 1) ? params.best_of : 1;
        moonshine_result* r = nullptr;
        double best_score = -1.0;
        for (int run = 0; run < n_runs; run++) {
            // Mix run index into the sticky seed override. Run 0 keeps
            // historical behaviour (seed_override=0 → audio-derived seed).
            moonshine_set_seed(ctx_, run == 0 ? 0 : ((uint64_t)run * 0x9E3779B97F4A7C15ULL));
            moonshine_result* cand = moonshine_transcribe_with_probs(ctx_, samples, n_samples);
            if (!cand)
                continue;
            double sum = 0.0;
            int cnt = 0;
            for (int i = 0; i < cand->n_tokens; i++) {
                sum += (double)cand->token_probs[i];
                cnt++;
            }
            double score = (cnt > 0) ? (sum / cnt) : 0.0;
            if (!r || score > best_score) {
                if (r)
                    moonshine_result_free(r);
                r = cand;
                best_score = score;
            } else {
                moonshine_result_free(cand);
            }
        }
        if (!r || !r->text || !r->text[0]) {
            if (r)
                moonshine_result_free(r);
            return out;
        }
        if (!params.no_prints && n_runs > 1)
            fprintf(stderr, "crispasr[moonshine]: best-of-%d picked score=%.4f\n", n_runs, best_score);

        crispasr_segment seg;
        seg.t0 = t_offset_cs;
        seg.t1 = t_offset_cs + (int64_t)((double)n_samples / 16000.0 * 100.0);
        seg.text = r->text;

        while (!seg.text.empty() && (seg.text.front() == ' ' || seg.text.front() == '\n'))
            seg.text.erase(seg.text.begin());
        while (!seg.text.empty() && (seg.text.back() == ' ' || seg.text.back() == '\n'))
            seg.text.pop_back();

        // Token-level confidence: one entry per emitted (non-EOS) decoder
        // step. No per-token timestamps (moonshine's decoder isn't time-aligned),
        // leave t0/t1 unset (-1).
        seg.tokens.reserve((size_t)r->n_tokens);
        for (int i = 0; i < r->n_tokens; i++) {
            crispasr_token tok;
            tok.id = r->token_ids[i];
            tok.confidence = r->token_probs[i];
            const char* piece = moonshine_token_text(ctx_, r->token_ids[i]);
            if (piece && piece[0])
                tok.text = piece;
            seg.tokens.push_back(std::move(tok));
        }

        moonshine_result_free(r);

        // --no-punctuation: post-strip ASCII punctuation + lowercase.
        if (!params.punctuation) {
            crispasr_strip_ascii_punctuation(seg.text);
            crispasr_lowercase_ascii(seg.text);
            for (auto& tok : seg.tokens) {
                crispasr_strip_ascii_punctuation(tok.text);
                crispasr_lowercase_ascii(tok.text);
            }
        }

        if (!seg.text.empty())
            out.push_back(std::move(seg));
        return out;
    }

    void shutdown() override {
        if (ctx_) {
            moonshine_free(ctx_);
            ctx_ = nullptr;
        }
    }

    ~MoonshineBackend() override { MoonshineBackend::shutdown(); }

private:
    moonshine_context* ctx_ = nullptr;
};

std::unique_ptr<CrispasrBackend> crispasr_make_moonshine_backend() {
    return std::make_unique<MoonshineBackend>();
}
