// crispasr_backend_omniasr.cpp — OmniASR backend adapter (CTC + LLM).

#include "crispasr_backend.h"
#include "crispasr_backend_utils.h"
#include "omniasr.h"
#include "whisper_params.h"

#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

class OmniasrBackend : public CrispasrBackend {
public:
    OmniasrBackend() = default;

    const char* name() const override { return "omniasr"; }
    uint32_t capabilities() const override { return CAP_TOKEN_CONFIDENCE | CAP_TEMPERATURE | CAP_PUNCTUATION_TOGGLE; }

    bool init(const whisper_params& params) override {
        omniasr_context_params cp = omniasr_context_default_params();
        cp.n_threads = params.n_threads;
        cp.max_new_tokens = params.max_new_tokens > 0 ? params.max_new_tokens : cp.max_new_tokens;
        cp.verbosity = params.no_prints ? 0 : 1;
        cp.temperature = params.temperature;
        cp.use_gpu = crispasr_backend_should_use_gpu(params);
        if (getenv("OMNIASR_DEBUG"))
            cp.verbosity = 2;
        // Pass language for LLM variant (e.g. "eng_Latn" from -l en)
        // The LLM model uses this for language conditioning
        if (!params.language.empty() && params.language != "auto")
            lang_str_ = params.language;
        if (!lang_str_.empty())
            cp.language = lang_str_.c_str();
        ctx_ = omniasr_init_from_file(params.model.c_str(), cp);
        return ctx_ != nullptr;
    }

    std::vector<crispasr_segment> transcribe(const float* samples, int n_samples, int64_t t_offset_cs,
                                             const whisper_params& params) override {
        std::vector<crispasr_segment> out;
        if (!ctx_)
            return out;

        // LLM variant: capture per-token confidence. CTC variant returns
        // nullptr from the with_probs path — fall back to the plain entry.
        // Best-of-N: when temperature > 0 and best_of > 1, run N seeded
        // decodes via the sticky seed override and keep the highest mean prob.
        const int n_runs = (params.temperature > 0.0f && params.best_of > 1) ? params.best_of : 1;
        omniasr_result* r = nullptr;
        double best_score = -1.0;
        for (int run = 0; run < n_runs; run++) {
            omniasr_set_seed(ctx_, run == 0 ? 0 : ((uint64_t)run * 0x9E3779B97F4A7C15ULL));
            omniasr_result* cand = omniasr_transcribe_with_probs(ctx_, samples, n_samples);
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
                    omniasr_result_free(r);
                r = cand;
                best_score = score;
            } else {
                omniasr_result_free(cand);
            }
        }
        if (!params.no_prints && n_runs > 1 && r)
            fprintf(stderr, "crispasr[omniasr]: best-of-%d picked score=%.4f\n", n_runs, best_score);
        crispasr_segment seg;
        seg.t0 = t_offset_cs;
        seg.t1 = t_offset_cs + (int64_t)(n_samples * 100 / 16000);
        if (r) {
            if (r->text)
                seg.text = r->text;
            seg.tokens.reserve((size_t)r->n_tokens);
            for (int i = 0; i < r->n_tokens; i++) {
                crispasr_token tok;
                tok.id = r->token_ids[i];
                tok.confidence = r->token_probs[i];
                const char* piece = omniasr_token_text(ctx_, r->token_ids[i]);
                if (piece && piece[0]) {
                    std::string p = piece;
                    std::string decoded;
                    for (size_t ci = 0; ci < p.size(); ci++) {
                        if ((unsigned char)p[ci] == 0xE2 && ci + 2 < p.size() && (unsigned char)p[ci + 1] == 0x96 &&
                            (unsigned char)p[ci + 2] == 0x81) {
                            decoded += ' ';
                            ci += 2;
                        } else {
                            decoded += p[ci];
                        }
                    }
                    tok.text = std::move(decoded);
                }
                seg.tokens.push_back(std::move(tok));
            }
            omniasr_result_free(r);
        } else {
            // CTC variant: text only.
            char* text = omniasr_transcribe(ctx_, samples, n_samples);
            if (!text)
                return out;
            seg.text = text;
            free(text);
        }
        // --no-punctuation: post-strip ASCII punctuation + lowercase. The
        // omniasr-llm variant emits mixed-case punctuated text; CTC variant
        // is already lowercase no-punc, so the strip is a no-op there.
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
            omniasr_free(ctx_);
            ctx_ = nullptr;
        }
    }

private:
    omniasr_context* ctx_ = nullptr;
    std::string lang_str_;
};

std::unique_ptr<CrispasrBackend> crispasr_make_omniasr_backend() {
    return std::make_unique<OmniasrBackend>();
}
