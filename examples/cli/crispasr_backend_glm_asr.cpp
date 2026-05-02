// crispasr_backend_glm_asr.cpp — GLM-ASR-Nano backend adapter.

#include "crispasr_backend.h"
#include "crispasr_backend_utils.h"
#include "glm_asr.h"
#include "whisper_params.h"

#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

class GlmAsrBackend : public CrispasrBackend {
public:
    GlmAsrBackend() = default;

    const char* name() const override { return "glm-asr"; }

    uint32_t capabilities() const override {
        return CAP_TIMESTAMPS_CTC | CAP_TEMPERATURE | CAP_LANGUAGE_DETECT | CAP_AUTO_DOWNLOAD | CAP_TOKEN_CONFIDENCE |
               CAP_PUNCTUATION_TOGGLE;
    }

    bool init(const whisper_params& params) override {
        glm_asr_context_params cp = glm_asr_context_default_params();
        cp.n_threads = params.n_threads;
        cp.verbosity = params.no_prints ? 0 : 1;
        cp.use_gpu = crispasr_backend_should_use_gpu(params);
        cp.temperature = params.temperature;
        ctx_ = glm_asr_init_from_file(params.model.c_str(), cp);
        return ctx_ != nullptr;
    }

    std::vector<crispasr_segment> transcribe(const float* samples, int n_samples, int64_t t_offset_cs,
                                             const whisper_params& params) override {
        std::vector<crispasr_segment> out;
        if (!ctx_)
            return out;

        // Best-of-N: when temperature > 0 and best_of > 1, run N seeded
        // decodes (process-global libc rand reseeded per run) and keep the
        // highest mean prob.
        const int n_runs = (params.temperature > 0.0f && params.best_of > 1) ? params.best_of : 1;
        glm_asr_result* r = nullptr;
        double best_score = -1.0;
        for (int run = 0; run < n_runs; run++) {
            if (n_runs > 1)
                glm_asr_set_seed(ctx_, (unsigned int)(run * 0x9E3779B9u + 1u));
            glm_asr_result* cand = glm_asr_transcribe_with_probs(ctx_, samples, n_samples);
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
                    glm_asr_result_free(r);
                r = cand;
                best_score = score;
            } else {
                glm_asr_result_free(cand);
            }
        }
        if (!r || !r->text)
            return out;
        if (!params.no_prints && n_runs > 1)
            fprintf(stderr, "crispasr[glm-asr]: best-of-%d picked score=%.4f\n", n_runs, best_score);

        crispasr_segment seg;
        seg.t0 = t_offset_cs;
        seg.t1 = t_offset_cs + (int64_t)((double)n_samples / 16000.0 * 100.0);
        seg.text = r->text;

        // Trim leading/trailing whitespace
        while (!seg.text.empty() && (seg.text.front() == ' ' || seg.text.front() == '\n'))
            seg.text.erase(seg.text.begin());
        while (!seg.text.empty() && (seg.text.back() == ' ' || seg.text.back() == '\n'))
            seg.text.pop_back();

        // GPT-2 byte-level BPE decoder: Ġ (U+0120, UTF-8 0xC4 0xA0) → space,
        // Ċ (U+010A, UTF-8 0xC4 0x8A) → newline. All other bytes pass through.
        auto decode_bpe_piece = [](const char* raw) -> std::string {
            std::string out;
            if (!raw)
                return out;
            for (size_t ci = 0; raw[ci] != '\0';) {
                unsigned char c = (unsigned char)raw[ci];
                if (c == 0xC4 && raw[ci + 1] != '\0') {
                    unsigned char c2 = (unsigned char)raw[ci + 1];
                    if (c2 == 0xA0) {
                        out += ' ';
                        ci += 2;
                        continue;
                    }
                    if (c2 == 0x8A) {
                        out += '\n';
                        ci += 2;
                        continue;
                    }
                }
                out += (char)c;
                ci++;
            }
            return out;
        };

        // Per-token confidence; no per-token timestamps (GLM-ASR's LLM
        // decoder isn't time-aligned).
        seg.tokens.reserve((size_t)r->n_tokens);
        for (int i = 0; i < r->n_tokens; i++) {
            crispasr_token tok;
            tok.id = r->token_ids[i];
            tok.confidence = r->token_probs[i];
            tok.text = decode_bpe_piece(glm_asr_token_text(ctx_, r->token_ids[i]));
            seg.tokens.push_back(std::move(tok));
        }
        glm_asr_result_free(r);

        // --no-punctuation: strip ASCII punctuation from segment text and per-token
        // pieces. GLM-ASR's LLM produces punctuated, capitalised English by
        // default; this matches the historical CTC-style "lowercase, no punc"
        // output users expect when they pass the toggle.
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
            glm_asr_free(ctx_);
            ctx_ = nullptr;
        }
    }

    ~GlmAsrBackend() override { GlmAsrBackend::shutdown(); }

private:
    glm_asr_context* ctx_ = nullptr;
};

std::unique_ptr<CrispasrBackend> crispasr_make_glm_asr_backend() {
    return std::make_unique<GlmAsrBackend>();
}
