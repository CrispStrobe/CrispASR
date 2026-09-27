// crispasr_backend_omniasr.cpp — OmniASR backend adapter (CTC + LLM).

#include "crispasr_backend.h"
#include "crispasr_backend_utils.h"
#include "crispasr_ctc_stitch.h"
#include "omniasr.h"
#include "whisper_params.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include "core/crispasr_env.h"

// CTC split window (seconds). Measured 2026-09-27: the official fairseq2
// pipeline and this port agree frame by frame (logits cos >= 0.99996 on
// omniASR CTC 300M-v2 / 1B-v2 at 11 s and 33 s) and BOTH degrade on long
// unsplit input - 300M-v2 already on 11 s jfk, 1B-v2 at 33 s. Across the window
// A/B (0 / 7 / 15 / 30 s x jfk, jfk x2, jfk x3, 31 s speech) 7 s was the most
// robust, so it stays the default.
static constexpr float kCtcDefaultWindowSec = 7.0f;

class OmniasrBackend : public CrispasrBackend {
public:
    OmniasrBackend() = default;
    // RAII cleanup so resources are freed on destruction even when a return path
    // skips the explicit backend->shutdown() (matches PocketTTSBackend).
    ~OmniasrBackend() override { OmniasrBackend::shutdown(); }

    const char* name() const override { return "omniasr"; }
    uint32_t capabilities() const override {
        // Beam-search applies only to the LLM variant; the matrix lists it
        // there. CTC variant ignores beam_size at the decode layer.
        // CAP_DIARIZE: framework post-step works on the segment list.
        // CAP_PUNCTUATION_TOGGLE intentionally NOT declared: omniasr's
        // CTC vocab is lowercase + unpunctuated by design (verified against
        // JFK on 2026-05-04 — output is "and so my fellow americas ask not
        // ..."), so there is nothing to toggle off. Re-add only if a
        // post-step casing/punctuation restorer is wired in.
        return CAP_TOKEN_CONFIDENCE | CAP_TEMPERATURE | CAP_BEAM_SEARCH | CAP_AUTO_DOWNLOAD | CAP_TIMESTAMPS_CTC |
               CAP_FLASH_ATTN | CAP_DIARIZE;
    }

    bool init(const whisper_params& params) override {
        omniasr_context_params cp = omniasr_context_default_params();
        cp.n_threads = params.n_threads;
        cp.max_new_tokens = params.max_new_tokens > 0 ? params.max_new_tokens : cp.max_new_tokens;
        cp.verbosity = params.no_prints ? 0 : 1;
        cp.temperature = params.temperature;
        cp.beam_size = params.beam_size > 0 ? params.beam_size : 1;
        cp.use_gpu = crispasr_backend_should_use_gpu(params);
        if (crispasr_env::get("CRISPASR_OMNIASR_DEBUG"))
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
        last_logits_ = {};
        if (!ctx_)
            return out;

        // CTC variant: split long input into overlapping windows, stitch the
        // per-window logit grids inside each overlap (at a frame both windows
        // call blank, else the midpoint - crispasr_ctc_stitch.h), and CTC-decode the
        // stitched grid ONCE. The old loop concatenated each window's TEXT —
        // overlap words came out twice, and fixed 5 s steps left a tiny tail
        // window (jfk 11 s -> 0-5, 4.5-9.5, 9-11 s) that decodes as garbage
        // ("...Ask what you can do. كan do fo يu اoutي." on every device/quant;
        // the regression's 0.25 WER gate was absorbing it). Windows are now
        // equal-sized, so none is much shorter than the rest.
        // CRISPASR_OMNIASR_CTC_CHUNK_SEC sets the window (0 = never split).
        // The LLM variant handles its own segmentation internally.
        const bool is_ctc = omniasr_is_ctc(ctx_);
        constexpr int SR = 16000;
        float win_sec = kCtcDefaultWindowSec;
        if (const char* e = crispasr_env::get("CRISPASR_OMNIASR_CTC_CHUNK_SEC"))
            win_sec = (float)atof(e);
        const int win = (int)(win_sec * SR);
        if (is_ctc && win > 0 && n_samples > win) {
            const int overlap = std::min(SR, win / 4);
            const int n_chunks = (n_samples - overlap + (win - overlap) - 1) / (win - overlap);
            const int step = (n_samples - overlap + n_chunks - 1) / n_chunks; // equal windows of step+overlap
            std::vector<crispasr_ctc_stitch::Window> wins;
            int V = 0;
            for (int c = 0; c < n_chunks; c++) {
                const int off = c * step;
                const int end = (c == n_chunks - 1) ? n_samples : std::min(n_samples, off + step + overlap);
                float* lg = nullptr;
                int Vc = 0, T = 0;
                char* t = omniasr_transcribe_with_logits(ctx_, samples + off, end - off, &lg, &Vc, &T);
                free(t);
                if (lg && Vc > 0 && T > 0 && (V == 0 || Vc == V)) {
                    V = Vc;
                    crispasr_ctc_stitch::Window w;
                    w.off = off;
                    w.end = end;
                    w.T = T;
                    w.lg.assign(lg, lg + (size_t)T * Vc);
                    wins.push_back(std::move(w));
                }
                free(lg);
            }
            crispasr_ctc_logits stitched;
            stitched.n_vocab = V;
            stitched.data = crispasr_ctc_stitch::stitch(wins, V, /*blank_id=*/0, &stitched.n_frames);
            if (stitched.n_frames == 0)
                return out;
            crispasr_segment seg;
            seg.t0 = t_offset_cs;
            seg.t1 = t_offset_cs + (int64_t)(n_samples * 100 / SR);
            if (char* text =
                    omniasr_ctc_decode_logits(ctx_, stitched.data.data(), stitched.n_vocab, stitched.n_frames)) {
                seg.text = text;
                free(text);
            }
            if (params.return_logits) {
                stitched.normalization = "logits";
                stitched.vocab.reserve((size_t)stitched.n_vocab);
                for (int i = 0; i < stitched.n_vocab; i++)
                    stitched.vocab.emplace_back(omniasr_token_text(ctx_, i));
                last_logits_ = std::move(stitched);
            }
            if (!seg.text.empty())
                out.push_back(std::move(seg));
            return out;
        }

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
        seg.t1 = t_offset_cs + (int64_t)(n_samples * 100 / SR);
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
            // CTC variant: text only, with optional raw logits.
            float* logits = nullptr;
            int n_vocab = 0;
            int n_frames = 0;
            char* text = params.return_logits
                             ? omniasr_transcribe_with_logits(ctx_, samples, n_samples, &logits, &n_vocab, &n_frames)
                             : omniasr_transcribe(ctx_, samples, n_samples);
            if (!text)
                return out;
            seg.text = text;
            free(text);
            if (params.return_logits && logits && n_vocab > 0 && n_frames > 0) {
                last_logits_.n_frames = n_frames;
                last_logits_.n_vocab = n_vocab;
                last_logits_.data.assign(logits, logits + (size_t)n_frames * n_vocab);
                last_logits_.normalization = "logits";
                last_logits_.vocab.reserve((size_t)n_vocab);
                for (int i = 0; i < n_vocab; i++) {
                    const char* piece = omniasr_token_text(ctx_, i);
                    last_logits_.vocab.emplace_back(piece ? piece : "");
                }
            }
            free(logits);
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

    const crispasr_ctc_logits* last_ctc_logits() const override {
        return last_logits_.data.empty() ? nullptr : &last_logits_;
    }

private:
    omniasr_context* ctx_ = nullptr;
    std::string lang_str_;
    crispasr_ctc_logits last_logits_;
};

std::unique_ptr<CrispasrBackend> crispasr_make_omniasr_backend() {
    return std::make_unique<OmniasrBackend>();
}
