// crispasr_backend_cohere.cpp — adapter for Cohere Transcribe.
//
// Wraps cohere_init_from_file + cohere_transcribe_ex. Cohere returns
// per-token confidence and linearly-interpolated timestamps but no word
// grouping, so we emit one segment per transcribe() call with tokens
// attached.
//
// Cohere's punctuation toggle is set on the context params at init() time,
// not per call, so this backend reads it from whisper_params once during init.
// CLI --diarize is a generic post-processing pass; do not map it to Cohere's
// experimental <|diarize|> decode prompt because that changes ASR text.

#include "crispasr_backend.h"
#include "crispasr_backend_utils.h"
#include "whisper_params.h"
#include "core/ngram_loop_fix.h"
#include "core/crispasr_env.h"
#include "text_lid_dispatch.h"

#include "cohere.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

namespace {

// RAII wrapper turning our text LID into the `agreement` signal the probe
// scorer wants: "how confident is a TEXT detector that this probe transcript
// really is in the language it was decoded under".
//
// The reference implementation (cohereX) uses langdetect, which returns a
// distribution and reads off the candidate's probability directly. Ours
// returns a ranked top-k, so we look the candidate up in the top 5 and use its
// score — same quantity, truncated. Absent from the top 5 counts as 0.
struct TextLidHook {
    text_lid_context* lid = nullptr;

    ~TextLidHook() {
        if (lid)
            text_lid_free(lid);
    }

    bool open(const whisper_params& p) {
        // Reuse whatever the user already asked for on --lid-on-transcript, so
        // we don't fetch a second detector; "auto" pulls the small cld3 GGUF.
        const std::string arg = p.lid_on_transcript.empty() ? std::string("auto") : p.lid_on_transcript;
        const std::string path = text_lid_resolve_path(arg, p.cache_dir, /*quiet=*/true);
        if (path.empty())
            return false;
        lid = text_lid_init_from_file(path.c_str(), p.n_threads);
        return lid != nullptr;
    }

    static float agreement(const char* text, const char* lang, void* user) {
        auto* self = static_cast<TextLidHook*>(user);
        if (!self || !self->lid || !text || !*text || !lang)
            return 0.0f;
        const char* labels[5] = {nullptr};
        float scores[5] = {0.0f};
        const int n = text_lid_predict_topk(self->lid, text, 5, labels, scores);
        for (int i = 0; i < n; i++) {
            if (!labels[i])
                continue;
            // Labels may carry a region suffix ("zh-Hans"); compare the base.
            std::string label(labels[i]);
            const size_t cut = label.find_first_of("-_");
            if (cut != std::string::npos)
                label.resize(cut);
            if (label == lang)
                return scores[i];
        }
        return 0.0f;
    }
};

class CohereBackend : public CrispasrBackend {
public:
    CohereBackend() = default;
    ~CohereBackend() override { CohereBackend::shutdown(); }

    const char* name() const override { return "cohere"; }

    uint32_t capabilities() const override {
        return CAP_TIMESTAMPS_NATIVE | CAP_WORD_TIMESTAMPS | CAP_TOKEN_CONFIDENCE | CAP_DIARIZE |
               CAP_PUNCTUATION_TOGGLE | CAP_FLASH_ATTN | CAP_TEMPERATURE | CAP_BEAM_SEARCH | CAP_PARALLEL_PROCESSORS |
               CAP_AUTO_DOWNLOAD | CAP_STREAM_DELTA | CAP_STREAM_UTTERANCE;
    }

    bool init(const whisper_params& p) override {
        cohere_context_params cp = cohere_context_default_params();
        cp.n_threads = p.n_threads;
        // Cohere: cast-on-read is 13% faster than flash on chunked
        // short-form (30s auto-chunks). Flash only wins on unchunked
        // long-form (>5 min). Force flash via CRISPASR_COHERE_FLASH=1.
        // See PERFORMANCE.md §5 (PLAN #73 closeout).
        cp.use_flash = (getenv("CRISPASR_COHERE_FLASH") != nullptr);
        cp.use_gpu = crispasr_backend_should_use_gpu(p);
        cp.no_punctuation = !p.punctuation;
        cp.diarize = false;
        cp.verbosity = p.no_prints ? 0 : 1;
        if (getenv("CRISPASR_VERBOSE") || crispasr_env::get("CRISPASR_COHERE_BENCH"))
            cp.verbosity = 2;

        ctx_ = cohere_init_from_file(p.model.c_str(), cp);
        if (!ctx_) {
            fprintf(stderr, "crispasr[cohere]: failed to load model '%s'\n", p.model.c_str());
            return false;
        }
        return true;
    }

    // Probe-based LID: transcribe a short clip once per language the model
    // declares and keep the best-scoring candidate (see cohere.h /
    // core/lid_probe.h). Preferred over external LID here for a correctness
    // reason, not just to skip a download: whisper-tiny knows 99 languages and
    // Cohere Transcribe accepts 14 (the Arabic finetune, two), so an external
    // detector can hand back a language this model was never trained on.
    //
    // Automatic only for a SMALL language set, purely for COST: one encode +
    // decode per candidate is ~37 s for 14 on an M1, against ~1 s for
    // whisper-tiny. Accuracy holds at 14 — measured on the real base model,
    // jfk.wav -> en (p=0.169) and an Arabic clip -> ar (p=0.254) — so
    // --lid-backend probe is safe to force, just slower.
    // (An earlier note here claimed accuracy degraded at 14. That was an
    // artifact of forcing a 14-language list onto the TWO-language Arabic
    // finetune, which translates when asked for a language it lacks.)
    bool detect_language(const float* samples, int n_samples, const whisper_params& p, std::string& out_lang,
                         float& out_confidence) override {
        if (!ctx_ || !samples || n_samples <= 0)
            return false;

        const int n_supported = cohere_n_supported_languages(ctx_);
        if (n_supported <= 0) {
            // Pre-metadata GGUF: no list to probe over. Reconverting it also
            // restores the -l validation, so say so once.
            if (!p.no_prints)
                fprintf(stderr, "crispasr[cohere]: this GGUF carries no supported-language list "
                                "(converted before that metadata existed) — falling back to external LID, "
                                "and `-l` cannot be validated against the model\n");
            return false;
        }

        const bool forced = (p.lid_backend == "probe" || p.lid_backend == "self");
        int max_langs = 4;
        if (const char* v = crispasr_env::get("CRISPASR_COHERE_PROBE_MAX_LANGS"))
            max_langs = atoi(v);
        if (!forced && n_supported > max_langs) {
            if (!p.no_prints)
                fprintf(stderr,
                        "crispasr[cohere]: %d supported languages — skipping the self-probe "
                        "(one encode+decode each). Force it with `--lid-backend probe`.\n",
                        n_supported);
            return false;
        }

        cohere_lid_params lp = cohere_lid_default_params();
        lp.verbosity = p.no_prints ? 0 : 1;

        // Optional agreement signal: run our own text LID over each probe
        // transcript. Strongest discriminator among same-script languages;
        // scoring degrades gracefully to length+diversity without it.
        TextLidHook hook;
        const char* textlid_env = crispasr_env::get("CRISPASR_COHERE_PROBE_TEXTLID");
        const bool want_textlid = !(textlid_env && (textlid_env[0] == '0' || textlid_env[0] == 'n' ||
                                                    textlid_env[0] == 'N' || textlid_env[0] == 'f'));
        if (want_textlid && hook.open(p)) {
            lp.text_lid = &TextLidHook::agreement;
            lp.text_lid_user = &hook;
        }

        char lang_buf[16] = {0};
        float conf = 0.0f;
        if (!cohere_detect_language(ctx_, samples, n_samples, lp, lang_buf, (int)sizeof(lang_buf), &conf))
            return false;

        out_lang = lang_buf;
        out_confidence = conf;
        return true;
    }

    bool prefers_vad() const override {
        // Cohere Transcribe transcribes NON-SPEECH as fabricated text, and our
        // chunker splits long audio at RMS minima without ever DROPPING the
        // quiet parts. Measured on the Arabic q4_k-imatrix build:
        //
        //   10 s of pure digital silence  -> "And I'm going to go ahead and do that."
        //   jfk.wav + 20 s of silence     -> that same sentence appended to an
        //                                    otherwise perfect transcript
        //
        // With --vad both go away (the second returns the clean transcript, the
        // first returns nothing). VAD also fixes the OTHER failure of fixed
        // windows: on a 60 s FLEURS clip the un-VAD'd run cut mid-sentence,
        // garbled a clause ("how acidic, basic, alkaline the cabbage juice is"
        // for "...the chemical is") and DROPPED a whole sentence that the VAD
        // run recovered. So this is a content win, not only a silence guard.
        //
        // Only arms the long-audio safeguard in crispasr_run.cpp, and only when
        // the user passed no --vad / --vad-model / --chunk-seconds.
        return true;
    }

    void warmup() override {
        if (!ctx_)
            return;
        std::vector<float> silence(8000, 0.0f);
        cohere_result* r = cohere_transcribe_ex(ctx_, silence.data(), (int)silence.size(), "en", 0);
        if (r)
            cohere_result_free(r);
    }
 void set_stream_delta(int delta_new_samples) override {
        if (ctx_)
            cohere_set_stream_delta(ctx_, delta_new_samples);
    }
    void save_utterance_cross_kv(int64_t s, int64_t e, int64_t w, int sr, int64_t gen) override { if (ctx_) cohere_save_utterance_cross_kv(ctx_, s, e, w, sr, gen); }
    void restore_utterance_cross_kv(int n, int64_t gen) override { if (ctx_) cohere_restore_utterance_cross_kv(ctx_, n, gen); }
    void clear_utterance_cross_kv() override { if (ctx_) cohere_clear_utterance_cross_kv(ctx_); }

    std::vector<crispasr_segment> transcribe(const float* samples, int n_samples, int64_t t_offset_cs,
                                             const whisper_params& params) override {
        std::vector<crispasr_segment> out;
        if (!ctx_)
            return out;

        // Sticky decode-time sampling controls.
        cohere_set_temperature(ctx_, params.temperature, params.seed);
        cohere_set_beam_size(ctx_, params.beam_size > 0 ? params.beam_size : 1);
        cohere_set_max_new_tokens(ctx_, params.max_new_tokens);
        cohere_set_frequency_penalty(ctx_, params.frequency_penalty);

        cohere_result* r = cohere_transcribe_ex(ctx_, samples, n_samples, params.language.c_str(), t_offset_cs);
        if (!r)
            return out;

        crispasr_segment seg;
        seg.t0 = t_offset_cs;
        seg.t1 = t_offset_cs;
        seg.text = core_ngram::fix_loops(r->text ? r->text : "");

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

        // issue #218 follow-up: fix_loops() above cleans seg.text, but
        // seg.words was built from the raw (un-collapsed) token stream, so
        // word-level output (SRT/VTT, JSON `words`) still shows every
        // repeated word even once the flat text looks clean. Filter
        // seg.words with the SAME collapse decision, in lockstep.
        {
            std::vector<std::string> word_texts;
            word_texts.reserve(seg.words.size());
            for (const auto& w : seg.words)
                word_texts.push_back(w.text);
            const std::vector<int> keep = core_ngram::fix_loops_keep_indices(word_texts);
            if (keep.size() != seg.words.size()) {
                std::vector<crispasr_word> filtered;
                filtered.reserve(keep.size());
                for (int idx : keep)
                    filtered.push_back(std::move(seg.words[idx]));
                seg.words = std::move(filtered);
            }
        }

        // Apply the exact same logic to seg.tokens for full JSON output parity.
        {
            std::vector<std::string> token_texts;
            token_texts.reserve(seg.tokens.size());
            for (const auto& t : seg.tokens)
                token_texts.push_back(t.text);
            const std::vector<int> keep = core_ngram::fix_loops_keep_indices(token_texts);
            if (keep.size() != seg.tokens.size()) {
                std::vector<crispasr_token> filtered;
                filtered.reserve(keep.size());
                for (int idx : keep)
                    filtered.push_back(std::move(seg.tokens[idx]));
                seg.tokens = std::move(filtered);
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
