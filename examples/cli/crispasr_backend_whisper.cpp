// crispasr_backend_whisper.cpp — CrispasrBackend wrapper around whisper.cpp.
//
// This adapter exists so `--backend whisper` (and --list-backends) can run
// whisper through the same dispatch layer as every other crispasr backend.
// It is NOT used by the default cli.cpp whisper path: when no --backend flag
// is passed and the model is a legacy ggml-*.bin file, the historical cli.cpp
// whisper code is still entered and produces the exact byte-identical output
// it always has (including whisper-specific features like -owts karaoke,
// full-mode JSON with DTW tokens, token scoring, grammar, stereo diarize,
// whisper-internal VAD, n_processors, and so on).
//
// This wrapper deliberately exposes a REDUCED feature set — everything the
// unified `crispasr_write_*` writers already support (txt/srt/vtt/csv/lrc/
// basic JSON) plus tinydiarize speaker_turn_next. Features that the unified
// pipeline doesn't understand yet (WTS, full JSON with DTW, n_processors,
// grammar, whisper-internal VAD) require the historical path and are listed
// as CAP_* bits the wrapper does NOT advertise. `crispasr_run.cpp` then
// warns on stderr when the user requests one of those via the unified path.
//
// Capability matrix reporting (--list-backends) used to read a hardcoded
// `kWhisperCaps` constant in crispasr_backend.cpp because the wrapper didn't
// exist. With this file in place the matrix reads the capability bitmask off
// an actual instance — so there's only one source of truth.

#include "crispasr_backend.h"
#include "crispasr_vad_cli.h" // crispasr_resolve_vad_model — auto-DL silero
#include "whisper_params.h"

#include "grammar-parser.h" // grammar_parser::parse_state::c_rules()
#include "whisper.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

class WhisperBackend : public CrispasrBackend {
public:
    WhisperBackend() = default;
    ~WhisperBackend() override { WhisperBackend::shutdown(); }

    const char* name() const override { return "whisper"; }

    uint32_t capabilities() const override {
        // What this wrapper can actually deliver when driven by
        // crispasr_run_backend. Features exclusive to the historical
        // cli.cpp path (WTS, JSON-full with DTW, n_processors,
        // whisper-internal VAD, stereo diarize) are intentionally
        // omitted; users who need them should run without --backend on
        // a ggml-*.bin file, which keeps the byte-identical historical
        // path.
        return CAP_TIMESTAMPS_NATIVE | CAP_WORD_TIMESTAMPS | CAP_TOKEN_CONFIDENCE | CAP_LANGUAGE_DETECT |
               CAP_TRANSLATE | CAP_TEMPERATURE | CAP_BEAM_SEARCH | CAP_GRAMMAR | CAP_FLASH_ATTN | CAP_VAD_INTERNAL |
               CAP_PARALLEL_PROCESSORS | CAP_DIARIZE | CAP_AUTO_DOWNLOAD;
    }

    bool init(const whisper_params& p) override {
        whisper_context_params cp = whisper_context_default_params();
        cp.use_gpu = p.use_gpu;
        cp.gpu_device = p.gpu_device;
        cp.flash_attn = p.flash_attn;

        ctx_ = whisper_init_from_file_with_params(p.model.c_str(), cp);
        if (!ctx_) {
            fprintf(stderr, "crispasr[whisper]: failed to load model '%s'\n", p.model.c_str());
            return false;
        }
        return true;
    }

    std::vector<crispasr_segment> transcribe(const float* samples, int n_samples, int64_t t_offset_cs,
                                             const whisper_params& p) override {
        std::vector<crispasr_segment> out;
        if (!ctx_)
            return out;

        whisper_full_params wp =
            whisper_full_default_params(p.beam_size > 1 ? WHISPER_SAMPLING_BEAM_SEARCH : WHISPER_SAMPLING_GREEDY);

        wp.print_realtime = false;
        wp.print_progress = false;
        wp.print_timestamps = false;
        wp.print_special = false;
        wp.translate = p.translate;
        wp.language = p.language.c_str();
        wp.detect_language = p.detect_language;
        wp.n_threads = p.n_threads;

        wp.token_timestamps = true; // needed for the word vector below
        wp.max_len = p.max_len;
        wp.split_on_word = p.split_on_word;
        wp.audio_ctx = p.audio_ctx;

        wp.tdrz_enable = p.tinydiarize;

        wp.initial_prompt = p.prompt.c_str();
        wp.carry_initial_prompt = p.carry_initial_prompt;

        wp.greedy.best_of = p.best_of;
        wp.beam_search.beam_size = p.beam_size;

        wp.temperature_inc = p.no_fallback ? 0.0f : p.temperature_inc;
        wp.temperature = p.temperature;

        wp.entropy_thold = p.entropy_thold;
        wp.logprob_thold = p.logprob_thold;
        wp.no_speech_thold = p.no_speech_thold;

        wp.suppress_regex = p.suppress_regex.empty() ? nullptr : p.suppress_regex.c_str();
        wp.suppress_nst = p.suppress_nst;

        wp.no_timestamps = p.no_timestamps;

        // Whisper-internal VAD. Note this is distinct from the Silero-VAD
        // slicing the crispasr dispatcher does upstream — that one produces
        // `crispasr_audio_slice` batches that we transcribe one slice at a
        // time. This is whisper's in-graph energy-based VAD that skips
        // silent regions while decoding a single call. The two compose
        // without interfering: the dispatcher's slicing is a no-op when
        // params.vad_model is empty, in which case all of whisper-internal
        // VAD's behaviour is preserved.
        // Resolve VAD model so `--vad` without `--vad-model` auto-downloads
        // ggml-silero-v5.1.2.bin to the cache, matching the non-whisper
        // backends' UX (#33). The resolved path must outlive the whisper
        // call, so we hold it on the stack here.
        const std::string vad_path = crispasr_resolve_vad_model(p);
        wp.vad = p.vad;
        wp.vad_model_path = vad_path.c_str();
        wp.vad_params.threshold = p.vad_threshold;
        wp.vad_params.min_speech_duration_ms = p.vad_min_speech_duration_ms;
        wp.vad_params.min_silence_duration_ms = p.vad_min_silence_duration_ms;
        wp.vad_params.max_speech_duration_s = p.vad_max_speech_duration_s;
        wp.vad_params.speech_pad_ms = p.vad_speech_pad_ms;
        wp.vad_params.samples_overlap = p.vad_samples_overlap;

        // Grammar. When the user passed --grammar + --grammar-rule, the CLI
        // has already parsed the GBNF and stashed it in params.grammar_parsed.
        // Convert to whisper's C interface and hand it through. The
        // grammar_rules vector must outlive whisper_full, so it lives here
        // on the stack frame of transcribe(), not inside an if block.
        std::vector<const whisper_grammar_element*> grammar_rules;
        const bool use_grammar = (!p.grammar_parsed.rules.empty() && !p.grammar_rule.empty());
        if (use_grammar) {
            grammar_rules = p.grammar_parsed.c_rules();
            auto it = p.grammar_parsed.symbol_ids.find(p.grammar_rule);
            if (it != p.grammar_parsed.symbol_ids.end()) {
                wp.grammar_rules = grammar_rules.data();
                wp.n_grammar_rules = grammar_rules.size();
                wp.i_start_rule = it->second;
                wp.grammar_penalty = p.grammar_penalty;
                // Beam search is required for grammar-constrained decoding.
                wp.strategy = WHISPER_SAMPLING_BEAM_SEARCH;
            } else {
                fprintf(stderr, "crispasr[whisper]: grammar rule '%s' not found — ignoring\n", p.grammar_rule.c_str());
            }
        }

        // When the user asked for n_processors > 1, delegate parallelism
        // to whisper_full_parallel which internally clones N decoder
        // states and splits the audio into N roughly equal chunks.
        // Each chunk runs concurrently and whisper stitches the
        // segments back together. When n_processors == 1 this falls
        // through to the plain whisper_full path so the default is
        // byte-identical with the historical cli.
        const int nproc = (p.n_processors > 1) ? p.n_processors : 1;
        const int rc = (nproc > 1) ? whisper_full_parallel(ctx_, wp, samples, n_samples, nproc)
                                   : whisper_full(ctx_, wp, samples, n_samples);
        if (rc != 0) {
            fprintf(stderr, "crispasr[whisper]: %s failed\n", nproc > 1 ? "whisper_full_parallel" : "whisper_full");
            return out;
        }

        const int n_segments = whisper_full_n_segments(ctx_);
        const whisper_token eot = whisper_token_eot(ctx_);
        out.reserve(n_segments);
        for (int i = 0; i < n_segments; ++i) {
            crispasr_segment s;
            s.text = whisper_full_get_segment_text(ctx_, i);
            s.t0 = whisper_full_get_segment_t0(ctx_, i) + t_offset_cs;
            s.t1 = whisper_full_get_segment_t1(ctx_, i) + t_offset_cs;
            s.speaker_turn_next = whisper_full_get_segment_speaker_turn_next(ctx_, i);

            const int nt = whisper_full_n_tokens(ctx_, i);
            s.tokens.reserve(nt);
            for (int j = 0; j < nt; ++j) {
                const auto d = whisper_full_get_token_data(ctx_, i, j);
                if (d.id >= eot)
                    continue; // drop special tokens for display
                crispasr_token t;
                t.id = d.id;
                t.text = whisper_token_to_str(ctx_, d.id);
                t.confidence = d.p;
                t.t0 = d.t0 >= 0 ? d.t0 + t_offset_cs : -1;
                t.t1 = d.t1 >= 0 ? d.t1 + t_offset_cs : -1;
                t.t_dtw = d.t_dtw;
                t.is_special = false;
                s.tokens.push_back(std::move(t));
            }

            // Synthesize a word vector by grouping adjacent tokens on a space
            // prefix in the token text. Whisper's BPE tokens typically begin
            // a new word with a leading space.
            crispasr_word cur;
            bool have_cur = false;
            for (const auto& t : s.tokens) {
                const std::string& txt = t.text;
                const bool starts_word = !txt.empty() && txt[0] == ' ';
                if (starts_word && have_cur) {
                    s.words.push_back(std::move(cur));
                    cur = {};
                    have_cur = false;
                }
                if (!have_cur) {
                    cur.t0 = t.t0 >= 0 ? t.t0 : s.t0;
                    have_cur = true;
                }
                cur.text += txt;
                cur.t1 = t.t1 >= 0 ? t.t1 : s.t1;
            }
            if (have_cur)
                s.words.push_back(std::move(cur));

            out.push_back(std::move(s));
        }
        return out;
    }

    // No stereo override here — diarize is now a generic
    // dispatcher-level post-step (crispasr_diarize.cpp). The whisper
    // wrapper just transcribes mono and lets the dispatcher label
    // segments based on stereo channel energy.
    void shutdown() override {
        if (ctx_) {
            whisper_free(ctx_);
            ctx_ = nullptr;
        }
    }

private:
    struct whisper_context* ctx_ = nullptr;
};

} // namespace

std::unique_ptr<CrispasrBackend> crispasr_make_whisper_backend() {
    return std::unique_ptr<CrispasrBackend>(new WhisperBackend());
}
