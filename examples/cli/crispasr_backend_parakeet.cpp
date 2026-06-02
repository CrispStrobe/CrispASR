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
        //
        // CAP_INTERNAL_CHUNKING intentionally NOT declared (2026-05-26,
        // PLAN #114 follow-up). The backend's own
        // `parakeet_transcribe_streamed` does handle long audio
        // internally (chunked encode + concat + single TDT decode), but
        // the empirical option matrix in PERFORMANCE.md shows that the
        // dispatcher's chunk-30 + overlap-save + LCS-merge dedup path
        // produces materially more content on long audio (v3+EN60:
        // 520→755 chars +45 %; ja+JA60: 1674→1942 +16 %; v3+JA60:
        // 605→660 +9 %). Without this flag the dispatcher's
        // auto-chunk-at-30s fallback fires for audio > 30 s. Short audio
        // (< 30 s) is unaffected — the dispatcher only auto-chunks past
        // the threshold, so 11 s JFK still gets one backend call on the
        // full audio.
        return CAP_TIMESTAMPS_NATIVE | CAP_WORD_TIMESTAMPS | CAP_TOKEN_CONFIDENCE | CAP_FLASH_ATTN |
               CAP_PUNCTUATION_TOGGLE | CAP_TEMPERATURE | CAP_BEAM_SEARCH | CAP_DIARIZE | CAP_PARALLEL_PROCESSORS |
               CAP_AUTO_DOWNLOAD | CAP_UNBOUNDED_INPUT;
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

    void warmup() override {
        if (!ctx_)
            return;
        // 0.5 s of silence at 16 kHz — touches mel, encoder, and decoder
        // graphs once so subsequent calls hit pre-allocated buffers.
        std::vector<float> silence(8000, 0.0f);
        parakeet_result* r = parakeet_transcribe_ex(ctx_, silence.data(), (int)silence.size(), 0);
        if (r)
            parakeet_result_free(r);
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
        parakeet_set_beam_size(ctx_, params.beam_size > 0 ? params.beam_size : 1);

        // PLAN #98: CTC-WS hotword phrase boost
        if (!params.hotwords.empty()) {
            auto hw = core_context_bias::parse_hotwords(params.hotwords);
            std::vector<const char*> ptrs;
            for (auto& s : hw)
                ptrs.push_back(s.c_str());
            parakeet_set_hotwords(ctx_, ptrs.data(), (int)ptrs.size(), params.hotwords_boost);
        }

        // Issue #89: encoding-path selection.
        //
        // parakeet_transcribe_ex (single-pass) routes the full audio through
        // one bidirectional FastConformer encoder pass. The attention is
        // numerically unstable past ~10-20 s in a way that depends on
        // per-feature z-norm statistics: two codec-quantized copies of the
        // same speech (≈0.3% RMS diff, 0.998 waveform corr) can flip the
        // encoder output std by 10-15 % and drive the TDT decoder into
        // emit-blank-forever past a few seconds. Repro: lenhone's clip in
        // issue #89, comment 4529025103 (60 s file → 20 s of output).
        //
        // parakeet_transcribe_streamed encodes the (globally-z-normed) mel
        // in overlapping 8 s windows and concatenates encoder outputs
        // before a single TDT decode. The attention is local-bounded so
        // codec-level perturbations don't amplify, and the decoder still
        // sees one contiguous encoder sequence (no LSTM cold-start).
        // On audio where single-pass works, streamed produces byte-
        // identical or near-identical text. On audio where single-pass
        // collapses, streamed still covers ~99 % of the clip.
        //
        // We therefore always go through the streamed path. Single-pass
        // is preserved as an opt-in escape hatch for callers who really
        // want the bidirectional-over-everything behaviour (e.g. for
        // bit-exact reproduction of upstream NeMo on test data) — set
        // `CRISPASR_PARAKEET_STREAM_THRESHOLD=999` to bypass streaming
        // for audio shorter than that.
        //
        // Env knobs:
        //   CRISPASR_PARAKEET_STREAM_THRESHOLD : single-pass for audio ≤
        //       this duration (seconds). Default 0 = always streamed.
        //   CRISPASR_PARAKEET_STREAM_CHUNK     : encoder chunk size (s).
        //       Default 0 = let the C library pick per-model: 8 s for the
        //       JA-only model (vocab=3072), 30 s for the multilingual / v3
        //       family (vocab=8192). Manual override here for the rare
        //       case where neither heuristic fits the audio at hand.
        //   CRISPASR_PARAKEET_STREAM_OVERLAP   : encoder overlap (s).
        //       Default 2. Covers FastConformer's receptive field at the
        //       chunk boundary; overlap frames from later chunks are
        //       discarded before decode.
        int stream_threshold_s = 0;
        int stream_chunk_s = 0; // 0 = let the C library pick per-model
        int stream_overlap_s = 2;
        if (const char* e = getenv("CRISPASR_PARAKEET_STREAM_THRESHOLD"))
            stream_threshold_s = std::max(0, atoi(e));
        if (const char* e = getenv("CRISPASR_PARAKEET_STREAM_CHUNK"))
            stream_chunk_s = std::max(2, atoi(e));
        if (const char* e = getenv("CRISPASR_PARAKEET_STREAM_OVERLAP"))
            stream_overlap_s = std::max(0, atoi(e));

        parakeet_result* r;
        const bool use_single_pass = stream_threshold_s > 0 && n_samples <= stream_threshold_s * 16000;
        if (use_single_pass) {
            r = parakeet_transcribe_ex(ctx_, samples, n_samples, t_offset_cs);
        } else {
            r = parakeet_transcribe_streamed(ctx_, samples, n_samples, t_offset_cs, stream_chunk_s, stream_overlap_s);
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
