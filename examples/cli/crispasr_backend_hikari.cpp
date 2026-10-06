// crispasr_backend_hikari.cpp — adapter for sbintuitions/hikari-medium.
//
// Simultaneous speech translation EN -> DE/JA/RU and streaming English ASR.
// The model decides once per 80 ms of audio (emit a token or WAIT), so the
// offline path is the streaming policy replayed over the file, and
// `--stream` uses the incremental session (create_realtime_session).
//
//   crispasr --backend hikari -m hikari-medium-q8_0.gguf -l en --tr-tl de -f talk.wav
//   crispasr --backend hikari -m ... -l en               (English transcription)
//
// Target language: --tr-tl, else -tl; none (or "en") = transcription.
// Policy knobs (upstream hikari-client defaults): HIKARI_WP_BASE (0),
// HIKARI_WP_BOOST (0.6), HIKARI_WP_DECAY (0.3), HIKARI_REP (40),
// HIKARI_CTX (337 tokens), HIKARI_TAIL_MS (2000 ms of trailing silence for
// files). The Silero speech probability drives the wait-penalty boost, and
// that boost is what makes the model emit at all during speech (without it,
// upstream and here, jfk 0-4 s is 48/48 WAIT). Model: HIKARI_VAD_MODEL, else
// --vad-model, else the default Silero (auto-download). HIKARI_VAD=0 = off.

#include "crispasr_backend.h"
#include "crispasr_backend_utils.h"
#include "crispasr_vad_cli.h"
#include "whisper_params.h"

#include "core/silero_context.h"
#include "hikari.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

float env_f(const char* k, float d) {
    const char* v = std::getenv(k);
    return (v && *v) ? (float)std::atof(v) : d;
}
int env_i(const char* k, int d) {
    const char* v = std::getenv(k);
    return (v && *v) ? std::atoi(v) : d;
}

// Stateful Silero over the newest 512 samples of each window — what
// model_wrapper.py's is_speech() does with the TorchScript model.
struct HikariVad {
    whisper_vad_context* vctx = nullptr;
    static float prob(const float* s, int n, void* user) {
        auto* self = static_cast<HikariVad*>(user);
        if (!self->vctx || !crispasr_silero_detect_continue(self->vctx, s, n))
            return 0.0f;
        const int np = whisper_vad_n_probs(self->vctx);
        return np > 0 ? whisper_vad_probs(self->vctx)[np - 1] : 0.0f;
    }
    void reset() {
        if (vctx) {
            float z = 0.0f;
            whisper_vad_detect_speech(vctx, &z, 0); // clears LSTM state + source context
        }
    }
    ~HikariVad() {
        if (vctx)
            whisper_vad_free(vctx);
    }
};

std::string token_text(hikari_context* ctx, int32_t id) {
    char* t = hikari_token_text(ctx, id);
    std::string s = t ? t : "";
    std::free(t);
    return s;
}

class HikariRealtimeSession final : public CrispasrRealtimeSession {
public:
    HikariRealtimeSession(hikari_context* ctx, HikariVad* vad) : ctx_(ctx), vad_(vad) { reset(); }

    bool append(const float* samples, int n_samples, bool flush, callback on_text) override {
        if (n_samples > 0 && hikari_stream_push(ctx_, samples, n_samples) < 0)
            return false;
        if (flush && hikari_stream_flush(ctx_) < 0)
            return false;
        char* t = hikari_stream_text(ctx_);
        std::string text = t ? t : "";
        std::free(t);
        const size_t lead = text.find_first_not_of(' ');
        text = lead == std::string::npos ? std::string() : text.substr(lead);
        if (flush || text != last_) {
            last_ = text;
            on_text(text, flush);
        }
        return true;
    }

    void reset() override {
        hikari_stream_reset(ctx_);
        if (vad_)
            vad_->reset();
        last_.clear();
    }

private:
    hikari_context* ctx_;
    HikariVad* vad_;
    std::string last_;
};

class HikariBackend : public CrispasrBackend {
public:
    ~HikariBackend() override { HikariBackend::shutdown(); }

    const char* name() const override { return "hikari"; }

    uint32_t capabilities() const override {
        return CAP_TRANSLATE | CAP_SRC_TGT_LANGUAGE | CAP_PUNCTUATION_NATIVE | CAP_STREAMING | CAP_UNBOUNDED_INPUT |
               CAP_TIMESTAMPS_NATIVE;
    }

    bool init(const whisper_params& p) override {
        hikari_context_params cp = hikari_context_default_params();
        cp.n_threads = p.n_threads;
        cp.verbosity = p.no_prints ? 0 : (env_i("HIKARI_VERBOSE", 0) ? 2 : 1);
        cp.use_gpu = crispasr_backend_should_use_gpu(p);
        ctx_ = hikari_init_from_file(p.model.c_str(), cp);
        if (!ctx_) {
            fprintf(stderr, "crispasr[hikari]: failed to load model '%s'\n", p.model.c_str());
            return false;
        }
        hikari_policy pol = hikari_default_policy();
        pol.baseline_wait_penalty = env_f("HIKARI_WP_BASE", pol.baseline_wait_penalty);
        pol.wait_penalty_boost = env_f("HIKARI_WP_BOOST", pol.wait_penalty_boost);
        pol.wait_penalty_decay = env_f("HIKARI_WP_DECAY", pol.wait_penalty_decay);
        pol.repetition_penalty = env_f("HIKARI_REP", pol.repetition_penalty);
        pol.decoder_context = env_i("HIKARI_CTX", pol.decoder_context);
        hikari_set_policy(ctx_, &pol);
        hikari_set_tail_silence_ms(ctx_, env_i("HIKARI_TAIL_MS", 2000));

        if (env_i("HIKARI_VAD", 1)) {
            // The speech flag only steers the wait-penalty boost, so a VAD
            // the user picked for slicing is honoured, and Silero otherwise.
            // HIKARI_VAD_MODEL names the file without turning on the
            // dispatcher's VAD slicing (which --vad-model would).
            whisper_params vp = p;
            vp.vad = true;
            const char* env_vad = std::getenv("HIKARI_VAD_MODEL");
            const std::string path = (env_vad && *env_vad) ? std::string(env_vad) : crispasr_resolve_vad_model(vp);
            whisper_vad_context_params vcp = whisper_vad_default_context_params();
            vcp.n_threads = 1;
            vad_.vctx = path.empty() ? nullptr : whisper_vad_init_from_file_with_params(path.c_str(), vcp);
            if (vad_.vctx && !crispasr_silero_enable_context(vad_.vctx)) {
                whisper_vad_free(vad_.vctx);
                vad_.vctx = nullptr;
            }
            if (vad_.vctx) {
                hikari_set_speech_prob_fn(ctx_, &HikariVad::prob, &vad_);
            } else if (!p.no_prints) {
                fprintf(stderr, "crispasr[hikari]: no Silero VAD available - wait-penalty boost disabled\n");
            }
        }
        return apply_task(p);
    }

    std::vector<crispasr_segment> transcribe(const float* samples, int n_samples, int64_t t_offset_cs,
                                             const whisper_params& params) override {
        std::vector<crispasr_segment> out;
        if (!ctx_ || !apply_task(params))
            return out;
        vad_.reset();
        char* t = hikari_transcribe(ctx_, samples, n_samples);
        if (!t)
            return out;
        std::free(t);

        // One segment per sentence. Times are EMISSION times (the audio the
        // decision had seen), which trail the speech by the model's lag.
        crispasr_segment seg;
        int64_t seg_start = t_offset_cs;
        const int n = hikari_stream_n_steps(ctx_);
        const int64_t t_end_audio = t_offset_cs + (int64_t)n_samples * 100 / 16000;
        auto close = [&](int64_t t1) {
            const size_t lead = seg.text.find_first_not_of(' ');
            if (lead == std::string::npos)
                return;
            seg.text = seg.text.substr(lead);
            seg.t0 = seg_start;
            seg.t1 = std::min(std::max(t1, seg_start), std::max(t_end_audio, seg_start));
            seg_start = seg.t1;
            out.push_back(std::move(seg));
            seg = crispasr_segment{};
        };
        for (int i = 0; i < n; i++) {
            const int32_t id = hikari_stream_step_token(ctx_, i);
            const std::string piece = token_text(ctx_, id);
            if (piece.empty())
                continue;
            const int64_t tcs = t_offset_cs + (int64_t)(hikari_stream_step_time(ctx_, i) * 100.0 + 0.5);
            crispasr_token tok;
            tok.text = piece;
            tok.id = id;
            tok.t0 = tok.t1 = tcs;
            seg.tokens.push_back(tok);
            seg.text += piece;
            const std::string tail = seg.text.size() >= 3 ? seg.text.substr(seg.text.size() - 3) : seg.text;
            const char last = seg.text.empty() ? 0 : seg.text.back();
            if (last == '.' || last == '!' || last == '?' || tail == "\xE3\x80\x82" || tail == "\xEF\xBC\x81" ||
                tail == "\xEF\xBC\x9F")
                close(tcs);
        }
        close(t_end_audio);
        return out;
    }

    std::unique_ptr<CrispasrRealtimeSession> create_realtime_session(const whisper_params& params) override {
        if (!ctx_ || !apply_task(params))
            return nullptr;
        return std::unique_ptr<CrispasrRealtimeSession>(new HikariRealtimeSession(ctx_, vad_.vctx ? &vad_ : nullptr));
    }

    bool prefers_realtime_session() const override { return true; }

    void shutdown() override {
        if (ctx_) {
            hikari_free(ctx_);
            ctx_ = nullptr;
        }
    }

private:
    bool apply_task(const whisper_params& p) {
        if (!p.language.empty() && p.language != "en" && p.language != "auto" && !warned_lang_) {
            fprintf(stderr, "crispasr[hikari]: the model takes English speech only; -l %s ignored\n",
                    p.language.c_str());
            warned_lang_ = true;
        }
        std::string tgt = !p.translate_target_lang.empty() ? p.translate_target_lang : p.target_lang;
        if (tgt.empty() && p.translate) {
            fprintf(stderr, "crispasr[hikari]: --translate needs a target: --tr-tl de|ja|ru\n");
            return false;
        }
        const bool translate = !tgt.empty() && tgt != "en";
        if (translate && tgt != "de" && tgt != "ja" && tgt != "ru") {
            fprintf(stderr, "crispasr[hikari]: target '%s' is not trained (de, ja, ru)\n", tgt.c_str());
            return false;
        }
        const std::string key = translate ? tgt : std::string("transcribe");
        if (key == task_key_)
            return true;
        if (hikari_set_task(ctx_, translate ? 1 : 0, tgt.c_str()) != 0)
            return false;
        task_key_ = key;
        return true;
    }

    hikari_context* ctx_ = nullptr;
    HikariVad vad_;
    std::string task_key_;
    bool warned_lang_ = false;
};

} // namespace

std::unique_ptr<CrispasrBackend> crispasr_make_hikari_backend() {
    return std::unique_ptr<CrispasrBackend>(new HikariBackend());
}
