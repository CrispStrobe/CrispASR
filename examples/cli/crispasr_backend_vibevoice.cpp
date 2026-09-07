// crispasr_backend_vibevoice.cpp — adapter for Microsoft VibeVoice-ASR.
//
// The runtime itself expects 24 kHz mono PCM. The unified CrispASR CLI
// standardizes on 16 kHz audio input, so this adapter performs the same
// simple linear 16k -> 24k upsample that other 24 kHz backends use.

#include "crispasr_backend.h"
#include "crispasr_backend_utils.h"
#include "whisper_params.h"

#include "vibevoice.h"
#include "core/vibevoice_transcript.h" // #300: Start/End/Speaker/Content → segments

#include <cctype>
#include <cstdlib>
#include <cstdio>
#include <string>
#include <utility>
#include <sys/stat.h>
#include <vector>
#include "core/crispasr_env.h"

namespace {

static bool ends_with_ci(const std::string& s, const std::string& suffix) {
    if (s.size() < suffix.size())
        return false;
    for (size_t i = 0; i < suffix.size(); i++) {
        char a = (char)std::tolower((unsigned char)s[s.size() - suffix.size() + i]);
        char b = (char)std::tolower((unsigned char)suffix[i]);
        if (a != b)
            return false;
    }
    return true;
}

static bool file_exists(const std::string& path) {
    struct stat st;
    return stat(path.c_str(), &st) == 0;
}

static void set_env_overwrite(const char* name, const char* value) {
#if defined(_WIN32)
    _putenv_s(name, value);
#else
    setenv(name, value, 1);
#endif
}

static std::vector<float> resample_16k_to_24k(const float* in, int n_in) {
    std::vector<float> out;
    if (!in || n_in <= 0)
        return out;

    const int n_out = (int)((double)n_in * 24000.0 / 16000.0);
    out.resize((size_t)n_out);
    for (int i = 0; i < n_out; ++i) {
        const double pos = (double)i * 16000.0 / 24000.0;
        int i0 = (int)pos;
        int i1 = i0 + 1;
        if (i0 < 0)
            i0 = 0;
        if (i1 >= n_in)
            i1 = n_in - 1;
        const float frac = (float)(pos - (double)i0);
        out[(size_t)i] = in[i0] * (1.0f - frac) + in[i1] * frac;
    }
    return out;
}

class VibeVoiceRealtimeSession final : public CrispasrRealtimeSession {
public:
    VibeVoiceRealtimeSession(vibevoice_context* ctx, const std::string& context)
        : stream_(vibevoice_stream_open(ctx, context.empty() ? nullptr : context.c_str())) {}
    ~VibeVoiceRealtimeSession() override { vibevoice_stream_free(stream_); }

    bool valid() const { return stream_ != nullptr; }

    bool append(const float* samples, int n_samples, bool flush, callback on_text) override {
        if (!stream_)
            return false;
        struct State {
            VibeVoiceRealtimeSession* self;
            callback* fn;
        } state{this, &on_text};
        auto chunk_cb = [](const char* chunk, void* user) {
            auto& state = *static_cast<State*>(user);
            if (chunk)
                state.self->text_ += chunk;
            if (!state.self->text_.empty())
                (*state.fn)(state.self->text_, false);
        };
        if (vibevoice_stream_feed(stream_, samples, n_samples, flush, chunk_cb, &state) < 0)
            return false;
        if (flush)
            on_text(text_, true);
        return true;
    }

    void reset() override {
        vibevoice_stream_reset(stream_);
        text_.clear();
    }

private:
    vibevoice_stream* stream_ = nullptr;
    std::string text_;
};

class VibeVoiceBackend : public CrispasrBackend {
public:
    VibeVoiceBackend(std::string backend_name, bool allow_generic_no_voice)
        : backend_name_(std::move(backend_name)), allow_generic_no_voice_(allow_generic_no_voice) {}
    ~VibeVoiceBackend() override { VibeVoiceBackend::shutdown(); }

    const char* name() const override { return backend_name_.c_str(); }

    uint32_t capabilities() const override {
        // ASR mode produces segments → framework -am + --diarize work.
        // CAP_PUNCTUATION_NATIVE: VibeVoice-ASR is an LLM decoder — its Content
        // is already punctuated AND sentence-cased. Without the flag the CLI
        // auto-ran FireRedPunc over it (#308's audit item, found while fixing
        // #300): the capitaliser turned "And" into "ANd" and a second full stop
        // landed on text that already ended in one ("country..").
        // CAP_TEMPERATURE intentionally NOT declared (#369). It was, and it was
        // a claim with nothing behind it: `params.temperature` is never plumbed
        // into vibevoice_context, and the ASR decode is a plain argmax over the
        // logits — no temperature, no top-p, no sampling of any kind. The
        // reporter of #369 spent time establishing from outside that `-tp 0.8`
        // with different `--seed` values returns character-identical output,
        // which is exactly what warn_unsupported() would have told them for
        // free. Dropping the cap makes `--temperature` print "unsupported by
        // this backend" instead of being silently ignored. Re-declare it only
        // together with a decode path that actually reads the value — cf.
        // crispasr_backend_gemma4_e2b.cpp, "so CAP_TEMPERATURE is real, not
        // just a claim". CAP_BEAM_SEARCH was already, correctly, absent.
        uint32_t caps = CAP_TIMESTAMPS_CTC | CAP_AUTO_DOWNLOAD | CAP_FLASH_ATTN | CAP_DIARIZE |
                        CAP_PUNCTUATION_NATIVE;
        // The streaming 1.5B checkpoint is ASR-only and has no acoustic
        // decoder. The generic alias keeps its historical dual-mode claim
        // until a loaded streaming checkpoint lets us narrow it.
        if (backend_name_ != "vibevoice-streaming" && !vibevoice_is_asr_streaming(ctx_))
            caps |= CAP_TTS;
        if (allow_generic_no_voice_)
            caps |= CAP_VOICE_CLONING;
        if (backend_name_ == "vibevoice-streaming" || vibevoice_is_asr_streaming(ctx_))
            caps |= CAP_STREAMING;
        return caps;
    }

    // VibeVoice operates at 24 kHz internally.
    int input_sample_rate() const override { return 24000; }

    bool init(const whisper_params& p) override {
        vibevoice_context_params cp = vibevoice_context_default_params();
        cp.n_threads = p.n_threads;
        // #315: only force a cap when the user explicitly set --max-new-tokens;
        // otherwise pass 0 so vibevoice_resolve_max_new_tokens() scales the ASR
        // budget by audio duration (whisper_params.max_new_tokens defaults to
        // 512, so the old `> 0` test always forced 512 and truncated long-form).
        // Mirrors every other AR backend adapter (canary/funasr/mimo/moonshine…).
        cp.max_new_tokens = p.max_new_tokens_explicit ? p.max_new_tokens : 0;
        cp.verbosity = p.no_prints ? 0 : 1;
        cp.use_gpu = crispasr_backend_should_use_gpu(p);
        cp.tts_steps = p.tts_steps;
        cp.cfg_scale = p.tts_cfg_scale;
        ctx_ = vibevoice_init_from_file(p.model.c_str(), cp);
        if (!ctx_) {
            fprintf(stderr, "crispasr[vibevoice]: failed to load model '%s'\n", p.model.c_str());
            return false;
        }
        // For the ASR backend ("vibevoice"), verify encoder tensors exist
        // immediately so the user gets a clear diagnostic before any audio is
        // processed.  TTS-only aliases ("vibevoice-tts", "vibevoice-1.5b")
        // legitimately lack these tensors and must not fail here.
        if ((backend_name_ == "vibevoice" || backend_name_ == "vibevoice-streaming") && !vibevoice_has_asr(ctx_)) {
            fprintf(stderr,
                    "crispasr[vibevoice]: error: '%s' is a TTS-only model (no at_enc.*/st_enc.* tensors).\n"
                    "  Use --backend vibevoice-tts for this model, or download the ASR model:\n"
                    "  crispasr -m auto --backend vibevoice --auto-download\n",
                    p.model.c_str());
            vibevoice_free(ctx_);
            ctx_ = nullptr;
            return false;
        }
        return true;
    }

    std::vector<crispasr_segment> transcribe(const float* samples, int n_samples, int64_t t_offset_cs,
                                             const whisper_params& params) override {
        std::vector<crispasr_segment> out;
        if (!ctx_ || !samples || n_samples <= 0)
            return out;

        // When the CLI already loaded at 24 kHz (via input_sample_rate()),
        // pass through directly — no lossy resample needed.
        const int sr = input_sample_rate();
        const float* vv_pcm = samples;
        int vv_n = n_samples;
        std::vector<float> pcm24_buf;
        if (sr != 24000) {
            pcm24_buf = resample_16k_to_24k(samples, n_samples);
            vv_pcm = pcm24_buf.data();
            vv_n = (int)pcm24_buf.size();
        }
        // ASR is greedy, so --seed only matters when the acoustic posterior is
        // being sampled (CRISPASR_VIBEVOICE_ASR_SAMPLE=1). It was never plumbed
        // through the transcribe path at all, which is part of why the knob
        // looked inert from outside (#369).
        vibevoice_set_seed(ctx_, (uint32_t)params.seed);
        const char* context = params.context.empty() ? nullptr : params.context.c_str();
        char* text = vibevoice_transcribe_with_context(ctx_, vv_pcm, vv_n, context);
        if (!text)
            return out;
        const std::string raw = text;
        std::free(text);

        const int64_t dur_cs = (int64_t)((double)n_samples * 100.0 / (double)sr);

        // #300: VibeVoice-ASR answers with a JSON array of utterances carrying
        // Start / End / Speaker / Content (that is what its prompt asks for).
        // Read it: one segment per utterance, native timings, and the speaker in
        // the structured `speaker` field — the "(Speaker N) " form moss-diarize
        // uses, which is what file-mode prefixing and the #300 streaming emit
        // both consume. Before this the whole blob came back as ONE segment's
        // text, so the labels were literal JSON in the transcript and
        // `seg.speaker` was never set (the streaming "speaker" field could not
        // fire for this backend at all).
        //
        // CRISPASR_VIBEVOICE_RAW_TRANSCRIPT=1 restores the pre-#300 behaviour
        // (one segment, raw model output) for anyone parsing the blob themselves.
        if (!crispasr_env::truthy("CRISPASR_VIBEVOICE_RAW_TRANSCRIPT")) {
            const std::vector<core_vibevoice::Utterance> utts = core_vibevoice::parse(raw);
            for (const auto& u : utts) {
                // The model's own non-speech markers are not transcript text.
                // Dropped rather than emitted so an SRT never carries a literal
                // "[Silence]" over speech, and so the CLI's non-silent-audio
                // warning can fire (#369).
                if (core_vibevoice::is_non_speech_marker(u.text))
                    continue;
                std::string t = u.text;
                while (!t.empty() && (unsigned char)t.front() <= ' ')
                    t.erase(t.begin());
                while (!t.empty() && (unsigned char)t.back() <= ' ')
                    t.pop_back();
                if (t.empty())
                    continue;
                crispasr_segment seg;
                seg.text = std::move(t);
                // The model times each utterance within THIS chunk, so the
                // chunk offset still has to be added; clamp into the chunk in
                // case a decode overshoots its own window.
                auto clamp_cs = [&](double sec, int64_t fallback) {
                    if (sec < 0.0)
                        return t_offset_cs + fallback;
                    int64_t cs = (int64_t)(sec * 100.0 + (sec < 0 ? -0.5 : 0.5));
                    if (cs < 0)
                        cs = 0;
                    if (cs > dur_cs)
                        cs = dur_cs;
                    return t_offset_cs + cs;
                };
                seg.t0 = clamp_cs(u.start_s, 0);
                seg.t1 = clamp_cs(u.end_s, dur_cs);
                if (seg.t1 < seg.t0)
                    seg.t1 = seg.t0;
                if (u.speaker >= 0) {
                    char spk[32];
                    snprintf(spk, sizeof(spk), "(Speaker %d) ", u.speaker);
                    seg.speaker = spk;
                }
                out.push_back(std::move(seg));
            }
            if (!out.empty())
                return out;
            // The blob DID parse; it just carried nothing but non-speech. Return
            // empty so the caller reports no transcript — falling through here
            // would hand back the raw JSON, which is how "[Silence]" reached the
            // user's transcript in the first place.
            if (!utts.empty())
                return out;
            // Nothing parsed — the model answered in prose, or the decode was
            // cut before the first complete object. Fall through and hand back
            // the raw string rather than dropping the transcript.
        }

        crispasr_segment seg;
        seg.text = raw;
        seg.t0 = t_offset_cs;
        seg.t1 = t_offset_cs + dur_cs;
        out.push_back(std::move(seg));
        return out;
    }

    std::unique_ptr<CrispasrRealtimeSession> create_realtime_session(const whisper_params& params) override {
        if (!vibevoice_is_asr_streaming(ctx_))
            return nullptr;
        auto session = std::make_unique<VibeVoiceRealtimeSession>(ctx_, params.context);
        if (!session->valid())
            return nullptr;
        return session;
    }

    std::vector<float> synthesize(const std::string& text, const whisper_params& params) override {
        if (ctx_) {
            vibevoice_set_tts_steps(ctx_, params.tts_steps);
            vibevoice_set_seed(ctx_, (uint32_t)params.seed);
            vibevoice_set_cfg_scale(ctx_, params.tts_cfg_scale);
        }
        // Voice resolution order:
        //   1. Bare-name in --voice-dir: <voice-dir>/<name>.gguf
        //      (matches qwen3-tts post-d35940b — server callers can pass the same
        //       {"voice": "<name>"} shape across all TTS backends.)
        //   2. Explicit --voice <path.gguf>: literal path to a voice GGUF.
        //      --voice <path.wav>: 1.5B WAV-cloning path (see wav_ref_path_ below).
        //   3. Per-language sibling pick (vibevoice-voice-<lang>-Spk1_woman.gguf
        //      → vibevoice-voice-<lang>-Spk0_man.gguf), if -l <lang> is set.
        //   4. Sibling vibevoice-voice-emma.gguf as English default (matches
        //      the auto-download companion).
        std::string voice_path = params.tts_voice;

        // (1) Bare name → voice-dir lookup. A "bare name" is a token with no path
        // separators and no .gguf/.wav extension (e.g. `voice: "vik"` from a server
        // request). Path-traversal sanitization mirrors the qwen3-tts adapter.
        if (!voice_path.empty() && !params.tts_voice_dir.empty() && voice_path.find('/') == std::string::npos &&
            voice_path.find('\\') == std::string::npos && !ends_with_ci(voice_path, ".gguf") &&
            !ends_with_ci(voice_path, ".wav")) {
            if (voice_path.find("..") != std::string::npos || voice_path.find('\0') != std::string::npos) {
                fprintf(stderr, "crispasr[%s]: voice name '%s' contains illegal characters (.. or NUL)\n",
                        backend_name_.c_str(), voice_path.c_str());
                return {};
            }
            const std::string gguf_path = params.tts_voice_dir + "/" + voice_path + ".gguf";
            const std::string wav_path = params.tts_voice_dir + "/" + voice_path + ".wav";
            if (file_exists(gguf_path)) {
                voice_path = gguf_path;
            } else if (file_exists(wav_path)) {
                voice_path = wav_path;
            } else {
                fprintf(stderr, "crispasr[%s]: warning: neither '%s' nor '%s' were found on disk\n",
                        backend_name_.c_str(), gguf_path.c_str(), wav_path.c_str());
            }
            // else: leave bare name; loader will fail with a clear error below.
        }

        // (2b) WAV reference → 1.5B cloning path. Don't attempt to load .wav as GGUF;
        // store it and expose via VIBEVOICE_VOICE_AUDIO before each synthesize call.
        if (ends_with_ci(voice_path, ".wav")) {
            wav_ref_path_ = voice_path;
            voice_path.clear();
        }

        // (3)+(4) Sibling auto-pick when nothing was explicitly provided.
        // The realtime model wants a prompt voice. The 1.5B base model can
        // synthesize a generic voice without one, so keep the fallback only for
        // the realtime path.
        if (!allow_generic_no_voice_ && voice_path.empty() && wav_ref_path_.empty()) {
            auto slash = params.model.find_last_of("/\\");
            std::string dir = (slash == std::string::npos) ? "." : params.model.substr(0, slash);
            auto try_sibling = [&](const std::string& fname) -> std::string {
                std::string p = dir + "/" + fname;
                return file_exists(p) ? p : std::string{};
            };
            const std::string& lang = params.language;
            if (!lang.empty() && lang != "auto" && lang.size() >= 2) {
                std::string l2 = lang.substr(0, 2);
                voice_path = try_sibling("vibevoice-voice-" + l2 + "-Spk1_woman.gguf");
                if (voice_path.empty())
                    voice_path = try_sibling("vibevoice-voice-" + l2 + "-Spk0_man.gguf");
            }
            if (voice_path.empty())
                voice_path = try_sibling("vibevoice-voice-emma.gguf");
        }

        // Per-call voice-key cache: reload only when the resolved path changes.
        // Replaces the old bool so server callers can switch voice per request.
        if (!voice_path.empty() && voice_path != last_voice_key_) {
            if (vibevoice_load_voice(ctx_, voice_path.c_str()) == 0) {
                last_voice_key_ = voice_path;
                if (params.tts_voice.empty() || params.tts_voice != voice_path)
                    fprintf(stderr, "crispasr[%s]: voice loaded '%s'\n", backend_name_.c_str(), voice_path.c_str());
            } else {
                fprintf(stderr,
                        "crispasr[%s]: voice '%s' could not be loaded; refusing to synthesise without a "
                        "voice prompt.\n",
                        backend_name_.c_str(), voice_path.c_str());
                return {};
            }
        }

        // Gate: the realtime model requires a prompt voice. The 1.5B base
        // model can run without one and generate its generic prior voice.
        if (!allow_generic_no_voice_ && last_voice_key_.empty() && wav_ref_path_.empty()) {
            const char* voice_wav_env = crispasr_env::get("CRISPASR_VIBEVOICE_VOICE_AUDIO");
            if (!voice_wav_env || !voice_wav_env[0]) {
                fprintf(stderr,
                        "crispasr[%s]: no voice prompt resolved (pass --voice <path.gguf>, --voice <path.wav>, "
                        "drop a <name>.gguf into --voice-dir, set VIBEVOICE_VOICE_AUDIO=<wav>, or place a "
                        "vibevoice-voice-*.gguf next to the model).\n",
                        backend_name_.c_str());
                return {};
            }
        }

        if (!ctx_ || text.empty())
            return {};

        // Expose --voice <path.wav> to vibevoice_synthesize via env var.
        if (!wav_ref_path_.empty())
            set_env_overwrite("VIBEVOICE_VOICE_AUDIO", wav_ref_path_.c_str());

        int n_samples = 0;
        float* pcm = vibevoice_synthesize(ctx_, text.c_str(), &n_samples);
        if (!pcm || n_samples <= 0)
            return {};
        std::vector<float> out(pcm, pcm + n_samples);
        std::free(pcm);
        return out;
    }

    void shutdown() override {
        if (ctx_) {
            vibevoice_free(ctx_);
            ctx_ = nullptr;
        }
        last_voice_key_.clear();
    }

private:
    vibevoice_context* ctx_ = nullptr;
    std::string last_voice_key_;
    std::string wav_ref_path_; // set when --voice <path.wav> is used (1.5B WAV cloning)
    std::string backend_name_ = "vibevoice";
    bool allow_generic_no_voice_ = false;
};

} // namespace

std::unique_ptr<CrispasrBackend> crispasr_make_vibevoice_backend() {
    return std::unique_ptr<CrispasrBackend>(new VibeVoiceBackend("vibevoice", false));
}

std::unique_ptr<CrispasrBackend> crispasr_make_vibevoice_streaming_backend() {
    return std::unique_ptr<CrispasrBackend>(new VibeVoiceBackend("vibevoice-streaming", false));
}

std::unique_ptr<CrispasrBackend> crispasr_make_vibevoice_tts_backend() {
    // TTS-only alias — skips the ASR encoder check in init().
    return std::unique_ptr<CrispasrBackend>(new VibeVoiceBackend("vibevoice-tts", false));
}

std::unique_ptr<CrispasrBackend> crispasr_make_vibevoice_1p5b_backend() {
    return std::unique_ptr<CrispasrBackend>(new VibeVoiceBackend("vibevoice-1.5b", true));
}
