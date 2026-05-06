// crispasr_backend_vibevoice.cpp — adapter for Microsoft VibeVoice-ASR.
//
// The runtime itself expects 24 kHz mono PCM. The unified CrispASR CLI
// standardizes on 16 kHz audio input, so this adapter performs the same
// simple linear 16k -> 24k upsample that other 24 kHz backends use.

#include "crispasr_backend.h"
#include "crispasr_backend_utils.h"
#include "whisper_params.h"

#include "vibevoice.h"

#include <cctype>
#include <cstdio>
#include <string>
#include <sys/stat.h>
#include <vector>

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

class VibeVoiceBackend : public CrispasrBackend {
public:
    VibeVoiceBackend() = default;
    ~VibeVoiceBackend() override { VibeVoiceBackend::shutdown(); }

    const char* name() const override { return "vibevoice"; }

    uint32_t capabilities() const override {
        // ASR mode produces segments → framework -am + --diarize work.
        return CAP_TIMESTAMPS_CTC | CAP_AUTO_DOWNLOAD | CAP_TEMPERATURE | CAP_FLASH_ATTN | CAP_TTS | CAP_DIARIZE;
    }

    bool init(const whisper_params& p) override {
        vibevoice_context_params cp = vibevoice_context_default_params();
        cp.n_threads = p.n_threads;
        cp.max_new_tokens = p.max_new_tokens > 0 ? p.max_new_tokens : cp.max_new_tokens;
        cp.verbosity = p.no_prints ? 0 : 1;
        cp.use_gpu = crispasr_backend_should_use_gpu(p);
        cp.tts_steps = p.tts_steps;
        ctx_ = vibevoice_init_from_file(p.model.c_str(), cp);
        if (!ctx_) {
            fprintf(stderr, "crispasr[vibevoice]: failed to load model '%s'\n", p.model.c_str());
            return false;
        }
        return true;
    }

    std::vector<crispasr_segment> transcribe(const float* samples, int n_samples, int64_t t_offset_cs,
                                             const whisper_params& /*params*/) override {
        std::vector<crispasr_segment> out;
        if (!ctx_ || !samples || n_samples <= 0)
            return out;

        const std::vector<float> pcm24 = resample_16k_to_24k(samples, n_samples);
        char* text = vibevoice_transcribe(ctx_, pcm24.data(), (int)pcm24.size());
        if (!text)
            return out;

        crispasr_segment seg;
        seg.text = text;
        seg.t0 = t_offset_cs;
        seg.t1 = t_offset_cs + (int64_t)((double)n_samples * 100.0 / 16000.0);
        out.push_back(std::move(seg));
        std::free(text);
        return out;
    }

    std::vector<float> synthesize(const std::string& text, const whisper_params& params) override {
        // Voice resolution order:
        //   1. Bare-name in --voice-dir: <voice-dir>/<name>.gguf
        //      (matches qwen3-tts post-d35940b — server-mode callers can
        //       pass the same {"voice": "<name>"} request shape across all
        //       TTS backends; the adapter does the filesystem lookup
        //       against `params.tts_voice_dir`.)
        //   2. Explicit --voice <path>: literal path to a voice GGUF.
        //   3. Per-language sibling pick (vibevoice-voice-<lang>-Spk1_woman.gguf
        //      → vibevoice-voice-<lang>-Spk0_man.gguf), if -l <lang> is set.
        //   4. Sibling vibevoice-voice-emma.gguf as English default (matches
        //      the auto-download companion).
        std::string voice_path = params.tts_voice;

        // (1) Bare name → voice-dir lookup. A "bare name" is a token with
        // no path separators and no `.gguf`/`.wav` extension — e.g.
        // `voice: "vik"` from a /v1/audio/speech request. Path-traversal
        // sanitization (reject `..` and NUL) mirrors the qwen3-tts adapter.
        if (!voice_path.empty() && !params.tts_voice_dir.empty() && voice_path.find('/') == std::string::npos &&
            voice_path.find('\\') == std::string::npos && !ends_with_ci(voice_path, ".gguf") &&
            !ends_with_ci(voice_path, ".wav")) {
            if (voice_path.find("..") != std::string::npos || voice_path.find('\0') != std::string::npos) {
                fprintf(stderr, "crispasr[vibevoice-tts]: voice name '%s' contains illegal characters (.. or NUL)\n",
                        voice_path.c_str());
                return {};
            }
            const std::string gguf_path = params.tts_voice_dir + "/" + voice_path + ".gguf";
            if (file_exists(gguf_path)) {
                voice_path = gguf_path;
            }
            // else: leave bare name; vibevoice_load_voice will fail with a
            // clearer "could not be loaded" path below.
        }

        if (voice_path.empty()) {
            auto slash = params.model.find_last_of("/\\");
            std::string dir = (slash == std::string::npos) ? "." : params.model.substr(0, slash);
            auto try_sibling = [&](const std::string& fname) -> std::string {
                std::string p = dir + "/" + fname;
                if (file_exists(p))
                    return p;
                return {};
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

        // Per-call voice-key cache. Replaces the previous `voice_loaded_`
        // boolean so server callers can switch voice per request just by
        // changing params.tts_voice. CLI single-shot behaviour is unchanged
        // (first call still pays one load); vibevoice_load_voice() replaces
        // the active voice when re-called.
        if (!voice_path.empty() && voice_path != last_voice_key_) {
            if (vibevoice_load_voice(ctx_, voice_path.c_str()) == 0) {
                last_voice_key_ = voice_path;
                if (params.tts_voice.empty() || params.tts_voice != voice_path)
                    fprintf(stderr, "crispasr[vibevoice-tts]: voice loaded '%s'\n", voice_path.c_str());
            } else {
                fprintf(stderr,
                        "crispasr[vibevoice-tts]: voice '%s' could not be loaded; "
                        "refusing to synthesise without a voice prompt.\n",
                        voice_path.c_str());
                return {};
            }
        }
        if (last_voice_key_.empty()) {
            fprintf(stderr, "crispasr[vibevoice-tts]: no voice prompt resolved (pass --voice <path>, "
                            "drop a <name>.gguf into --voice-dir, or place a vibevoice-voice-*.gguf "
                            "next to the model).\n");
            return {};
        }
        if (!ctx_ || text.empty())
            return {};
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
};

} // namespace

std::unique_ptr<CrispasrBackend> crispasr_make_vibevoice_backend() {
    return std::unique_ptr<CrispasrBackend>(new VibeVoiceBackend());
}
