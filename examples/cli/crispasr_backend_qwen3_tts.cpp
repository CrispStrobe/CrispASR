// crispasr_backend_qwen3_tts.cpp — adapter for Qwen3-TTS-12Hz.
//
// Two-GGUF runtime: the talker LM (loaded from --model) and a separate
// 12 Hz RVQ codec (loaded via --codec-model, or auto-discovered as a
// sibling of the talker, or via the auto-download companion file).
// Voice cloning takes either a baked voice-pack GGUF or a reference
// WAV plus its transcription (--voice ref.wav --ref-text "...").

#include "crispasr_backend.h"
#include "crispasr_backend_utils.h"
#include "whisper_params.h"

#include "qwen3_tts.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <sys/stat.h>

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

// Look for a sibling codec file next to the talker. The auto-download
// path drops both files into the same cache dir, so this hits in most
// real-world setups.
static std::string discover_codec(const std::string& model_path) {
    auto dir_of = [](const std::string& p) -> std::string {
        auto sep = p.find_last_of("/\\");
        return (sep == std::string::npos) ? std::string(".") : p.substr(0, sep);
    };
    const std::string dir = dir_of(model_path);
    static const char* candidates[] = {
        "qwen3-tts-tokenizer-12hz.gguf",
        "qwen3-tts-tokenizer.gguf",
        "qwen3-tts-codec.gguf",
    };
    for (const char* name : candidates) {
        std::string p = dir + "/" + name;
        if (file_exists(p))
            return p;
    }
    return "";
}

class Qwen3TtsBackend : public CrispasrBackend {
public:
    Qwen3TtsBackend() = default;
    ~Qwen3TtsBackend() override { Qwen3TtsBackend::shutdown(); }

    const char* name() const override { return "qwen3-tts"; }

    uint32_t capabilities() const override { return CAP_TTS | CAP_AUTO_DOWNLOAD | CAP_TEMPERATURE | CAP_FLASH_ATTN; }

    std::vector<crispasr_segment> transcribe(const float* /*samples*/, int /*n_samples*/, int64_t /*t_offset_cs*/,
                                             const whisper_params& /*params*/) override {
        fprintf(stderr, "crispasr[qwen3-tts]: transcription is not supported by this backend\n");
        return {};
    }

    bool init(const whisper_params& p) override {
        qwen3_tts_context_params cp = qwen3_tts_context_default_params();
        cp.n_threads = p.n_threads;
        cp.verbosity = p.no_prints ? 0 : 1;
        cp.use_gpu = crispasr_backend_should_use_gpu(p);
        cp.temperature = p.temperature;
        ctx_ = qwen3_tts_init_from_file(p.model.c_str(), cp);
        if (!ctx_) {
            fprintf(stderr, "crispasr[qwen3-tts]: failed to load talker '%s'\n", p.model.c_str());
            return false;
        }

        // Resolve the codec GGUF.
        std::string codec_path = p.tts_codec_model;
        if (codec_path.empty())
            codec_path = discover_codec(p.model);
        if (codec_path.empty()) {
            fprintf(stderr, "crispasr[qwen3-tts]: no codec model found. Pass --codec-model PATH or place "
                            "qwen3-tts-tokenizer-12hz.gguf next to the talker.\n");
            return false;
        }
        if (qwen3_tts_set_codec_path(ctx_, codec_path.c_str()) != 0) {
            fprintf(stderr, "crispasr[qwen3-tts]: failed to load codec '%s'\n", codec_path.c_str());
            return false;
        }
        if (!p.no_prints)
            fprintf(stderr, "crispasr[qwen3-tts]: codec loaded from '%s'\n", codec_path.c_str());
        return true;
    }

    std::vector<float> synthesize(const std::string& text, const whisper_params& params) override {
        if (!ctx_ || text.empty())
            return {};

        // Voice prompt: load once on first synthesise call. Three paths:
        //   --voice <name>       → CustomVoice fixed-speaker selection
        //                          (only when the loaded model is CustomVoice)
        //   --voice X.gguf       → baked voice pack (Base)
        //   --voice X.wav --ref-text "..." → runtime ECAPA + codec encoder (Base)
        if (!voice_loaded_) {
            if (qwen3_tts_is_custom_voice(ctx_)) {
                // CustomVoice: --voice is a speaker NAME (e.g. "vivian").
                // If absent, default to the first speaker in the table.
                std::string spk_name = params.tts_voice;
                if (spk_name.empty() || ends_with_ci(spk_name, ".wav") || ends_with_ci(spk_name, ".gguf")) {
                    if (!spk_name.empty() && !params.no_prints) {
                        fprintf(stderr, "crispasr[qwen3-tts]: CustomVoice expects a speaker NAME for --voice, "
                                        "got '%s' — falling back to first speaker.\n",
                                spk_name.c_str());
                    }
                    const char* first = qwen3_tts_get_speaker_name(ctx_, 0);
                    if (!first) {
                        fprintf(stderr,
                                "crispasr[qwen3-tts]: CustomVoice model has no speakers in the GGUF metadata\n");
                        return {};
                    }
                    spk_name = first;
                }
                if (qwen3_tts_set_speaker_by_name(ctx_, spk_name.c_str()) != 0) {
                    return {};
                }
                if (!params.no_prints) {
                    fprintf(stderr, "crispasr[qwen3-tts]: CustomVoice speaker = '%s' (available: ", spk_name.c_str());
                    int n = qwen3_tts_n_speakers(ctx_);
                    for (int i = 0; i < n; i++) {
                        fprintf(stderr, "%s%s", i ? ", " : "", qwen3_tts_get_speaker_name(ctx_, i));
                    }
                    fprintf(stderr, ")\n");
                }
            } else if (!params.tts_voice.empty()) {
                const std::string& v = params.tts_voice;
                if (ends_with_ci(v, ".wav")) {
                    if (params.tts_ref_text.empty()) {
                        fprintf(stderr, "crispasr[qwen3-tts]: --voice is a WAV but --ref-text was not set. "
                                        "Provide the reference transcription so the talker can match it.\n");
                        return {};
                    }
                    if (qwen3_tts_set_voice_prompt_with_text(ctx_, v.c_str(), params.tts_ref_text.c_str()) != 0) {
                        fprintf(stderr, "crispasr[qwen3-tts]: failed to set voice prompt from '%s'\n", v.c_str());
                        return {};
                    }
                } else {
                    if (qwen3_tts_load_voice_pack(ctx_, v.c_str()) != 0) {
                        fprintf(stderr, "crispasr[qwen3-tts]: failed to load voice pack '%s'\n", v.c_str());
                        return {};
                    }
                }
            }
            voice_loaded_ = true;
        }

        int n = 0;
        float* pcm = qwen3_tts_synthesize(ctx_, text.c_str(), &n);
        if (!pcm || n <= 0)
            return {};
        std::vector<float> out(pcm, pcm + n);
        qwen3_tts_pcm_free(pcm);
        return out;
    }

    void shutdown() override {
        if (ctx_) {
            qwen3_tts_free(ctx_);
            ctx_ = nullptr;
        }
        voice_loaded_ = false;
    }

private:
    qwen3_tts_context* ctx_ = nullptr;
    bool voice_loaded_ = false;
};

} // namespace

std::unique_ptr<CrispasrBackend> crispasr_make_qwen3_tts_backend() {
    return std::unique_ptr<CrispasrBackend>(new Qwen3TtsBackend());
}
