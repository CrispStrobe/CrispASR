// crispasr_backend_cosyvoice3.cpp — adapter for FunAudioLLM/Fun-CosyVoice3-0.5B-2512 TTS.
//
// Four-GGUF runtime: LLM (-m), flow (sibling), CAMPPlus (sibling), HiFT
// (sibling), plus a voices.gguf carrying baked voice-clone bundles. The
// flow path can be overridden via --codec-model; CAMPPlus / HiFT / voices
// auto-discover as siblings of the LLM (or via the
// COSYVOICE3_CAMPPLUS_PATH / COSYVOICE3_HIFT_PATH /
// COSYVOICE3_VOICES_PATH env vars).

#include "crispasr_backend.h"
#include "crispasr_backend_utils.h"
#include "whisper_params.h"

#include "cosyvoice3_tts.h"

#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <sys/stat.h>
#include <vector>

namespace {

bool ends_with_ci(const std::string& s, const std::string& suffix) {
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

bool file_exists(const std::string& path) {
    struct stat st;
    return stat(path.c_str(), &st) == 0;
}

std::string dir_of(const std::string& path) {
    auto sep = path.find_last_of("/\\");
    return (sep == std::string::npos) ? std::string(".") : path.substr(0, sep);
}

std::string discover_sibling(const std::string& base_dir, const std::vector<const char*>& candidates) {
    for (const char* name : candidates) {
        std::string p = base_dir + "/" + name;
        if (file_exists(p))
            return p;
    }
    return "";
}

class CosyVoice3TtsBackend : public CrispasrBackend {
public:
    CosyVoice3TtsBackend() = default;
    ~CosyVoice3TtsBackend() override { CosyVoice3TtsBackend::shutdown(); }

    const char* name() const override { return "cosyvoice3-tts"; }

    uint32_t capabilities() const override {
        return CAP_TTS | CAP_VOICE_CLONING | CAP_AUTO_DOWNLOAD | CAP_TEMPERATURE | CAP_FLASH_ATTN;
    }

    int tts_sample_rate() const override { return 24000; }

    std::vector<crispasr_segment> transcribe(const float* /*samples*/, int /*n_samples*/, int64_t /*t_offset_cs*/,
                                             const whisper_params& /*params*/) override {
        fprintf(stderr, "crispasr[cosyvoice3-tts]: transcription is not supported by this backend\n");
        return {};
    }

    bool init(const whisper_params& p) override {
        cosyvoice3_tts_context_params cp = cosyvoice3_tts_context_default_params();
        cp.n_threads = p.n_threads;
        cp.verbosity = p.no_prints ? 0 : 1;
        cp.use_gpu = crispasr_backend_should_use_gpu(p);
        // CV3 trains with `ras_sampling(top_p=0.8, top_k=25, ...)`; greedy
        // (temperature=0) falls into the documented "silent_tokens"
        // loop within ~5 steps. Default to a non-zero temperature so
        // `cosyvoice3_tts_sample_ras` engages, but honour an explicit
        // user override (including --temperature 0 for diff testing).
        cp.temperature = p.temperature > 0.0f ? p.temperature : 0.8f;
        cp.seed = (uint64_t)p.seed;
        cp.max_tokens = p.max_new_tokens;

        ctx_ = cosyvoice3_tts_init_from_file(p.model.c_str(), cp);
        if (!ctx_) {
            fprintf(stderr, "crispasr[cosyvoice3-tts]: failed to load LLM '%s'\n", p.model.c_str());
            return false;
        }

        const std::string base_dir = dir_of(p.model);

        // ---- Flow ----
        std::string flow_path = p.tts_codec_model;
        if (flow_path.empty() || flow_path == "auto" || flow_path == "default") {
            flow_path = discover_sibling(base_dir, {
                                                       "cosyvoice3-flow-f16.gguf",
                                                       "cosyvoice3-flow-q8_0.gguf",
                                                       "cosyvoice3-flow.gguf",
                                                   });
        }
        if (flow_path.empty()) {
            fprintf(stderr, "crispasr[cosyvoice3-tts]: no flow GGUF found. Place "
                            "cosyvoice3-flow-f16.gguf next to the LLM, or pass --codec-model PATH.\n");
            return false;
        }
        if (cosyvoice3_tts_init_flow_from_file(ctx_, flow_path.c_str()) != 0) {
            fprintf(stderr, "crispasr[cosyvoice3-tts]: failed to load flow '%s'\n", flow_path.c_str());
            return false;
        }

        // ---- CAMPPlus ----
        std::string campplus_path;
        const char* env_campplus = getenv("COSYVOICE3_CAMPPLUS_PATH");
        if (env_campplus && env_campplus[0])
            campplus_path = env_campplus;
        if (campplus_path.empty()) {
            campplus_path = discover_sibling(base_dir, {
                                                           "cosyvoice3-campplus-f16.gguf",
                                                           "cosyvoice3-campplus.gguf",
                                                       });
        }
        if (campplus_path.empty()) {
            fprintf(stderr, "crispasr[cosyvoice3-tts]: no CAMPPlus GGUF found. Place "
                            "cosyvoice3-campplus-f16.gguf next to the LLM, or set "
                            "COSYVOICE3_CAMPPLUS_PATH.\n");
            return false;
        }
        if (cosyvoice3_tts_init_campplus_from_file(ctx_, campplus_path.c_str()) != 0) {
            fprintf(stderr, "crispasr[cosyvoice3-tts]: failed to load CAMPPlus '%s'\n", campplus_path.c_str());
            return false;
        }

        // ---- HiFT ----
        std::string hift_path;
        const char* env_hift = getenv("COSYVOICE3_HIFT_PATH");
        if (env_hift && env_hift[0])
            hift_path = env_hift;
        if (hift_path.empty()) {
            hift_path = discover_sibling(base_dir, {
                                                       "cosyvoice3-hift-f16.gguf",
                                                       "cosyvoice3-hift.gguf",
                                                   });
        }
        if (hift_path.empty()) {
            fprintf(stderr, "crispasr[cosyvoice3-tts]: no HiFT GGUF found. Place "
                            "cosyvoice3-hift-f16.gguf next to the LLM, or set "
                            "COSYVOICE3_HIFT_PATH.\n");
            return false;
        }
        if (cosyvoice3_tts_init_hift_from_file(ctx_, hift_path.c_str()) != 0) {
            fprintf(stderr, "crispasr[cosyvoice3-tts]: failed to load HiFT '%s'\n", hift_path.c_str());
            return false;
        }

        // ---- speech_tokenizer_v3 ----
        std::string s3tok_path = discover_sibling(base_dir, {
                                                                "cosyvoice3-s3tok-f16.gguf",
                                                                "cosyvoice3-s3tok.gguf",
                                                            });
        if (!s3tok_path.empty()) {
            if (cosyvoice3_tts_init_s3tok_from_file(ctx_, s3tok_path.c_str()) != 0) {
                fprintf(stderr, "crispasr[cosyvoice3-tts]: failed to load s3tok '%s'\n", s3tok_path.c_str());
                return false;
            }
        } else if (!p.no_prints) {
            fprintf(stderr, "crispasr[cosyvoice3-tts]: no s3tok GGUF found. Runtime WAV cloning will fall back to the "
                            "Python voice-bake bridge until cosyvoice3-s3tok-f16.gguf is present.\n");
        }

        // ---- Voices ----
        std::string voices_path;
        const char* env_voices = getenv("COSYVOICE3_VOICES_PATH");
        if (env_voices && env_voices[0])
            voices_path = env_voices;
        if (voices_path.empty()) {
            voices_path = discover_sibling(base_dir, {
                                                         "cosyvoice3-voices.gguf",
                                                         "voices.gguf",
                                                     });
        }
        if (voices_path.empty()) {
            fprintf(stderr, "crispasr[cosyvoice3-tts]: no voices GGUF found. Run "
                            "models/convert-cosyvoice3-voices-to-gguf.py and place the "
                            "output next to the LLM, or set COSYVOICE3_VOICES_PATH.\n");
            return false;
        }
        if (cosyvoice3_tts_init_voices_from_file(ctx_, voices_path.c_str()) != 0) {
            fprintf(stderr, "crispasr[cosyvoice3-tts]: failed to load voices '%s'\n", voices_path.c_str());
            return false;
        }
        if (!p.no_prints) {
            int nv = cosyvoice3_tts_n_voices(ctx_);
            fprintf(stderr, "crispasr[cosyvoice3-tts]: %d voice(s) available:", nv);
            for (int i = 0; i < nv; i++) {
                fprintf(stderr, " %s", cosyvoice3_tts_voice_name(ctx_, i));
            }
            fprintf(stderr, "\n");
        }
        return true;
    }

    std::vector<float> synthesize(const std::string& text, const whisper_params& params) override {
        if (!ctx_ || text.empty())
            return {};
        cosyvoice3_tts_set_temperature(ctx_, params.temperature > 0.0f ? params.temperature : 0.8f);
        cosyvoice3_tts_set_seed(ctx_, (uint64_t)params.seed);

        int n = 0;
        float* pcm = nullptr;
        if (ends_with_ci(params.tts_voice, ".wav")) {
            if (params.tts_ref_text.empty()) {
                fprintf(stderr, "crispasr[cosyvoice3-tts]: --voice is a WAV but --ref-text was not set.\n");
                return {};
            }
            pcm = cosyvoice3_tts_synth_from_wav(ctx_, text.c_str(), params.tts_voice.c_str(),
                                                params.tts_ref_text.c_str(), &n);
        } else {
            const char* voice = params.tts_voice.empty() ? nullptr : params.tts_voice.c_str();
            pcm = cosyvoice3_tts_synth(ctx_, text.c_str(), voice, &n);
        }
        if (!pcm || n <= 0) {
            if (pcm)
                free(pcm);
            return {};
        }
        std::vector<float> out(pcm, pcm + n);
        free(pcm);
        return out;
    }

    void shutdown() override {
        if (ctx_) {
            cosyvoice3_tts_free(ctx_);
            ctx_ = nullptr;
        }
    }

private:
    struct cosyvoice3_tts_context* ctx_ = nullptr;
};

} // namespace

std::unique_ptr<CrispasrBackend> crispasr_make_cosyvoice3_tts_backend() {
    return std::unique_ptr<CrispasrBackend>(new CosyVoice3TtsBackend());
}
