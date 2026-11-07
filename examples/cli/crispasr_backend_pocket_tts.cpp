// crispasr_backend_pocket_tts.cpp -- adapter for kyutai/pocket-tts
// (100M continuous-latent AR TTS, 24 kHz, MIT/CC-BY-4.0).
//
// Single-GGUF runtime: the combined model contains the FlowLM backbone,
// consistency head, Mimi VAE decoder, and SentencePiece tokenizer.
// Voice cloning requires encoder weights (--voice-cloning at convert time).
//
// Pocket TTS is architecturally unique: it generates continuous 32-dim
// float vectors at 12.5 Hz via one-step Lagrangian Self Distillation,
// NOT discrete token IDs. There is no codebook lookup or RVQ.

#include "crispasr_backend.h"
#include "crispasr_backend_utils.h"
#include "crispasr_model_mgr_cli.h"
#include "crispasr_model_registry.h"
#include "whisper_params.h"

#include "pocket_tts.h"

#include "core/audio_resample.h"
#include "core/wav_reader.h"

#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

static bool ends_with_ci(const std::string& s, const std::string& suffix) {
    if (s.size() < suffix.size())
        return false;
    for (size_t i = 0; i < suffix.size(); ++i) {
        if (std::tolower((unsigned char)s[s.size() - suffix.size() + i]) != std::tolower((unsigned char)suffix[i]))
            return false;
    }
    return true;
}

static bool file_exists(const std::string& path) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f)
        return false;
    fclose(f);
    return true;
}

// Resolve `--voice` for pocket-tts. Accept reference WAVs and Kyutai's official
// prepared `.safetensors` embeddings. A bare name from a server request
// (`{"voice": "alba"}`) resolves against --voice-dir, preferring the official
// embedding and then a locally recorded WAV. Mirrors the
// bare-name resolution in the vibevoice / qwen3-tts adapters. Absolute paths and
// names already carrying a `.wav` extension are passed through untouched.
static std::string resolve_pocket_voice_path(const whisper_params& params) {
    std::string voice_path = params.tts_voice;
    if (voice_path.empty() || params.tts_voice_dir.empty())
        return voice_path;
    // Only rewrite a bare token: no path separators, no explicit .wav extension.
    if (voice_path.find('/') != std::string::npos || voice_path.find('\\') != std::string::npos ||
        ends_with_ci(voice_path, ".wav") || ends_with_ci(voice_path, ".safetensors")) {
        return voice_path;
    }
    // Path-traversal sanitisation (the server already guards network requests,
    // but a direct CLI caller does not).
    if (voice_path.find("..") != std::string::npos || voice_path.find('\0') != std::string::npos) {
        fprintf(stderr, "crispasr[pocket-tts]: voice name '%s' contains illegal characters (.. or NUL)\n",
                voice_path.c_str());
        return voice_path;
    }
    const std::string embedding_path = params.tts_voice_dir + "/" + voice_path + ".safetensors";
    if (file_exists(embedding_path))
        return embedding_path;
    const std::string wav_path = params.tts_voice_dir + "/" + voice_path + ".wav";
    if (file_exists(wav_path))
        return wav_path;
    fprintf(stderr, "crispasr[pocket-tts]: warning: voice '%s' not found at '%s'\n", voice_path.c_str(),
            wav_path.c_str());
    return voice_path; // leave as-is; the WAV reader below emits a clear error.
}

class PocketTTSBackend : public CrispasrBackend {
public:
    PocketTTSBackend() = default;
    ~PocketTTSBackend() override { PocketTTSBackend::shutdown(); }

    const char* name() const override { return "pocket-tts"; }

    // Own --voice-dir resolution (prefers a .safetensors embedding over a .wav); skip the shared #584 step.
    bool resolves_voice_dir_itself() const override { return true; }

    uint32_t capabilities() const override { return CAP_TTS | CAP_AUTO_DOWNLOAD | CAP_TEMPERATURE | CAP_VOICE_CLONING; }

    int tts_sample_rate() const override { return 24000; }

    std::vector<crispasr_segment> transcribe(const float* /*samples*/, int /*n_samples*/, int64_t /*t_offset_cs*/,
                                             const whisper_params& /*params*/) override {
        fprintf(stderr, "crispasr[pocket-tts]: transcription is not supported by this backend\n");
        return {};
    }

    bool init(const whisper_params& p) override {
        pocket_tts_context_params cp = pocket_tts_context_default_params();
        cp.n_threads = p.n_threads;
        cp.verbosity = p.no_prints ? 0 : 1;
        cp.use_gpu = crispasr_backend_should_use_gpu(p);
        cp.seed = p.seed;

        // The CLI's global --temperature defaults to 0.0 (whisper-style
        // greedy ASR). For pocket-tts, the default is 0.7; only override
        // when the user explicitly passes a non-zero value.
        if (p.temperature > 0.0f) {
            cp.temperature = p.temperature;
        }

        std::string model_path = p.model;
        model_path = crispasr_resolve_model_cli(model_path, p.backend, p.no_prints, p.cache_dir, p.auto_download, "");

        ctx_ = pocket_tts_init_from_file(model_path.c_str(), cp);
        if (!ctx_) {
            fprintf(stderr, "crispasr[pocket-tts]: failed to load model '%s'\n", model_path.c_str());
            return false;
        }

        // Pocket TTS produces near-silence without voice conditioning.
        // Auto-load samples/jfk.wav as default voice if --voice not specified.
        if (p.tts_voice.empty()) {
            const char* fallbacks[] = {"samples/jfk.wav", "../../samples/jfk.wav", nullptr};
            bool loaded = false;
            for (const char** fb = fallbacks; *fb; fb++) {
                FILE* f = fopen(*fb, "rb");
                if (f) {
                    fclose(f);
                    std::vector<float> ref_pcm;
                    int ref_sr = 0;
                    if (crispasr::core::read_wav_mono_pcm16(*fb, ref_pcm, ref_sr) && !ref_pcm.empty()) {
                        if (ref_sr != 24000)
                            ref_pcm =
                                core_audio::resample_polyphase(ref_pcm.data(), (int)ref_pcm.size(), ref_sr, 24000);
                        if (pocket_tts_set_voice(ctx_, ref_pcm.data(), (int)ref_pcm.size()) == 0) {
                            if (!p.no_prints)
                                fprintf(stderr, "crispasr[pocket-tts]: auto-loaded voice from '%s'\n", *fb);
                            loaded = true;
                            break;
                        }
                    }
                }
            }
            if (!loaded && !p.no_prints) {
                fprintf(stderr, "crispasr[pocket-tts]: WARNING: no --voice specified; "
                                "output will be near-silent without voice conditioning\n");
            }
        }

        return true;
    }

    std::vector<float> synthesize(const std::string& text, const whisper_params& params) override {
        if (!ctx_ || text.empty()) {
            return {};
        }

        // Apply runtime parameter overrides
        if (params.temperature > 0.0f) {
            pocket_tts_set_temperature(ctx_, params.temperature);
        }
        pocket_tts_set_seed(ctx_, params.seed);

        // Load voice conditioning if --voice points to a WAV file
        // (voice cloning requires encoder weights in the GGUF). A bare name is
        // resolved against --voice-dir so `--server`/`{"voice": "<name>"}` and
        // multi-voice CLI runs work (issue #255). Cache on the resolved path so
        // repeated requests with the same voice don't re-load the reference.
        std::string voice_path = resolve_pocket_voice_path(params);
        if (!voice_path.empty() && voice_path != last_voice_key_) {
            if (ends_with_ci(voice_path, ".safetensors")) {
                int rc = pocket_tts_load_voice_embedding(ctx_, voice_path.c_str());
                if (rc != 0) {
                    if (!params.no_prints)
                        fprintf(stderr, "crispasr[pocket-tts]: prepared voice embedding failed (rc=%d)\n", rc);
                    return {};
                }
            } else {
                std::vector<float> ref_pcm;
                int ref_sr = 0;
                if (!crispasr::core::read_wav_mono_pcm16(voice_path, ref_pcm, ref_sr)) {
                    fprintf(stderr, "crispasr[pocket-tts]: failed to read voice reference '%s'\n", voice_path.c_str());
                    last_voice_key_ = voice_path;
                    return {};
                }
                // Resample to 24 kHz if needed
                if (ref_sr != 24000) {
                    ref_pcm = core_audio::resample_polyphase(ref_pcm.data(), (int)ref_pcm.size(), ref_sr, 24000);
                }
                int rc = pocket_tts_set_voice(ctx_, ref_pcm.data(), (int)ref_pcm.size());
                if (rc != 0 && !params.no_prints) {
                    fprintf(stderr, "crispasr[pocket-tts]: voice cloning failed (rc=%d)\n", rc);
                }
            }
            last_voice_key_ = voice_path;
        }

        int n = 0;
        float* pcm = pocket_tts_synthesize(ctx_, text.c_str(), &n);
        if (!pcm || n <= 0) {
            return {};
        }
        std::vector<float> out(pcm, pcm + n);
        pocket_tts_pcm_free(pcm);
        return out;
    }

    void shutdown() override {
        if (ctx_) {
            pocket_tts_free(ctx_);
            ctx_ = nullptr;
        }
        last_voice_key_.clear();
    }

private:
    pocket_tts_context* ctx_ = nullptr;
    std::string last_voice_key_;
};

} // namespace

std::unique_ptr<CrispasrBackend> crispasr_make_pocket_tts_backend() {
    return std::unique_ptr<CrispasrBackend>(new PocketTTSBackend());
}
