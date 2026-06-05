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

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

class PocketTTSBackend : public CrispasrBackend {
public:
    PocketTTSBackend() = default;
    ~PocketTTSBackend() override { PocketTTSBackend::shutdown(); }

    const char* name() const override { return "pocket-tts"; }

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
        // (voice cloning requires encoder weights in the GGUF)
        if (!params.tts_voice.empty() && params.tts_voice != last_voice_key_) {
            std::vector<float> ref_pcm;
            int ref_sr = 0;
            if (!crispasr::core::read_wav_mono_pcm16(params.tts_voice, ref_pcm, ref_sr)) {
                fprintf(stderr, "crispasr[pocket-tts]: failed to read voice reference '%s'\n",
                        params.tts_voice.c_str());
            } else {
                // Resample to 24 kHz if needed
                if (ref_sr != 24000) {
                    ref_pcm = core_audio::resample_polyphase(ref_pcm.data(), (int)ref_pcm.size(), ref_sr, 24000);
                }
                int rc = pocket_tts_set_voice(ctx_, ref_pcm.data(), (int)ref_pcm.size());
                if (rc != 0 && !params.no_prints) {
                    fprintf(stderr, "crispasr[pocket-tts]: voice cloning failed (rc=%d)\n", rc);
                }
            }
            last_voice_key_ = params.tts_voice;
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
