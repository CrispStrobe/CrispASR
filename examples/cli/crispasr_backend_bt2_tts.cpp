// crispasr_backend_bt2_tts.cpp — CLI adapter for Breeze TTS 2 (#412).
//
// Backend key: `bt2-tts`. It is deliberately NOT led by "Breeze": §4 of the
// BreezeBlue Research and Non-Commercial License Agreement bars the licensor's
// marks as the PRIMARY name of a derivative, while descriptive attribution —
// this comment, the registry description, the model card — is permitted.
//
// ⚠ The WEIGHTS are non-commercial. The registry entry carries the licence
// prose, so crispasr_license_requires_acceptance() makes -m auto demand
// CRISPASR_ACCEPT_LICENSE before the download.
//
// Capabilities that ship here: plain text-to-speech, and Voice Clone via
// --voice ref.wav (+ --ref-text). Voice Design (--tts-instruct) and Voice
// Direction need 2-3 CFG prompt branches per decode step, which this build
// does not implement; those requests are REFUSED rather than quietly answered
// with an unguided result. See breeze_tts_2_capabilities().
//
// The codec is not in the model file: Breeze's bundled audio tokenizer is
// bit-identical to Qwen3-TTS-Tokenizer-12Hz, which CrispASR already ships, so
// it is resolved as a registry companion exactly like the qwen3-tts talker's.

#include "crispasr_backend.h"
#include "crispasr_backend_utils.h"
#include "crispasr_model_mgr_cli.h"
#include "crispasr_model_registry.h"
#include "crispasr_tts_ref_text.h"
#include "crispasr_voice_provenance.h"
#include "whisper_params.h"

#include "breeze_tts_2.h"
#include "core/audio_resample.h"
#include "core/wav_reader.h"

#include <cstdio>
#include <string>
#include <sys/stat.h>
#include <vector>

namespace {

bool file_exists(const std::string& p) {
    struct stat st;
    return !p.empty() && stat(p.c_str(), &st) == 0;
}

// The codec commonly sits next to the model, under its canonical name.
std::string discover_codec(const std::string& model_path) {
    const size_t slash = model_path.find_last_of("/\\");
    const std::string dir = (slash == std::string::npos) ? std::string(".") : model_path.substr(0, slash);
    for (const char* name : {"qwen3-tts-tokenizer-12hz.gguf", "qwen3-tts-tokenizer-12hz-f16.gguf"}) {
        const std::string cand = dir + "/" + name;
        if (file_exists(cand))
            return cand;
    }
    return {};
}

class Bt2TtsBackend : public CrispasrBackend {
public:
    bool init(const whisper_params& p) override {
        // Resolve the codec BEFORE loading the model: without it the backend
        // can still produce codes, but it cannot produce audio, and finding
        // that out after a multi-second model load is a worse failure.
        std::string codec_path = p.tts_codec_model;
        if (!codec_path.empty() && codec_path != "auto" && codec_path != "default") {
            codec_path = crispasr_resolve_model_cli(codec_path, p.backend, p.no_prints, p.cache_dir, p.auto_download,
                                                    p.tts_codec_quant);
        } else {
            codec_path.clear();
        }
        if (codec_path.empty())
            codec_path = discover_codec(p.model);
        if (codec_path.empty()) {
            CrispasrRegistryEntry entry;
            if (crispasr_registry_lookup(p.backend, entry, p.tts_codec_quant) && !entry.companion_filename.empty()) {
                codec_path = crispasr_resolve_model_cli(entry.companion_filename, p.backend, p.no_prints, p.cache_dir,
                                                        p.auto_download, p.tts_codec_quant);
            }
        }
        if (codec_path.empty()) {
            fprintf(stderr, "crispasr[bt2-tts]: no codec model found. Pass --codec-model PATH or place "
                            "qwen3-tts-tokenizer-12hz.gguf next to the model.\n");
            return false;
        }

        auto cp = breeze_tts_2_context_default_params();
        cp.n_threads = p.n_threads;
        cp.verbosity = p.no_prints ? 0 : (p.verbose ? 2 : 1);
        cp.use_gpu = p.use_gpu; // #292-adjacent: never let this default to false
        if (p.temperature > 0.0f)
            cp.temperature = p.temperature;
        cp.seed = (uint64_t)p.seed;
        cp.max_new_tokens = p.max_new_tokens_explicit ? p.max_new_tokens : 0;
        cp.codec_path = codec_path.c_str();

        ctx_ = breeze_tts_2_init_from_file(p.model.c_str(), cp);
        if (!ctx_) {
            fprintf(stderr, "crispasr[bt2-tts]: failed to load '%s'\n", p.model.c_str());
            return false;
        }
        if (!p.no_prints)
            fprintf(stderr, "crispasr[bt2-tts]: codec = '%s'\n", codec_path.c_str());

        // Voice Clone: --voice ref.wav (+ --ref-text, else auto-transcribed).
        if (!p.tts_voice.empty()) {
            const std::string voice = p.tts_voice_dir.empty()
                                          ? p.tts_voice
                                          : crispasr_voice::resolve_voice_path(p.tts_voice, p.tts_voice_dir);
            std::vector<float> pcm;
            int sr = 0;
            if (!crispasr::core::read_wav_mono_pcm16(voice, pcm, sr) || pcm.empty()) {
                fprintf(stderr, "crispasr[bt2-tts]: cannot read --voice '%s' — ignoring\n", voice.c_str());
            } else {
                // The codec is a 24 kHz model; feeding it anything else
                // silently changes the frame rate of the reference.
                if (sr != 24000)
                    pcm = core_audio::resample_polyphase(pcm.data(), (int)pcm.size(), sr, 24000);
                ref_pcm_ = std::move(pcm);
                ref_text_ = p.tts_ref_text;
                if (ref_text_.empty())
                    ref_text_ = crispasr_ref_text::resolve_cached(voice, ref_pcm_, p, "crispasr[bt2-tts]", ".bt2-ref");
                if (ref_text_.empty()) {
                    // The reference TEXT is not decoration: the clone prompt is
                    // ref_text + ref audio + target text, and a missing
                    // transcript misaligns the audio against the text encoder.
                    fprintf(stderr, "crispasr[bt2-tts]: --voice given but no --ref-text and auto-transcription "
                                    "failed — dropping the reference rather than cloning against an empty "
                                    "transcript\n");
                    ref_pcm_.clear();
                } else if (!p.no_prints) {
                    fprintf(stderr, "crispasr[bt2-tts]: voice = '%s'\n", voice.c_str());
                }
            }
        }

        if (!p.tts_instruct.empty()) {
            fprintf(stderr, "crispasr[bt2-tts]: --tts-instruct (Voice Design/Direction) needs classifier-free guidance "
                            "over 2-3 prompt branches, which this build does not implement. Refusing rather than "
                            "synthesizing an unguided result that would sound fine and ignore the instruction.\n");
            return false;
        }
        return true;
    }

    void shutdown() override {
        if (ctx_) {
            breeze_tts_2_free(ctx_);
            ctx_ = nullptr;
        }
    }

    const char* name() const override { return "bt2-tts"; }
    uint32_t capabilities() const override { return CAP_TTS | CAP_VOICE_CLONING | CAP_AUTO_DOWNLOAD | CAP_TEMPERATURE; }
    int input_sample_rate() const override { return 24000; }
    int tts_sample_rate() const override { return 24000; }

    std::vector<float> synthesize(const std::string& text, const whisper_params& p) override {
        if (!ctx_ || text.empty())
            return {};
        if (p.seed != 0)
            breeze_tts_2_set_seed(ctx_, (uint64_t)p.seed);
        int n = 0;
        float* pcm = ref_pcm_.empty()
                         ? breeze_tts_2_synthesize(ctx_, text.c_str(), &n)
                         : breeze_tts_2_synthesize_with_reference(ctx_, text.c_str(), ref_pcm_.data(),
                                                                  (int)ref_pcm_.size(), ref_text_.c_str(), &n);
        if (!pcm || n <= 0)
            return {};
        std::vector<float> out(pcm, pcm + n);
        breeze_tts_2_pcm_free(pcm);
        return out;
    }

    std::vector<crispasr_segment> transcribe(const float*, int, int64_t, const whisper_params&) override { return {}; }

private:
    breeze_tts_2_context* ctx_ = nullptr;
    std::vector<float> ref_pcm_;
    std::string ref_text_;
};

} // namespace

std::unique_ptr<CrispasrBackend> crispasr_create_bt2_tts_backend() {
    return std::make_unique<Bt2TtsBackend>();
}
