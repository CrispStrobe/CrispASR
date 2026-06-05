// crispasr_backend_f5_tts.cpp — adapter for SWivid/F5-TTS.
//
// Single-GGUF runtime: DiT + Vocos in one file. Voice cloning via
// --voice <ref.wav> --ref-text "transcript of ref audio".
// When --ref-text is omitted, the reference audio is automatically
// transcribed using whisper (tiny.en) to estimate the ref transcript.
// No codec model needed (Vocos vocoder is part of the GGUF).

#include "crispasr_backend.h"
#include "crispasr_backend_utils.h"
#include "crispasr_model_mgr_cli.h"
#include "crispasr_model_registry.h"
#include "whisper_params.h"

#include "f5_tts.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

// Minimal WAV reader → mono float PCM at original sample rate.
// Returns empty on failure. Sets *out_sr to the file's sample rate.
static std::vector<float> read_wav_mono(const std::string& path, int* out_sr) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f)
        return {};
    char riff[4];
    fread(riff, 1, 4, f);
    if (memcmp(riff, "RIFF", 4) != 0) {
        fclose(f);
        return {};
    }
    fseek(f, 8, SEEK_SET); // skip file size
    char wave[4];
    fread(wave, 1, 4, f);
    if (memcmp(wave, "WAVE", 4) != 0) {
        fclose(f);
        return {};
    }
    // Find fmt and data chunks
    int sr = 0, bits = 0, channels = 0;
    std::vector<float> pcm;
    while (!feof(f)) {
        char id[4];
        if (fread(id, 1, 4, f) != 4)
            break;
        uint32_t sz;
        if (fread(&sz, 4, 1, f) != 1)
            break;
        if (memcmp(id, "fmt ", 4) == 0) {
            uint16_t fmt;
            fread(&fmt, 2, 1, f);
            uint16_t ch;
            fread(&ch, 2, 1, f);
            channels = ch;
            uint32_t s;
            fread(&s, 4, 1, f);
            sr = (int)s;
            fseek(f, 6, SEEK_CUR); // byte rate + block align
            uint16_t b;
            fread(&b, 2, 1, f);
            bits = b;
            if (sz > 16)
                fseek(f, sz - 16, SEEK_CUR);
        } else if (memcmp(id, "data", 4) == 0) {
            int n_samples = (int)(sz / (bits / 8) / channels);
            pcm.resize(n_samples);
            for (int i = 0; i < n_samples; i++) {
                float sum = 0;
                for (int c = 0; c < channels; c++) {
                    if (bits == 16) {
                        int16_t v;
                        fread(&v, 2, 1, f);
                        sum += (float)v / 32768.0f;
                    } else if (bits == 32) {
                        int32_t v;
                        fread(&v, 4, 1, f);
                        sum += (float)v / 2147483648.0f;
                    }
                }
                pcm[i] = sum / (float)channels;
            }
            break;
        } else {
            fseek(f, sz, SEEK_CUR);
        }
    }
    fclose(f);
    *out_sr = sr;
    return pcm;
}

// Linear-interpolation resample
static std::vector<float> resample_linear(const std::vector<float>& in, int sr_in, int sr_out) {
    if (sr_in == sr_out)
        return in;
    int n_out = (int)((float)in.size() * (float)sr_out / (float)sr_in);
    std::vector<float> out(n_out);
    for (int i = 0; i < n_out; i++) {
        float pos = (float)i * (float)sr_in / (float)sr_out;
        int idx = (int)pos;
        float frac = pos - (float)idx;
        if (idx + 1 < (int)in.size())
            out[i] = in[idx] * (1.0f - frac) + in[idx + 1] * frac;
        else if (idx < (int)in.size())
            out[i] = in[idx];
    }
    return out;
}

// Auto-transcribe reference audio using any CrispASR backend.
// The backend name defaults to "whisper" but can be overridden via
// --ref-asr (e.g. --ref-asr parakeet, --ref-asr moonshine).
// The model is resolved via the registry (auto-download if needed).
static std::string transcribe_ref_audio(const std::vector<float>& pcm_16k, const whisper_params& p,
                                        const std::string& asr_backend) {
    auto backend = crispasr_create_backend(asr_backend);
    if (!backend) {
        if (!p.no_prints)
            fprintf(stderr, "crispasr[f5-tts]: unknown ASR backend '%s' for ref-text transcription\n",
                    asr_backend.c_str());
        return "";
    }

    // Build minimal params for the ASR backend — just model + threads.
    // Resolve the model path via the registry (handles auto-download).
    whisper_params asr_p = {};
    asr_p.n_threads = p.n_threads;
    asr_p.no_prints = true; // suppress ASR model load chatter
    asr_p.language = p.language.empty() ? "en" : p.language;
    asr_p.auto_download = true;
    asr_p.model = crispasr_resolve_model_cli("auto", asr_backend, /*quiet=*/true,
                                             /*cache_dir=*/"", /*auto_download=*/true);

    if (!backend->init(asr_p)) {
        if (!p.no_prints)
            fprintf(stderr, "crispasr[f5-tts]: failed to init '%s' for ref-text transcription\n", asr_backend.c_str());
        return "";
    }

    auto segments = backend->transcribe(pcm_16k.data(), (int)pcm_16k.size(), 0, asr_p);
    backend->shutdown();

    std::string result;
    for (const auto& seg : segments) {
        if (!result.empty())
            result += " ";
        result += seg.text;
    }

    // Trim leading/trailing whitespace
    size_t start = result.find_first_not_of(" \t\n\r");
    size_t end = result.find_last_not_of(" \t\n\r");
    if (start == std::string::npos)
        return "";
    return result.substr(start, end - start + 1);
}

class F5TtsBackend : public CrispasrBackend {
public:
    ~F5TtsBackend() override { F5TtsBackend::shutdown(); }

    const char* name() const override { return "f5-tts"; }

    uint32_t capabilities() const override { return CAP_TTS | CAP_VOICE_CLONING | CAP_AUTO_DOWNLOAD; }

    std::vector<crispasr_segment> transcribe(const float* /*samples*/, int /*n_samples*/, int64_t /*t_offset_cs*/,
                                             const whisper_params& /*params*/) override {
        fprintf(stderr, "crispasr[f5-tts]: transcription is not supported by this backend\n");
        return {};
    }

    bool init(const whisper_params& p) override {
        f5_tts_params fp = f5_tts_default_params();
        fp.n_threads = p.n_threads;
        fp.verbosity = p.no_prints ? 0 : 1;
        fp.seed = p.seed;

        ctx_ = f5_tts_init_from_file(p.model.c_str(), fp);
        if (!ctx_) {
            fprintf(stderr, "crispasr[f5-tts]: failed to load model '%s'\n", p.model.c_str());
            return false;
        }

        // Load reference audio for voice cloning
        if (!p.tts_voice.empty()) {
            int wav_sr = 0;
            auto ref_pcm = read_wav_mono(p.tts_voice, &wav_sr);
            if (ref_pcm.empty() || wav_sr <= 0) {
                fprintf(stderr, "crispasr[f5-tts]: failed to load reference audio '%s'\n", p.tts_voice.c_str());
                return false;
            }

            // Resample to 24kHz
            auto ref_24k = resample_linear(ref_pcm, wav_sr, 24000);

            // RMS normalize to 0.1 (matching Python reference)
            float rms = 0.0f;
            for (float s : ref_24k)
                rms += s * s;
            rms = sqrtf(rms / (float)ref_24k.size());
            if (rms < 0.1f && rms > 1e-10f) {
                float scale = 0.1f / rms;
                for (float& s : ref_24k)
                    s *= scale;
            }

            // Auto-transcribe reference audio when --ref-text is missing.
            // F5-TTS needs the ref transcript to estimate speech rate for
            // duration calculation; without it, output length is wrong.
            std::string ref_text_str = p.tts_ref_text;
            if (ref_text_str.empty()) {
                // Resample ref to 16kHz for whisper
                auto ref_16k = resample_linear(ref_pcm, wav_sr, 16000);
                std::string asr_name = p.tts_ref_asr.empty() ? "whisper" : p.tts_ref_asr;
                if (!p.no_prints) {
                    fprintf(stderr, "crispasr[f5-tts]: --ref-text not set, auto-transcribing via %s...\n",
                            asr_name.c_str());
                }
                ref_text_str = transcribe_ref_audio(ref_16k, p, asr_name);
                if (ref_text_str.empty()) {
                    if (!p.no_prints) {
                        fprintf(stderr, "crispasr[f5-tts]: auto-transcription returned empty; "
                                        "duration estimate may be inaccurate\n");
                    }
                } else if (!p.no_prints) {
                    fprintf(stderr, "crispasr[f5-tts]: auto-transcribed ref: '%s'\n", ref_text_str.c_str());
                }
            }

            const char* ref_text = ref_text_str.empty() ? "" : ref_text_str.c_str();
            if (f5_tts_set_reference(ctx_, ref_24k.data(), (int)ref_24k.size(), ref_text) != 0) {
                fprintf(stderr, "crispasr[f5-tts]: failed to set reference audio\n");
                return false;
            }

            if (!p.no_prints) {
                fprintf(stderr, "crispasr[f5-tts]: loaded ref audio '%s' (%d@%dHz → %d@24kHz) ref_text='%s'\n",
                        p.tts_voice.c_str(), (int)ref_pcm.size(), wav_sr, (int)ref_24k.size(), ref_text);
            }
        }

        return true;
    }

    std::vector<float> synthesize(const std::string& text, const whisper_params& p) override {
        if (!ctx_)
            return {};

        // Update runtime params
        if (p.seed > 0)
            f5_tts_set_seed(ctx_, p.seed);
        if (p.tts_speed > 0.0f)
            f5_tts_set_speed(ctx_, p.tts_speed);
        // 75c-opt-2: native f5 knobs
        if (p.tts_num_steps >= 0)
            f5_tts_set_ode_steps(ctx_, p.tts_num_steps);
        if (p.tts_cfg_scale >= 0.0f)
            f5_tts_set_cfg_strength(ctx_, p.tts_cfg_scale);

        float* pcm = nullptr;
        int sr = 0;
        int n = f5_tts_synthesize(ctx_, text.c_str(), &pcm, &sr);
        if (n <= 0 || !pcm)
            return {};

        std::vector<float> result(pcm, pcm + n);
        free(pcm);
        return result;
    }

    int tts_sample_rate() const override { return ctx_ ? f5_tts_sample_rate(ctx_) : 24000; }

    void shutdown() override {
        if (ctx_) {
            f5_tts_free(ctx_);
            ctx_ = nullptr;
        }
    }

private:
    f5_tts_context* ctx_ = nullptr;
};

} // namespace

// ── Factory registration ────────────────────────────────────────────

std::unique_ptr<CrispasrBackend> crispasr_make_f5_tts_backend() {
    return std::make_unique<F5TtsBackend>();
}
