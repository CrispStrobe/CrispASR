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
#include "core/tts_voice_policy.h"
#include "core/tts_ref_cache.h"
#include "crispasr_tts_ref_text.h"
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
    (void)!fread(riff, 1, 4, f);
    if (memcmp(riff, "RIFF", 4) != 0) {
        fclose(f);
        return {};
    }
    fseek(f, 8, SEEK_SET); // skip file size
    char wave[4];
    (void)!fread(wave, 1, 4, f);
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
            (void)!fread(&fmt, 2, 1, f);
            uint16_t ch;
            (void)!fread(&ch, 2, 1, f);
            channels = ch;
            uint32_t s;
            (void)!fread(&s, 4, 1, f);
            sr = (int)s;
            fseek(f, 6, SEEK_CUR); // byte rate + block align
            uint16_t b;
            (void)!fread(&b, 2, 1, f);
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
                        (void)!fread(&v, 2, 1, f);
                        sum += (float)v / 32768.0f;
                    } else if (bits == 32) {
                        int32_t v;
                        (void)!fread(&v, 4, 1, f);
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
        fp.use_gpu = p.use_gpu;

        ctx_ = f5_tts_init_from_file(p.model.c_str(), fp);
        if (!ctx_) {
            fprintf(stderr, "crispasr[f5-tts]: failed to load model '%s'\n", p.model.c_str());
            return false;
        }

        // Voice cloning is intentionally NOT applied here (at startup): the
        // HTTP server owns ONE backend instance and passes a per-request
        // `voice=` in params.tts_voice, which only takes effect if re-applied
        // at synth time (see prepare_voice below). Baking --voice at init alone
        // made every /v1/audio/speech request serve the startup voice (or the
        // built-in default when no --voice was given).

        return true;
    }

    // Load + encode the reference WAV (voice cloning), re-loading only when the
    // path changes so single-shot and server callers pay the encode once.
    // Mirrors moss-tts prepare_voice: the server passes a per-request `voice=`
    // into params.tts_voice which MUST be consumed at synth time. "default" /
    // "auto" mean "use the built-in reference" (kept as-is); an empty voice
    // also keeps the current reference.
    void prepare_voice(const whisper_params& p) {
        // Decision lives in core/tts_voice_policy.h with tests: dedupe an
        // identical repeat, treat ""/default/auto as the built-in reference, and
        // apply anything else. Exact-match sentinels on purpose — a file really
        // can be called Default.wav.
        const auto action = core_tts_voice::decide(last_voice_, p.tts_voice);
        if (action == core_tts_voice::Action::Unchanged)
            return;
        last_voice_ = p.tts_voice;
        if (action == core_tts_voice::Action::Builtin)
            return; // keep f5's built-in reference

        int wav_sr = 0;
        auto ref_pcm = read_wav_mono(p.tts_voice, &wav_sr);
        if (ref_pcm.empty() || wav_sr <= 0) {
            fprintf(stderr, "crispasr[f5-tts]: failed to load reference audio '%s'\n", p.tts_voice.c_str());
            last_voice_.clear();
            return;
        }

        // Resample to the MODEL's mel sample rate (24 kHz for stock F5/Vocos,
        // 16 kHz for Raon sbhifigan16k — #387). The mel front-end uses
        // hp.sample_rate, so the reference PCM must match it or the mel is
        // computed on wrongly-rated audio.
        const int model_sr = f5_tts_sample_rate(ctx_);
        auto ref_24k = resample_linear(ref_pcm, wav_sr, model_sr);

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
            // The reference transcript is stable per voice clip, and
            // auto-transcription loads + runs a whole ASR model. Cache it
            // next to the voice as "<voice>.f5reftext" so later runs skip
            // Whisper entirely. Disable with CRISPASR_TTS_REF_CACHE=0.
            const std::string cache_path = crispasr_ref_cache::path_for(p.tts_voice, ".f5reftext");
            const bool cache_enabled = !crispasr_ref_cache::disabled();
            std::vector<uint32_t> shape;
            std::vector<uint8_t> payload;
            if (cache_enabled && crispasr_ref_cache::load(cache_path, p.tts_voice, "f5-reftext", shape, payload)) {
                ref_text_str.assign((const char*)payload.data(), payload.size());
                if (!p.no_prints) {
                    fprintf(stderr, "crispasr[f5-tts]: using cached ref transcript '%s': '%s'\n", cache_path.c_str(),
                            ref_text_str.c_str());
                }
            } else {
                // Resample ref to 16kHz for whisper
                auto ref_16k = resample_linear(ref_pcm, wav_sr, 16000);
                std::string asr_name = p.tts_ref_asr.empty() ? "whisper" : p.tts_ref_asr;
                if (!p.no_prints) {
                    fprintf(stderr, "crispasr[f5-tts]: --ref-text not set, auto-transcribing via %s...\n",
                            asr_name.c_str());
                }
                ref_text_str = crispasr_ref_text::transcribe(ref_16k, p, asr_name, "crispasr[f5-tts]");
                if (cache_enabled && !ref_text_str.empty()) {
                    crispasr_ref_cache::save(cache_path, "f5-reftext", {(uint32_t)ref_text_str.size()},
                                             ref_text_str.data(), ref_text_str.size());
                }
            }
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
            last_voice_.clear();
            return;
        }

        if (!p.no_prints) {
            fprintf(stderr, "crispasr[f5-tts]: loaded ref audio '%s' (%d@%dHz → %d@%dHz) ref_text='%s'\n",
                    p.tts_voice.c_str(), (int)ref_pcm.size(), wav_sr, (int)ref_24k.size(), model_sr, ref_text);
        }
    }

    std::vector<float> synthesize(const std::string& text, const whisper_params& p) override {
        if (!ctx_)
            return {};

        // Per-request voice cloning: the server copies the request's `voice`
        // into params.tts_voice, so (re)apply the reference here whenever it
        // changes. Sentinels/empty keep the built-in reference.
        prepare_voice(p);

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
    std::string last_voice_; // cache key for the loaded reference voice
};

} // namespace

// ── Factory registration ────────────────────────────────────────────

std::unique_ptr<CrispasrBackend> crispasr_make_f5_tts_backend() {
    return std::make_unique<F5TtsBackend>();
}
