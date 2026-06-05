// crispasr_backend_melotts.cpp — MeloTTS (VITS2) backend adapter.
// With OpenVoice2 Tone Color Converter for voice cloning via --voice.

#include "crispasr_backend.h"
#include "crispasr_backend_utils.h"
#include "melotts.h"
#include "openvoice2.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// Read a mono WAV file into float PCM + sample rate.
// Handles WAVs with extra chunks before the data chunk.
static bool read_wav_mono(const char* path, std::vector<float>& pcm, int& sr) {
    FILE* f = fopen(path, "rb");
    if (!f)
        return false;

    // Read RIFF header (12 bytes)
    uint8_t riff[12];
    if (fread(riff, 1, 12, f) != 12) {
        fclose(f);
        return false;
    }
    if (memcmp(riff, "RIFF", 4) != 0 || memcmp(riff + 8, "WAVE", 4) != 0) {
        fclose(f);
        return false;
    }

    int bits = 16;
    sr = 16000;

    // Walk chunks until we find "data"
    while (!feof(f)) {
        uint8_t chunk_hdr[8];
        if (fread(chunk_hdr, 1, 8, f) != 8)
            break;
        uint32_t chunk_size = *(uint32_t*)(chunk_hdr + 4);

        if (memcmp(chunk_hdr, "fmt ", 4) == 0) {
            uint8_t fmt[40] = {};
            size_t to_read = chunk_size < 40 ? chunk_size : 40;
            if (fread(fmt, 1, to_read, f) != to_read) {
                fclose(f);
                return false;
            }
            sr = *(int32_t*)(fmt + 4);
            bits = *(int16_t*)(fmt + 14);
            if (chunk_size > to_read)
                fseek(f, (long)(chunk_size - to_read), SEEK_CUR);
        } else if (memcmp(chunk_hdr, "data", 4) == 0) {
            int n_samples = (int)(chunk_size / (bits / 8));
            pcm.resize(n_samples);
            if (bits == 16) {
                std::vector<int16_t> buf(n_samples);
                size_t got = fread(buf.data(), 2, n_samples, f);
                pcm.resize(got);
                for (size_t i = 0; i < got; i++)
                    pcm[i] = buf[i] / 32768.0f;
            } else if (bits == 32) {
                size_t got = fread(pcm.data(), 4, n_samples, f);
                pcm.resize(got);
            }
            fclose(f);
            return !pcm.empty();
        } else {
            // Skip unknown chunk
            fseek(f, (long)chunk_size, SEEK_CUR);
        }
    }

    fclose(f);
    return false;
}

class MelottsBackend : public CrispasrBackend {
public:
    const char* name() const override { return "melotts"; }

    uint32_t capabilities() const override { return CAP_TTS | (ov2_ctx_ ? CAP_VOICE_CLONING : 0); }

    bool init(const whisper_params& p) override {
        struct melotts_params mp = melotts_default_params();
        mp.n_threads = p.n_threads;
        mp.verbosity = p.no_prints ? 0 : 1;
        mp.use_gpu = crispasr_backend_should_use_gpu(p);
        mp.speaker_id = p.tts_speaker_id;
        if (p.seed > 0)
            mp.seed = (uint32_t)p.seed;

        ctx_ = melotts_init_from_file(p.model.c_str(), mp);
        if (!ctx_)
            return false;

        // Load BERT companion model for contextual conditioning.
        // Sources (in priority order):
        //   1. --codec-model <path>  (explicit)
        //   2. MELOTTS_BERT env var
        //   3. bert-base-uncased.gguf next to the melotts model
        //   4. Auto-download from registry companion
        {
            std::string bert_path;
            if (!p.tts_codec_model.empty()) {
                bert_path = p.tts_codec_model;
            } else {
                const char* env = std::getenv("MELOTTS_BERT");
                if (env && *env) {
                    bert_path = env;
                } else {
                    // Look next to the melotts model
                    std::string model_dir = p.model;
                    size_t sep = model_dir.find_last_of("/\\");
                    if (sep != std::string::npos)
                        model_dir.resize(sep + 1);
                    else
                        model_dir = "./";
                    std::string candidate = model_dir + "bert-base-uncased.gguf";
                    FILE* test = fopen(candidate.c_str(), "rb");
                    if (test) {
                        fclose(test);
                        bert_path = candidate;
                    }
                    // Also try the cache dir
                    if (bert_path.empty() && !p.cache_dir.empty()) {
                        candidate = p.cache_dir + "/bert-base-uncased.gguf";
                        test = fopen(candidate.c_str(), "rb");
                        if (test) {
                            fclose(test);
                            bert_path = candidate;
                        }
                    }
                }
            }
            if (!bert_path.empty()) {
                if (melotts_load_bert(ctx_, bert_path.c_str())) {
                    if (!p.no_prints)
                        fprintf(stderr, "melotts: BERT conditioning enabled\n");
                } else if (!p.no_prints) {
                    fprintf(stderr, "melotts: BERT model not loaded (quality will be reduced)\n");
                }
            }
        }

        // Try to load OpenVoice2 TCC model for voice cloning.
        // Look for openvoice2-tcc-*.gguf next to the melotts model.
        if (!p.tts_voice.empty()) {
            std::string model_dir = p.model;
            size_t slash = model_dir.find_last_of("/\\");
            if (slash != std::string::npos)
                model_dir.resize(slash + 1);
            else
                model_dir = "./";

            // Try common names
            for (const char* name : {"openvoice2-tcc-f16.gguf", "openvoice2-tcc-q4_k.gguf", "openvoice2-tcc.gguf"}) {
                std::string tcc_path = model_dir + name;
                FILE* test = fopen(tcc_path.c_str(), "rb");
                if (test) {
                    fclose(test);
                    auto cp = openvoice2_context_default_params();
                    cp.n_threads = p.n_threads;
                    cp.verbosity = p.no_prints ? 0 : 1;
                    ov2_ctx_ = openvoice2_init_from_file(tcc_path.c_str(), cp);
                    if (ov2_ctx_) {
                        const char* dd = std::getenv("OV2_DUMP_DIR");
                        if (dd)
                            openvoice2_set_dump_dir(ov2_ctx_, dd);
                        if (!p.no_prints)
                            fprintf(stderr, "melotts: OpenVoice2 TCC loaded from '%s'\n", tcc_path.c_str());
                    }
                    break;
                }
            }

            if (!ov2_ctx_) {
                fprintf(stderr, "melotts: warning: --voice specified but no OpenVoice2 TCC "
                                "model found. Place openvoice2-tcc-f16.gguf next to the melotts model.\n");
            }

            // Load reference audio
            int ref_sr = 0;
            if (read_wav_mono(p.tts_voice.c_str(), ref_pcm_, ref_sr)) {
                ref_sr_ = ref_sr;
                if (!p.no_prints)
                    fprintf(stderr, "melotts: loaded ref voice '%s' (%d samples @ %d Hz)\n", p.tts_voice.c_str(),
                            (int)ref_pcm_.size(), ref_sr_);
            } else {
                fprintf(stderr, "melotts: failed to read voice reference '%s'\n", p.tts_voice.c_str());
            }
        }

        return true;
    }

    std::vector<float> synthesize(const std::string& text, const whisper_params& params) override {
        if (!ctx_)
            return {};

        // Apply runtime params
        if (params.tts_speed > 0) {
            melotts_set_length_scale(ctx_, 1.0f / params.tts_speed);
        }

        float* pcm = nullptr;
        int n = melotts_synthesize(ctx_, text.c_str(), &pcm, nullptr);
        if (!pcm || n <= 0)
            return {};

        std::vector<float> out(pcm, pcm + n);
        melotts_pcm_free(pcm);

        // Voice cloning via OpenVoice2 if reference audio provided
        if (ov2_ctx_ && !ref_pcm_.empty()) {
            int src_sr = ctx_ ? melotts_sample_rate(ctx_) : 44100;
            float* cloned_pcm = nullptr;
            int n_cloned = 0;

            if (openvoice2_convert(ov2_ctx_, out.data(), (int)out.size(), src_sr, ref_pcm_.data(), (int)ref_pcm_.size(),
                                   ref_sr_, &cloned_pcm, &n_cloned)) {
                out.assign(cloned_pcm, cloned_pcm + n_cloned);
                free(cloned_pcm);
                // Output sample rate changes to OpenVoice2's rate (22050)
                ov2_output_sr_ = 22050;
            } else {
                fprintf(stderr, "melotts: OpenVoice2 voice conversion failed, "
                                "returning original audio\n");
            }
        }

        return out;
    }

    int tts_sample_rate() const override {
        if (ov2_output_sr_ > 0)
            return ov2_output_sr_;
        return ctx_ ? melotts_sample_rate(ctx_) : 44100;
    }

    std::vector<crispasr_segment> transcribe(const float*, int, int64_t, const whisper_params&) override {
        return {}; // TTS-only
    }

    void shutdown() override {
        if (ov2_ctx_) {
            openvoice2_free(ov2_ctx_);
            ov2_ctx_ = nullptr;
        }
        if (ctx_) {
            melotts_free(ctx_);
            ctx_ = nullptr;
        }
    }

private:
    melotts_context* ctx_ = nullptr;
    openvoice2_context* ov2_ctx_ = nullptr;
    std::vector<float> ref_pcm_;
    int ref_sr_ = 0;
    int ov2_output_sr_ = 0;
};

std::unique_ptr<CrispasrBackend> crispasr_make_melotts_backend() {
    return std::unique_ptr<CrispasrBackend>(new MelottsBackend());
}
