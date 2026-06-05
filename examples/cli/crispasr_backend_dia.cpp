// crispasr_backend_dia.cpp — adapter for nari-labs/Dia-1.6B
// (Llama-style text encoder + AR decoder with cross-attention + DAC codec).
//
// Dialogue-style TTS with [S1]/[S2] speaker tags. 44.1 kHz output.
// The DAC codec can be embedded in the main GGUF or loaded from a
// separate file (--codec-model), reusing the same DAC weights as
// the Zonos port.

#include "crispasr_backend.h"
#include "crispasr_backend_utils.h"
#include "crispasr_model_mgr_cli.h"
#include "crispasr_model_registry.h"
#include "whisper_params.h"

#include "dia_tts.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <sys/stat.h>
#include <vector>

namespace {

static bool file_exists(const std::string& path) {
    struct stat st;
    return stat(path.c_str(), &st) == 0;
}

// Look for a sibling DAC codec file next to the main model.
static std::string discover_dac_codec(const std::string& model_path) {
    auto dir_of = [](const std::string& p) -> std::string {
        auto sep = p.find_last_of("/\\");
        return (sep == std::string::npos) ? std::string(".") : p.substr(0, sep);
    };
    const std::string dir = dir_of(model_path);
    static const char* candidates[] = {
        "dac-44khz.gguf",
        "dac_44khz.gguf",
        "dia-dac-44khz.gguf",
        "dac-decoder.gguf",
    };
    for (const char* name : candidates) {
        std::string p = dir + "/" + name;
        if (file_exists(p)) {
            return p;
        }
    }
    return "";
}

class DiaBackend : public CrispasrBackend {
public:
    DiaBackend() = default;
    ~DiaBackend() override { DiaBackend::shutdown(); }

    const char* name() const override { return "dia"; }

    uint32_t capabilities() const override { return CAP_TTS | CAP_AUTO_DOWNLOAD | CAP_TEMPERATURE; }

    int tts_sample_rate() const override { return 44100; }

    std::vector<crispasr_segment> transcribe(const float* /*samples*/, int /*n_samples*/, int64_t /*t_offset_cs*/,
                                             const whisper_params& /*params*/) override {
        fprintf(stderr, "crispasr[dia]: transcription is not supported by this backend\n");
        return {};
    }

    bool init(const whisper_params& p) override {
        dia_tts_context_params cp = dia_tts_context_default_params();
        cp.n_threads = p.n_threads;
        cp.verbosity = p.no_prints ? 0 : 1;
        cp.use_gpu = crispasr_backend_should_use_gpu(p);
        cp.seed = p.seed;

        // Temperature: Dia default is 1.2 (unlike whisper's 0.0).
        // Only override if the user explicitly sets a non-zero value.
        if (p.temperature > 0.0f) {
            cp.temperature = p.temperature;
        }

        ctx_ = dia_tts_init_from_file(p.model.c_str(), cp);
        if (!ctx_) {
            fprintf(stderr, "crispasr[dia]: failed to load model '%s'\n", p.model.c_str());
            return false;
        }

        // DAC codec: check if embedded in main GGUF, otherwise look for separate file
        std::string codec_path = p.tts_codec_model;
        if (!codec_path.empty() && codec_path != "auto" && codec_path != "default") {
            codec_path = crispasr_resolve_model_cli(codec_path, p.backend, p.no_prints, p.cache_dir, p.auto_download,
                                                    p.tts_codec_quant);
        } else {
            codec_path.clear();
        }
        if (codec_path.empty()) {
            codec_path = discover_dac_codec(p.model);
        }
        if (codec_path.empty()) {
            CrispasrRegistryEntry entry;
            if (crispasr_registry_lookup(p.backend, entry, p.tts_codec_quant) && !entry.companion_filename.empty()) {
                codec_path = crispasr_resolve_model_cli(entry.companion_filename, p.backend, p.no_prints, p.cache_dir,
                                                        p.auto_download, p.tts_codec_quant);
            }
        }
        if (!codec_path.empty()) {
            dia_tts_set_codec_path(ctx_, codec_path.c_str());
            if (!p.no_prints) {
                fprintf(stderr, "crispasr[dia]: DAC codec path = '%s'\n", codec_path.c_str());
            }
        } else if (!p.no_prints) {
            fprintf(stderr, "crispasr[dia]: no separate DAC codec found (expecting embedded in main GGUF)\n");
        }

        return true;
    }

    std::vector<float> synthesize(const std::string& text, const whisper_params& params) override {
        if (!ctx_ || text.empty()) {
            return {};
        }

        if (params.temperature > 0.0f) {
            dia_tts_set_temperature(ctx_, params.temperature);
        }
        dia_tts_set_seed(ctx_, params.seed);

        int n = 0;
        float* pcm = dia_tts_synthesize(ctx_, text.c_str(), &n);
        if (!pcm || n <= 0) {
            return {};
        }
        std::vector<float> out(pcm, pcm + n);
        dia_tts_pcm_free(pcm);
        return out;
    }

    void shutdown() override {
        if (ctx_) {
            dia_tts_free(ctx_);
            ctx_ = nullptr;
        }
    }

private:
    dia_tts_context* ctx_ = nullptr;
};

} // namespace

std::unique_ptr<CrispasrBackend> crispasr_make_dia_backend() {
    return std::unique_ptr<CrispasrBackend>(new DiaBackend());
}
