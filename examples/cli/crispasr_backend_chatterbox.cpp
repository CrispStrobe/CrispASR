// crispasr_backend_chatterbox.cpp — adapter for Chatterbox TTS (base, turbo,
// Kartoffelbox). Two-GGUF runtime: T3 AR model (--model) and S3Gen
// encoder+vocoder (--codec-model).

#include "crispasr_backend.h"
#include "crispasr_backend_utils.h"
#include "crispasr_cache.h"
#include "crispasr_model_mgr_cli.h"
#include "crispasr_model_registry.h"
#include "whisper_params.h"

#include "chatterbox.h"

#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/stat.h>
#include <string>
#include <vector>

namespace {

static bool file_exists(const std::string& path) {
    struct stat st;
    return stat(path.c_str(), &st) == 0;
}

static bool contains_ci(const std::string& haystack, const std::string& needle) {
    if (needle.empty() || haystack.size() < needle.size())
        return false;
    for (size_t i = 0; i + needle.size() <= haystack.size(); ++i) {
        bool match = true;
        for (size_t j = 0; j < needle.size(); ++j) {
            unsigned char a = (unsigned char)haystack[i + j];
            unsigned char b = (unsigned char)needle[j];
            if ((char)std::tolower(a) != (char)std::tolower(b)) {
                match = false;
                break;
            }
        }
        if (match)
            return true;
    }
    return false;
}

static bool is_auto_companion_arg(const std::string& arg) {
    return arg.empty() || arg == "auto" || arg == "default";
}

static std::string discover_s3gen(const whisper_params& p) {
    if (!p.tts_codec_model.empty() && p.tts_codec_model != "auto" && p.tts_codec_model != "default") {
        return crispasr_resolve_model_cli(p.tts_codec_model, p.backend, p.no_prints, p.cache_dir, p.auto_download,
                                          p.tts_codec_quant);
    }

    const bool turbo_like = contains_ci(p.backend, "turbo") || contains_ci(p.backend, "kartoffel") ||
                            contains_ci(p.model, "turbo") || contains_ci(p.model, "kartoffel");
    const char* const* candidates = nullptr;
    static const char* turbo_candidates[] = {
        "chatterbox-turbo-s3gen-f16.gguf",
        nullptr,
    };
    static const char* base_candidates[] = {
        "chatterbox-s3gen-q8_0.gguf",
        "chatterbox-s3gen-f16.gguf",
        nullptr,
    };
    candidates = turbo_like ? turbo_candidates : base_candidates;

    auto try_dir = [&](const std::string& dir) -> std::string {
        for (const char* const* it = candidates; *it; ++it) {
            const std::string path = dir + "/" + *it;
            if (file_exists(path))
                return path;
        }
        return "";
    };

    auto sep = p.model.find_last_of("/\\");
    const std::string model_dir = (sep == std::string::npos) ? std::string(".") : p.model.substr(0, sep);
    std::string path = try_dir(model_dir);
    if (!path.empty())
        return path;

    const std::string cache_dir = crispasr_cache::dir(p.cache_dir);
    path = try_dir(cache_dir);
    if (!path.empty())
        return path;

    CrispasrRegistryEntry entry;
    if (crispasr_registry_lookup(p.backend, entry, p.tts_codec_quant) && !entry.companion_filename.empty()) {
        return crispasr_resolve_model_cli(entry.companion_filename, p.backend, p.no_prints, p.cache_dir,
                                          p.auto_download, p.tts_codec_quant);
    }
    if (p.auto_download || !is_auto_companion_arg(p.tts_codec_model)) {
        const char* wanted = turbo_like ? "chatterbox-turbo-s3gen-f16.gguf" : "chatterbox-s3gen-q8_0.gguf";
        return crispasr_resolve_model_cli(wanted, p.backend, p.no_prints, p.cache_dir, p.auto_download,
                                          p.tts_codec_quant);
    }

    return "";
}

class ChatterboxBackend : public CrispasrBackend {
public:
    ChatterboxBackend() = default;
    ~ChatterboxBackend() override { ChatterboxBackend::shutdown(); }

    const char* name() const override { return "chatterbox"; }

    uint32_t capabilities() const override {
        return CAP_TTS | CAP_AUTO_DOWNLOAD | CAP_TEMPERATURE | CAP_FLASH_ATTN | CAP_VOICE_CLONING;
    }

    std::vector<crispasr_segment> transcribe(const float* /*samples*/, int /*n_samples*/, int64_t /*t_offset_cs*/,
                                             const whisper_params& /*params*/) override {
        fprintf(stderr, "crispasr[chatterbox]: transcription is not supported by this backend\n");
        return {};
    }

    bool init(const whisper_params& p) override {
        chatterbox_context_params cp = chatterbox_context_default_params();
        cp.n_threads = p.n_threads;
        cp.verbosity = p.no_prints ? 0 : 1;
        cp.use_gpu = p.use_gpu;
        ctx_ = chatterbox_init_from_file(p.model.c_str(), cp);
        if (!ctx_) {
            fprintf(stderr, "crispasr[chatterbox]: failed to load T3 model '%s'\n", p.model.c_str());
            return false;
        }
        std::string s3gen_path = discover_s3gen(p);
        if (!s3gen_path.empty()) {
            if (chatterbox_set_s3gen_path(ctx_, s3gen_path.c_str()) != 0) {
                fprintf(stderr, "crispasr[chatterbox]: failed to load S3Gen '%s'\n", s3gen_path.c_str());
                return false;
            }
            s3gen_loaded_ = true;
            if (!p.no_prints) {
                fprintf(stderr, "crispasr[chatterbox]: S3Gen loaded from '%s'\n", s3gen_path.c_str());
            }
        }
        return true;
    }

    std::vector<float> synthesize(const std::string& text, const whisper_params& /*params*/) override {
        if (!ctx_ || text.empty())
            return {};
        if (!s3gen_loaded_) {
            fprintf(stderr, "crispasr[chatterbox]: S3Gen not loaded. Pass --codec-model <s3gen.gguf>\n");
            return {};
        }
        int n = 0;
        float* pcm = chatterbox_synthesize(ctx_, text.c_str(), &n);
        if (!pcm || n <= 0)
            return {};
        std::vector<float> out(pcm, pcm + n);
        chatterbox_pcm_free(pcm);
        return out;
    }

    void shutdown() override {
        if (ctx_) {
            chatterbox_free(ctx_);
            ctx_ = nullptr;
        }
        s3gen_loaded_ = false;
    }

private:
    chatterbox_context* ctx_ = nullptr;
    bool s3gen_loaded_ = false;
};

} // namespace

std::unique_ptr<CrispasrBackend> crispasr_make_chatterbox_backend() {
    return std::unique_ptr<CrispasrBackend>(new ChatterboxBackend());
}
