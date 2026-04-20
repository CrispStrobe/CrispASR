// crispasr_lid_cli.cpp — CLI LID shim.
//
// Translates CLI flags into library calls for whisper-tiny /
// silero-native, and keeps the sherpa-onnx subprocess fallback + model
// auto-download here where the CLI's cache / UX policy lives.

#include "crispasr_lid_cli.h"
#include "crispasr_cache.h"
#include "crispasr_lid.h" // shared library header (src/)
#include "whisper_params.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#include <sys/stat.h>
#define close _close
#define popen _popen
#define pclose _pclose
static int mkstemps(char* t, int s) {
    (void)s;
    return _mktemp_s(t, strlen(t) + 1) == 0 ? _open(t, _O_CREAT | _O_WRONLY, 0600) : -1;
}
#else
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace {

std::string expand_home(const std::string& p) {
    if (p.empty() || p[0] != '~')
        return p;
    const char* home = std::getenv("HOME");
    if (!home || !*home)
        return p;
    return std::string(home) + p.substr(1);
}

// Default whisper LID model: multilingual tiny. 75 MB, fast.
constexpr const char* kWhisperLidDefaultUrl = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.bin";
constexpr const char* kWhisperLidDefaultFile = "ggml-tiny.bin";
constexpr const char* kSileroLidDefaultUrl =
    "https://huggingface.co/cstr/silero-lid-lang95-GGUF/resolve/main/silero-lid-lang95-f32.gguf";
constexpr const char* kSileroLidDefaultFile = "silero-lid-lang95-f32.gguf";
constexpr const char* kFireredLidDefaultUrl =
    "https://huggingface.co/cstr/firered-lid-GGUF/resolve/main/firered-lid-f16.gguf";
constexpr const char* kFireredLidDefaultFile = "firered-lid-f16.gguf";
constexpr const char* kEcapaLidDefaultUrl =
    "https://huggingface.co/cstr/ecapa-lid-107-GGUF/resolve/main/ecapa-lid-107.gguf";
constexpr const char* kEcapaLidDefaultFile = "ecapa-lid-107.gguf";

std::string resolve_whisper_lid_model(const whisper_params& p) {
    if (!p.lid_model.empty())
        return expand_home(p.lid_model);
    return crispasr_cache::ensure_cached_file(kWhisperLidDefaultFile, kWhisperLidDefaultUrl, p.no_prints,
                                              "crispasr[lid]", p.cache_dir);
}

std::string resolve_silero_lid_model(const whisper_params& p) {
    if (!p.lid_model.empty() && p.lid_model != "auto")
        return expand_home(p.lid_model);
    return crispasr_cache::ensure_cached_file(kSileroLidDefaultFile, kSileroLidDefaultUrl, p.no_prints,
                                              "crispasr[lid-silero]", p.cache_dir);
}

std::string resolve_firered_lid_model(const whisper_params& p) {
    if (!p.lid_model.empty() && p.lid_model != "auto")
        return expand_home(p.lid_model);
    return crispasr_cache::ensure_cached_file(kFireredLidDefaultFile, kFireredLidDefaultUrl, p.no_prints,
                                              "crispasr[lid-firered]", p.cache_dir);
}

std::string resolve_ecapa_lid_model(const whisper_params& p) {
    if (!p.lid_model.empty() && p.lid_model != "auto")
        return expand_home(p.lid_model);
    return crispasr_cache::ensure_cached_file(kEcapaLidDefaultFile, kEcapaLidDefaultUrl, p.no_prints,
                                              "crispasr[lid-ecapa]", p.cache_dir);
}

// Common English-name → ISO 639-1 mapping for the sherpa subprocess path.
std::string to_iso_code(const std::string& s) {
    static const std::pair<const char*, const char*> map[] = {
        {"english", "en"},    {"german", "de"},     {"french", "fr"},     {"spanish", "es"},   {"italian", "it"},
        {"portuguese", "pt"}, {"dutch", "nl"},      {"russian", "ru"},    {"polish", "pl"},    {"czech", "cs"},
        {"turkish", "tr"},    {"arabic", "ar"},     {"hindi", "hi"},      {"japanese", "ja"},  {"korean", "ko"},
        {"chinese", "zh"},    {"mandarin", "zh"},   {"cantonese", "zh"},  {"ukrainian", "uk"}, {"swedish", "sv"},
        {"norwegian", "no"},  {"danish", "da"},     {"finnish", "fi"},    {"greek", "el"},     {"hebrew", "he"},
        {"thai", "th"},       {"vietnamese", "vi"}, {"indonesian", "id"}, {"malay", "ms"},     {"romanian", "ro"},
        {"hungarian", "hu"},  {"bulgarian", "bg"},  {"serbian", "sr"},    {"slovak", "sk"},    {"slovenian", "sl"},
        {"croatian", "hr"},
    };
    std::string lo = s;
    for (char& c : lo)
        c = (char)std::tolower((unsigned char)c);
    for (auto& p : map)
        if (lo == p.first)
            return p.second;
    if (lo.size() == 2)
        return lo;
    return lo;
}

bool detect_with_sherpa(const float* samples, int n_samples, const whisper_params& p, crispasr_lid_result& out) {
    const char* env_bin = std::getenv("CRISPASR_SHERPA_LID_BIN");
    const std::string bin =
        env_bin && *env_bin ? std::string(env_bin) : std::string("sherpa-onnx-offline-language-identification");

    char tmpl[] = "/tmp/crispasr-lid-XXXXXX.wav";
    int fd = mkstemps(tmpl, 4);
    if (fd < 0) {
        fprintf(stderr, "crispasr[lid]: mkstemps failed\n");
        return false;
    }
    close(fd);
    const std::string wav_path = tmpl;

    FILE* f = fopen(wav_path.c_str(), "wb");
    if (!f) {
        std::remove(wav_path.c_str());
        return false;
    }
    const uint32_t sr = 16000;
    const uint16_t ch = 1;
    const uint16_t bps = 16;
    const uint32_t byte_rate = sr * ch * bps / 8;
    const uint16_t block_align = ch * bps / 8;
    const uint32_t data_bytes = (uint32_t)n_samples * block_align;
    const uint32_t riff_size = 36 + data_bytes;
    auto w32 = [&](uint32_t v) { fwrite(&v, 4, 1, f); };
    auto w16 = [&](uint16_t v) { fwrite(&v, 2, 1, f); };
    fwrite("RIFF", 1, 4, f);
    w32(riff_size);
    fwrite("WAVE", 1, 4, f);
    fwrite("fmt ", 1, 4, f);
    w32(16);
    w16(1);
    w16(ch);
    w32(sr);
    w32(byte_rate);
    w16(block_align);
    w16(bps);
    fwrite("data", 1, 4, f);
    w32(data_bytes);
    std::vector<int16_t> pcm(n_samples);
    for (int i = 0; i < n_samples; i++) {
        float v = samples[i];
        if (v > 1.0f)
            v = 1.0f;
        if (v < -1.0f)
            v = -1.0f;
        pcm[i] = (int16_t)(v * 32767.0f);
    }
    fwrite(pcm.data(), sizeof(int16_t), pcm.size(), f);
    fclose(f);

    std::ostringstream cmd;
    cmd << bin << " --whisper-model='" << p.lid_model << "' '" << wav_path << "' 2>&1";
    if (!p.no_prints)
        fprintf(stderr, "crispasr[lid]: %s\n", cmd.str().c_str());

    std::unique_ptr<FILE, int (*)(FILE*)> pipe(popen(cmd.str().c_str(), "r"), pclose);
    if (!pipe) {
        fprintf(stderr, "crispasr[lid]: failed to spawn sherpa LID subprocess\n");
        std::remove(wav_path.c_str());
        return false;
    }

    char linebuf[512];
    std::string detected;
    while (fgets(linebuf, sizeof(linebuf), pipe.get())) {
        std::string line = linebuf;
        while (!line.empty() && (line.back() == '\n' || line.back() == '\r' || line.back() == ' '))
            line.pop_back();
        auto pos = line.find("Detected language:");
        if (pos != std::string::npos) {
            detected = line.substr(pos + std::string("Detected language:").size());
            size_t s = 0;
            while (s < detected.size() && detected[s] == ' ')
                s++;
            detected = detected.substr(s);
            break;
        }
    }
    std::remove(wav_path.c_str());

    if (detected.empty()) {
        fprintf(stderr, "crispasr[lid]: sherpa subprocess produced no 'Detected language:' line\n");
        return false;
    }

    out.lang_code = to_iso_code(detected);
    out.confidence = 1.0f; // sherpa reports argmax only
    out.source = "silero";
    if (!p.no_prints)
        fprintf(stderr, "crispasr[lid]: sherpa → %s (%s)\n", detected.c_str(), out.lang_code.c_str());
    return true;
}

} // namespace

bool crispasr_detect_language_cli(const float* samples, int n_samples, const whisper_params& params,
                                  crispasr_lid_result& out) {
    out = {};
    if (!samples || n_samples <= 0)
        return false;

    std::string be = params.lid_backend;
    if (be.empty())
        be = "whisper";

    if (be == "whisper" || be == "whisper-tiny") {
        CrispasrLidOptions opts;
        opts.method = CrispasrLidMethod::Whisper;
        opts.model_path = resolve_whisper_lid_model(params);
        opts.n_threads = params.n_threads;
        opts.use_gpu = params.use_gpu;
        opts.gpu_device = params.gpu_device;
        opts.flash_attn = params.flash_attn;
        opts.verbose = !params.no_prints;
        CrispasrLidResult r;
        if (!crispasr_detect_language(samples, n_samples, opts, r))
            return false;
        out.lang_code = r.lang_code;
        out.confidence = r.confidence;
        out.source = r.source;
        return true;
    }

    if (be == "silero") {
        const std::string model_path = resolve_silero_lid_model(params);
        // Native GGUF path first (faster, smaller, no extra runtime).
        if (!model_path.empty() && model_path.size() >= 5 &&
            model_path.compare(model_path.size() - 5, 5, ".gguf") == 0) {
            CrispasrLidOptions opts;
            opts.method = CrispasrLidMethod::Silero;
            opts.model_path = model_path;
            opts.n_threads = params.n_threads;
            opts.verbose = !params.no_prints;
            CrispasrLidResult r;
            if (crispasr_detect_language(samples, n_samples, opts, r)) {
                out.lang_code = r.lang_code;
                out.confidence = r.confidence;
                out.source = r.source;
                return true;
            }
            fprintf(stderr, "crispasr[lid]: silero-native returned null — "
                            "falling back to sherpa subprocess\n");
        }
        // Sherpa-ONNX subprocess fallback (for ONNX models).
        return detect_with_sherpa(samples, n_samples, params, out);
    }

    if (be == "firered") {
        const std::string model_path = resolve_firered_lid_model(params);
        if (!model_path.empty()) {
            CrispasrLidOptions opts;
            opts.method = CrispasrLidMethod::Firered;
            opts.model_path = model_path;
            opts.n_threads = params.n_threads;
            opts.verbose = !params.no_prints;
            CrispasrLidResult r;
            if (crispasr_detect_language(samples, n_samples, opts, r)) {
                out.lang_code = r.lang_code;
                out.confidence = r.confidence;
                out.source = r.source;
                return true;
            }
        }
        fprintf(stderr, "crispasr[lid]: firered LID failed\n");
        return false;
    }

    if (be == "ecapa") {
        const std::string model_path = resolve_ecapa_lid_model(params);
        if (!model_path.empty()) {
            CrispasrLidOptions opts;
            opts.method = CrispasrLidMethod::Ecapa;
            opts.model_path = model_path;
            opts.n_threads = params.n_threads;
            opts.verbose = !params.no_prints;
            CrispasrLidResult r;
            if (crispasr_detect_language(samples, n_samples, opts, r)) {
                out.lang_code = r.lang_code;
                out.confidence = r.confidence;
                out.source = r.source;
                return true;
            }
        }
        fprintf(stderr, "crispasr[lid]: ecapa LID failed\n");
        return false;
    }

    fprintf(stderr, "crispasr[lid]: unknown --lid-backend '%s' (expected 'whisper', 'silero', 'firered', or 'ecapa')\n",
            be.c_str());
    return false;
}
