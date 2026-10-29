// crispasr_lid_cli.cpp — CLI LID shim.
//
// Translates CLI flags into library calls for whisper-tiny /
// silero-native, and keeps the sherpa-onnx subprocess fallback + model
// auto-download here where the CLI's cache / UX policy lives.

#include "crispasr_lid_cli.h"
#include "crispasr_backend.h"
#include "crispasr_cache.h"
#include "crispasr_lid.h" // shared library header (src/)
#include "core/lid_recording.h"
#include "crispasr_subprocess.h"
#include "whisper_params.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#ifdef _WIN32
#include <fcntl.h>
#include <direct.h>
#include <io.h>
#include <sys/stat.h>
#define close _close
#define mkdir(d, m) _mkdir(d)
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

std::string scratch_dir() {
    const char* env = std::getenv("CRISPASR_SCRATCH_DIR");
    std::string d = (env && *env) ? std::string(env) : crispasr_cache::dir() + "/scratch";
    mkdir(d.c_str(), 0755);
    return d;
}
constexpr const char* kEcapaLidDefaultFile = "ecapa-lid-107.gguf";

std::string resolve_whisper_lid_model(const whisper_params& p) {
    // --lid-model belongs to the selected provider. A Silero GGUF (or
    // another provider's checkpoint) must never be reused by Whisper when
    // that provider declines a window and we take the default fallback.
    const bool selected_whisper =
        p.lid_backend.empty() || p.lid_backend == "whisper" || p.lid_backend == "whisper-tiny";
    if (selected_whisper && !p.lid_model.empty() && p.lid_model != "auto")
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

    std::string tmpl_s = scratch_dir() + "/crispasr-lid-XXXXXX.wav";
    std::vector<char> tmpl(tmpl_s.begin(), tmpl_s.end());
    tmpl.push_back('\0');
    int fd = mkstemps(tmpl.data(), 4);
    if (fd < 0) {
        fprintf(stderr, "crispasr[lid]: mkstemps failed\n");
        return false;
    }
    close(fd);
    const std::string wav_path = tmpl.data();

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

    const std::vector<std::string> args = {
        bin,
        "--whisper-model=" + p.lid_model,
        wav_path,
    };
    if (!p.no_prints)
        fprintf(stderr, "crispasr[lid]: %s\n", crispasr_cli_process::join_cmdline(args).c_str());

    const int timeout_sec =
        crispasr_cli_process::timeout_from_audio_samples("CRISPASR_SHERPA_LID_TIMEOUT_SEC", n_samples);
    const auto run = crispasr_cli_process::run_capture_stdout(args, timeout_sec, true);
    if (run.timed_out) {
        fprintf(stderr, "crispasr[lid]: sherpa subprocess timed out after %d s\n", timeout_sec);
        std::remove(wav_path.c_str());
        return false;
    }
    if (run.spawn_failed) {
        fprintf(stderr, "crispasr[lid]: failed to spawn sherpa LID subprocess\n");
        std::remove(wav_path.c_str());
        return false;
    }
    if (run.exit_code != 0) {
        fprintf(stderr, "crispasr[lid]: sherpa subprocess exited with code %d\n", run.exit_code);
        std::remove(wav_path.c_str());
        return false;
    }

    std::string detected;
    std::istringstream lines(run.output);
    for (std::string line; std::getline(lines, line);) {
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

bool crispasr_backend_probe_language(CrispasrBackend& backend, const float* samples, int n_samples,
                                     const whisper_params& params, crispasr_lid_result& out) {
    out = {};
    if (!samples || n_samples <= 0)
        return false;

    // An explicitly named external backend wins — the probe is only the
    // default when the user expressed no preference (or asked for it).
    const std::string& be = params.lid_backend;
    if (!be.empty() && be != "probe" && be != "self")
        return false;

    if (params.lid_full_recording) {
        core_lid::WindowResult result;
        const bool ok = core_lid::detect_recording(
            samples, n_samples,
            [&](const float* window, int count, core_lid::WindowResult& vote) {
                if (!backend.detect_language(window, count, params, vote.lang_code, vote.confidence))
                    return false;
                vote.source = std::string(backend.name()) + "-probe";
                return true;
            },
            result);
        if (!ok)
            return false;
        out.lang_code = result.lang_code;
        out.confidence = result.confidence;
        out.source = result.source;
        return true;
    }
    std::string lang;
    float confidence = 0.0f;
    if (!backend.detect_language(samples, n_samples, params, lang, confidence) || lang.empty())
        return false;
    out.lang_code = lang;
    out.confidence = confidence;
    out.source = std::string(backend.name()) + "-probe";
    return true;
}

static bool detect_whisper_cli(const float* samples, int n_samples, const whisper_params& params,
                               crispasr_lid_result& out) {
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

static bool crispasr_detect_language_cli_window(const float* samples, int n_samples, const whisper_params& params,
                                                crispasr_lid_result& out) {
    out = {};
    if (!samples || n_samples <= 0)
        return false;

    std::string be = params.lid_backend;
    if (be.empty())
        be = "whisper";
    if (be == "probe" || be == "self") {
        // Reached only when the backend's own probe declined or failed —
        // crispasr_backend_probe_language() already had its turn. Fall back to
        // the external default rather than erroring on a valid flag value.
        fprintf(stderr, "crispasr[lid]: this backend has no self-probe (or it returned nothing) — "
                        "falling back to whisper-tiny LID\n");
        be = "whisper";
    }

    if (be == "whisper" || be == "whisper-tiny")
        return detect_whisper_cli(samples, n_samples, params, out);

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
            // #409: inconclusive (confidence-gated) or failed. Sherpa would
            // re-run the SAME model and reach the same non-answer, so go
            // straight to the robust default instead of returning garbage.
            fprintf(stderr, "crispasr[lid]: silero LID inconclusive or failed — "
                            "falling back to whisper-tiny LID\n");
            return detect_whisper_cli(samples, n_samples, params, out);
        }
        // Sherpa-ONNX subprocess fallback (for ONNX models).
        if (detect_with_sherpa(samples, n_samples, params, out))
            return true;
        fprintf(stderr, "crispasr[lid]: silero (sherpa) LID failed — "
                        "falling back to whisper-tiny LID\n");
        return detect_whisper_cli(samples, n_samples, params, out);
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

bool crispasr_detect_language_cli(const float* samples, int n_samples, const whisper_params& params,
                                  crispasr_lid_result& out) {
    if (!params.lid_full_recording)
        return crispasr_detect_language_cli_window(samples, n_samples, params, out);

    core_lid::WindowResult result;
    const bool ok = core_lid::detect_recording(
        samples, n_samples,
        [&](const float* window, int count, core_lid::WindowResult& vote) {
            crispasr_lid_result one;
            if (!crispasr_detect_language_cli_window(window, count, params, one))
                return false;
            vote.lang_code = one.lang_code;
            vote.confidence = one.confidence;
            vote.source = one.source;
            return true;
        },
        result);
    if (!ok)
        return false;
    out.lang_code = result.lang_code;
    out.confidence = result.confidence;
    out.source = result.source;
    return true;
}
