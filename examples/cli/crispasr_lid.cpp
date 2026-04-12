// crispasr_lid.cpp — implementation of the optional LID pre-step.
//
// See crispasr_lid.h for the interface contract.
//
// Two backends:
//   * whisper — uses the whisper.cpp C API directly on a ggml-tiny.bin
//               multilingual model. No extra deps. 99-language auto-detect.
//               This is the default.
//   * silero  — shells out to sherpa-onnx-offline-language-identification
//               pointed at a user-supplied LID ONNX model. Same
//               subprocess pattern as --diarize-method sherpa (see
//               crispasr_diarize.cpp): avoids linking onnxruntime into
//               crispasr while still letting users opt into Silero's
//               95-lang classifier or sherpa's Whisper-based LID when
//               the 75 MB whisper-tiny weights are too heavy.
//
// === whisper-tiny details ===
//
// Initialises a multilingual whisper_context from a ggml-*.bin file
// (default ggml-tiny.bin, auto-downloaded to ~/.cache/crispasr on first
// use via curl/wget). Pads the first 30 s of input to exactly 480 000
// samples, computes the mel spectrogram, runs a single encoder pass,
// and picks the argmax over whisper_lang_auto_detect()'s per-language
// probabilities. Frees the context before returning. For longer-running
// multi-file jobs the model load + encode is done per invocation; a
// later commit can add a process-lifetime cache keyed on the model
// path if the reload cost becomes noticeable.

#include "crispasr_lid.h"
#include "crispasr_cache.h"
#include "whisper_params.h"

#include "whisper.h"
#include "silero_lid.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <memory>
#include <sstream>
#include <string>
#ifdef _WIN32
#include <io.h>
#include <fcntl.h>
#define close _close
static int mkstemps(char * t, int s) {
    (void)s; return _mktemp_s(t, strlen(t)+1) == 0 ? _open(t, _O_CREAT|_O_WRONLY, 0600) : -1;
}
#else
#include <unistd.h>
#endif
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

// -----------------------------------------------------------------------
// Shared helpers
// -----------------------------------------------------------------------

std::string expand_home(const std::string & p) {
    if (p.empty() || p[0] != '~') return p;
    const char * home = std::getenv("HOME");
    if (!home || !*home) return p;
    return std::string(home) + p.substr(1);
}

// -----------------------------------------------------------------------
// Backend 1: whisper-tiny LID via whisper.h
// -----------------------------------------------------------------------

// Default model — multilingual tiny. 75 MB, fast, and accurate enough to
// pick between Whisper's trained languages for a 30-second clip.
constexpr const char * kWhisperLidDefaultUrl =
    "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.bin";
constexpr const char * kWhisperLidDefaultFile = "ggml-tiny.bin";

// Resolve the LID model path. If params.lid_model is set we use that
// directly. Otherwise delegate to the shared cache helper which checks
// ~/.cache/crispasr and auto-downloads on miss.
std::string resolve_whisper_lid_model(const whisper_params & p) {
    if (!p.lid_model.empty()) {
        return expand_home(p.lid_model);
    }
    return crispasr_cache::ensure_cached_file(
        kWhisperLidDefaultFile, kWhisperLidDefaultUrl, p.no_prints,
        "crispasr[lid]");
}

// Process-lifetime cache: keep the whisper LID context around between
// invocations so batch runs (multiple -f inputs, or multiple slices of
// one long input) don't re-load the 75 MB ggml-tiny.bin every time.
// Cache is keyed on the model path — if the user switches --lid-model
// mid-run, we free the old context and load the new one.
struct WhisperLidCache {
    whisper_context * ctx        = nullptr;
    std::string       model_path;
    bool              use_gpu    = false;
    int               gpu_device = 0;
    bool              flash_attn = true;
};

WhisperLidCache & whisper_lid_cache() {
    static WhisperLidCache c;
    return c;
}

bool detect_with_whisper_tiny(
    const float * samples, int n_samples,
    const whisper_params & p,
    crispasr_lid_result & out)
{
    const std::string model_path = resolve_whisper_lid_model(p);
    if (model_path.empty()) return false;

    WhisperLidCache & c = whisper_lid_cache();

    // Invalidate the cache whenever the caller changes the model path or
    // any of the cparams we pass to whisper_init_from_file_with_params.
    // Fresh state + fresh model == fresh context, same as before.
    const bool cache_miss = (c.ctx == nullptr) ||
                            (c.model_path != model_path) ||
                            (c.use_gpu    != p.use_gpu)  ||
                            (c.gpu_device != p.gpu_device) ||
                            (c.flash_attn != p.flash_attn);

    if (cache_miss) {
        if (c.ctx) {
            whisper_free(c.ctx);
            c.ctx = nullptr;
        }
        whisper_context_params cp = whisper_context_default_params();
        cp.use_gpu    = p.use_gpu;
        cp.gpu_device = p.gpu_device;
        cp.flash_attn = p.flash_attn;

        c.ctx = whisper_init_from_file_with_params(model_path.c_str(), cp);
        if (!c.ctx) {
            fprintf(stderr, "crispasr[lid]: failed to load '%s'\n", model_path.c_str());
            return false;
        }

        if (!whisper_is_multilingual(c.ctx)) {
            fprintf(stderr,
                    "crispasr[lid]: model '%s' is English-only — pass a multilingual "
                    "ggml-*.bin via --lid-model\n",
                    model_path.c_str());
            whisper_free(c.ctx);
            c.ctx = nullptr;
            return false;
        }

        c.model_path = model_path;
        c.use_gpu    = p.use_gpu;
        c.gpu_device = p.gpu_device;
        c.flash_attn = p.flash_attn;
    }

    whisper_context * ctx = c.ctx;

    // Whisper's encoder expects exactly 30 s (480 000 samples). Pad with
    // zeros if the input is shorter; truncate if it's longer. LID only
    // looks at the first 30 s anyway, so we don't need to keep the tail.
    constexpr int SR  = 16000;
    constexpr int NEED = SR * 30;
    std::vector<float> pcm((size_t)NEED, 0.0f);
    const int n_use = std::min(n_samples, NEED);
    std::memcpy(pcm.data(), samples, (size_t)n_use * sizeof(float));

    // From here down we operate on the cached context. The cache owns
    // ctx's lifetime — no whisper_free() here on the success or the
    // transient-failure paths. Only the cache-invalidation branch above
    // ever frees it.

    if (whisper_pcm_to_mel(ctx, pcm.data(), NEED, p.n_threads) != 0) {
        fprintf(stderr, "crispasr[lid]: pcm_to_mel failed\n");
        return false;
    }
    if (whisper_encode(ctx, 0, p.n_threads) != 0) {
        fprintf(stderr, "crispasr[lid]: encode failed\n");
        return false;
    }

    const int n_langs = whisper_lang_max_id() + 1;
    std::vector<float> probs((size_t)n_langs, 0.0f);
    const int lang_id = whisper_lang_auto_detect(
        ctx, /*offset_ms=*/0, p.n_threads, probs.data());
    if (lang_id < 0 || lang_id >= n_langs) {
        fprintf(stderr, "crispasr[lid]: whisper_lang_auto_detect failed\n");
        return false;
    }

    out.lang_code  = whisper_lang_str(lang_id);
    out.confidence = probs[lang_id];
    out.source     = "whisper";

    if (!p.no_prints) {
        fprintf(stderr,
                "crispasr[lid]: detected '%s' (p=%.3f) via whisper-tiny\n",
                out.lang_code.c_str(), out.confidence);
    }

    return true;
}

// -----------------------------------------------------------------------
// Backend 2: Silero LID — via sherpa-onnx-offline-language-identification
// -----------------------------------------------------------------------
//
// Shell out to an externally-installed
// `sherpa-onnx-offline-language-identification` binary (k2-fsa/sherpa-onnx)
// that hosts either the Silero 95-language classifier or one of the
// Whisper-based multilingual LID models. Same tradeoff as the sherpa
// diarize wrapper in crispasr_diarize.cpp: we avoid linking onnxruntime
// (a ~90 MB shared lib) into the crispasr binary while still letting
// users opt into a much smaller, faster LID model than whisper-tiny.
//
// CLI knobs in whisper_params:
//   --lid-backend silero
//   --lid-model   PATH     (path to the sherpa LID model, mandatory here)
//
// The binary prints a single line like:
//     "Detected language: English"
// or an ISO code depending on the version. We parse a generous set of
// formats and map common English-name → ISO-639-1 codes below.
bool detect_with_silero(
    const float * samples, int n_samples,
    const whisper_params & p,
    crispasr_lid_result & out)
{
    if (p.lid_model.empty()) {
        fprintf(stderr,
            "crispasr[lid]: --lid-backend silero needs --lid-model PATH\n"
            "               (GGUF from convert-silero-lid-to-gguf.py, or\n"
            "                a sherpa-onnx LID model for the subprocess fallback)\n");
        return false;
    }

    // ---- Try native GGUF path first (if the model is a .gguf file) ----
    const std::string & model_path = p.lid_model;
    if (model_path.size() >= 5 &&
        model_path.compare(model_path.size() - 5, 5, ".gguf") == 0)
    {
        silero_lid_context * lid = silero_lid_init(model_path.c_str(), p.n_threads);
        if (lid) {
            float conf = 0;
            const char * lang = silero_lid_detect(lid, samples, n_samples, &conf);
            if (lang && *lang) {
                // Parse "NN, Language" format → extract ISO code
                std::string full = lang;
                // The lang_dict has format "en, English" — extract before comma
                auto comma = full.find(',');
                std::string code = (comma != std::string::npos)
                    ? full.substr(0, comma) : full;
                // Trim whitespace
                while (!code.empty() && code.back() == ' ') code.pop_back();
                while (!code.empty() && code.front() == ' ') code.erase(code.begin());
                // Trim digits (lang_dict keys are "30" for English index)
                bool is_digit = true;
                for (char c : code) if (!isdigit(c)) { is_digit = false; break; }
                if (is_digit) {
                    // Index-based — the full string IS the language name
                    code = full;
                }
                out.lang_code  = code;
                out.confidence = conf;
                out.source     = "silero-native";
                if (!p.no_prints) {
                    fprintf(stderr, "crispasr[lid]: silero-native → %s (%.2f)\n",
                            full.c_str(), conf);
                }
                silero_lid_free(lid);
                return true;
            }
            silero_lid_free(lid);
            fprintf(stderr, "crispasr[lid]: silero-native returned null — "
                            "falling back to sherpa subprocess\n");
        }
    }

    // ---- Sherpa subprocess fallback (for ONNX models) ----

    // Default binary name; user can override via $CRISPASR_SHERPA_LID_BIN
    // for installs that don't put sherpa on $PATH.
    const char * env_bin = std::getenv("CRISPASR_SHERPA_LID_BIN");
    const std::string bin = env_bin && *env_bin
        ? std::string(env_bin)
        : std::string("sherpa-onnx-offline-language-identification");

    // Write a temp 16 kHz mono int16 WAV — sherpa reads a file path.
    char tmpl[] = "/tmp/crispasr-lid-XXXXXX.wav";
    int fd = mkstemps(tmpl, 4);
    if (fd < 0) {
        fprintf(stderr, "crispasr[lid]: mkstemps failed\n");
        return false;
    }
    close(fd);
    std::string wav_path = tmpl;
    {
        FILE * f = fopen(wav_path.c_str(), "wb");
        if (!f) { std::remove(wav_path.c_str()); return false; }
        const uint32_t sr = 16000;
        const uint16_t ch = 1;
        const uint16_t bps = 16;
        const uint32_t byte_rate = sr * ch * bps / 8;
        const uint16_t block_align = ch * bps / 8;
        const uint32_t data_bytes = (uint32_t)n_samples * block_align;
        const uint32_t riff_size = 36 + data_bytes;
        auto w32 = [&](uint32_t v) { fwrite(&v, 4, 1, f); };
        auto w16 = [&](uint16_t v) { fwrite(&v, 2, 1, f); };
        fwrite("RIFF", 1, 4, f); w32(riff_size); fwrite("WAVE", 1, 4, f);
        fwrite("fmt ", 1, 4, f); w32(16); w16(1); w16(ch);
        w32(sr); w32(byte_rate); w16(block_align); w16(bps);
        fwrite("data", 1, 4, f); w32(data_bytes);
        std::vector<int16_t> pcm(n_samples);
        for (int i = 0; i < n_samples; i++) {
            float v = samples[i];
            if (v >  1.0f) v =  1.0f;
            if (v < -1.0f) v = -1.0f;
            pcm[i] = (int16_t)(v * 32767.0f);
        }
        fwrite(pcm.data(), sizeof(int16_t), pcm.size(), f);
        fclose(f);
    }

    std::ostringstream cmd;
    cmd << bin
        << " --whisper-model='" << p.lid_model << "'"
        << " '" << wav_path << "' 2>&1";
    if (!p.no_prints) {
        fprintf(stderr, "crispasr[lid]: %s\n", cmd.str().c_str());
    }

    std::unique_ptr<FILE, int(*)(FILE*)> pipe(popen(cmd.str().c_str(), "r"), pclose);
    if (!pipe) {
        fprintf(stderr, "crispasr[lid]: failed to spawn sherpa LID subprocess\n");
        std::remove(wav_path.c_str());
        return false;
    }
    char linebuf[512];
    std::string detected;
    while (fgets(linebuf, sizeof(linebuf), pipe.get())) {
        std::string line = linebuf;
        // Strip trailing whitespace.
        while (!line.empty() &&
               (line.back() == '\n' || line.back() == '\r' || line.back() == ' '))
            line.pop_back();
        // Accept either: "Detected language: English"
        // or a plain ISO/English name on its own line.
        auto pos = line.find("Detected language:");
        if (pos != std::string::npos) {
            detected = line.substr(pos + std::string("Detected language:").size());
            // Trim leading spaces
            size_t s = 0;
            while (s < detected.size() && detected[s] == ' ') s++;
            detected = detected.substr(s);
            break;
        }
    }
    std::remove(wav_path.c_str());

    if (detected.empty()) {
        fprintf(stderr,
            "crispasr[lid]: sherpa subprocess produced no 'Detected language:' line\n"
            "               (check that sherpa-onnx-offline-language-identification\n"
            "               is installed and the --lid-model path is correct)\n");
        return false;
    }

    // Map common English names → ISO 639-1. Sherpa sometimes prints the
    // 2-letter code directly (e.g. "en"), in which case we pass through.
    auto to_iso = [](const std::string & s) -> std::string {
        static const std::pair<const char *, const char *> map[] = {
            {"english", "en"}, {"german", "de"}, {"french", "fr"},
            {"spanish", "es"}, {"italian", "it"}, {"portuguese", "pt"},
            {"dutch", "nl"},   {"russian", "ru"}, {"polish", "pl"},
            {"czech", "cs"},   {"turkish", "tr"}, {"arabic", "ar"},
            {"hindi", "hi"},   {"japanese", "ja"}, {"korean", "ko"},
            {"chinese", "zh"}, {"mandarin", "zh"},{"cantonese", "zh"},
            {"ukrainian", "uk"}, {"swedish", "sv"}, {"norwegian", "no"},
            {"danish", "da"}, {"finnish", "fi"}, {"greek", "el"},
            {"hebrew", "he"}, {"thai", "th"}, {"vietnamese", "vi"},
            {"indonesian", "id"}, {"malay", "ms"}, {"romanian", "ro"},
            {"hungarian", "hu"}, {"bulgarian", "bg"}, {"serbian", "sr"},
            {"slovak", "sk"}, {"slovenian", "sl"}, {"croatian", "hr"},
        };
        std::string lo = s;
        for (char & c : lo) c = (char)std::tolower((unsigned char)c);
        for (auto & p : map) if (lo == p.first) return p.second;
        // If already ISO-ish (2 chars, lowercase), pass through.
        if (lo.size() == 2) return lo;
        return lo;
    };

    out.lang_code  = to_iso(detected);
    out.confidence = 1.0f;  // sherpa only reports argmax, not a score
    out.source     = "silero";
    if (!p.no_prints) {
        fprintf(stderr,
                "crispasr[lid]: sherpa → %s (%s)\n",
                detected.c_str(), out.lang_code.c_str());
    }
    return true;
}

} // namespace

// -----------------------------------------------------------------------
// Public entrypoint
// -----------------------------------------------------------------------

bool crispasr_detect_language(
    const float * samples, int n_samples,
    const whisper_params & params,
    crispasr_lid_result & out)
{
    out = {};
    if (!samples || n_samples <= 0) return false;

    // Pick the backend. An explicit --lid-backend takes priority; when
    // empty we default to whisper-tiny (safest, no Python, no extra
    // model formats to maintain).
    std::string be = params.lid_backend;
    if (be.empty()) be = "whisper";

    if (be == "whisper" || be == "whisper-tiny") {
        return detect_with_whisper_tiny(samples, n_samples, params, out);
    }
    if (be == "silero") {
        return detect_with_silero(samples, n_samples, params, out);
    }

    fprintf(stderr,
            "crispasr[lid]: unknown --lid-backend '%s' (expected 'whisper' or 'silero')\n",
            be.c_str());
    return false;
}
