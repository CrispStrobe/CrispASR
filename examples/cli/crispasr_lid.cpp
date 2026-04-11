// crispasr_lid.cpp — implementation of the optional LID pre-step.
//
// See crispasr_lid.h for the interface contract.
//
// Runtime is 100% C++ — no Python, no shell-outs to torch, no ONNX
// runtime dependency. The whisper-tiny path uses the whisper.cpp C API
// directly. The silero path is a placeholder until a native GGUF port
// of Silero's language detector lands (tracked in TODO.md): the flag is
// accepted but currently errors out with an actionable message.
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
#include "whisper_params.h"

#include "whisper.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <unistd.h>
#include <sys/stat.h>

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

bool file_exists(const std::string & p) {
    return access(p.c_str(), F_OK) == 0;
}

std::string default_cache_dir() {
    const char * home = std::getenv("HOME");
    std::string dir = (home && *home) ? home : "/tmp";
    dir += "/.cache/crispasr";
    mkdir(dir.c_str(), 0755);
    return dir;
}

// Try curl then wget to fetch a URL into the given destination path.
// Returns true on success. Used only for the whisper-tiny default
// model auto-download. Same two-tool fallback as crispasr_model_mgr.cpp.
bool fetch_url(const std::string & url, const std::string & dst,
               bool no_prints) {
    if (!no_prints) {
        fprintf(stderr, "crispasr[lid]: downloading %s\n", url.c_str());
    }
    const std::string cmd_curl =
        "curl -fL --progress-bar -o " + dst + " " + url;
    if (std::system(cmd_curl.c_str()) == 0 && file_exists(dst)) return true;
    const std::string cmd_wget =
        "wget -q --show-progress -O " + dst + " " + url;
    if (std::system(cmd_wget.c_str()) == 0 && file_exists(dst)) return true;
    fprintf(stderr, "crispasr[lid]: failed to download (curl + wget both failed)\n");
    return false;
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
// directly. Otherwise we check for the default file in the crispasr
// cache dir, auto-downloading it on first use.
std::string resolve_whisper_lid_model(const whisper_params & p) {
    if (!p.lid_model.empty()) {
        return expand_home(p.lid_model);
    }
    const std::string dst = default_cache_dir() + "/" + kWhisperLidDefaultFile;
    if (file_exists(dst)) return dst;
    if (!fetch_url(kWhisperLidDefaultUrl, dst, p.no_prints)) return "";
    return dst;
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
// Backend 2: Silero LID — reserved slot for a future native GGUF port
// -----------------------------------------------------------------------
//
// Silero publishes a ~5 MB language-classifier model (silero_lang_detector
// / silero_lang_detector_95) as a TorchScript + ONNX bundle. To run it
// inside crispasr without any Python or ONNX-runtime dependency we'd
// need to:
//
//   1. Write a one-time converter (models/convert-silero-lid-to-gguf.py)
//      that loads the TorchScript or ONNX export, extracts the weights
//      and the language-id lookup table, and writes them as a GGUF
//      tensor archive using the same gguf Python package we already use
//      for the other converters.
//
//   2. Add a loader + inference wrapper that mirrors whisper_vad's
//      ggml-based forward pass for Silero VAD — the VAD converter is
//      already in tree (models/convert-silero-vad-to-ggml.py) and shows
//      the pattern: load weights via core_gguf::load_weights, run a
//      small ggml graph per chunk, softmax + argmax on the language
//      logits.
//
// Until that lands, the flag is accepted but the handler returns false
// with an actionable error so users don't wait on a stuck run. Track
// progress in TODO.md under the "native Silero LID port" item.
bool detect_with_silero(
    const float * /*samples*/, int /*n_samples*/,
    const whisper_params & /*p*/,
    crispasr_lid_result & /*out*/)
{
    fprintf(stderr,
            "crispasr[lid]: --lid-backend silero is not implemented yet.\n"
            "               A native GGUF port of Silero's language detector\n"
            "               is on the TODO. For now use --lid-backend whisper\n"
            "               (the default) which ships a 75 MB ggml-tiny.bin\n"
            "               multilingual model with 99-language auto-detect.\n");
    return false;
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
