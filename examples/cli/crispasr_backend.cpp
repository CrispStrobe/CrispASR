// crispasr_backend.cpp — backend factory, auto-detection, and helpers.

#include "crispasr_backend.h"
#include "whisper_params.h"

// Forward declarations of per-backend constructors. Each is implemented in
// its own crispasr_backend_X.cpp file and compiled only if the backend's
// library is linked in.
std::unique_ptr<CrispasrBackend> crispasr_make_whisper_backend();
std::unique_ptr<CrispasrBackend> crispasr_make_parakeet_backend();
std::unique_ptr<CrispasrBackend> crispasr_make_canary_backend();
std::unique_ptr<CrispasrBackend> crispasr_make_cohere_backend();
std::unique_ptr<CrispasrBackend> crispasr_make_granite_backend();
std::unique_ptr<CrispasrBackend> crispasr_make_granite_nle_backend();
std::unique_ptr<CrispasrBackend> crispasr_make_voxtral_backend();
std::unique_ptr<CrispasrBackend> crispasr_make_voxtral4b_backend();
std::unique_ptr<CrispasrBackend> crispasr_make_qwen3_backend();
std::unique_ptr<CrispasrBackend> crispasr_make_fastconformer_ctc_backend();
std::unique_ptr<CrispasrBackend> crispasr_make_wav2vec2_backend();
std::unique_ptr<CrispasrBackend> crispasr_make_vibevoice_backend();
std::unique_ptr<CrispasrBackend> crispasr_make_qwen3_tts_backend();
std::unique_ptr<CrispasrBackend> crispasr_make_orpheus_backend();
std::unique_ptr<CrispasrBackend> crispasr_make_chatterbox_backend();
std::unique_ptr<CrispasrBackend> crispasr_make_kokoro_backend();
std::unique_ptr<CrispasrBackend> crispasr_make_glm_asr_backend();
std::unique_ptr<CrispasrBackend> crispasr_make_kyutai_stt_backend();
std::unique_ptr<CrispasrBackend> crispasr_make_firered_asr_backend();
std::unique_ptr<CrispasrBackend> crispasr_make_moonshine_backend();
std::unique_ptr<CrispasrBackend> crispasr_make_moonshine_streaming_backend();
std::unique_ptr<CrispasrBackend> crispasr_make_gemma4_e2b_backend();
std::unique_ptr<CrispasrBackend> crispasr_make_omniasr_backend();
std::unique_ptr<CrispasrBackend> crispasr_make_mimo_asr_backend();

#include "ggml.h"
#include "gguf.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Factory
// ---------------------------------------------------------------------------

std::unique_ptr<CrispasrBackend> crispasr_create_backend(const std::string& name) {
    if (name == "whisper")
        return crispasr_make_whisper_backend();
    if (name == "parakeet")
        return crispasr_make_parakeet_backend();
    if (name == "canary")
        return crispasr_make_canary_backend();
    if (name == "cohere")
        return crispasr_make_cohere_backend();
    if (name == "granite" || name == "granite-4.1" || name == "granite-4.1-plus")
        return crispasr_make_granite_backend();
    if (name == "granite-4.1-nar" || name == "granite-nar")
        return crispasr_make_granite_nle_backend();
    if (name == "voxtral")
        return crispasr_make_voxtral_backend();
    if (name == "voxtral4b")
        return crispasr_make_voxtral4b_backend();
    if (name == "qwen3")
        return crispasr_make_qwen3_backend();
    if (name == "fastconformer-ctc")
        return crispasr_make_fastconformer_ctc_backend();
    if (name == "wav2vec2" || name == "hubert" || name == "data2vec")
        return crispasr_make_wav2vec2_backend();
    if (name == "vibevoice" || name == "vibevoice-tts")
        return crispasr_make_vibevoice_backend();
    if (name == "qwen3-tts" || name == "qwen3_tts" || name == "qwen3tts" || name == "qwen3-tts-customvoice" ||
        name == "qwen3tts-customvoice" || name == "qwen3-tts-cv" || name == "qwen3-tts-1.7b-base" ||
        name == "qwen3-tts-1.7b" || name == "qwen3-tts-1.7b-customvoice" || name == "qwen3-tts-1.7b-cv" ||
        name == "qwen3-tts-1.7b-voicedesign" || name == "qwen3-tts-voicedesign" || name == "qwen3-tts-vd")
        return crispasr_make_qwen3_tts_backend();
    if (name == "orpheus" || name == "orpheus-tts" || name == "orpheus3b" || name == "kartoffel-orpheus" ||
        name == "kartoffel_orpheus" || name == "kartoffel-orpheus-de-natural" ||
        name == "kartoffel-orpheus-de-synthetic" || name == "kartoffel-orpheus-natural" ||
        name == "kartoffel-orpheus-synthetic" || name == "lex-au-orpheus-de" || name == "lex-au-orpheus")
        return crispasr_make_orpheus_backend();
    if (name == "chatterbox" || name == "chatterbox-tts" || name == "chatterbox-base" ||
        name == "chatterbox-turbo" || name == "chatterbox_turbo" || name == "kartoffelbox" ||
        name == "kartoffelbox-turbo" || name == "kartoffelbox_turbo" || name == "lahgtna" ||
        name == "lahgtna-chatterbox" || name == "lahgtna-chatterbox-v1")
        return crispasr_make_chatterbox_backend();
    if (name == "kokoro" || name == "styletts2" || name == "styletts2-ljspeech" || name == "kokoro-tts")
        return crispasr_make_kokoro_backend();
    if (name == "glm-asr" || name == "glmasr" || name == "glm" || name == "glm_asr")
        return crispasr_make_glm_asr_backend();
    if (name == "kyutai-stt" || name == "kyutai" || name == "moshi-stt")
        return crispasr_make_kyutai_stt_backend();
    if (name == "firered-asr" || name == "firered")
        return crispasr_make_firered_asr_backend();
    if (name == "moonshine-streaming")
        return crispasr_make_moonshine_streaming_backend();
    if (name == "gemma4-e2b" || name == "gemma4e2b" || name == "gemma4")
        return crispasr_make_gemma4_e2b_backend();
    if (name == "moonshine")
        return crispasr_make_moonshine_backend();
    if (name == "omniasr" || name == "omniasr-ctc" || name == "omniasr-llm")
        return crispasr_make_omniasr_backend();
    if (name == "mimo-asr" || name == "mimo_asr" || name == "mimoasr")
        return crispasr_make_mimo_asr_backend();

    fprintf(stderr, "crispasr: error: unknown backend '%s'\n", name.c_str());
    return nullptr;
}

std::vector<std::string> crispasr_list_backends() {
    return {
        "whisper",
        "parakeet",
        "canary",
        "cohere",
        "granite",
        "granite-4.1",
        "granite-4.1-plus",
        "granite-4.1-nar",
        "voxtral",
        "voxtral4b",
        "qwen3",
        "fastconformer-ctc",
        "wav2vec2",
        "hubert",
        "data2vec",
        "vibevoice",
        "qwen3-tts",
        "qwen3-tts-customvoice",
        "qwen3-tts-1.7b-base",
        "qwen3-tts-1.7b-customvoice",
        "qwen3-tts-1.7b-voicedesign",
        "orpheus",
        "lex-au-orpheus-de",
        "kartoffel-orpheus-de-natural",
        "kartoffel-orpheus-de-synthetic",
        "chatterbox",
        "chatterbox-turbo",
        "kartoffelbox-turbo",
        "lahgtna-chatterbox",
        "kokoro",
        "glm-asr",
        "kyutai-stt",
        "firered-asr",
        "moonshine",
        "moonshine-streaming",
        "gemma4-e2b",
        "omniasr",
        "omniasr-llm",
        "mimo-asr",
    };
}

// ---------------------------------------------------------------------------
// Capability matrix for --list-backends
// ---------------------------------------------------------------------------

struct feature_col {
    const char* label;
    uint32_t flag;
};

static constexpr feature_col kFeatures[] = {
    {"ts-native", CAP_TIMESTAMPS_NATIVE},
    {"ts-ctc", CAP_TIMESTAMPS_CTC},
    {"word-ts", CAP_WORD_TIMESTAMPS},
    {"tok-conf", CAP_TOKEN_CONFIDENCE},
    {"lang-detect", CAP_LANGUAGE_DETECT},
    {"translate", CAP_TRANSLATE},
    {"diarize", CAP_DIARIZE},
    {"grammar", CAP_GRAMMAR},
    {"temperature", CAP_TEMPERATURE},
    {"beam", CAP_BEAM_SEARCH},
    {"flash", CAP_FLASH_ATTN},
    {"punctuation", CAP_PUNCTUATION_TOGGLE},
    {"src/tgt lang", CAP_SRC_TGT_LANGUAGE},
    {"auto-dl", CAP_AUTO_DOWNLOAD},
    {"tts", CAP_TTS},
};

void crispasr_print_backend_matrix() {
    const auto backends = crispasr_list_backends();

    // Column widths
    size_t name_w = 8;
    for (const auto& b : backends)
        if (b.size() > name_w)
            name_w = b.size();

    // Header
    printf("crispasr backends (%zu):\n\n", backends.size());
    printf("  %-*s", (int)name_w, "backend");
    for (const auto& f : kFeatures)
        printf(" %-12s", f.label);
    printf("\n  ");
    for (size_t i = 0; i < name_w; i++)
        printf("-");
    for (size_t i = 0; i < sizeof(kFeatures) / sizeof(kFeatures[0]); i++) {
        printf(" ------------");
    }
    printf("\n");

    // Each row: instantiate the backend just to read its capability bitmask.
    for (const auto& name : backends) {
        uint32_t caps = 0;
        auto be = crispasr_create_backend(name);
        if (be)
            caps = be->capabilities();
        // backend destroyed when unique_ptr goes out of scope
        printf("  %-*s", (int)name_w, name.c_str());
        for (const auto& f : kFeatures) {
            printf(" %-12s", (caps & f.flag) ? "   Y" : "    -");
        }
        printf("\n");
    }
    printf("\nUse --backend NAME to force a specific backend. When omitted, the\n");
    printf("backend is auto-detected from GGUF metadata or the filename.\n");
    printf("\n");
    printf("Language detection: backends that don't advertise lang-detect\n");
    printf("natively (cohere, canary, granite, voxtral, voxtral4b) can still\n");
    printf("accept `-l auto` via the LID pre-step. Pick the provider with\n");
    printf("--lid-backend whisper|silero (whisper-tiny is the default).\n");
}

// Stable cap-name slugs for JSON output. These map to the kFeatures
// table above but use long, kebab-cased names that are nicer for
// machine consumers (test-all-backends.py, etc.) than the table
// labels which are tuned for terminal-column-width.
struct cap_slug {
    const char* slug;
    uint32_t flag;
};
static constexpr cap_slug kCapSlugs[] = {
    {"timestamps-native", CAP_TIMESTAMPS_NATIVE},
    {"timestamps-ctc", CAP_TIMESTAMPS_CTC},
    {"word-timestamps", CAP_WORD_TIMESTAMPS},
    {"token-confidence", CAP_TOKEN_CONFIDENCE},
    {"language-detect", CAP_LANGUAGE_DETECT},
    {"translate", CAP_TRANSLATE},
    {"diarize", CAP_DIARIZE},
    {"grammar", CAP_GRAMMAR},
    {"temperature", CAP_TEMPERATURE},
    {"beam-search", CAP_BEAM_SEARCH},
    {"flash-attn", CAP_FLASH_ATTN},
    {"punctuation-toggle", CAP_PUNCTUATION_TOGGLE},
    {"src-tgt-language", CAP_SRC_TGT_LANGUAGE},
    {"auto-download", CAP_AUTO_DOWNLOAD},
    {"parallel-processors", CAP_PARALLEL_PROCESSORS},
    {"vad-internal", CAP_VAD_INTERNAL},
    {"tts", CAP_TTS},
};

void crispasr_print_backend_matrix_json() {
    const auto backends = crispasr_list_backends();

    printf("{\n  \"backends\": [\n");
    bool first_backend = true;
    for (const auto& name : backends) {
        uint32_t caps = 0;
        auto be = crispasr_create_backend(name);
        if (be)
            caps = be->capabilities();
        // backend destroyed when unique_ptr goes out of scope

        if (!first_backend)
            printf(",\n");
        first_backend = false;
        printf("    {\n");
        printf("      \"name\": \"%s\",\n", name.c_str());
        printf("      \"caps_bitmask\": %u,\n", caps);
        printf("      \"caps\": [");
        bool first_cap = true;
        for (const auto& c : kCapSlugs) {
            if (caps & c.flag) {
                if (!first_cap)
                    printf(", ");
                first_cap = false;
                printf("\"%s\"", c.slug);
            }
        }
        printf("]\n");
        printf("    }");
    }
    printf("\n  ]\n}\n");
}

// ---------------------------------------------------------------------------
// GGUF auto-detection
// ---------------------------------------------------------------------------

// Read the "general.architecture" key from a GGUF file and map it to a
// backend name. Uses gguf_init_from_file() — which lives in ggml — so this
// is cheap: only the metadata is parsed, not the weight tensors.
//
// Mappings are based on the value that each model's converter writes into
// the GGUF file. When a converter doesn't write this key we fall back to
// filename heuristics.
std::string crispasr_detect_backend_from_gguf(const std::string& model_path) {
    if (model_path.empty())
        return "";

    // ---- Pass 1: filename heuristics ----
    //
    // Try filename matching first. This avoids two problems:
    //   1. Whisper's legacy ggml-*.bin files are not GGUF; calling
    //      gguf_init_from_file() on them prints a confusing stderr warning.
    //   2. It's a fast path that covers nearly every real-world case
    //      (users consistently name their models after the architecture).
    // Extract filename (after last / or \) for matching — avoid false
    // positives from directory names (e.g. "test_cohere/" matching "cohere").
    std::string fname = model_path;
    auto sep = fname.find_last_of("/\\");
    if (sep != std::string::npos)
        fname = fname.substr(sep + 1);
    auto contains_ci = [&](const char* needle) {
        std::string lo;
        lo.reserve(fname.size());
        for (char c : fname)
            lo += (char)std::tolower((unsigned char)c);
        return lo.find(needle) != std::string::npos;
    };

    if (contains_ci("voxtral") && contains_ci("4b"))
        return "voxtral4b";
    if (contains_ci("voxtral"))
        return "voxtral";
    if (contains_ci("parakeet"))
        return "parakeet";
    // Check "fastconformer-ctc" / "stt_en_fc_ctc" style filenames before
    // the broader "canary" match so users who drop a NeMo standalone
    // model next to a canary aligner pick the right backend.
    if (contains_ci("fastconformer") && contains_ci("ctc"))
        return "fastconformer-ctc";
    if (contains_ci("omniasr"))
        return "omniasr";
    if (contains_ci("wav2vec2"))
        return "wav2vec2";
    if (contains_ci("hubert"))
        return "wav2vec2";
    if (contains_ci("data2vec"))
        return "wav2vec2";
    if (contains_ci("vibevoice"))
        return "vibevoice";
    if (contains_ci("fireredpunc"))
        return "fireredpunc";
    if (contains_ci("canary"))
        return "canary";
    if (contains_ci("cohere"))
        return "cohere";
    if (contains_ci("qwen3") && contains_ci("asr"))
        return "qwen3";
    if (contains_ci("qwen3") && contains_ci("tts"))
        return "qwen3-tts";
    if (contains_ci("orpheus") || contains_ci("kartoffel-orpheus") || contains_ci("kartoffel_orpheus"))
        return "orpheus";
    if (contains_ci("kartoffelbox") || contains_ci("lahgtna") || contains_ci("chatterbox"))
        return "chatterbox";
    if (contains_ci("kokoro"))
        return "kokoro";
    if (contains_ci("styletts") && contains_ci("ljspeech"))
        return "kokoro";
    if (contains_ci("granite") && contains_ci("nar"))
        return "granite-4.1-nar";
    if (contains_ci("granite") && contains_ci("speech"))
        return "granite";
    if (contains_ci("glm") && contains_ci("asr"))
        return "glm-asr";
    if (contains_ci("kyutai") && contains_ci("stt"))
        return "kyutai-stt";
    if (contains_ci("moshi") && contains_ci("stt"))
        return "kyutai-stt";
    if (contains_ci("firered") && (contains_ci("asr") || contains_ci("lid")))
        return "firered-asr";
    if (contains_ci("gemma") && (contains_ci("e2b") || contains_ci("4-e2b")))
        return "gemma4-e2b";
    if (contains_ci("moonshine") && contains_ci("streaming"))
        return "moonshine-streaming";
    if (contains_ci("moonshine"))
        return "moonshine";
    if (contains_ci("ggml-") && contains_ci(".bin"))
        return "whisper";

    // ---- Pass 2: GGUF metadata ----
    //
    // Only reached when the filename didn't clearly identify a backend.
    // Reads just the "general.architecture" key; no weight tensors.
    struct gguf_init_params gip = {/*.no_alloc=*/true, /*.ctx=*/nullptr};
    gguf_context* gctx = gguf_init_from_file(model_path.c_str(), gip);
    if (!gctx)
        return "";

    std::string result;
    const int key = gguf_find_key(gctx, "general.architecture");
    if (key >= 0) {
        const char* arch = gguf_get_val_str(gctx, key);
        if (arch) {
            const std::string a = arch;
            if (a == "whisper")
                result = "whisper";
            else if (a == "parakeet")
                result = "parakeet";
            else if (a == "parakeet-tdt" || a == "parakeet-ja" || a == "parakeet_ja")
                result = "parakeet";
            else if (a == "canary")
                result = "canary";
            else if (a == "canary-ctc")
                result = "canary";
            else if (a == "cohere")
                result = "cohere";
            else if (a == "cohere-transcribe")
                result = "cohere";
            else if (a == "qwen3-asr" || a == "qwen3_asr" || a == "qwen3asr")
                result = "qwen3";
            else if (a == "qwen3-tts" || a == "qwen3_tts" || a == "qwen3tts")
                result = "qwen3-tts";
            else if (a == "orpheus")
                result = "orpheus";
            else if (a == "kokoro" || a == "styletts2" || a == "styletts2-ljspeech")
                result = "kokoro";
            else if (a == "voxtral")
                result = "voxtral";
            else if (a == "voxtral4b" || a == "voxtral-4b" || a == "voxtral_4b")
                result = "voxtral4b";
            else if (a == "granite-speech" || a == "granite_speech" || a == "granitespeech")
                result = "granite";
            else if (a == "granite-nle" || a == "granite_nle" || a == "granitenle")
                result = "granite-4.1-nar";
            else if (a == "wav2vec2" || a == "wav2vec2-ctc")
                result = "wav2vec2";
            else if (a == "vibevoice" || a == "vibevoice-asr" || a == "vibevoice_asr")
                result = "vibevoice";
            else if (a == "fastconformer-ctc" || a == "stt-fastconformer-ctc" || a == "stt_fastconformer_ctc")
                result = "fastconformer-ctc";
            else if (a == "glmasr" || a == "glm-asr" || a == "glm_asr")
                result = "glm-asr";
            else if (a == "kyutai-stt" || a == "kyutai_stt" || a == "kyutaistt")
                result = "kyutai-stt";
            else if (a == "firered-asr" || a == "firered_asr" || a == "firereadasr" || a == "firered-lid" ||
                     a == "firered_lid")
                result = "firered-asr";
            else if (a == "moonshine_streaming")
                return "moonshine-streaming";
            else if (a == "gemma4e2b" || a == "gemma4_e2b")
                return "gemma4-e2b";
            else if (a == "moonshine" || a == "moonshine-tiny" || a == "moonshine-base")
                result = "moonshine";
            else if (a == "omniasr-ctc" || a == "omniasr_ctc" || a == "omniasr" || a == "omniasr-llm" ||
                     a == "omniasr_llm")
                result = "omniasr";
            else if (a == "hubert" || a == "hubert-ctc")
                result = "wav2vec2";
            else if (a == "data2vec" || a == "data2vec-audio" || a == "data2vec_audio")
                result = "wav2vec2";
            else if (a == "fireredpunc")
                result = "fireredpunc";
            else if (a == "mimo_asr" || a == "mimo-asr")
                result = "mimo-asr";
        }
    }
    gguf_free(gctx);
    return result;
}
