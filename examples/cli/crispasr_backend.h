// crispasr_backend.h — abstract backend interface for the unified crispasr CLI.
//
// Each model in src/ (parakeet, canary, cohere, qwen3-asr, voxtral, voxtral4b,
// granite_speech) is wrapped by a backend that converts its native result type
// into the common crispasr_segment vector. The whisper backend is a thin
// adapter over whisper_full_parallel() that reads whisper_context segments out
// into the same vector.
//
// The main CLI (cli.cpp) parses args into whisper_params, then either takes
// the historical whisper code path (when params.backend == "" or "whisper")
// or dispatches to crispasr_run_backend() which drives the pipeline:
//   load audio -> VAD slice -> backend->transcribe() -> write outputs.

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

// Forward declaration — defined in cli.cpp. We intentionally reuse the
// existing whisper_params struct (extended with a few new fields) instead of
// introducing a parallel crispasr_params, so users keep the same interface
// they already know from whisper-cli.
struct whisper_params;

// ---------------------------------------------------------------------------
// Common result types
// ---------------------------------------------------------------------------

struct crispasr_token_alt {
    std::string text;
    float       prob = 0.0f;        // probability [0,1]
    int32_t     id = -1;
};

struct crispasr_token {
    std::string text;
    float       confidence = -1.0f; // [0,1], -1 if unavailable
    int64_t     t0 = -1;            // centiseconds, absolute; -1 if unavailable
    int64_t     t1 = -1;
    int32_t     id = -1;            // backend-specific token id, -1 if unavailable
    int64_t     t_dtw = -1;         // whisper DTW token time, -1 if unused
    bool        is_special = false; // whisper: token id >= eot; skipped by wts
    std::vector<crispasr_token_alt> alts; // top-N alternatives (--alt mode)
};

struct crispasr_word {
    std::string text;
    int64_t     t0 = 0;             // centiseconds, absolute
    int64_t     t1 = 0;
};

struct crispasr_segment {
    std::string                 text;
    int64_t                     t0 = 0; // centiseconds, absolute
    int64_t                     t1 = 0;
    std::string                 speaker;          // empty if no diarization
    bool                        speaker_turn_next = false; // whisper tinydiarize
    std::vector<crispasr_word>  words;            // may be empty
    std::vector<crispasr_token> tokens;           // may be empty
};

// ---------------------------------------------------------------------------
// Capability bitmask
// ---------------------------------------------------------------------------

enum crispasr_capability : uint32_t {
    CAP_TIMESTAMPS_NATIVE  = 1u << 0,  // model produces segment timestamps natively
    CAP_TIMESTAMPS_CTC     = 1u << 1,  // can use CTC aligner for timestamps
    CAP_WORD_TIMESTAMPS    = 1u << 2,  // word-level timestamps available
    CAP_TOKEN_CONFIDENCE   = 1u << 3,  // per-token probability
    CAP_LANGUAGE_DETECT    = 1u << 4,  // auto language detection
    CAP_TRANSLATE          = 1u << 5,  // speech translation
    CAP_DIARIZE            = 1u << 6,  // speaker diarization
    CAP_GRAMMAR            = 1u << 7,  // GBNF grammar constraints
    CAP_TEMPERATURE        = 1u << 8,  // temperature/sampling control
    CAP_BEAM_SEARCH        = 1u << 9,  // beam search
    CAP_FLASH_ATTN         = 1u << 10, // flash attention toggle
    CAP_PUNCTUATION_TOGGLE = 1u << 11, // can enable/disable punctuation
    CAP_SRC_TGT_LANGUAGE   = 1u << 12, // separate source/target language (canary)
    CAP_AUTO_DOWNLOAD      = 1u << 13, // supports -m auto via HF hub
    CAP_PARALLEL_PROCESSORS= 1u << 14, // whisper-style n_processors
    CAP_VAD_INTERNAL       = 1u << 15, // backend handles VAD internally (whisper)
};

// ---------------------------------------------------------------------------
// Backend interface
// ---------------------------------------------------------------------------

class CrispasrBackend {
public:
    virtual ~CrispasrBackend() = default;

    // Human-readable name ("whisper", "parakeet", "canary", ...).
    virtual const char * name() const = 0;

    // Bitmask of crispasr_capability flags.
    virtual uint32_t capabilities() const = 0;

    // Load the model and prepare internal state. Returns false on failure.
    // Params are passed by const-ref — backends should only read the fields
    // they care about.
    virtual bool init(const whisper_params & params) = 0;

    // Transcribe a single audio slice of 16 kHz mono PCM samples.
    // t_offset_cs is the absolute start of this slice in centiseconds; all
    // returned segment/word/token timestamps must be absolute (include the
    // offset).
    virtual std::vector<crispasr_segment> transcribe(
        const float * samples, int n_samples,
        int64_t t_offset_cs,
        const whisper_params & params) = 0;

    // Optional stereo-aware overload for backends that can split stereo
    // channels for diarization (currently: whisper). Default
    // implementation falls through to mono transcribe(); override when
    // your backend can use stereo. The two channel buffers each have
    // n_samples_per_channel elements at 16 kHz.
    virtual std::vector<crispasr_segment> transcribe_stereo(
        const float * left_samples,
        const float * right_samples,
        int n_samples_per_channel,
        int64_t t_offset_cs,
        const whisper_params & params)
    {
        // Mono fallback: average L+R into a temporary buffer and dispatch
        // through the main transcribe(). Backends that don't override
        // this method get sane behaviour without any extra wiring.
        std::vector<float> mono((size_t)n_samples_per_channel);
        for (int i = 0; i < n_samples_per_channel; i++) {
            mono[(size_t)i] = 0.5f * (left_samples[i] + right_samples[i]);
        }
        return transcribe(mono.data(), n_samples_per_channel, t_offset_cs, params);
    }

    // Release all resources.
    virtual void shutdown() = 0;
};

// ---------------------------------------------------------------------------
// Factory + auto-detection
// ---------------------------------------------------------------------------

// Create a backend by name. Returns nullptr if the name is not recognised or
// the backend was not compiled in. Caller owns the returned pointer.
std::unique_ptr<CrispasrBackend> crispasr_create_backend(const std::string & name);

// Detect the backend from GGUF metadata. Reads the "general.architecture"
// key using gguf_init_from_file() and maps it to a backend name. Returns
// an empty string if detection fails.
std::string crispasr_detect_backend_from_gguf(const std::string & model_path);

// List the backend names that were compiled into this binary.
std::vector<std::string> crispasr_list_backends();

// Print a human-readable capability matrix for the available backends.
// Called by --list-backends in cli.cpp.
void crispasr_print_backend_matrix();

// ---------------------------------------------------------------------------
// Top-level entry point
// ---------------------------------------------------------------------------

// Drive the non-whisper pipeline end-to-end: resolve model path, create
// backend, load audio, segment via VAD (or fixed chunks), transcribe,
// print to stdout, write output files. Returns a process exit code.
//
// Invoked from cli.cpp main() when params.backend is set to a non-whisper
// backend. The whisper path in cli.cpp is left completely untouched.
int crispasr_run_backend(const whisper_params & params);
