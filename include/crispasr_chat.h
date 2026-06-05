// crispasr_chat.h — public C ABI for text → text chat / LLM inference.
//
// Sits next to crispasr.h (the ASR + audio surface). Implemented in
// src/chat.cpp on top of the private `crispasr-llama-core` static lib
// (vendored llama.cpp). Downstream consumers see ONLY POD structs and
// opaque handles — no llama.h types leak into this header.
//
// Threading
// ---------
//   One call at a time per `crispasr_chat_session_t` — the session
//   serialises its own context internally with a mutex. The intended
//   server pattern is one session per worker thread; multiple sessions
//   over one process are fully supported.
//
// Memory
// ------
//   The KV cache persists across `crispasr_chat_generate` calls inside
//   one session, so multi-turn chats don't re-prefill the full history.
//   Use `crispasr_chat_reset` to flush.
//
// Strings out of this ABI come from malloc — free with
// `crispasr_chat_string_free`.

#ifndef CRISPASR_CHAT_H
#define CRISPASR_CHAT_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef CRISPASR_SHARED
#ifdef _WIN32
#ifdef CRISPASR_BUILD
#define CRISPASR_CHAT_API __declspec(dllexport)
#else
#define CRISPASR_CHAT_API __declspec(dllimport)
#endif
#else
#define CRISPASR_CHAT_API __attribute__((visibility("default")))
#endif
#else
#define CRISPASR_CHAT_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

// ---------------------------------------------------------------------------
// Error reporting
// ---------------------------------------------------------------------------
// Every entry point that can fail accepts a `crispasr_chat_error*` (may be
// NULL). On success the struct is left untouched. On failure `code` is
// set non-zero and `message` carries a short null-terminated diagnostic.
typedef struct crispasr_chat_error {
    int32_t code;
    char message[256];
} crispasr_chat_error;

// ---------------------------------------------------------------------------
// Messages
// ---------------------------------------------------------------------------
// `role` is one of "system", "user", "assistant", "tool" — matches the
// OpenAI chat schema; the chat-template layer will translate the role
// names into whatever the model's template expects.
typedef struct crispasr_chat_message {
    const char* role;
    const char* content;
} crispasr_chat_message;

// ---------------------------------------------------------------------------
// Open params (per-session, model-level)
// ---------------------------------------------------------------------------
// All fields have well-defined defaults via `crispasr_chat_open_params_default`.
typedef struct crispasr_chat_open_params {
    int32_t n_threads;       // generation threads      (default: physical cores)
    int32_t n_threads_batch; // batch / prefill threads (default: n_threads)
    int32_t n_ctx;           // context window in tokens — 0 = model default
    int32_t n_batch;         // logical batch size      (default: 512)
    int32_t n_ubatch;        // physical micro-batch    (default: 512)
    int32_t n_gpu_layers;    // -1 = all, 0 = CPU only  (default: -1)
    bool use_mmap;           // default: true
    bool use_mlock;          // default: false
    bool embeddings;         // future use; keep false  (default: false)
    // If non-NULL overrides the template baked into the GGUF. NULL =
    // read `tokenizer.chat_template` from the model; if absent fall back
    // to "chatml". The string is copied — caller may free immediately.
    const char* chat_template;
} crispasr_chat_open_params;

// Populate `out` with sensible defaults. Safe to call before _open.
CRISPASR_CHAT_API void crispasr_chat_open_params_default(crispasr_chat_open_params* out);

// ---------------------------------------------------------------------------
// Generate params (per-call, sampler-level)
// ---------------------------------------------------------------------------
// Phase 2 + 3 fields: top_k / top_p / repeat_penalty / seed.
typedef struct crispasr_chat_generate_params {
    int32_t max_tokens;    // hard cap on tokens generated (default: 256)
    float temperature;     // 0.0 = greedy                 (default: 0.8)
    int32_t top_k;         // 0 = disabled                 (default: 40)
    float top_p;           // 1.0 = disabled               (default: 0.95)
    float min_p;           // 0.0 = disabled               (default: 0.05)
    float repeat_penalty;  // 1.0 = disabled               (default: 1.1)
    int32_t repeat_last_n; // -1 = ctx size, 0 = disabled  (default: 64)
    uint32_t seed;         // RNG seed; 0 = random         (default: 0)

    // Stop sequences: NULL = none. Generation halts (output is truncated
    // BEFORE the match) the first time any of these substrings appears
    // in the accumulated decoded output.
    const char* const* stop;
    size_t n_stop;

    // If true, the system / user portion is prefilled but assistant
    // generation is suppressed — useful for measuring prompt cost.
    bool prefill_only;
} crispasr_chat_generate_params;

CRISPASR_CHAT_API void crispasr_chat_generate_params_default(crispasr_chat_generate_params* out);

// ---------------------------------------------------------------------------
// Session lifecycle
// ---------------------------------------------------------------------------
typedef struct crispasr_chat_session crispasr_chat_session;
typedef crispasr_chat_session* crispasr_chat_session_t;

// Open a session from a GGUF chat model on disk. Returns NULL on failure
// and fills `err` when non-NULL. `params` may be NULL — defaults apply.
CRISPASR_CHAT_API crispasr_chat_session_t crispasr_chat_open(const char* model_path,
                                                             const crispasr_chat_open_params* params,
                                                             crispasr_chat_error* err);

// Free the session and its KV cache. Safe to call with NULL.
CRISPASR_CHAT_API void crispasr_chat_close(crispasr_chat_session_t s);

// Clear the KV cache so the next _generate re-prefills from scratch. Call
// when starting a new conversation in a reused session.
CRISPASR_CHAT_API int32_t crispasr_chat_reset(crispasr_chat_session_t s, crispasr_chat_error* err);

// ---------------------------------------------------------------------------
// One-shot generate
// ---------------------------------------------------------------------------
// Applies the model's chat template to `messages`, prefills, runs
// generation to `max_tokens` or a stop sequence, and returns a freshly
// malloc'd UTF-8 string holding the assistant's reply. Free with
// `crispasr_chat_string_free`. Returns NULL on failure (sets `err`).
CRISPASR_CHAT_API char* crispasr_chat_generate(crispasr_chat_session_t s, const crispasr_chat_message* messages,
                                               size_t n_messages, const crispasr_chat_generate_params* params,
                                               crispasr_chat_error* err);

// ---------------------------------------------------------------------------
// Streaming generate
// ---------------------------------------------------------------------------
// Fires `on_token` once per detokenised UTF-8 chunk (typically one piece
// per llama_token). The chunk pointer is valid only during the callback.
// `user` is forwarded verbatim. Returns 0 on clean completion (including
// stop-sequence / EOG termination) and non-zero on failure.
typedef void (*crispasr_chat_on_token)(const char* utf8_chunk, void* user);

CRISPASR_CHAT_API int32_t crispasr_chat_generate_stream(crispasr_chat_session_t s,
                                                        const crispasr_chat_message* messages, size_t n_messages,
                                                        const crispasr_chat_generate_params* params,
                                                        crispasr_chat_on_token on_token, void* user,
                                                        crispasr_chat_error* err);

// ---------------------------------------------------------------------------
// Memory + introspection
// ---------------------------------------------------------------------------
// Returns the name of the chat template the session resolved against
// (e.g. "chatml", "llama3", "gemma"). Pointer is owned by the session
// and stays valid until _close.
CRISPASR_CHAT_API const char* crispasr_chat_template_name(crispasr_chat_session_t s);

// Returns the context window in tokens.
CRISPASR_CHAT_API int32_t crispasr_chat_n_ctx(crispasr_chat_session_t s);

// Pre-flight memory estimate for a GGUF chat model on disk. Returns the
// approximate working-set in bytes (weights + KV cache + activations) or
// 0 if it could not be estimated. Mirrors the shape of crispasr's
// existing `crispasr_memory_estimate_*` family; lets CrisperWeaver's
// pre-flight guard short-circuit on low-RAM devices.
CRISPASR_CHAT_API size_t crispasr_chat_memory_estimate(const char* model_path, const crispasr_chat_open_params* params,
                                                       crispasr_chat_error* err);

// Free a malloc'd string returned by `crispasr_chat_generate`.
CRISPASR_CHAT_API void crispasr_chat_string_free(char* s);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // CRISPASR_CHAT_H
