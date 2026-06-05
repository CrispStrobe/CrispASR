// chat.cpp — implementation of the public crispasr_chat_* C ABI.
//
// Sits on the private `crispasr-llama-core` static lib (vendored llama.cpp).
// llama.h types stay inside this translation unit — none leak into
// include/crispasr_chat.h. See docs/prompts/chat-abi.md for the
// full design rationale.
//
// Threading
//   One `crispasr_chat_session` carries its own mutex; concurrent calls
//   on the same handle serialise. Multiple sessions in the same process
//   run independently.
//
// KV cache
//   Persisted across `crispasr_chat_generate` calls inside one session
//   so a multi-turn chat doesn't re-prefill the history. `_reset` calls
//   `llama_memory_clear` on the KV state.

#include "crispasr_chat.h"

#include "llama.h"

#include <algorithm>
#include <atomic>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace {

// ---------------------------------------------------------------------------
// Process-wide one-shot init for the llama backend. llama_backend_init
// registers ggml ops + memory allocators globally; calling it twice is a
// no-op but doing so under multi-session race is safer behind once_flag.
// ---------------------------------------------------------------------------
std::once_flag g_llama_backend_init_flag;
void ensure_llama_backend_init() {
    std::call_once(g_llama_backend_init_flag, []() { llama_backend_init(); });
}

void set_err(crispasr_chat_error* err, int32_t code, const char* fmt, ...) {
    if (!err) {
        return;
    }
    err->code = code;
    va_list ap;
    va_start(ap, fmt);
    std::vsnprintf(err->message, sizeof(err->message), fmt, ap);
    va_end(ap);
}

// Detokenize one token to UTF-8. `special=true` so chat templates'
// control tokens are visible to stop-sequence scanning when needed; the
// caller decides what to strip from final output (we strip EOG/EOT).
std::string piece_to_string(const llama_vocab* vocab, llama_token token, bool special) {
    std::vector<char> buf(64);
    int32_t n = llama_token_to_piece(vocab, token, buf.data(), (int32_t)buf.size(), 0, special);
    if (n < 0) {
        buf.resize(-n);
        n = llama_token_to_piece(vocab, token, buf.data(), (int32_t)buf.size(), 0, special);
        if (n < 0) {
            return {};
        }
    }
    return std::string(buf.data(), (size_t)n);
}

std::vector<llama_token> tokenize(const llama_vocab* vocab, const std::string& text, bool add_special,
                                  bool parse_special) {
    if (text.empty()) {
        return {};
    }
    // First call with n_tokens_max=0 returns -required_size.
    int32_t n = -llama_tokenize(vocab, text.c_str(), (int32_t)text.size(), nullptr, 0, add_special, parse_special);
    if (n <= 0) {
        return {};
    }
    std::vector<llama_token> out((size_t)n);
    int32_t written = llama_tokenize(vocab, text.c_str(), (int32_t)text.size(), out.data(), (int32_t)out.size(),
                                     add_special, parse_special);
    if (written < 0) {
        return {};
    }
    out.resize((size_t)written);
    return out;
}

// Find the earliest occurrence of any `stop` substring inside `acc`.
// Returns std::string::npos if none. On match, sets `stop_idx` to the
// index in `params->stop` that matched (purely informative).
size_t find_first_stop(const std::string& acc, const char* const* stop, size_t n_stop, size_t* stop_idx) {
    size_t earliest = std::string::npos;
    for (size_t i = 0; i < n_stop; ++i) {
        if (!stop[i] || !*stop[i]) {
            continue;
        }
        const size_t pos = acc.find(stop[i]);
        if (pos != std::string::npos && pos < earliest) {
            earliest = pos;
            if (stop_idx) {
                *stop_idx = i;
            }
        }
    }
    return earliest;
}

// Build the prompt string by applying the model's chat template via
// llama.cpp's pre-defined list (chatml / llama3 / gemma / qwen / …).
// Returns false on failure.
bool apply_chat_template(const char* tmpl, const crispasr_chat_message* msgs, size_t n, bool add_ass,
                         std::string& out) {
    if (n == 0) {
        out.clear();
        return true;
    }
    // llama.cpp's signature expects llama_chat_message (same shape as ours).
    std::vector<llama_chat_message> chat;
    chat.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        chat.push_back({msgs[i].role ? msgs[i].role : "user", msgs[i].content ? msgs[i].content : ""});
    }
    // First call with length=0 returns required buffer size (or negative on bad template).
    int32_t need = llama_chat_apply_template(tmpl, chat.data(), chat.size(), add_ass, nullptr, 0);
    if (need < 0) {
        return false;
    }
    out.resize((size_t)need);
    int32_t written =
        llama_chat_apply_template(tmpl, chat.data(), chat.size(), add_ass, out.data(), (int32_t)out.size());
    if (written < 0) {
        return false;
    }
    out.resize((size_t)written);
    return true;
}

// Resolve the chat template name. Caller-override > GGUF meta > chatml.
// Returned pointer is owned by either `s->tmpl_owned` or the model — we
// pin it into `s->tmpl_owned` either way for stable lifetime.
const char* resolve_template(const llama_model* model, const char* override_tmpl, std::string& tmpl_owned) {
    if (override_tmpl && *override_tmpl) {
        tmpl_owned = override_tmpl;
        return tmpl_owned.c_str();
    }
    const char* baked = llama_model_chat_template(model, /*name=*/nullptr);
    if (baked && *baked) {
        tmpl_owned = baked;
        return tmpl_owned.c_str();
    }
    tmpl_owned = "chatml";
    return tmpl_owned.c_str();
}

} // namespace

// ---------------------------------------------------------------------------
// Session impl
// ---------------------------------------------------------------------------
struct crispasr_chat_session {
    llama_model* model = nullptr;
    llama_context* ctx = nullptr;
    const llama_vocab* vocab = nullptr;

    std::string tmpl; // resolved chat template name
    int32_t n_ctx = 0;
    int32_t n_threads = 1;
    int32_t n_threads_batch = 1;

    // History of tokens already in the KV cache for this conversation.
    // We tokenise + decode only the NEW prefix on each _generate call;
    // a divergent history triggers a full reset.
    std::vector<llama_token> history;

    // Mutex serialising one-call-at-a-time per session.
    std::mutex mu;
};

// ---------------------------------------------------------------------------
// Default params
// ---------------------------------------------------------------------------
extern "C" void crispasr_chat_open_params_default(crispasr_chat_open_params* out) {
    if (!out) {
        return;
    }
    const int32_t hw = (int32_t)std::max(1u, std::thread::hardware_concurrency());
    out->n_threads = std::min(hw, 8);
    out->n_threads_batch = out->n_threads;
    out->n_ctx = 0; // model default
    out->n_batch = 512;
    out->n_ubatch = 512;
    out->n_gpu_layers = -1; // all
    out->use_mmap = true;
    out->use_mlock = false;
    out->embeddings = false;
    out->chat_template = nullptr;
}

extern "C" void crispasr_chat_generate_params_default(crispasr_chat_generate_params* out) {
    if (!out) {
        return;
    }
    out->max_tokens = 256;
    out->temperature = 0.8f;
    out->top_k = 40;
    out->top_p = 0.95f;
    out->min_p = 0.05f;
    out->repeat_penalty = 1.10f;
    out->repeat_last_n = 64;
    out->seed = 0;
    out->stop = nullptr;
    out->n_stop = 0;
    out->prefill_only = false;
}

// ---------------------------------------------------------------------------
// Open / close
// ---------------------------------------------------------------------------
extern "C" crispasr_chat_session_t crispasr_chat_open(const char* model_path, const crispasr_chat_open_params* params,
                                                      crispasr_chat_error* err) {
    if (!model_path || !*model_path) {
        set_err(err, 1, "model_path is null or empty");
        return nullptr;
    }
    ensure_llama_backend_init();

    crispasr_chat_open_params p;
    crispasr_chat_open_params_default(&p);
    if (params) {
        p = *params;
    }

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = p.n_gpu_layers;
    mparams.use_mmap = p.use_mmap;
    mparams.use_mlock = p.use_mlock;

    llama_model* model = llama_model_load_from_file(model_path, mparams);
    if (!model) {
        set_err(err, 2, "llama_model_load_from_file failed for %s", model_path);
        return nullptr;
    }

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = (uint32_t)std::max(0, p.n_ctx);
    cparams.n_batch = (uint32_t)std::max(1, p.n_batch);
    cparams.n_ubatch = (uint32_t)std::max(1, p.n_ubatch);
    cparams.n_threads = std::max(1, p.n_threads);
    cparams.n_threads_batch = std::max(1, p.n_threads_batch);
    cparams.embeddings = p.embeddings;

    llama_context* ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        llama_model_free(model);
        set_err(err, 3, "llama_init_from_model failed");
        return nullptr;
    }

    // cppcheck-suppress legacyUninitvar
    // ^ false positive: `new (std::nothrow) T{}` value-initializes; `s` is
    //   either nullptr or fully zeroed before the null check below.
    auto* s = new (std::nothrow) crispasr_chat_session{};
    if (!s) {
        llama_free(ctx);
        llama_model_free(model);
        set_err(err, 4, "out of memory");
        return nullptr;
    }
    s->model = model;
    s->ctx = ctx;
    s->vocab = llama_model_get_vocab(model);
    s->n_ctx = (int32_t)llama_n_ctx(ctx);
    s->n_threads = cparams.n_threads;
    s->n_threads_batch = cparams.n_threads_batch;
    (void)resolve_template(model, p.chat_template, s->tmpl);
    return s;
}

extern "C" void crispasr_chat_close(crispasr_chat_session_t s) {
    if (!s) {
        return;
    }
    if (s->ctx) {
        llama_free(s->ctx);
    }
    if (s->model) {
        llama_model_free(s->model);
    }
    delete s;
}

extern "C" int32_t crispasr_chat_reset(crispasr_chat_session_t s, crispasr_chat_error* err) {
    if (!s) {
        set_err(err, 1, "session is null");
        return 1;
    }
    std::lock_guard<std::mutex> guard(s->mu);
    llama_memory_clear(llama_get_memory(s->ctx), /*data=*/true);
    s->history.clear();
    return 0;
}

extern "C" const char* crispasr_chat_template_name(crispasr_chat_session_t s) {
    return s ? s->tmpl.c_str() : nullptr;
}

extern "C" int32_t crispasr_chat_n_ctx(crispasr_chat_session_t s) {
    return s ? s->n_ctx : 0;
}

// ---------------------------------------------------------------------------
// Generation core — shared by one-shot and streaming variants.
// ---------------------------------------------------------------------------
namespace {

// Build a sampler chain matching `params`. Caller owns the result via
// llama_sampler_free. Order matches llama.cpp's recommended layout in
// examples/main: penalties → top_k → top_p → min_p → temp → dist.
llama_sampler* build_sampler_chain(const llama_vocab* vocab, const crispasr_chat_generate_params& gp) {
    llama_sampler_chain_params sp = llama_sampler_chain_default_params();
    sp.no_perf = true;
    llama_sampler* chain = llama_sampler_chain_init(sp);
    if (!chain) {
        return nullptr;
    }
    if (gp.repeat_penalty != 1.0f && gp.repeat_last_n != 0) {
        llama_sampler_chain_add(
            chain, llama_sampler_init_penalties(gp.repeat_last_n, gp.repeat_penalty, /*freq=*/0.0f, /*present=*/0.0f));
    }
    if (gp.temperature <= 0.0f) {
        // Greedy — ignores other sampling params.
        llama_sampler_chain_add(chain, llama_sampler_init_greedy());
        return chain;
    }
    if (gp.top_k > 0) {
        llama_sampler_chain_add(chain, llama_sampler_init_top_k(gp.top_k));
    }
    if (gp.top_p > 0.0f && gp.top_p < 1.0f) {
        llama_sampler_chain_add(chain, llama_sampler_init_top_p(gp.top_p, /*min_keep=*/1));
    }
    if (gp.min_p > 0.0f) {
        llama_sampler_chain_add(chain, llama_sampler_init_min_p(gp.min_p, /*min_keep=*/1));
    }
    llama_sampler_chain_add(chain, llama_sampler_init_temp(gp.temperature));
    llama_sampler_chain_add(chain, llama_sampler_init_dist(gp.seed));
    (void)vocab;
    return chain;
}

// Common generation loop. `on_token`, if non-null, fires once per
// detokenised piece. `out` accumulates the full text. Returns 0 on
// success, non-zero on decode failure.
int32_t generate_loop(crispasr_chat_session* s, const std::vector<llama_token>& prompt_new,
                      const crispasr_chat_generate_params& gp, crispasr_chat_on_token on_token, void* user,
                      std::string& out, crispasr_chat_error* err) {
    // -- Prefill the prompt prefix in one (or several) batches. --
    if (!prompt_new.empty()) {
        // Mutable copy because llama_batch_get_one takes a non-const ptr.
        std::vector<llama_token> tokens = prompt_new;
        llama_batch batch = llama_batch_get_one(tokens.data(), (int32_t)tokens.size());
        if (llama_decode(s->ctx, batch) != 0) {
            set_err(err, 10, "llama_decode failed during prefill");
            return 10;
        }
        s->history.insert(s->history.end(), prompt_new.begin(), prompt_new.end());
    }
    if (gp.prefill_only) {
        return 0;
    }

    // -- Build sampler chain for this generate call. --
    std::unique_ptr<llama_sampler, decltype(&llama_sampler_free)> smpl(build_sampler_chain(s->vocab, gp),
                                                                       &llama_sampler_free);
    if (!smpl) {
        set_err(err, 11, "sampler chain init failed");
        return 11;
    }

    // -- Decode loop. --
    const int32_t max_tokens = gp.max_tokens > 0 ? gp.max_tokens : 256;
    llama_token new_token = 0;
    for (int32_t i = 0; i < max_tokens; ++i) {
        new_token = llama_sampler_sample(smpl.get(), s->ctx, -1);

        if (llama_vocab_is_eog(s->vocab, new_token)) {
            break;
        }

        const std::string piece = piece_to_string(s->vocab, new_token, /*special=*/false);
        if (!piece.empty()) {
            out.append(piece);
            if (on_token) {
                on_token(piece.c_str(), user);
            }
        }
        s->history.push_back(new_token);

        // Stop-sequence handling: truncate to the match boundary, fire
        // no further callbacks, and exit cleanly. Honour even if the
        // match straddles the most recent piece.
        if (gp.n_stop > 0 && gp.stop) {
            size_t which = 0;
            const size_t pos = find_first_stop(out, gp.stop, gp.n_stop, &which);
            if (pos != std::string::npos) {
                out.resize(pos);
                break;
            }
        }

        // Decode the just-sampled token to advance the KV cache.
        llama_batch batch = llama_batch_get_one(&new_token, 1);
        if (llama_decode(s->ctx, batch) != 0) {
            set_err(err, 12, "llama_decode failed during generation");
            return 12;
        }
    }
    return 0;
}

// Prefill helper: build the chat-templated prompt, tokenize, and return
// the NEW token suffix (the part not already in `s->history`). If the
// new prompt diverges from history we wipe the KV cache and start over.
int32_t prepare_prompt(crispasr_chat_session* s, const crispasr_chat_message* messages, size_t n_messages,
                       std::vector<llama_token>& out_new, crispasr_chat_error* err) {
    std::string formatted;
    if (!apply_chat_template(s->tmpl.c_str(), messages, n_messages, /*add_ass=*/true, formatted)) {
        set_err(err, 20, "llama_chat_apply_template failed for template '%s'", s->tmpl.c_str());
        return 20;
    }
    // First message on a fresh session adds BOS; afterwards we want to
    // continue mid-stream so add_special=false.
    const bool fresh = s->history.empty();
    std::vector<llama_token> full = tokenize(s->vocab, formatted, /*add_special=*/fresh, /*parse_special=*/true);
    if (full.empty()) {
        set_err(err, 21, "tokenize produced no tokens");
        return 21;
    }

    if (fresh) {
        out_new = std::move(full);
        return 0;
    }

    // Compare against history: largest common prefix.
    size_t common = 0;
    const size_t n_cmp = std::min(s->history.size(), full.size());
    for (; common < n_cmp; ++common) {
        if (s->history[common] != full[common]) {
            break;
        }
    }
    if (common < s->history.size()) {
        // History diverged — flush KV cache and re-prefill from scratch.
        llama_memory_clear(llama_get_memory(s->ctx), /*data=*/true);
        s->history.clear();
        out_new = std::move(full);
        return 0;
    }
    // History is a clean prefix; only decode the new suffix.
    out_new.assign(full.begin() + (ptrdiff_t)common, full.end());
    return 0;
}

} // namespace

// ---------------------------------------------------------------------------
// Public generate entrypoints
// ---------------------------------------------------------------------------
extern "C" char* crispasr_chat_generate(crispasr_chat_session_t s, const crispasr_chat_message* messages,
                                        size_t n_messages, const crispasr_chat_generate_params* params,
                                        crispasr_chat_error* err) {
    if (!s) {
        set_err(err, 1, "session is null");
        return nullptr;
    }
    if (n_messages > 0 && !messages) {
        set_err(err, 1, "messages is null but n_messages > 0");
        return nullptr;
    }
    std::lock_guard<std::mutex> guard(s->mu);

    crispasr_chat_generate_params gp;
    crispasr_chat_generate_params_default(&gp);
    if (params) {
        gp = *params;
    }

    std::vector<llama_token> prompt_new;
    if (int32_t rc = prepare_prompt(s, messages, n_messages, prompt_new, err); rc != 0) {
        return nullptr;
    }
    std::string out;
    if (int32_t rc = generate_loop(s, prompt_new, gp, /*on_token=*/nullptr, /*user=*/nullptr, out, err); rc != 0) {
        return nullptr;
    }
    char* dup = (char*)std::malloc(out.size() + 1);
    if (!dup) {
        set_err(err, 30, "out of memory");
        return nullptr;
    }
    std::memcpy(dup, out.data(), out.size());
    dup[out.size()] = '\0';
    return dup;
}

extern "C" int32_t crispasr_chat_generate_stream(crispasr_chat_session_t s, const crispasr_chat_message* messages,
                                                 size_t n_messages, const crispasr_chat_generate_params* params,
                                                 crispasr_chat_on_token on_token, void* user,
                                                 crispasr_chat_error* err) {
    if (!s) {
        set_err(err, 1, "session is null");
        return 1;
    }
    if (n_messages > 0 && !messages) {
        set_err(err, 1, "messages is null but n_messages > 0");
        return 1;
    }
    std::lock_guard<std::mutex> guard(s->mu);

    crispasr_chat_generate_params gp;
    crispasr_chat_generate_params_default(&gp);
    if (params) {
        gp = *params;
    }

    std::vector<llama_token> prompt_new;
    if (int32_t rc = prepare_prompt(s, messages, n_messages, prompt_new, err); rc != 0) {
        return rc;
    }
    std::string sink;
    return generate_loop(s, prompt_new, gp, on_token, user, sink, err);
}

extern "C" void crispasr_chat_string_free(char* s) {
    if (s) {
        std::free(s);
    }
}

// ---------------------------------------------------------------------------
// Memory estimate
// ---------------------------------------------------------------------------
// We use llama.cpp's `no_alloc` model-load path to read the file's KV
// metadata without allocating tensor backing memory, then size up KV +
// activations from there. Activations stay an approximation — getting
// it exact would require building the graph, and the pre-flight guard
// just wants "≤ available RAM / VRAM, with margin".
extern "C" size_t crispasr_chat_memory_estimate(const char* model_path, const crispasr_chat_open_params* params,
                                                crispasr_chat_error* err) {
    if (!model_path || !*model_path) {
        set_err(err, 1, "model_path is null");
        return 0;
    }
    ensure_llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    mparams.use_mmap = true;
    mparams.vocab_only = false;
    mparams.no_alloc = true;  // metadata only — tensor data not faulted in
    mparams.n_gpu_layers = 0; // we don't want to provision a backend
    llama_model* model = llama_model_load_from_file(model_path, mparams);
    if (!model) {
        set_err(err, 2, "llama_model_load_from_file failed");
        return 0;
    }

    // Approximation: weights size ≈ on-disk file size (mmap-friendly).
    size_t weights = 0;
    if (FILE* f = std::fopen(model_path, "rb")) {
        std::fseek(f, 0, SEEK_END);
        const long off = std::ftell(f);
        if (off > 0) {
            weights = (size_t)off;
        }
        std::fclose(f);
    }

    // KV cache: n_ctx * n_layer * (n_embd_k + n_embd_v) * sizeof(fp16).
    // The exposed accessors give us what we need without llama-impl.
    const int32_t n_ctx = params && params->n_ctx > 0 ? params->n_ctx : llama_model_n_ctx_train(model);
    const int32_t n_layer = llama_model_n_layer(model);
    const int32_t n_embd_k = llama_model_n_embd(model); // overestimate for non-GQA
    const size_t kv_bytes = (size_t)std::max(0, n_ctx) * (size_t)std::max(0, n_layer) * (size_t)std::max(0, n_embd_k) *
                            2 * 2; // 2 caches × fp16

    // Activations + overhead: rule-of-thumb 256 MB margin.
    constexpr size_t overhead = 256ull * 1024ull * 1024ull;

    llama_model_free(model);
    return weights + kv_bytes + overhead;
}
