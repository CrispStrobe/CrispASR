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
//   That mutex cannot also govern teardown. `crispasr_chat_close` frees
//   the context, the model and the session itself, and a generation holds
//   the mutex for as long as it decodes — seconds — so a close that merely
//   took the mutex would still free the session under a second call that
//   was already blocked behind that generation. Teardown therefore runs on
//   its own short-lived lock and a count of the calls currently inside the
//   session: `close` retires the handle so no further call enters, waits
//   for the count to fall to zero, and only then frees.
//
// KV cache
//   Persisted across `crispasr_chat_generate` calls inside one session
//   so a multi-turn chat doesn't re-prefill the history. `_reset` calls
//   `llama_memory_clear` on the KV state.

#include "crispasr_chat.h"

#include "llama.h"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <memory>
#include <mutex>
#include <string>
#include <system_error>
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
// With n == 0 the result is the template's own opening — for add_ass that
// is the bare assistant prefix, which is what a token count of an empty
// conversation should report. Returns false on failure.
bool apply_chat_template(const char* tmpl, const crispasr_chat_message* msgs, size_t n, bool add_ass,
                         std::string& out) {
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

// Text a template appends after the assistant opening that the built-in
// renderer (llama_chat_apply_template, which matches templates by family and
// does not evaluate Jinja) leaves out. Qwen3 / Qwen3.5 templates open the
// reply with an empty `<think>\n\n</think>\n\n` unless `enable_thinking`
// is set: that empty block IS their non-thinking mode. Rendered without it,
// the model writes the block itself — four or more generated tokens per
// reply, and on a reasoning checkpoint sometimes a real one. Only templates
// that spell out this exact branch get it; the caller's override is taken
// as written.
std::string generation_suffix(const llama_model* model, const char* override_tmpl) {
    if (override_tmpl && *override_tmpl)
        return {};
    const char* baked = llama_model_chat_template(model, /*name=*/nullptr);
    if (!baked)
        return {};
    const std::string t = baked;
    if (t.find("enable_thinking") != std::string::npos && t.find("'<think>\\n\\n</think>\\n\\n'") != std::string::npos)
        return "<think>\n\n</think>\n\n";
    return {};
}

} // namespace

// ---------------------------------------------------------------------------
// Session impl
// ---------------------------------------------------------------------------
struct crispasr_chat_session {
    llama_model* model = nullptr;
    llama_context* ctx = nullptr;
    const llama_vocab* vocab = nullptr;

    std::string tmpl;       // resolved chat template name
    std::string gen_suffix; // appended after the assistant opening (generation_suffix)
    int32_t n_ctx = 0;
    int32_t n_threads = 1;
    int32_t n_threads_batch = 1;

    // History of tokens already in the KV cache for this conversation.
    // We tokenise + decode only the NEW suffix on each _generate call;
    // a divergent history is truncated back to the shared prefix.
    std::vector<llama_token> history;

    // Caller's cancellation hook, consulted between prompt batches and
    // before each sampled token. Also handed to llama_set_abort_callback,
    // which reaches the CPU backend's in-graph check.
    crispasr_chat_abort_callback abort_cb = nullptr;
    void* abort_user = nullptr;

    // Mutex serialising one-call-at-a-time per session.
    std::mutex mu;

    // Teardown accounting, held only long enough to read `retiring` and move
    // `in_use` — never across a native call, so counting a call in does not
    // serialise it against anything.
    std::mutex life;
    std::condition_variable idle;
    int32_t in_use = 0;
    bool retiring = false;
};

namespace {

// Holds the session open for the span of one call. Every entry point that
// dereferences a session takes one; `crispasr_chat_close` waits for the
// count to reach zero before it frees anything.
//
// A hold taken on a session already retired is refused rather than granted,
// so a call that reaches the session during a close is declined instead of
// running against memory about to be freed. That is a diagnostic for a caller
// that has already broken the header's ordering rule, not a guarantee: this
// check lives in the memory being freed, so a call descheduled just before it
// locks `s_->life` can have the session freed underneath it and then lock a
// destroyed mutex. Closing that window needs storage that outlives the
// session, which is the caller's own lock — see crispasr_chat_close.
class session_hold {
public:
    explicit session_hold(crispasr_chat_session* s) : s_(s) {
        if (!s_) {
            return;
        }
        std::lock_guard<std::mutex> g(s_->life);
        if (s_->retiring) {
            s_ = nullptr;
            return;
        }
        s_->in_use += 1;
    }

    ~session_hold() {
        if (!s_) {
            return;
        }
        std::lock_guard<std::mutex> g(s_->life);
        s_->in_use -= 1;
        if (s_->in_use == 0) {
            s_->idle.notify_all();
        }
    }

    session_hold(const session_hold&) = delete;
    session_hold& operator=(const session_hold&) = delete;

    // False when the session is retiring, i.e. a close is already waiting to
    // free it and this call must not proceed.
    bool held() const { return s_ != nullptr; }

private:
    crispasr_chat_session* s_;
};

// The public callback returns false to abort; every call site asks the
// opposite question, so invert once here.
bool abort_requested(crispasr_chat_session* s) {
    return s->abort_cb && !s->abort_cb(s->abort_user);
}

// Adapter for llama_set_abort_callback, whose callback returns true to
// abort. A static trampoline rather than a cast of the caller's function
// pointer: the two types differ in meaning, and casting them would be
// undefined behaviour the moment ggml called through it.
bool abort_trampoline(void* data) {
    auto* s = static_cast<crispasr_chat_session*>(data);
    return s && abort_requested(s);
}

} // namespace

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
    // cppcheck-suppress uninitvar
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
    s->gen_suffix = generation_suffix(model, p.chat_template);
    return s;
}

extern "C" void crispasr_chat_close(crispasr_chat_session_t s) {
    if (!s) {
        return;
    }
    {
        std::unique_lock<std::mutex> lk(s->life);
        if (s->retiring) {
            // A second close that got this far while the first is still
            // waiting. Returning is better than freeing twice, but it is a
            // partial mitigation for a caller that has already broken the
            // rule, not support for closing twice: a second close blocked on
            // `life` can be beaten to the delete by the first and wake on a
            // destroyed mutex. Close exactly once.
            return;
        }
        // Retire first, so a call arriving from here on is declined rather
        // than counted in and waited for, and the wait below terminates.
        s->retiring = true;
        s->idle.wait(lk, [s] { return s->in_use == 0; });
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
    session_hold hold(s);
    if (!hold.held()) {
        set_err(err, 5, "session is closing");
        return 5;
    }
    std::lock_guard<std::mutex> guard(s->mu);
    llama_memory_clear(llama_get_memory(s->ctx), /*data=*/true);
    s->history.clear();
    return 0;
}

extern "C" const char* crispasr_chat_template_name(crispasr_chat_session_t s) {
    // The hold covers the read, not the pointer: what comes back points into
    // the session's own string and dies with the session, exactly as before.
    // Copy it before the next close, as the header says.
    session_hold hold(s);
    return hold.held() ? s->tmpl.c_str() : nullptr;
}

extern "C" int32_t crispasr_chat_n_ctx(crispasr_chat_session_t s) {
    session_hold hold(s);
    return hold.held() ? s->n_ctx : 0;
}

extern "C" int32_t crispasr_chat_count_tokens(crispasr_chat_session_t s, const crispasr_chat_message* messages,
                                              size_t n_messages, crispasr_chat_error* err) {
    if (!s) {
        set_err(err, 1, "session is null");
        return -1;
    }
    if (n_messages > 0 && !messages) {
        set_err(err, 1, "messages is null but n_messages > 0");
        return -1;
    }
    session_hold hold(s);
    if (!hold.held()) {
        set_err(err, 5, "session is closing");
        return -1;
    }
    std::lock_guard<std::mutex> guard(s->mu);

    // Same rendering and same tokenizer flags prepare_prompt uses on a
    // fresh session, so the number is the prompt that session prefills.
    std::string formatted;
    if (!apply_chat_template(s->tmpl.c_str(), messages, n_messages, /*add_ass=*/true, formatted)) {
        set_err(err, 20, "llama_chat_apply_template failed for template '%s'", s->tmpl.c_str());
        return -1;
    }
    if (formatted.empty()) {
        // The template rendered to nothing, which several of the vendored ones
        // do for an empty message array: they emit only from inside their loop
        // over the messages and have no `add_ass` opening of their own. A fresh
        // session would prefill nothing here, so nothing is what this reports.
        // Failing instead would make a legitimate query model-dependent, and
        // inventing a token would count one the model never sees.
        return 0;
    }
    formatted += s->gen_suffix;
    const std::vector<llama_token> tokens = tokenize(s->vocab, formatted, /*add_special=*/true, /*parse_special=*/true);
    if (tokens.empty()) {
        set_err(err, 21, "tokenize produced no tokens for a non-empty prompt");
        return -1;
    }
    return (int32_t)tokens.size();
}

extern "C" void crispasr_chat_set_abort_callback(crispasr_chat_session_t s, crispasr_chat_abort_callback cb,
                                                 void* user) {
    if (!s) {
        return;
    }
    session_hold hold(s);
    if (!hold.held()) {
        return;
    }
    std::lock_guard<std::mutex> guard(s->mu);
    s->abort_cb = cb;
    s->abort_user = user;
    // Registering after open still reaches every backend that supports an
    // in-graph abort — llama_context::set_abort_callback re-registers on
    // each of them. Only the CPU backend implements it today, so the
    // checks in generate_loop are what cover the accelerated tiers.
    llama_set_abort_callback(s->ctx, cb ? &abort_trampoline : nullptr, cb ? s : nullptr);
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
        // llama_decode asserts rather than erroring when a batch holds more
        // than n_batch tokens, so walk the prompt in n_batch-sized pieces.
        // Pieces decoded in order into the same sequence continue the KV
        // cache, so positions need no bookkeeping here.
        // Mutable copy because llama_batch_get_one takes a non-const ptr.
        std::vector<llama_token> tokens = prompt_new;
        const int32_t n_total = (int32_t)tokens.size();
        const int32_t n_piece_max = std::max<int32_t>(1, (int32_t)llama_n_batch(s->ctx));
        for (int32_t off = 0; off < n_total; off += n_piece_max) {
            // Checked between pieces so a cancel during a long prefill costs
            // one prompt batch rather than the whole prompt.
            if (abort_requested(s)) {
                llama_memory_clear(llama_get_memory(s->ctx), /*data=*/true);
                s->history.clear();
                set_err(err, CRISPASR_CHAT_ERR_ABORTED, "generation aborted by callback during prefill");
                return CRISPASR_CHAT_ERR_ABORTED;
            }
            const int32_t n_piece = std::min(n_piece_max, n_total - off);
            llama_batch batch = llama_batch_get_one(tokens.data() + off, n_piece);
            if (llama_decode(s->ctx, batch) != 0) {
                // On the CPU backend the abort lands inside the graph, so a
                // decode that failed under an abort request is a cancel
                // rather than a fault.
                const bool cancelled = abort_requested(s);
                // Drop the partial prefill rather than leaving the session
                // holding a prompt prefix its history does not describe.
                llama_memory_clear(llama_get_memory(s->ctx), /*data=*/true);
                s->history.clear();
                if (cancelled) {
                    set_err(err, CRISPASR_CHAT_ERR_ABORTED, "generation aborted by callback during prefill");
                    return CRISPASR_CHAT_ERR_ABORTED;
                }
                set_err(err, 10, "llama_decode failed during prefill");
                return 10;
            }
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
        // Before sampling, and outside the piece branch below: on Metal and
        // CUDA a running batch cannot be interrupted, so this is the only
        // place a cancel is honoured on those backends.
        if (abort_requested(s)) {
            // The cache holds half an assistant turn that no later prompt
            // can reuse, so flush it rather than hand the next call a
            // prefix it would have to diverge from anyway.
            llama_memory_clear(llama_get_memory(s->ctx), /*data=*/true);
            s->history.clear();
            set_err(err, CRISPASR_CHAT_ERR_ABORTED, "generation aborted by callback");
            return CRISPASR_CHAT_ERR_ABORTED;
        }

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
            // As in prefill: an in-graph abort on the CPU backend surfaces
            // here as a failed decode, and is a cancel rather than a fault.
            const bool cancelled = abort_requested(s);
            // A failed decode drops this sequence's cache entries, so the
            // history no longer describes what is in the cache — clear both.
            llama_memory_clear(llama_get_memory(s->ctx), /*data=*/true);
            s->history.clear();
            if (cancelled) {
                set_err(err, CRISPASR_CHAT_ERR_ABORTED, "generation aborted by callback");
                return CRISPASR_CHAT_ERR_ABORTED;
            }
            set_err(err, 12, "llama_decode failed during generation");
            return 12;
        }
    }
    return 0;
}

// Prefill helper: build the chat-templated prompt, tokenize, and return
// the NEW token suffix (the part not already in `s->history`). Where the
// new prompt diverges from the history, the KV cache is truncated to
// their common prefix so only the divergent tail is re-decoded.
int32_t prepare_prompt(crispasr_chat_session* s, const crispasr_chat_message* messages, size_t n_messages,
                       std::vector<llama_token>& out_new, crispasr_chat_error* err) {
    if (n_messages == 0) {
        // Nothing to answer: the template would render a bare assistant
        // prefix and the model would continue from nowhere.
        set_err(err, 21, "no messages to prefill");
        return 21;
    }
    std::string formatted;
    if (!apply_chat_template(s->tmpl.c_str(), messages, n_messages, /*add_ass=*/true, formatted)) {
        set_err(err, 20, "llama_chat_apply_template failed for template '%s'", s->tmpl.c_str());
        return 20;
    }
    formatted += s->gen_suffix;
    // `messages` is the whole conversation, so it is tokenized exactly as
    // a just-opened session would tokenize it, leading BOS included. That
    // is what makes it comparable to the history token for token below.
    std::vector<llama_token> full = tokenize(s->vocab, formatted, /*add_special=*/true, /*parse_special=*/true);
    if (full.empty()) {
        set_err(err, 21, "tokenize produced no tokens");
        return 21;
    }

    if (s->history.empty()) {
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
        // The history diverged, but everything before `common` is prompt
        // the cache already holds and the new prompt still asks for — a
        // fixed instruction block, in the usual case. Drop only the tail.
        if (common == full.size()) {
            // The new prompt is a strict prefix of the history: with no
            // suffix left to decode, generation would sample from logits
            // belonging to a token that is no longer the last one. Hold a
            // token back so at least one is always decoded.
            --common;
        }
        // A cache type that refuses a partial removal reports false and
        // leaves its contents alone, so the full clear stays the fallback.
        if (common > 0 && llama_memory_seq_rm(llama_get_memory(s->ctx), 0, (llama_pos)common, -1)) {
            s->history.resize(common);
        } else {
            llama_memory_clear(llama_get_memory(s->ctx), /*data=*/true);
            s->history.clear();
            common = 0;
        }
        // The cache holds exactly the tokens the history now names, either
        // way, so the suffix decodes onto a prefix that describes it.
        out_new.assign(full.begin() + (ptrdiff_t)common, full.end());
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
    session_hold hold(s);
    if (!hold.held()) {
        set_err(err, 5, "session is closing");
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
    session_hold hold(s);
    if (!hold.held()) {
        set_err(err, 5, "session is closing");
        return 5;
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

// The canonical Art. 50(1) disclosure for conversational products built on this
// ABI. Deliberately parallel to crispasr_session_disclaimer_text() on the audio
// side — same job, different modality — and kept in ONE place so the CLI, the
// server, the Flutter binding and downstream integrators cannot each invent
// their own wording. tests/test-compliance-wiring.cpp pins it against drift.
extern "C" const char* crispasr_chat_ai_disclosure_text(void) {
    return "You are interacting with an AI system. Responses are generated by artificial intelligence.";
}

// ---------------------------------------------------------------------------
// Memory estimate
// ---------------------------------------------------------------------------
// We use llama.cpp's `no_alloc` model-load path to read the file's KV
// metadata without allocating tensor backing memory, then size up KV +
// activations from there. Activations stay an approximation — getting
// it exact would require building the graph, and the pre-flight guard
// just wants "≤ available RAM / VRAM, with margin".
//
// `no_alloc` requires `use_mmap = false`. With mmap on, llama_model::load_tensors
// takes the branch that wraps a backend buffer directly over the mapped file
// region, and that branch opens with GGML_ASSERT(!ml.no_alloc) — the two are
// mutually exclusive by construction. With mmap off it takes the branch
// `no_alloc` was written for: a zero-byte dummy buffer that every weight tensor
// points at, then an early return before any tensor data is read. llama.cpp's
// own device-memory probe sets the same pair.
//
// `vocab_only` is NOT a substitute. load_hparams returns as soon as it sees that
// flag, before it reads the context length, embedding length and block count —
// precisely the three values the KV term below is built from — so the estimate
// would silently collapse to file size + overhead while reporting success.
extern "C" size_t crispasr_chat_memory_estimate(const char* model_path, const crispasr_chat_open_params* params,
                                                crispasr_chat_error* err) {
    if (!model_path || !*model_path) {
        set_err(err, 1, "model_path is null");
        return 0;
    }
    ensure_llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    mparams.use_mmap = false; // required by no_alloc — see above
    mparams.vocab_only = false;
    mparams.no_alloc = true;  // metadata only — tensor data not read
    mparams.n_gpu_layers = 0; // we don't want to provision a backend
    llama_model* model = llama_model_load_from_file(model_path, mparams);
    if (!model) {
        set_err(err, 2, "llama_model_load_from_file failed");
        return 0;
    }

    // Approximation: weights size ≈ on-disk file size (mmap-friendly).
    //
    // std::filesystem::file_size, not fseek/ftell: ftell returns `long`, which
    // is 32-bit on 64-bit Windows, so every GGUF over 2 GiB — which is most of
    // the ones worth guarding against — would fail to be represented and leave
    // the weights term at zero while the estimate still reported success. An
    // estimate that omits the model it is deciding whether to load is worse
    // than no estimate, so a size we cannot read is an error, not a zero.
    std::error_code ec;
    const std::uintmax_t file_bytes = std::filesystem::file_size(std::filesystem::path(model_path), ec);
    if (ec || file_bytes > (std::uintmax_t)SIZE_MAX) {
        llama_model_free(model);
        set_err(err, 3, "could not read the model file's size");
        return 0;
    }
    const size_t weights = (size_t)file_bytes;

    // KV cache: n_ctx * n_layer * (n_embd_k + n_embd_v) * sizeof(fp16).
    // The exposed accessors give us what we need without llama-impl.
    //
    // Rounded up to a multiple of 256 the way llama_context's own constructor
    // rounds it (GGML_PAD(cparams.n_ctx, 256)), so a request of, say, 1000
    // tokens is sized as the 1024 the runtime will actually allocate.
    const int32_t n_ctx_req = params && params->n_ctx > 0 ? params->n_ctx : llama_model_n_ctx_train(model);
    // Widened before rounding: the padded value of a context near INT32_MAX
    // does not fit back into an int32_t.
    const uint64_t n_ctx = ((uint64_t)std::max(0, n_ctx_req) + 255u) & ~(uint64_t)255u;
    const int32_t n_layer = llama_model_n_layer(model);
    // n_embd is the full attention width; on a grouped-query model the per-layer
    // K/V width is a fraction of it, so this over-reports there. Over-reporting is
    // the safe direction for a "will this fit?" guard.
    const int32_t n_embd_k = llama_model_n_embd(model);
    const size_t kv_bytes =
        (size_t)n_ctx * (size_t)std::max(0, n_layer) * (size_t)std::max(0, n_embd_k) * 2 * 2; // 2 caches × fp16

    // Activations + overhead: rule-of-thumb 256 MB margin.
    constexpr size_t overhead = 256ull * 1024ull * 1024ull;

    llama_model_free(model);
    return weights + kv_bytes + overhead;
}
