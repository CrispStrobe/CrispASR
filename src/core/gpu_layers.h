// core/gpu_layers.h — how many LLM layers a backend keeps on the GPU.
//
// The layer-offload backends (PLAN #69a) used to read CRISPASR_N_GPU_LAYERS
// directly, so the session ABI's `n_gpu_layers` argument
// (crispasr_session_open_with_params, Python `Session(n_gpu_layers=...)`)
// was stored and never read by any backend (#610). The session open now sets
// a thread-local override for the duration of the model load, and every
// backend resolves the value here.
//
// Precedence: an explicit session value >= 0, else CRISPASR_N_GPU_LAYERS,
// else -1 ("all layers on the GPU").

#pragma once

#include <cstdlib>

namespace core_gpu_layers {

inline int& tls_override() {
    static thread_local int v = -1;
    return v;
}

// Layer count to keep on the GPU, or -1 for all of them.
inline int resolve() {
    if (tls_override() >= 0)
        return tls_override();
    if (const char* s = std::getenv("CRISPASR_N_GPU_LAYERS"))
        if (s[0])
            return std::atoi(s);
    return -1;
}

// Sets the override for one model load and restores the previous value.
struct Scope {
    int prev;
    explicit Scope(int n) : prev(tls_override()) { tls_override() = n; }
    ~Scope() { tls_override() = prev; }
    Scope(const Scope&) = delete;
    Scope& operator=(const Scope&) = delete;
};

} // namespace core_gpu_layers
