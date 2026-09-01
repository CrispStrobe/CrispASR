// htdemucs_gates.h — path selection for htdemucs (#413/#414), pure and
// unit-testable.
//
// Three coupled decisions, resolved ONCE per init from the environment, the
// caller's use_gpu intent, and whether a REAL (non-CPU) GPU backend exists:
//
//   use_graph   — run the ggml graph path instead of the legacy CPU/BLAS one
//   use_fused   — single fused graph (encoder+transformer+decoder on-device);
//                 only meaningful when use_graph is set
//   gpu_backend — place the graph on the GPU backend
//
// Measured facts the AUTO defaults encode (PR #414, RTX 3090 Ti + earlier
// M1/Kaggle numbers): fused-graph-on-GPU is ~20x faster than CPU/BLAS
// (RTF 0.37 vs 7.4); per-layer graphs — on GPU or CPU — are SLOWER than
// CPU/BLAS (host<->device roundtrip per layer / graph overhead). So AUTO
// picks graph+fused+GPU exactly when a real GPU is present and permitted,
// and plain CPU/BLAS otherwise. Never a slower-than-before configuration.
//
// Env semantics (each var: unset = AUTO, "0" = force off, else force on):
//   CRISPASR_HTDEMUCS_GPU    — GPU permission; explicit value beats the
//                              caller's use_gpu in BOTH directions (expert
//                              override; note params.use_gpu defaults true
//                              from the CLI, so "params wins" would make
//                              GPU=0 dead — the #414 review catch)
//   CRISPASR_HTDEMUCS_GGML   — graph path
//   CRISPASR_HTDEMUCS_FUSED  — fused graph (forced on implies graph unless
//                              graph is explicitly forced off)
#pragma once

#include <cstdlib>

namespace htdemucs_gates {

struct Resolved {
    bool use_graph = false;
    bool use_fused = false;
    bool gpu_backend = false;
};

inline Resolved resolve(const char* env_gpu, const char* env_ggml, const char* env_fused, bool caller_use_gpu,
                        bool have_real_gpu) {
    bool want_gpu = caller_use_gpu;
    if (env_gpu && *env_gpu)
        want_gpu = atoi(env_gpu) != 0;
    const bool gpu = want_gpu && have_real_gpu;

    Resolved r;
    const bool ggml_forced = env_ggml && *env_ggml;
    const bool fused_forced = env_fused && *env_fused;
    const bool fused_forced_on = fused_forced && atoi(env_fused) != 0;

    r.use_graph =
        ggml_forced ? (atoi(env_ggml) != 0) : (gpu || fused_forced_on); // FUSED=1 alone implies the graph it needs
    r.use_fused = fused_forced ? fused_forced_on : gpu;                 // AUTO: fused exactly on GPU
    r.use_fused = r.use_fused && r.use_graph;                           // fused cannot outlive the graph path
    r.gpu_backend = gpu && r.use_graph;                                 // BLAS path is CPU by construction
    return r;
}

} // namespace htdemucs_gates
