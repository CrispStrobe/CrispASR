// backend_path_gates.h — the shared graph/fused/GPU path-selection table.
//
// WHY THIS EXISTS. htdemucs (#413/#414) and mel-band-roformer (#422) had
// byte-for-byte identical resolve() implementations — the two bodies differed
// only in one trailing comment word ("BLAS path" vs "CPU path"). Two copies of
// a decision table drift the moment anyone fixes one of them, and there is a
// known wart waiting to be fixed (see NON-NUMERIC VALUES below), so the
// divergence was a matter of time rather than chance. Hoisted here so both
// backends resolve through one implementation and the unit tables in
// tests/test-htdemucs-gates.cpp and tests/test-mel-band-gates.cpp both cover it.
//
// The three decisions are coupled, and resolved ONCE per init from the
// environment, the caller's use_gpu intent, and whether a REAL (non-CPU) GPU
// backend exists:
//
//   use_graph   — run the ggml graph path instead of the legacy CPU/BLAS one
//   use_fused   — single fused graph on-device; only meaningful with use_graph
//   gpu_backend — place the graph on the GPU backend
//
// Env semantics (each var: unset = AUTO, "0" = force off, else force on):
//   <PREFIX>_GPU    — GPU permission. An explicit value beats the caller's
//                     use_gpu in BOTH directions. This matters because the CLI
//                     forwards params.use_gpu defaulted true, so a "params
//                     wins" rule would make GPU=0 dead — the #414 review catch,
//                     fix 208b2d59.
//   <PREFIX>_GGML   — graph path
//   <PREFIX>_FUSED  — fused graph; forced on implies the graph it needs unless
//                     the graph is explicitly forced off.
//
// NON-NUMERIC VALUES — a known wart, preserved deliberately. atoi() means
// <PREFIX>_GPU=true reads as 0, i.e. OFF, contradicting the documented
// "else = force on". Both backends have always behaved this way and scripts may
// depend on it; changing it is a behaviour change that belongs in its own
// commit with its own justification, not smuggled into a refactor. Recorded
// here so the next reader finds the decision rather than rediscovering the bug.
//
// Callers keep their own namespaced wrappers (htdemucs_gates::resolve,
// mel_band_gates::resolve) so no call site or test changes; the wrappers are
// one-liners forwarding here.
#pragma once

#include <cstdlib>

namespace core_backend_gates {

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
    r.gpu_backend = gpu && r.use_graph;                                 // the legacy path is CPU by construction
    return r;
}

} // namespace core_backend_gates
