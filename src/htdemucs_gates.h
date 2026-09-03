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

// The decision table itself lives in core/backend_path_gates.h — it was
// byte-identical to mel_band_gates.h and two copies drift. This header keeps
// the htdemucs-specific documentation above and the htdemucs_gates:: names
// that callers and tests already use.
#include "core/backend_path_gates.h"

namespace htdemucs_gates {

using Resolved = core_backend_gates::Resolved;

inline Resolved resolve(const char* env_gpu, const char* env_ggml, const char* env_fused, bool caller_use_gpu,
                        bool have_real_gpu) {
    return core_backend_gates::resolve(env_gpu, env_ggml, env_fused, caller_use_gpu, have_real_gpu);
}

} // namespace htdemucs_gates
