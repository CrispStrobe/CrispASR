// src/core/ggml_cpu_backend.h — CPU-backend access that survives GGML_BACKEND_DL.
//
// Issue #355: the shipped `-cuda` tarball carries libcuda.so.1 as a hard
// DT_NEEDED, so on a host without the NVIDIA driver the loader kills the
// process with exit 127 before main() runs and the advertised CPU fallback
// never gets a chance. The fix ggml provides is GGML_BACKEND_DL: every backend
// becomes a dlopen'd module, so a CUDA backend that cannot load simply is not
// registered and ggml_backend_init_best() picks CPU.
//
// Two things block that in this tree. The first is CMake — every per-model
// library links `ggml-cuda` / `ggml-metal` explicitly (a MODULE target cannot
// be linked), handled by the crispasr_link_ggml_* interface targets in the
// top-level CMakeLists.
//
// The second is this file's reason to exist. Under DL the CPU backend is a
// module too, so `ggml_backend_cpu_init`, `ggml_backend_is_cpu` and friends are
// not linkable — and this tree calls them at ~424 sites across 104 files. All
// six have exact registry equivalents; these wrappers are those, selected at
// compile time.
//
// **Without GGML_BACKEND_DL every wrapper is the direct call it replaces**, so
// the default build is unchanged — same symbols, same codegen, no runtime
// lookup. That is deliberate: the DL path is new and unproven on CUDA/HIP/
// Vulkan hardware, so it must not be able to alter the build everyone ships.
#pragma once

#include "ggml-backend.h"

#include <cstdio>
#include <cstring>

// ggml-cpu.h is included in BOTH branches: `ggml_cplan` is a plain struct and
// the declarations cost nothing. Only the symbols are unlinkable under DL.
#include "ggml-cpu.h"

#ifndef GGML_BACKEND_DL
#ifdef GGML_USE_METAL
#include "ggml-metal.h"
#endif
#endif

namespace core_cpu_backend {

#ifndef GGML_BACKEND_DL

// Statically linked CPU backend: call straight through. Identical to what the
// call sites did before, so the shipped build is byte-for-byte the same.
inline ggml_backend_t init() {
    return ggml_backend_cpu_init();
}
// Null-tolerant in BOTH branches. Under DL the registry lookup can legitimately
// fail, so call sites are allowed to pass null; the direct path must then not
// abort inside ggml. test-cpu-backend-shim caught this asymmetry.
inline bool is_cpu(ggml_backend_t b) {
    return b && ggml_backend_is_cpu(b);
}
inline void set_n_threads(ggml_backend_t b, int n) {
    if (b) {
        ggml_backend_cpu_set_n_threads(b, n);
    }
}
inline ggml_backend_buffer_type_t buffer_type() {
    return ggml_backend_cpu_buffer_type();
}

// Was the CPU backend BUILT with `name` (an ISA flag as ggml spells it:
// "AVX", "AVX2", "AVX512", "FMA", "F16C", "BMI2", "NEON", ...)?
//
// #380 needs this to compare the shipped build's ISA against the host's, and
// #403's release exposed why it belongs here: `ggml_cpu_has_*()` are symbols in
// libggml-cpu, so calling them directly fails to LINK under GGML_BACKEND_DL —
// which is exactly the configuration the CUDA packages are built in (#355).
inline bool has_feature(const char* name) {
    if (!name) {
        return false;
    }
    struct entry {
        const char* name;
        int (*fn)(void);
    };
    static const entry k[] = {
        {"AVX", ggml_cpu_has_avx},   {"AVX2", ggml_cpu_has_avx2}, {"AVX512", ggml_cpu_has_avx512},
        {"FMA", ggml_cpu_has_fma},   {"F16C", ggml_cpu_has_f16c}, {"BMI2", ggml_cpu_has_bmi2},
        {"NEON", ggml_cpu_has_neon}, {"SSE3", ggml_cpu_has_sse3}, {"SSSE3", ggml_cpu_has_ssse3},
    };
    for (const entry& e : k) {
        if (std::strcmp(e.name, name) == 0) {
            return e.fn() != 0;
        }
    }
    return false;
}
inline ggml_backend_reg_t reg() {
    return ggml_backend_cpu_reg();
}
inline void set_threadpool(ggml_backend_t b, ggml_threadpool_t tp) {
    if (b) {
        ggml_backend_cpu_set_threadpool(b, tp);
    }
}
inline bool is_metal(ggml_backend_t b) {
#ifdef GGML_USE_METAL
    return ggml_backend_is_metal(b);
#else
    (void)b;
    return false;
#endif
}

// Compute a graph on the CPU. Direct call: identical to every existing site.
inline enum ggml_status compute(ggml_context* ctx, ggml_cgraph* gf, int n_threads) {
    return ggml_graph_compute_with_ctx(ctx, gf, n_threads);
}

// The CPU backend's quantise-from-F32 kernel for `type`, or nullptr when it has
// none. Callers already fall back to ggml_get_type_traits()->from_float_ref,
// which lives in ggml-base and is always linkable.
inline ggml_from_float_t from_float_for(enum ggml_type type) {
    const auto* t = ggml_get_type_traits_cpu(type);
    return t ? t->from_float : nullptr;
}

// Plan + compute, kept separate so a hot loop can size its work buffer once and
// reuse it across frames (the VAD per-frame and wav2vec2 per-layer paths).
inline ggml_cplan plan(ggml_cgraph* gf, int n_threads, ggml_threadpool_t pool = nullptr) {
    return ggml_graph_plan(gf, n_threads, pool);
}
inline enum ggml_status compute_planned(ggml_cgraph* gf, ggml_cplan* p, int /*n_threads*/) {
    return ggml_graph_compute(gf, p);
}

// Shared CPU worker pool. Reused across calls so a per-frame graph does not
// spawn and join threads every time.
inline ggml_threadpool_t threadpool_new(int n_threads) {
    ggml_threadpool_params tpp = ggml_threadpool_params_default(n_threads);
    return ggml_threadpool_new(&tpp);
}
inline void threadpool_free(ggml_threadpool_t p) {
    ggml_threadpool_free(p);
}

#else

// Dynamic backends: the CPU backend is a module like any other, reached
// through the registry.
//
// The registry is empty until ggml_backend_load_all() has dlopen'd the modules.
// Relying on the entry points to have done that first is exactly the kind of
// ordering assumption that holds until it doesn't: nothing enforces it, and the
// failure is silent — cpu_device() returns null, every wrapper takes its
// null-guard, and compute() quietly returns GGML_STATUS_FAILED instead of
// computing. test-cpu-backend-shim hit precisely that, failing 5 of 7 cases
// under DL while passing all 7 without it.
//
// So load here instead, on first use. The function-local static makes it exactly
// once and thread-safe, and it is idempotent with the entry points' own call —
// whichever runs first wins and the other is a no-op.
inline ggml_backend_dev_t cpu_device() {
    static const bool loaded = [] {
        ggml_backend_load_all();
        return true;
    }();
    (void)loaded;
    ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    // Issue #405: when the host CPU is below the ISA floor of EVERY shipped
    // libggml-cpu module, ggml_backend_score() refuses them all and this
    // lookup returns null. That is a terminal condition for the process
    // (nothing computes without a CPU backend), and downstream it used to
    // surface as a bare GGML_ASSERT abort. Say what actually happened, once.
    if (!dev) {
        static const bool warned = [] {
            fprintf(stderr, "crispasr: no CPU ggml backend registered — the libggml-cpu module(s) shipped "
                            "next to this binary were either not found or refused because this host's CPU "
                            "lacks the instruction-set features they were built for. Use the "
                            "crispasr-*-cpu-legacy artifact, or build from source on this machine.\n");
            return true;
        }();
        (void)warned;
    }
    return dev;
}

inline ggml_backend_t init() {
    ggml_backend_dev_t dev = cpu_device();
    return dev ? ggml_backend_dev_init(dev, nullptr) : nullptr;
}

inline bool is_cpu(ggml_backend_t b) {
    if (!b) {
        return false;
    }
    ggml_backend_dev_t dev = ggml_backend_get_device(b);
    return dev && ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU;
}

inline void set_n_threads(ggml_backend_t b, int n) {
    if (!b) {
        return;
    }
    ggml_backend_dev_t dev = ggml_backend_get_device(b);
    ggml_backend_reg_t r = dev ? ggml_backend_dev_backend_reg(dev) : nullptr;
    if (!r) {
        return;
    }
    // The registry exposes it as a named proc rather than a linked symbol.
    typedef void (*set_n_threads_fn)(ggml_backend_t, int);
    auto fn = (set_n_threads_fn)ggml_backend_reg_get_proc_address(r, "ggml_backend_set_n_threads");
    if (fn) {
        fn(b, n);
    }
}

inline ggml_backend_buffer_type_t buffer_type() {
    ggml_backend_dev_t dev = cpu_device();
    return dev ? ggml_backend_dev_buffer_type(dev) : nullptr;
}

inline ggml_backend_reg_t reg() {
    ggml_backend_dev_t dev = cpu_device();
    return dev ? ggml_backend_dev_backend_reg(dev) : nullptr;
}

// Under DL the ISA predicates are unlinkable, but the registry publishes the
// same information: ggml_backend_cpu_get_proc_address exposes
// "ggml_backend_get_features", whose NULL-terminated array carries exactly the
// names ggml compiled in ("AVX2", "FMA", "AVX512", ...). Absent name => absent
// feature, which is also the right answer when the CPU module failed to load.
inline bool has_feature(const char* name) {
    if (!name) {
        return false;
    }
    ggml_backend_reg_t r = reg();
    if (!r) {
        return false;
    }
    auto fn = (ggml_backend_get_features_t)ggml_backend_reg_get_proc_address(r, "ggml_backend_get_features");
    if (!fn) {
        return false;
    }
    for (ggml_backend_feature* f = fn(r); f && f->name; ++f) {
        if (std::strcmp(f->name, name) == 0) {
            return true;
        }
    }
    return false;
}

// ⚠ No-op under DL, and this is a real functional difference rather than an
// oversight: ggml's CPU registry exposes `ggml_backend_set_n_threads` through
// get_proc_address but NOT the threadpool setter (see
// ggml_backend_cpu_get_proc_address in ggml/src/ggml-cpu/ggml-cpu.cpp — the
// list is n_threads, extra_bufts, features, abort_callback, numa_init,
// is_numa). So a DL build cannot install the shared worker pool and falls back
// to ggml's own per-call threading. Exposing it needs a fork patch to that
// switch; until then this is why the DL path stays opt-in.
inline void set_threadpool(ggml_backend_t, ggml_threadpool_t) {}

inline bool is_metal(ggml_backend_t b) {
    if (!b) {
        return false;
    }
    ggml_backend_dev_t dev = ggml_backend_get_device(b);
    ggml_backend_reg_t r = dev ? ggml_backend_dev_backend_reg(dev) : nullptr;
    const char* n = r ? ggml_backend_reg_name(r) : nullptr;
    return n && std::strcmp(n, "Metal") == 0;
}

// Compute a graph on the CPU, through a backend instead of the unlinkable
// ggml_graph_compute_with_ctx. Execution stays on the CPU — this routes the
// same work through the CPU backend module rather than moving it anywhere.
//
// The backend is thread_local, not a shared static: a ggml backend is not safe
// for concurrent use, and several of these call sites run inside worker threads.
// One instance per thread, created on first use and reused after that.
inline enum ggml_status compute(ggml_context* ctx, ggml_cgraph* gf, int n_threads) {
    (void)ctx; // the backend owns its own plan; no scratch context needed
    static thread_local ggml_backend_t tls = nullptr;
    if (!tls) {
        tls = init();
        if (!tls) {
            return GGML_STATUS_FAILED;
        }
    }
    set_n_threads(tls, n_threads);
    return ggml_backend_graph_compute(tls, gf);
}

// Not reachable under DL — ggml_get_type_traits_cpu lives in the CPU module.
// Returning nullptr sends the caller down its existing from_float_ref path,
// which is what it already does when the CPU backend has no kernel for a type.
inline ggml_from_float_t from_float_for(enum ggml_type) {
    return nullptr;
}

// ggml_graph_plan is a CPU-module symbol, so under DL there is no caller-owned
// plan to make. Report a zero work_size: every call site guards its work-buffer
// allocation on `work_size > 0`, so they allocate nothing and the backend does
// its own planning inside compute_planned below.
//
// ⚠ That costs the optimisation those sites exist for — "no scheduler, no
// threadpool churn" per frame becomes a fresh plan per call. Measured cost is
// unknown on this path; it is one more reason the DL build stays opt-in.
inline ggml_cplan plan(ggml_cgraph*, int, ggml_threadpool_t = nullptr) {
    ggml_cplan p = {};
    return p;
}
inline enum ggml_status compute_planned(ggml_cgraph* gf, ggml_cplan*, int n_threads) {
    static thread_local ggml_backend_t tls = nullptr;
    if (!tls) {
        tls = init();
        if (!tls) {
            return GGML_STATUS_FAILED;
        }
    }
    set_n_threads(tls, n_threads);
    return ggml_backend_graph_compute(tls, gf);
}

// No shared pool under DL. ggml_threadpool_new lives in the CPU module, and
// even if it were reachable the registry does not expose the setter that would
// attach it (see set_threadpool above). Returning nullptr makes every caller
// take its existing "no pool" branch — they all guard on the pointer — so the
// backend falls back to ggml's own per-call threading.
inline ggml_threadpool_t threadpool_new(int) {
    return nullptr;
}
inline void threadpool_free(ggml_threadpool_t) {}

#endif

} // namespace core_cpu_backend
