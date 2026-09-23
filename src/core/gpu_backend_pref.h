#pragma once
// gpu_backend_pref.h — process-global GPU backend preference
//
// Issue #214: `--gpu-backend vulkan` was ignored because every backend
// called `ggml_backend_init_best()` which unconditionally picks CUDA
// over Vulkan when both are compiled in. This header provides:
//
//   crispasr_set_gpu_backend_pref("vulkan")  — set once at startup
//   crispasr_init_gpu_backend()              — drop-in replacement for
//                                              ggml_backend_init_best()
//
// The preference is matched against ggml backend registry names
// (case-insensitive). Common values: "cuda", "vulkan", "metal", "cpu".
// Empty or null = auto (same as ggml_backend_init_best).

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"                    // core_cpu_backend::init() for the `--gpu-backend cpu` short-circuit
#include "metal_pipeline_cache_policy.h" // cap on the ggml-metal MTLBinaryArchive open cost (PLAN #88)

#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>
#include "core/ggml_cpu_backend.h"

namespace crispasr_gpu_pref {

// The preference is a simple global — set once at process start,
// read many times. Thread-safe via the string copy in get().
inline std::string& pref_storage() {
    static std::string s;
    return s;
}

inline std::mutex& pref_mutex() {
    static std::mutex m;
    return m;
}

} // namespace crispasr_gpu_pref

// Set the GPU backend preference. Call once at startup, before any
// backend init_from_file. Empty string = auto.
inline void crispasr_set_gpu_backend_pref(const char* name) {
    std::lock_guard<std::mutex> lock(crispasr_gpu_pref::pref_mutex());
    crispasr_gpu_pref::pref_storage() = name ? name : "";
}

inline std::string crispasr_get_gpu_backend_pref() {
    std::lock_guard<std::mutex> lock(crispasr_gpu_pref::pref_mutex());
    return crispasr_gpu_pref::pref_storage();
}

// Case-insensitive prefix check: does `haystack` start with `needle`?
inline bool ci_starts_with(const char* haystack, const char* needle) {
    for (; *needle; ++haystack, ++needle) {
        if (!*haystack)
            return false;
        if (tolower((unsigned char)*haystack) != tolower((unsigned char)*needle))
            return false;
    }
    return true;
}

// Drop-in replacement for ggml_backend_init_best().
// If a gpu_backend preference is set, iterate registered devices and
// pick the first GPU/iGPU device whose name starts with the preference
// (e.g. "vulkan" matches "Vulkan0", "Vulkan1", …).
// Falls back to ggml_backend_init_best() when no preference is set or
// the preferred backend isn't found.
inline ggml_backend_t crispasr_init_gpu_backend() {
    std::string pref = crispasr_get_gpu_backend_pref();

    // PLAN #88 follow-up (logic synced from CrispEmbed's T18/G4 fixes): bound
    // the ggml-metal MTLBinaryArchive open cost for every lane that reaches a
    // GPU device through this helper. The archive open costs ~1 ms/MB (683 MB
    // observed on the shared dev box = ~680 ms of fixed init). Skipped when the
    // preference is "cpu", because then no GPU device is created and the
    // diagnostic would fire spuriously. apply() is idempotent.
    // CRISPASR_METAL_PIPELINE_CACHE_MAX_MB=0 restores the uncapped behaviour.
    {
        const bool pref_is_cpu = !pref.empty() && pref.size() <= 3 && ci_starts_with("cpu", pref.c_str());
        if (!pref_is_cpu)
            core_metal_cache::apply();
    }

    // T18 sync (CrispEmbed): `--gpu-backend cpu` used to fall THROUGH to the
    // GPU. The loops below only ever consider GPU/iGPU devices, so "cpu"
    // matched nothing and this returned ggml_backend_init_best() — i.e. Metal
    // on an M1, silently costing the device init the flag exists to avoid.
    // CRISPASR_GPU_PREF_CPU_LEGACY=1 restores the old fall-through for A/B.
    if (!pref.empty() && ci_starts_with("cpu", pref.c_str()) && pref.size() <= 3) {
        const char* legacy = std::getenv("CRISPASR_GPU_PREF_CPU_LEGACY");
        if (!(legacy && legacy[0] && legacy[0] != '0')) {
            // core_cpu_backend::init(), NOT ggml_backend_dev_by_type(...CPU):
            // the registry lookup enumerates every registered device, which
            // constructs the Metal device as a side effect (measured on
            // CrispEmbed: it still ran ggml_metal_device_init and cost
            // ~29 ms). The direct constructor touches no registry, so nothing
            // GPU is created.
            ggml_backend_t cpu = core_cpu_backend::init();
            if (cpu) {
                fprintf(stderr, "%s: --gpu-backend cpu — using the CPU backend, no GPU device initialised\n", __func__);
                return cpu;
            }
        }
    }

    // ggml names the Apple backend "MTL" (registry) / "MTL0" (device), so the
    // natural `--gpu-backend metal` would never prefix-match. Alias it.
    if (pref.size() >= 3 && ci_starts_with("metal", pref.c_str()))
        pref = "mtl";

    if (!pref.empty()) {
        // Iterate all registered devices and find the first GPU whose
        // name starts with the preference string.
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            ggml_backend_dev_t dev = ggml_backend_dev_get(i);
            enum ggml_backend_dev_type dt = ggml_backend_dev_type(dev);
            if (dt != GGML_BACKEND_DEVICE_TYPE_GPU && dt != GGML_BACKEND_DEVICE_TYPE_IGPU)
                continue;
            const char* dev_name = ggml_backend_dev_name(dev);
            if (ci_starts_with(dev_name, pref.c_str())) {
                ggml_backend_t result = ggml_backend_dev_init(dev, nullptr);
                if (result) {
                    fprintf(stderr, "%s: using preferred GPU backend: %s\n", __func__, dev_name);
                    return result;
                }
                fprintf(stderr, "%s: preferred GPU device '%s' failed to init, trying fallback\n", __func__, dev_name);
            }
        }

        // Also try matching against the registry (backend library) name,
        // e.g. the user writes "vulkan" and the registry name is "Vulkan".
        for (size_t i = 0; i < ggml_backend_reg_count(); ++i) {
            ggml_backend_reg_t reg = ggml_backend_reg_get(i);
            const char* reg_name = ggml_backend_reg_name(reg);
            if (!ci_starts_with(reg_name, pref.c_str()))
                continue;
            // Found the registry — pick the first GPU device from it.
            for (size_t j = 0; j < ggml_backend_reg_dev_count(reg); ++j) {
                ggml_backend_dev_t dev = ggml_backend_reg_dev_get(reg, j);
                enum ggml_backend_dev_type dt = ggml_backend_dev_type(dev);
                if (dt != GGML_BACKEND_DEVICE_TYPE_GPU && dt != GGML_BACKEND_DEVICE_TYPE_IGPU)
                    continue;
                ggml_backend_t result = ggml_backend_dev_init(dev, nullptr);
                if (result) {
                    fprintf(stderr, "%s: using preferred GPU backend: %s (via registry '%s')\n", __func__,
                            ggml_backend_dev_name(dev), reg_name);
                    return result;
                }
            }
        }

        fprintf(stderr,
                "%s: WARNING: --gpu-backend '%s' requested but no matching "
                "GPU device found, falling back to auto\n",
                __func__, pref.c_str());
    }

    return ggml_backend_init_best();
}

// ── Can this backend actually run the model's matmuls? ─────────────────────
//
// A model that drives a single backend with ggml_gallocr + ggml_backend_graph_compute
// (rather than a ggml_backend_sched with a CPU fallback) has NO graceful path for
// an op the device declines: ggml aborts the process. Measured, not theorised —
// GitHub's hosted macos-14 runner exposes an "Apple Paravirtual device"
// (MTLGPUFamilyApple5) that reports
//
//     simdgroup reduction   = false
//     simdgroup matrix mul. = false
//
// so ggml-metal supports no MUL_MAT at all there, and the first encoder GEMM
// killed hft-parity-dump with "unsupported op 'MUL_MAT'" plus a backtrace.
//
// So ASK before committing. This probes GGML_OP_MUL_MAT for every weight dtype
// the GGUF actually contains, which is the op every such model's arithmetic is
// made of and the one that varies by dtype (a device can have an F16 kernel and
// no q4_0 one). It builds no data and allocates nothing but tensor headers.
//
// This is narrower than "the device supports this model": a device that has
// MUL_MAT but lacks, say, POOL_2D would still abort later. The complete answer
// is a ggml_backend_sched with the CPU as the fallback backend, which is a
// graph-lifecycle change rather than an init-time one. This catches the case
// that actually occurs — a device with no matmul kernels is a device with
// nothing useful to offer a transcription model — and it turns an abort into a
// CPU fallback and a warning.
inline bool crispasr_backend_supports_mul_mat(ggml_backend_t backend, const std::vector<ggml_type>& weight_types) {
    if (!backend)
        return false;
    // 256 rows: a multiple of every ggml block size in use (32 for q4_0/q8_0,
    // 256 for the K-quants), so the probe tensor is legal for each dtype.
    const int64_t K = 256, N = 256, M = 64;
    struct ggml_init_params ip = {/*.mem_size =*/ggml_tensor_overhead() * 16,
                                  /*.mem_buffer =*/nullptr,
                                  /*.no_alloc =*/true};
    ggml_context* c = ggml_init(ip);
    if (!c)
        return false;
    bool ok = true;
    for (ggml_type wt : weight_types) {
        if (ggml_blck_size(wt) <= 0 || K % ggml_blck_size(wt) != 0)
            continue; // not a shape this probe can express; leave it to the device
        ggml_tensor* w = ggml_new_tensor_2d(c, wt, K, N);
        ggml_tensor* x = ggml_new_tensor_2d(c, GGML_TYPE_F32, K, M);
        ggml_tensor* y = ggml_mul_mat(c, w, x);
        if (!ggml_backend_supports_op(backend, y)) {
            fprintf(stderr, "%s: %s cannot MUL_MAT a %s weight — falling back to the CPU backend\n", __func__,
                    ggml_backend_name(backend), ggml_type_name(wt));
            ok = false;
            break;
        }
    }
    ggml_free(c);
    return ok;
}
