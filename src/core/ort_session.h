// ort_session.h — onnxruntime session creation helper for the diarization
// backends (pyannote-segmentation-3.0 / TitaNet-Large ONNX).
//
// PR feature: CRISPASR_USE_ONNXRUNTIME. When enabled, model paths ending in
// ".onnx" are executed by onnxruntime instead of the ggml CPU runtime.
// GPU support: the CUDA execution provider is attached when the linked
// onnxruntime library exports it (i.e. the onnxruntime-gpu build) and the
// user has not set CRISPASR_ORT_FORCE_CPU=1. The CUDA EP is resolved with
// dlsym so the same binary works against both the CPU-only and the CUDA
// onnxruntime distributions (the CPU build does not export the CUDA append
// function, which would otherwise be a link-time error).
//
// SPDX-License-Identifier: MIT
#pragma once

#if defined(CRISPASR_USE_ONNXRUNTIME)

#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <string>

#if defined(__linux__)
#include <dlfcn.h>
#endif

namespace crispasr_ort {

// Single process-wide ORT environment (thread-safe static init).
inline Ort::Env& ort_env() {
    static Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "crispasr-ort");
    return env;
}

// Attach the CUDA execution provider if the linked onnxruntime exports it.
// Returns true when the EP was appended successfully. Device id 0.
inline bool try_append_cuda(Ort::SessionOptions& so) {
#if defined(__linux__)
    using AppendCudaFn = OrtStatus* (*)(OrtSessionOptions*, int);
    static AppendCudaFn fn = []() -> AppendCudaFn {
        void* sym = dlsym(RTLD_DEFAULT, "OrtSessionOptionsAppendExecutionProvider_CUDA");
        return reinterpret_cast<AppendCudaFn>(sym);
    }();
    if (!fn)
        return false;
    // SessionOptions converts implicitly to OrtSessionOptions* (Base<T> operator).
    OrtStatus* st = fn(so, 0);
    if (st != nullptr) {
        Ort::GetApi().ReleaseStatus(st);
        return false;
    }
    return true;
#else
    (void)so;
    return false;
#endif
}

// Create a session. Tries CUDA first (unless force_cpu), falls back to CPU.
// `out_provider` receives "CUDA" or "CPU" for logging.
inline Ort::Session create_session(const std::string& model_path, int n_threads, bool force_cpu,
                                   std::string& out_provider) {
    const int threads = std::max(1, n_threads);
    if (!force_cpu) {
        try {
            Ort::SessionOptions so;
            so.SetIntraOpNumThreads(threads);
            so.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
            if (try_append_cuda(so)) {
                Ort::Session sess(ort_env(), model_path.c_str(), so);
                out_provider = "CUDA";
                return sess;
            }
        } catch (const Ort::Exception& e) {
            fprintf(stderr, "crispasr[ort]: CUDA EP unavailable (%s), falling back to CPU\n", e.what());
        }
    }
    Ort::SessionOptions so;
    so.SetIntraOpNumThreads(threads);
    so.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    Ort::Session sess(ort_env(), model_path.c_str(), so);
    out_provider = "CPU";
    return sess;
}

// File-name dispatch helper: ONNX models are selected by the ".onnx" suffix,
// exactly like the GGUF path is selected by ".gguf".
inline bool is_onnx_path(const char* path) {
    if (!path)
        return false;
    std::string s(path);
    if (s.size() < 5)
        return false;
    std::string suffix = s.substr(s.size() - 5);
    std::transform(suffix.begin(), suffix.end(), suffix.begin(),
                   [](unsigned char c) { return (char)std::tolower(c); });
    return suffix == ".onnx";
}

} // namespace crispasr_ort

#endif // CRISPASR_USE_ONNXRUNTIME
