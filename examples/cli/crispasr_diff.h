// crispasr_diff.h — shared ground-truth diff harness.
//
// Companion to tools/dump_reference.py. A diff-test driver loads a
// GGUF archive produced by the Python dumper through this module, then
// asks it to compare any C++ ggml tensor (or raw float buffer) against
// the corresponding named reference tensor. Replaces the per-test
// inline `load_npy_f32` + ad-hoc metric code that every diff test used
// to carry (~50 lines each).
//
// Usage:
//
//   crispasr_diff::Ref ref;
//   if (!ref.load("/tmp/voxtral-ref.gguf")) return 1;
//
//   // ... run your C++ forward pass ...
//
//   auto r = ref.compare("mel_spectrogram", cpp_mel_data, cpp_mel_nelem);
//   printf("mel: max_abs=%.2e cos_min=%.6f  %s\n",
//          r.max_abs, r.cos_min, r.is_pass(0.999f) ? "PASS" : "FAIL");
//
// The returned Report has max_abs / mean_abs / rms / cos_min /
// top1_match_rate fields and an `is_pass(cos_threshold)` helper. Test
// drivers stop caring about NPY format, shape checking, or cosine loop
// writing — they just name the stage and compare.
//
// ---------------------------------------------------------------------------
// Per-layer diff recipe (what cracked issue #37 — see LEARNINGS.md)
// ---------------------------------------------------------------------------
// When an encoder runtime produces fluent-but-wrong output and the
// final encoder cos < 0.999, follow this protocol to localise the bug:
//
//   1. PYTHON SIDE — dump per-layer reference activations.
//      In the backend's reference_backends/<name>.py, register forward
//      hooks on every encoder submodule of interest (subsampling,
//      each conformer/transformer layer, attention, conv, ffn).
//      Use `tools/reference_backends/_hooks.py` (capture_modules /
//      drop_hooks / finalize) — it normalises (B, T, D) -> (T, D)
//      and matches crispasr's flat row-major layout.
//
//      Add the stage names to DEFAULT_STAGES so they get captured
//      automatically: pre_encode_output, encoder_layer_0..N-1, etc.
//
//   2. C++ SIDE — add a per-layer dump entry point to the runtime.
//      Mirror the production encoder graph but tag every layer's
//      output with `ggml_set_name("dump_layer_K")` + `ggml_set_output()`.
//      Run the graph once, then read each tagged tensor back via
//      `ggml_backend_tensor_get`. Allocate one (T_enc * d_model) F32
//      buffer per dumped stage. See parakeet_run_encoder_dump() in
//      src/parakeet.cpp for a worked example (~50 LOC).
//
//   3. DIFF HARNESS — wire stages into crispasr_diff_main.cpp.
//      Call ref.compare("encoder_layer_K", our_K_buf, n_elem) for each
//      stage. The first K where cos_mean drops below ~0.999 is where
//      the bug lives. Reading the runtime's loader / build_block code
//      against that layer's GGUF tensors usually finds it in minutes.
//
//   4. ISOLATE MEL VS ENCODER — add an "encoder_output_ref_mel" stage
//      that feeds the REFERENCE mel into the C++ encoder (not our
//      C++ mel). If cos matches the production encoder_output cos,
//      mel propagation isn't the issue and the bug is encoder-internal.
//      See parakeet_encoder_with_ref_mel_r() in crispasr_diff_main.cpp.
//
// The same recipe applies to LLM stages (per-layer hidden states),
// audio projector outputs, RVQ codec stages — anywhere a sequential
// pipeline can drift from a Python reference.

#pragma once

#include "crispasr_backend.h" // (unrelated, but makes linkage consistent)

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace crispasr_diff {

// Result of a single tensor comparison. All metrics computed over the
// intersection of the two tensors (truncated to min element count if
// shapes differ — the caller usually validates shapes separately).
struct Report {
    bool found = false;         // the name existed in the archive
    size_t n_elem = 0;          // number of elements compared
    float max_abs = 0.0f;       // max |cpp[i] - ref[i]|
    float mean_abs = 0.0f;      // mean |cpp[i] - ref[i]|
    float rms = 0.0f;           // sqrt(mean((cpp-ref)^2))
    float cos_min = 1.0f;       // worst per-row cosine similarity (-1 .. 1)
    float cos_mean = 1.0f;      // average per-row cosine similarity
    int top1_match = 0;         // logits only: tokens matching ref argmax
    int top1_total = 0;         // logits only: total tokens compared
    std::vector<int64_t> shape; // shape of the ref tensor

    bool is_pass(float cos_threshold = 0.999f) const { return found && cos_min >= cos_threshold; }
};

// Ground-truth archive loaded from a crispasr reference GGUF.
//
// Owns a ggml_context + backend buffer holding all reference tensors.
// Destructor releases them. Copy is disabled; move is allowed.
class Ref {
public:
    Ref() = default;
    ~Ref();

    Ref(const Ref&) = delete;
    Ref& operator=(const Ref&) = delete;
    Ref(Ref&& other) noexcept : impl_(other.impl_) { other.impl_ = nullptr; }
    Ref& operator=(Ref&& other) noexcept {
        if (this != &other) {
            delete impl_;
            impl_ = other.impl_;
            other.impl_ = nullptr;
        }
        return *this;
    }

    // Load the GGUF archive. Returns false with stderr on failure.
    bool load(const std::string& path);

    // Has a tensor with this name been loaded?
    bool has(const std::string& name) const;

    // Retrieve the raw reference data as float. Returns a pointer owned
    // by the Ref object (valid until it's destructed) and the element
    // count. Returns {nullptr, 0} if the name is missing.
    std::pair<const float*, size_t> get_f32(const std::string& name) const;

    // Shape of a reference tensor as GGUF stored it. Empty vector if
    // the name is missing.
    std::vector<int64_t> shape(const std::string& name) const;

    // Compare a raw float buffer against the named reference tensor.
    // The caller is responsible for providing data in the same logical
    // layout the Python dumper used (the Python side writes F32
    // row-major C-order, which matches what the C++ backends produce
    // when they call ggml_backend_tensor_get).
    //
    // When cmp_type == COS, the rows are the last dimension of the
    // reference tensor, and cos_min / cos_mean reflect per-row cosine
    // similarity. When cmp_type == L2, rows aren't used and cos_* are
    // left at their defaults.
    enum CompareMode { COS_LAST_DIM, L2_ONLY };
    Report compare(const std::string& name, const float* data, size_t n_elem, CompareMode mode = COS_LAST_DIM) const;

    // Convenience: compare the argmax-over-last-dim of `data` against
    // the argmax-over-last-dim of the named reference. Used for LLM
    // logits ("does the C++ path produce the same greedy token as the
    // PyTorch reference?"). Populates report.top1_match / top1_total.
    Report compare_argmax(const std::string& name, const float* data, size_t n_elem) const;

    // Metadata keys from the Python side (backend name, model path,
    // audio path, generated text). Empty string if missing.
    std::string meta(const std::string& key) const;

    // List the tensor names in the archive.
    std::vector<std::string> tensor_names() const;

public:
    // Exposed publicly so the free helper `cache_f32` in the .cpp can
    // reach it; treat as an implementation detail.
    struct Impl;

private:
    Impl* impl_ = nullptr;
};

} // namespace crispasr_diff
