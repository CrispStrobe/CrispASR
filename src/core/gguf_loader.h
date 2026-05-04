// src/core/gguf_loader.h — shared GGUF weight loading scaffolding.
//
// Every model implementation in src/ has its own copy of the "open a
// GGUF file, read its hyperparameters, allocate a backend buffer, mmap
// the weight data, and build a name -> tensor lookup map" dance. The
// code is ~40-60 lines per model and is essentially identical across
// them, with only the model-specific prefix and tensor naming scheme
// changing.
//
// This helper extracts the shared scaffolding. What stays model-specific:
//
//   * Hyperparameter reading (each model has its own hparams struct
//     and GGUF key prefix, e.g. "parakeet.n_layers" vs "voxtral.n_layers").
//   * Vocabulary / tokenizer loading (varies by tokenizer type).
//   * The actual per-field assignment loop that pulls tensors out of
//     the map and stores them in per-layer struct fields.
//
// What this helper does for the model:
//
//   * Opens the GGUF file in two passes (metadata, then tensor alloc).
//   * Provides scalar / string / array reader helpers with defaults.
//   * Allocates the backend buffer and mmap-copies the weight data.
//   * Builds the std::map<std::string, ggml_tensor *> tensor
//     lookup map and returns it in a WeightLoad struct.
//   * Provides require() / try_get() tensor lookup helpers that log a
//     sensible error message when a required tensor is missing.
//
// Usage pattern (each model's *_model_load function):
//
//   // 1. Metadata pass — read hyperparameters.
//   gguf_context * meta = core_gguf::open_metadata(path);
//   if (!meta) return false;
//   hp.n_layers = core_gguf::kv_u32(meta, "mymodel.n_layers", hp.n_layers);
//   // ... other hparams
//   core_gguf::load_vocab_array(meta, "tokenizer.ggml.tokens", vocab);
//   core_gguf::free_metadata(meta);
//
//   // 2. Weight pass — allocate backend buffer, mmap, build tensor map.
//   core_gguf::WeightLoad wl;
//   if (!core_gguf::load_weights(path, backend, wl)) return false;
//   model.ctx     = wl.ctx;
//   model.buf     = wl.buf;
//   model.tensors = std::move(wl.tensors);
//
//   // 3. Bind named tensors into struct fields.
//   model.attn.q_w = core_gguf::require(model.tensors, "encoder.attn.q.weight", "mymodel");
//   ... etc.

#pragma once

#include "ggml.h"
#include "ggml-backend.h"
#include "gguf.h"

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace core_gguf {

// ---------------------------------------------------------------------------
// Pass 1: metadata (hyperparameters + vocab).
// ---------------------------------------------------------------------------

// Open the GGUF for metadata-only reading. Returns a gguf_context owned
// by the caller — free with free_metadata() when done reading keys.
// Returns nullptr and prints an error to stderr on failure.
gguf_context* open_metadata(const char* path);

// Free a gguf_context obtained from open_metadata().
void free_metadata(gguf_context* gctx);

// Scalar key readers with defaults. All return the default value when
// the key is absent or the type doesn't match.
uint32_t kv_u32(gguf_context* gctx, const char* key, uint32_t default_val);
int32_t kv_i32(gguf_context* gctx, const char* key, int32_t default_val);
float kv_f32(gguf_context* gctx, const char* key, float default_val);
bool kv_bool(gguf_context* gctx, const char* key, bool default_val);
std::string kv_str(gguf_context* gctx, const char* key, const char* default_val);

// Read a string array (e.g. tokenizer.ggml.tokens). Returns an empty
// vector when the key is missing or has the wrong type.
std::vector<std::string> kv_str_array(gguf_context* gctx, const char* key);

// ---------------------------------------------------------------------------
// Pass 2: tensor allocation + weight data copy.
// ---------------------------------------------------------------------------

struct WeightLoad {
    ggml_context* ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    // PLAN #69a layer offload: optional second backend buffer for tensors
    // routed off-GPU. Non-null only when load_weights_split() was used.
    ggml_backend_buffer_t buf_cpu = nullptr;
    std::map<std::string, ggml_tensor*> tensors;
};

// Load all tensor metadata + weights into a new ggml_context backed by
// a newly-allocated backend buffer. On success the WeightLoad struct is
// populated and the caller takes ownership of ctx/buf (typically moving
// them into the model struct).
//
// model_tag is used only in error messages ("parakeet: ...").
bool load_weights(const char* path, ggml_backend_t backend, const char* model_tag, WeightLoad& out);

// PLAN #69a: layer-residency-aware weight loader. Tensors for which
// `is_gpu(tensor_name, user) == true` go on the GPU backend; the rest
// go on the CPU backend. ggml_backend_sched then auto-routes ops to
// follow weight residency, giving llama.cpp's `--n-gpu-layers` behaviour.
//
// Caller takes ownership of `out.ctx`, `out.buf` (gpu partition), and
// `out.buf_cpu` (cpu partition). All three must be freed by the caller
// or via free_weights() / free_weights_split() at shutdown.
//
// Falls back to the legacy alloc+copy path internally — the mmap
// optimisations in load_weights() require contiguous tensor regions
// that the split partition can't satisfy. Acceptable trade-off: users
// who set N_GPU_LAYERS are accepting the extra RAM hit to fit the
// model at all.
//
// Returns false on any allocation / load failure with a logged
// stderr message; partial state is freed before returning.
using IsGpuTensor = bool (*)(const char* tensor_name, void* user);
bool load_weights_split(const char* path, ggml_backend_t gpu_backend, ggml_backend_t cpu_backend, IsGpuTensor is_gpu,
                        void* user, const char* model_tag, WeightLoad& out);

// PLAN #69a — generic predicate for the "blk.<N>." tensor naming used
// by every LLM-decode backend in src/. Pass a pointer to the layer
// threshold as `user`; tensors whose layer index >= *threshold are
// routed to CPU, everything else (audio enc, projection, embeddings,
// output_norm, blk.<n>.* with n < threshold) stays on GPU.
//
// Returns -1 for non-layered tensors so backends with a different
// naming scheme can build their own predicate around this primitive.
int blk_layer_of(const char* tensor_name);

// Convenience predicate: true if `tensor_name` should live on GPU,
// given a threshold of N. `user` must point to an int holding N.
bool is_gpu_tensor_blk(const char* tensor_name, void* user);

// Free a WeightLoad's resources. Call when the model is being destroyed
// and the buffer/context are not held elsewhere.
void free_weights(WeightLoad& wl);

// PLAN #60g: hint the kernel that the mmap'd weight region is now being
// accessed in random order (e.g., the per-layer KV revisit pattern of
// decode steps), and that readahead is therefore wasted IO. No-op if
// the buffer wasn't allocated through one of our mmap paths or if
// CRISPASR_GGUF_MMAP wasn't set. Safe to call multiple times.
//
// Recommended use: after prefill completes, before entering the decode
// loop. See PLAN #60g for the rationale.
void mmap_advise_random(ggml_backend_buffer_t buf);

// ---------------------------------------------------------------------------
// Tensor lookup helpers
// ---------------------------------------------------------------------------

// Look up a tensor by name. Returns nullptr (silently) if missing.
ggml_tensor* try_get(const std::map<std::string, ggml_tensor*>& tensors, const char* name);

// Look up a tensor by name. Prints an error to stderr if missing but
// still returns nullptr — the caller decides whether a missing tensor
// is fatal.
ggml_tensor* require(const std::map<std::string, ggml_tensor*>& tensors, const char* name, const char* model_tag);

// Build a shell command that produces the formatted tensor name for a
// per-layer lookup. Avoids the snprintf(buf, sizeof(buf), "...", i) line
// that every loader repeats.
std::string format_layer_name(const char* fmt, int i);
std::string format_layer_name(const char* fmt, int i, int j);

} // namespace core_gguf
