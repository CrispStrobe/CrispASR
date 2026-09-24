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

// Read a float32 array (e.g. tokenizer.ggml.scores). Returns an empty vector
// when the key is missing or has the wrong type.
std::vector<float> kv_f32_array(gguf_context* gctx, const char* key);

// ---------------------------------------------------------------------------
// Pass 2: tensor allocation + weight data copy.
// ---------------------------------------------------------------------------

// CROSS-REPO TENSOR-MAP CONTRACT (read before changing the map type!)
// ------------------------------------------------------------------
// `core/gguf_loader.{h,cpp}` exists in BOTH CrispASR and CrispEmbed. When
// CrispEmbed builds, it compiles CrispASR's `crisp_audio`/`crisp_lid` sources
// against CrispEmbed's copy of this header (they link `crispembed-core`).
// CrispASR standalone prefers `std::map`; CrispEmbed prefers
// `std::unordered_map` (faster). A consumer doing
// `ctx.tensors = std::move(wl.tensors)` needs its field to be the SAME type as
// `WeightLoad::tensors` — but that type differs per repo. Hard-coding either
// type in the consumer broke the other build, causing the repeated
// std::map<->unordered_map flip-flop (commits e6693b23/ad869798/d1bd3b91/
// 1e4f1184/844f89d3, …).
//
// FIX: expose the type as a single alias `core_gguf::tensor_map`. Each repo's
// header defines it as its own choice; consumers (crisp_audio/audio_tower,
// crisp_lid/lid_cld3) declare `core_gguf::tensor_map tensors;` so it tracks
// whichever gguf_loader.h is compiled. Do NOT hard-code the map type in those
// consumer structs again — change this alias instead.
using tensor_map = std::map<std::string, ggml_tensor*>;

struct WeightLoad {
    ggml_context* ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    // PLAN #69a layer offload: optional second backend buffer for tensors
    // routed off-GPU. Non-null only when load_weights_split() was used.
    ggml_backend_buffer_t buf_cpu = nullptr;
    // Issue #276: extra buffers from chunked allocation in load_weights_split().
    // AMD Vulkan (proprietary driver) caps per-allocation at 2 GiB, so a
    // partition is allocated in chunks of at most 1.5 GiB. The first GPU/CPU
    // buffer is in buf/buf_cpu; any overflow chunks are listed here.
    //
    // READ-ONLY. These handles are owned by the loader and are released with
    // the primary buffer of their own partition, so freeing one here is a
    // double free. The list is for inspecting how a load was partitioned.
    std::vector<ggml_backend_buffer_t> split_bufs;
    tensor_map tensors;
};

// Load all tensor metadata + weights into a new ggml_context backed by
// a newly-allocated backend buffer. On success the WeightLoad struct is
// populated and the caller takes ownership of ctx/buf (typically moving
// them into the model struct).
//
// model_tag is used only in error messages ("parakeet: ...").
bool load_weights(const char* path, ggml_backend_t backend, const char* model_tag, WeightLoad& out);

// Load only tensors accepted by `include_tensor`. Metadata keys remain
// available through the caller's separate open_metadata() pass, while the
// returned ggml context and any allocated backend buffers contain only the
// selected tensors. This is intended for independently usable components
// embedded in a larger GGUF (for example an audio codec stored alongside an
// LLM), where loading the parent model would waste RAM/VRAM. Whole-file
// preload, readahead, and mlock hints are deliberately suppressed for this
// path so excluded parent tensors do not become resident as a side effect.
using IncludeTensor = bool (*)(const char* tensor_name, void* user);
bool load_weights_filtered(const char* path, ggml_backend_t backend, IncludeTensor include_tensor, void* user,
                           const char* model_tag, WeightLoad& out);

// PLAN #69a: layer-residency-aware weight loader. Tensors for which
// `is_gpu(tensor_name, user) == true` go on the GPU backend; the rest
// go on the CPU backend. ggml_backend_sched then auto-routes ops to
// follow weight residency, giving llama.cpp's `--n-gpu-layers` behaviour.
//
// Caller takes ownership of `out.ctx`, `out.buf` (gpu partition), and
// `out.buf_cpu` (cpu partition), and must free all three — the buffers with
// release_weight_buffer(), the context with ggml_free() — or hand the whole
// WeightLoad to free_weights(). Overflow chunks are NOT the caller's to free:
// see the note on WeightLoad::split_bufs.
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

// load_weights() onto `backend`, except tensors that backend cannot bind go to
// `cpu_backend`. "Cannot bind" is the backend's own supports_op() on the weight
// leaf - the exact check ggml_backend_sched later aborts on ("pre-allocated
// tensor ... in a buffer that cannot run the operation"). Vulkan rejects any
// tensor larger than maxStorageBufferRange, which a device can allocate but not
// bind: qwen3-tts's 622 MB text embedding on lavapipe (#337). When every tensor
// fits this IS load_weights() (mmap fast path, out.buf_cpu stays null);
// otherwise it is load_weights_split() and the caller owns out.buf_cpu too.
// `n_offloaded` (optional) receives the number of tensors routed to the CPU.
bool load_weights_fit(const char* path, ggml_backend_t backend, ggml_backend_t cpu_backend, const char* model_tag,
                      WeightLoad& out, int* n_offloaded = nullptr);

// ---------------------------------------------------------------------------
// ggml CPU repack ("extra") buffer type — docs/ggml-optimisation-playbook.md §4
// ---------------------------------------------------------------------------
//
// ggml's repacked int8 GEMM (ggml/src/ggml-cpu/repack.cpp) is several times
// faster than the generic quantised path, and it is reached ONLY by putting
// the weight in the CPU device's *extra* buffer type. Dispatch is keyed on the
// weight's buffer type inside ggml_compute_forward itself
// (ggml-cpu.c: ggml_cpu_extra_compute_forward), so no ggml_backend_sched is
// required — a model on a raw gallocr gets it too.
//
// Measured on the CrispASR VPS (Xeon Skylake-SP, AVX-512F/DQ/CD/BW/VL,
// NO AVX-512 VNNI, NO AMX), single thread, interleaved A/B, best-of-40:
//
//   q4_0  generic 1.22-1.44x f32   ->  repacked 0.76-0.97x f32   (1.4-1.7x)
//   q4_K  generic 1.83-3.02x f32   ->  repacked 0.48-0.66x f32   (3.2-5.6x)
//   q8_0  generic 1.08-1.31x f32   ->  NO x86 repack kernel exists
//
// Reproduce with `crispasr-repack-probe`. Four things an adopter must know:
//
//  1. **q8_0 has no x86 repack kernel at all.** ggml_repack_get_optimal_repack_type
//     gates GGML_TYPE_Q8_0 on NEON + dotprod/i8mm (or RISC-V), with no AVX2 or
//     AVX-512 branch. Every q8_0 model in this tree therefore gains nothing from
//     this path on x86 — it needs q4_0 or q4_K. On arm64 with dotprod, q8_0 does
//     have a kernel, so this is an ISA-specific statement, not a general one.
//  2. **The repack buffer type's set_tensor dereferences tensor->extra
//     unconditionally.** When it has no kernel for a given (type, shape, ISA) it
//     leaves extra null, so writing such a tensor is a null dereference, not a
//     graceful fallback. Tensors must be classified BEFORE they are written,
//     which is what repack_buft_accepts() below is for.
//  3. **The buffer type supports only MUL_MAT and MUL_MAT_ID**, with a 2-D
//     src[0] and an F32 src[1]. (Not GET_ROWS — src/crispasr.cpp's comment
//     saying otherwise is stale against this ggml version.) The loader cannot
//     tell which tensors are used that way; only the model can, hence the
//     predicate.
//  4. **This path gives up the zero-copy mmap of load_weights().** Verified,
//     not inferred: the mmap path binds tensor->data straight into the file map
//     and never calls set_tensor at all, while repacking is precisely a
//     set_tensor that rewrites the bytes into a buffer the buft owns. The two
//     cannot coexist for the same tensor. Expect load-time and RSS cost in
//     exchange for GEMM throughput, and measure both.

// Returns true when the CPU repack buffer type on THIS host has a repacked
// kernel for a 2-D tensor of this type and shape — i.e. when putting such a
// tensor in that buffer type is safe and useful. False when no repack buffer
// type is offered at all. Cheap: allocates and frees one tensor's worth of
// memory to ask ggml rather than duplicating its dispatch table.
bool repack_buft_accepts(ggml_type type, int64_t ne0, int64_t ne1);

// Load weights onto `cpu_backend`, placing every tensor for which
// `is_matmul_weight(name, user)` is true AND repack_buft_accepts() agrees into
// ggml's CPU repack buffer type, and everything else into the default one.
//
// The predicate must be true ONLY for tensors used exclusively as src[0] of a
// 2-D MUL_MAT / MUL_MAT_ID with an F32 activation. A tensor in the repack
// buffer type has had its bytes rewritten into an interleaved layout: reading
// it back, copying it, or using it in any other op will not do what you mean.
//
// `n_repacked`, when non-null, receives how many tensors actually landed in the
// repack buffer type. Zero is a normal outcome — no repack buft on this host,
// or no tensor of a type it has a kernel for — and not an error; the load still
// succeeds via the default buffer type.
//
// Ownership matches load_weights_split(): out.buf is the repack partition,
// out.buf_cpu the default one, and free_weights() releases both.
using IsMatmulWeight = bool (*)(const char* tensor_name, void* user);
bool load_weights_repack(const char* path, ggml_backend_t cpu_backend, IsMatmulWeight is_matmul_weight, void* user,
                         const char* model_tag, WeightLoad& out, int* n_repacked = nullptr);

// PLAN #69a — generic predicate for the "<prefix><N>." tensor naming
// used by every LLM-decode backend in src/. Each backend has its own
// prefix:
//   "blk."          (voxtral, voxtral4b, qwen3_asr, granite_speech, gemma4_e2b, mimo_asr)
//   "llm.blk."      (glm_asr)
//   "talker.blk."   (orpheus)
//   "dec."          (omniasr)
// Returns -1 if the tensor name doesn't match `<prefix><integer>.`.
int blk_layer_of_with_prefix(const char* tensor_name, const char* prefix);

// Convenience for backends using bare "blk.<N>." (the most common
// scheme). Equivalent to blk_layer_of_with_prefix(name, "blk.").
int blk_layer_of(const char* tensor_name);

// Configurable predicate: tensors named `<cfg.prefix><N>.<rest>` go to
// CPU iff N >= cfg.threshold; anything else (different prefix, no
// integer, threshold-violating layer) stays on GPU. Pass a pointer
// to a LayerSplitConfig as `user`.
struct LayerSplitConfig {
    const char* prefix; // e.g. "blk.", "llm.blk.", "talker.blk.", "dec."
    int threshold;      // N — first CPU-resident layer
};
bool is_gpu_tensor_with_prefix(const char* tensor_name, void* user);

// Bare "blk." convenience: pass a pointer to an int (the threshold)
// as `user`. Equivalent to is_gpu_tensor_with_prefix() with
// LayerSplitConfig{ "blk.", *N }.
bool is_gpu_tensor_blk(const char* tensor_name, void* user);

// Free a backend buffer that came from one of this loader's weight-loading
// entry points, releasing the host mmap behind it when there is one.
//
// USE THIS INSTEAD OF ggml_backend_buffer_free() FOR ANY BUFFER OBTAINED FROM
// load_weights() / load_weights_filtered() / load_weights_split(), including
// after the buffer has been moved into a model struct. On a non-CPU backend
// advertising `buffer_from_host_ptr` (Apple-Silicon Metal), load_weights hands
// the device a host mmap it does not own — `buffer_from_host_ptr` has no
// deallocator parameter, so freeing the backend buffer alone leaves the whole
// weight file mapped, resident and dirty for the life of the process.
//
// Semantics:
//   * A null handle is a no-op, and the caller's handle is nulled on return,
//     so a second call cannot double-free or double-unmap.
//   * A buffer with no recorded mapping is the ordinary case — the CPU mmap
//     path unmaps through its own free callback and the legacy alloc+copy
//     path maps nothing — and is released like any other backend buffer.
//   * The backend buffer is freed before the region is unmapped, so a
//     device-side view of the pages never outlives them.
//   * Any overflow chunks of the same partition (issue #276) are released
//     with it, so a split load needs no separate teardown.
void release_weight_buffer(ggml_backend_buffer_t& buf);

// Free a WeightLoad's resources. Call when the model is being destroyed
// and the buffer/context are not held elsewhere. Releases every buffer
// through release_weight_buffer().
void free_weights(WeightLoad& wl);

// PLAN #60g: hint the kernel that the mmap'd weight region is now being
// accessed in random order (e.g., the per-layer KV revisit pattern of
// decode steps), and that readahead is therefore wasted IO. No-op if
// the buffer wasn't allocated through one of our mmap paths (e.g.,
// when `CRISPASR_GGUF_MMAP=0` opts out of the default mmap loader).
// Safe to call multiple times.
//
// Recommended use: after prefill completes, before entering the decode
// loop. See PLAN #60g for the rationale.
void mmap_advise_random(ggml_backend_buffer_t buf);

// ---------------------------------------------------------------------------
// Tensor lookup helpers
// ---------------------------------------------------------------------------

// Look up a tensor by name. Returns nullptr (silently) if missing.
// Uses `tensor_map` (see the cross-repo contract note above) so the signature
// tracks the per-repo map choice automatically.
ggml_tensor* try_get(const tensor_map& tensors, const char* name);

// Look up a tensor by name. Prints an error to stderr if missing but
// still returns nullptr — the caller decides whether a missing tensor
// is fatal.
ggml_tensor* require(const tensor_map& tensors, const char* name, const char* model_tag);

// Build a shell command that produces the formatted tensor name for a
// per-layer lookup. Avoids the snprintf(buf, sizeof(buf), "...", i) line
// that every loader repeats.
std::string format_layer_name(const char* fmt, int i);
std::string format_layer_name(const char* fmt, int i, int j);

} // namespace core_gguf
