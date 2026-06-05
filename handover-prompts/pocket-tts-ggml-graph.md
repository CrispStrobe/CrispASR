# Pocket-TTS: rewrite manual CPU compute → ggml graph (GPU-capable)

## What this is

The `pocket-tts` backend (`src/pocket_tts.cpp`, 2206 LOC) is the only TTS backend
in CrispASR that does NOT use ggml compute graphs. Instead it has ~38 hand-written
C++ functions (`linear_f32`, `layer_norm`, `rms_norm`, `apply_rope_inplace`,
`conv1d_eager`, `conv_transpose1d_eager`, `vec_mul`, `softmax`, etc.) that do
element-wise loops on raw `float*` buffers read from ggml tensors via
`ggml_backend_tensor_get`.

This means: no SIMD matmul, no GPU offload, no `ggml_backend_sched`. The model is
1B params (Llama backbone) and ~15× slower than it should be.

**Goal:** Replace the manual compute with proper ggml graphs so the backend can use
`ggml_backend_sched` + `ggml_backend_init_best()` for Metal/CUDA acceleration,
matching every other backend in the project.

## Architecture (3 stages, all must be converted)

### Stage 1: Backbone (Llama AR transformer)
- **File:** `backbone_forward_step()` at line ~948
- **Architecture:** Llama-1B (d=1024, 16 heads, 6 layers, RoPE, pre-norm LN, GELU FFN, fused QKV)
- **Pattern:** per-token autoregressive with KV cache — called once per generated token
- **Weights:** `backbone_layers[i].attn_in_proj` (3*D,D), `attn_out_proj` (D,D),
  `attn_norm_w/b`, `ff1/ff2` (D,FF)/(FF,D), `ff_norm_w/b`
- **KV cache:** manual `pocket_tts_kv_cache` struct with `k[n_layers * max_seq * D]`
- **ggml equivalent:** `ggml_mul_mat` for projections, `ggml_rope_ext` for RoPE,
  `ggml_flash_attn_ext` for causal attention, `ggml_view_1d` for KV cache slots.
  Mirror kokoro.cpp or dia_tts.cpp KV-cached AR loops.

### Stage 2: Flow Net (DiT-like denoiser)
- **File:** `flow_net_forward()` at line ~1205
- **Architecture:** DiT with timestep embeddings, cross-attention to backbone output,
  ~10 Euler ODE steps (configurable via `POCKET_LSD_STEPS`)
- **Weights:** `flow_net.time_embeds[2]`, `flow_net.blocks[i]` with self-attn +
  cross-attn + FFN
- **Pattern:** Fixed number of forward passes (not autoregressive), each takes the
  full latent sequence. No KV cache.
- **ggml equivalent:** Build one graph per ODE step, or build the full DiT block
  graph and re-run with different inputs.

### Stage 3: Mimi Decoder (SEANet convolutions + transformer)
- **File:** `mimi_decode()` at line ~1436
- **Architecture:** Quantizer lookup → Conv1d stack with ELU activations →
  ConvTranspose1d upsampling (strides [8,6,5,4] = 960×) → residual blocks →
  2-layer transformer with LayerScale → final Conv1d → PCM @ 24kHz
- **Weights:** `model.mimi_dec_*` tensors, `dec_transformer_layers[i]`
- **Special ops:** Causal Conv1d (left-padding only), ELU activation, ConvTranspose1d,
  LayerScale (learnable per-element scale after attention/FFN)
- **Pattern:** Single forward pass over the full latent sequence. Per-frame KV-cached
  transformer decode is used for the 2-layer internal transformer.
- **ggml equivalent:** `ggml_conv_1d`, `ggml_conv_transpose_1d`, `ggml_elu` (or
  custom), `ggml_flash_attn_ext`. The causal padding is `pad_left = (kernel-1)*dilation`.
  Use `core/hifigan.h` conv helpers where applicable.

## Approach (DO NOT skip any step)

### Step 0 — worktree
```bash
git worktree add ../crispasr-pocket-ggml -b pocket-ggml main
cd ../crispasr-pocket-ggml
```

### Step 1 — reference dumps (ALREADY EXIST)
There are existing reference dumps at `/mnt/storage/pocket-tts/dumps*/`. The env var
`POCKET_DUMP_DIR` dumps per-stage intermediates. Verify the current manual-compute
backend matches these before changing anything:
```bash
POCKET_DUMP_DIR=/mnt/storage/pocket-tts/dumps-baseline \
./build/bin/crispasr --backend pocket-tts \
    -m /mnt/storage/pocket-tts/pocket-tts-english-novc-f16.gguf \
    --tts "Hello there." --tts-output /tmp/pocket_baseline.wav --seed 42
```

### Step 2 — convert ONE stage at a time
**Do the Mimi decoder first** — it's the simplest (single forward pass, no AR loop,
no KV cache). It's also the most compute-intensive (960× upsampling).

For each stage:
1. Build the ggml graph alongside the existing manual code
2. Run both, compare outputs (cosine similarity)
3. When cos ≈ 1.0, delete the manual code
4. Move to the next stage

### Step 3 — add ggml_backend_sched
Once all three stages use ggml graphs:
1. Add `ggml_backend_t backend` + `ggml_backend_sched_t sched` to context
2. `backend = params.use_gpu ? ggml_backend_init_best() : backend_cpu`
3. Load weights via `core_gguf::load_weights(path, backend, ...)`
4. Replace `ggml_gallocr` with `sched_reset` + `sched_alloc_graph` + `sched_graph_compute`
5. Wire `p.use_gpu = g_open_use_gpu_tls` in `src/crispasr_c_api.cpp`

### Step 4 — validate
ASR roundtrip: generate WAV → whisper transcribe → text should match input.

## Key files
- `src/pocket_tts.cpp` — the runtime (2206 LOC, everything is here)
- `src/pocket_tts.h` — public C API header
- `examples/cli/crispasr_backend_pocket.cpp` — CLI adapter (already working)
- Model: `/mnt/storage/pocket-tts/pocket-tts-english-novc-f16.gguf`
- Existing dumps: `/mnt/storage/pocket-tts/dumps*/`

## What NOT to change
- The weight loading (`load_flow_lm_tensors`, `load_mimi_decoder_tensors`) — these
  already use ggml tensors stored in ctx. The weights are already ggml tensors.
- The tokenizer (`load_tokenizer`, `bpe_encode`) — pure CPU text processing, fine.
- The hparams struct — already reads from GGUF metadata correctly.
- The C API (`pocket_tts_init_from_file`, `pocket_tts_synthesize`, `pocket_tts_free`,
  etc.) — keep the same public interface.

## Critical caveats

1. **The weights are already ggml tensors.** `load_flow_lm_tensors` stores them as
   `ggml_tensor*` in the model struct. The manual code reads them with
   `tensor_f32_data()` which does `ggml_backend_tensor_get` into a cached F32 buffer.
   For ggml graphs, just reference the tensor directly in `ggml_mul_mat(W, x)`.

2. **Causal Conv1d needs left-only padding.** The Mimi decoder uses causal convolutions
   where `padding = (kernel_size - 1) * dilation` on the LEFT only (no right padding).
   `ggml_conv_1d` pads symmetrically. You must either: (a) pre-pad the input with zeros
   on the left and use `padding=0`, or (b) pad symmetrically and trim the right side.
   The existing `conv1d_eager` does left-only padding manually.

3. **ELU activation.** ggml has `ggml_elu` — verify it matches PyTorch's
   `ELU(alpha=1.0)`: `x if x > 0 else alpha * (exp(x) - 1)`.

4. **KV cache for backbone AR loop.** The backbone runs per-token with a KV cache.
   For the ggml graph version, allocate KV cache tensors in the model context (not
   malloc'd float arrays), and use `ggml_view_1d` / `ggml_cpy` to update cache slots.
   Mirror how kokoro.cpp or dia_tts.cpp handle their AR KV caches.

5. **KV cache for Mimi decoder transformer.** The 2-layer internal transformer also
   has a KV cache (`dec_xfmr_kv`). This runs per-frame during Mimi decoding.

6. **F16 tensor data.** Many weights are F16. The manual code converts via
   `tensor_f32_data()` with a cache. With ggml graphs this is handled automatically —
   `ggml_mul_mat` supports mixed F16/F32.

7. **RoPE parameters.** The backbone uses `max_period = 1e6` (not the standard 10000).
   Pass this via `ggml_rope_ext` freq_base parameter.

8. **ggml_conv_1d input layout.** ggml expects `ne[0]=T` (spatial), `ne[1]=Cin`
   (channels). The tensor data stores `ne[0]` contiguously. See LEARNINGS.md
   "FastPitch Bug 3" for the full explanation.

9. **ggml_conv_1d output is 3D.** Returns `(OL, OC, N)` even for N=1. Squeeze to 2D
   with `ggml_reshape_2d` before adding bias. See LEARNINGS.md "FastPitch Bug 4".

10. **No hardcoded absolute paths.** Use env vars / runtime-chosen dirs.

11. **ALWAYS work in a worktree.** Never edit the shared checkout.

12. **Format with clang-format v18** (`./tools/format.sh --fix`) before committing.

13. **No Co-Authored-By lines** in commits (repo convention).

14. **`git stash && git pull --rebase cohere main && git stash pop`** before every commit.

## Model architecture summary

```
Text → BPE tokenize (9967 vocab)
     → Embedding(9967, 1024)
     → Backbone Transformer (6L, 1024D, 16H, RoPE, GELU FFN, pre-norm LN)
       [AR loop: generate latent codes token by token, KV-cached]
     → Codebook heads: 8 Linear(1024, 2048) → argmax → code indices
     → Per-frame:
         Flow Net (DiT): noise + backbone_out → Euler ODE (10 steps)
           → latent vectors (1024-d per frame)
     → Mimi Decoder:
         Quantizer lookup: 8 codebooks × Embedding(2048, 128) → sum → (128, T)
         Linear(128, 1024) projection
         SEANet: Conv1d(1024, 512, k=7) + 4× [ELU + ConvTr(↑8/6/5/4) + ResBlocks]
           + 2L Transformer (d=512, 8H, LayerScale)
           + ELU + Conv1d(32, 1, k=7) → PCM @ 24kHz
```

## Estimated scope
- ~800-1200 new lines of ggml graph-building code
- ~600 lines of manual compute code deleted
- 3 major compute stages to convert
- Mimi decoder is the best starting point (no AR loop, largest compute)
