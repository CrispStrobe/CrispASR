**Title:** `ggml-sched : fix cross-backend copy insertion for small mixed-backend graphs`

---

## Root cause

The ggml scheduler's backend assignment algorithm (passes 2-4 in
`ggml_backend_sched_split_graph`) uses a "flood fill" heuristic that
expands GPU assignments outward from GPU-anchored nodes. For large
graphs with many GPU-weight-referencing ops, this correctly assigns
most compute to GPU and inserts CPU→GPU copy nodes at split boundaries.

For **small graphs** where the only CPU-resident tensor is an
embedding table (fed to `ggml_get_rows`), the flood fill fails:

1. Pass 1 assigns `get_rows` to CPU (CUDA doesn't support k-quant
   GET_ROWS — Q4_K etc.)
2. Pass 2 "expand gpu down/up" skips CPU-assigned nodes (lines
   1117-1119: `if (*node_backend_id == sched->n_backends - 1) { cur_backend_id = -1; }`)
3. The downstream `rms_norm → mul(h, gpu_weight)` chain gets assigned
   to GPU via pass 2's expand-down from some later GPU-anchored op.
4. But the `get_rows` output (CPU) feeding into `rms_norm` (GPU)
   should trigger a copy node at the split boundary.
5. **The copy IS inserted** (pass 5, line 1382-1417), but it uses
   `ggml_dup_tensor_layout` which creates a tensor with the *same
   layout* as the source. The source is CPU-allocated, so the copy
   tensor's buffer type matches CPU. When `compute_splits` runs the
   GPU split, it calls `ggml_backend_tensor_copy` which dispatches to
   `cudaMemcpyAsync(DeviceToDevice)` — but the source data is
   actually on host memory → **`CUDA error: invalid argument`** or
   **`illegal memory access`**.

The fundamental issue: `ggml_dup_tensor_layout` copies the tensor
metadata but doesn't allocate a buffer on the *destination* backend.
The copy tensor inherits the source's buffer type, so when the
scheduler later allocates it, it may end up on the wrong backend.

## Symptom

Any model with:
- Quantized embedding table (Q4_K) on CPU (split-load)
- Matmul weights on GPU
- A minimal decode graph: `get_rows(embed_cpu) → rms_norm → mul_mat(gpu_weight) → ...`

Fails with one of:
- `ggml_cuda_compute_forward: ADD failed` (CUDA can't add CPU+GPU tensors)
- `ggml_cuda_cpy: invalid argument` (cudaMemcpyDeviceToDevice on host pointer)
- `CUDA error: an illegal memory access was encountered`

Works if the graph is large enough (e.g. prefill with audio+text
branches) because the flood fill has enough GPU-anchored ops to pull
all assignments GPU-ward.

## Affected models

Any model using `core_gguf::load_weights_split` with k-quant embeddings
on CPU and matmul weights on GPU. Currently: mimo-asr (workaround: reuse
prefill graph for decode). Potentially affects any future model with the
same split-load pattern.

## Workaround (in CrispASR)

mimo-asr decode steps reuse the prefill graph (which has enough GPU ops
for the flood fill to work) instead of the minimal T=1 step graph. The
audio branch computes zero (masked out) — negligible overhead. See
commit `ec3ba861`.

## Proposed fix

In `ggml_backend_sched_split_graph` pass 5, when creating a copy tensor
for a cross-backend input, the copy should be allocated on the
*destination* backend's buffer type, not the source's:

```c
// Current (buggy): inherits source layout/buffer
struct ggml_tensor * tensor_copy = ggml_dup_tensor_layout(sched->ctx, src);

// Fix: after creating the copy, ensure it will be allocated on the
// destination backend by setting its buffer type hint
// (this may require adding a buffer_type field to the copy tracking)
```

Alternatively, in `compute_splits` when executing the copy between
splits, use `ggml_backend_tensor_copy` which should handle
cross-backend copies via host staging — but verify it doesn't
short-circuit to `cudaMemcpyDeviceToDevice` when both tensors
happen to have CUDA buffer types.

## Confirmed on

- Tesla P100 (sm_60) — Kaggle GPU kernel v6-v12
- RTX 3090 (sm_86) — RunPod (the workaround was validated here)

## Repro

```bash
# Build CrispASR with CUDA
cmake -B build -DGGML_CUDA=ON && cmake --build build --target crispasr-cli

# Run mimo-asr with GPU + split-load (embed on CPU, matmul on GPU)
# This uses the prefill-graph workaround, so it works:
build/bin/crispasr --backend mimo-asr -m auto --auto-download \
    -f samples/jfk.wav

# To reproduce the raw bug, revert ec3ba861 (prefill-graph workaround)
# and use the T=1 step graph with gpu_embed_split=true
```

## Cross-refs

- CrispASR LEARNINGS.md §"ggml scheduler tightened cross-backend tensor resolution"
- upstream-prs/10 (related: Metal sched buffer reuse drift)
- upstream-prs/11 (related: Metal sched NaN at large T)
- CrispASR commit `ec3ba861` for the workaround
