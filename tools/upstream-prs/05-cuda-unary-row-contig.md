**Title:** `ggml-cuda : support row-contiguous (strided) unary ops`

**Status (2026-05-11):** patch applied to our vendored ggml,
**gated behind `GGML_CUDA_CRISPASR_UNARY_ROWS=OFF` (default OFF)**.
Build + bench A/B vs `POST+CUDA_GRAPHS q8_0` baseline `3.063 s`
not yet completed (session paused mid-build at ~117/~600 .obj).
Resume steps: `tools/upstream-prs/RESUME-A1000-phase1.md`.

---

`ggml_cuda_op_unary`'s support gate at `ggml/src/ggml-cuda/ggml-cuda.cu`
returns `ggml_is_contiguous(op->src[0])` (with a maintainer-authored
TODO directly above saying it "should become
`ggml_is_contiguous_rows`"). Strided-row views of contiguous tensors
— specifically the GLU gate half of a `(2*d, T)` matmul output, as
used in FastConformer's convolution module — fail this check and
fall back to CPU.

Effect on parakeet-tdt-0.6b-v3 on RTX A1000 Laptop (sm_86) + Windows
WDDM: the encoder graph builds 24 conformer layers × one `sigmoid`
on a strided view per layer × 15 chunks per long-clip run × 3 runs
= 1 080 CPU dispatches inserted into an otherwise all-CUDA encoder.
Each CPU dispatch inserts two `cudaMemcpyAsync` round-trips and one
`cudaStreamSynchronize` at the `ggml_backend_sched` boundary. Total
overhead: ~280 ms / run = ~9 % of the post-CUDA-Graphs wallclock on
this hardware.

Fix: relax the support gate to `ggml_is_contiguous_rows` (matches the
existing TODO), and add a strided-rows variant of `unary_op_kernel`
that indexes by destination element and decomposes the linear index
into `(i00, i01, i02, i03)`, then loads from `src` via per-dim strides
`nb01 / nb02 / nb03`. The fully-contiguous path stays bit-identical
(the dispatch picks the original linear-index kernel when
`ggml_is_contiguous(src0)` is true).

Patch: `05-cuda-unary-row-contig.patch` (2 files in
`ggml/src/ggml-cuda/`: `unary.cu` and `ggml-cuda.cu`).

**Verification.** Design only as of 2026-05-11 — the patch is applied
to our vendored ggml under a CMake compile gate; A/B bench against
the `3.063 s` POST+CUDA_GRAPHS baseline is the next step before
flipping the default. Once benched:

- `GGML_SCHED_DEBUG=2` should show the 24 conformer-layer
  `GGML_OP_UNARY(sigmoid)` ops move from `[CPU]` back to `[CUDA0]`
  post-patch (today they're listed as `## SPLIT #1: CPU # 4 inputs:`
  at every conformer layer in the dump captured at
  `bench-issue81/sched-debug.log`).
- WER on `samples/jfk.wav` must stay `0.000` (bit-identical
  transcript) — the new strided kernel is mathematically equivalent
  to the contiguous kernel; any divergence is a kernel-correctness
  bug.
- `test-backend-ops` for `sigmoid` / `silu` / `gelu` etc. should
  pass on the existing contiguous shapes (the dispatch picks the
  original kernel when `ggml_is_contiguous(src0)` is true) — add a
  strided-view case to exercise the new path.

Wallclock target: noticeable fraction of the ~10 % wallclock budget
attributable to the 24 CPU-UNARY splits per chunk. Conservative
estimate: 5-10 % improvement (3.063 → ~2.8 s long-clip mean).
Expected vs. measured will be recorded in PERFORMANCE.md "A1000
Ampere CUDA A/B" once the bench runs.

**Why this is the right scope.** The kernel rewrite is small (~30
LOC, one helper template + one dispatch fork) and additive — the
contiguous fast path is preserved and bit-identical to mainline.
The strided path uses standard `(nb01, nb02, nb03)` stride math
already used by many other ggml-cuda ops, so it's not introducing
a new pattern.

**Why the maintainer TODO is correct.** Most other ggml-cuda ops
(MUL, ADD, NORM, SCALE, …) already accept non-contiguous sources
because their kernels iterate by row using `nb01`. Only the unary
ops kept the strict `is_contiguous` gate because their kernel uses
flat linear indexing. This patch brings unary up to parity.
