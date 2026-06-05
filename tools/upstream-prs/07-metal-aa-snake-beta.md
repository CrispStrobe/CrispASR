**Title:** `metal : kernel_aa_snake_beta — fused anti-aliased SnakeBeta for BigVGAN v2`

**Status:** drafted, not implemented yet. Slot reserved alongside 05/06 which
are still in flight on `issue81-phase1-uar-wip`.

---

## What this is

NVIDIA ships a fused CUDA kernel for the BigVGAN v2 anti-aliased SnakeBeta
activation (used by IndexTTS-1.5, VoxCPM2, and increasingly the SOTA neural
vocoders): `indextts/BigVGAN/alias_free_activation/cuda/anti_alias_activation_cuda.cu`
in the upstream IndexTTS repo. One launch does the full
`replicate-pad → 2× upsample FIR → SnakeBeta → 2× downsample FIR` chain,
keeping every intermediate in registers — no HBM round trips between stages.

CrispASR currently expresses the same chain as a `ggml_map_custom1` CPU
custom op (`src/indextts_voc.cpp:aa_snake_beta_op`). On M1 unified memory
that path is fast in absolute terms (≈ 6.6 s vocoder for ≈ 6.7 s of audio)
but it forces ggml-backend-sched to fall back the entire vocoder onto CPU —
the AA op has no Metal kernel, so mixing it with a Metal-resident BigVGAN
graph triggers a Metal → CPU → Metal sync per AMP block (~20 sites per
generate), which on bench measures ≈ 25 % SLOWER than keeping the whole
vocoder on CPU. We work around this in `indextts_voc_init` by auto-falling
the vocoder backend to CPU whenever `use_aa = true` (the default since the
raw / non-AA path emits audible aliasing — see commit `cd21faea`).

Net: today, on Metal, the vocoder runs on CPU only because of this one op.

## Proposed kernel

Port the upstream CUDA kernel to MSL. Same threadgroup layout works:

- One threadgroup per `(channel × seq_chunk × batch)`.
- `threads_per_threadgroup = 128`, each thread owns `BUFFER_SIZE` output
  samples (e.g. 8 — match upstream's choice for the equivalent SM occupancy).
- Filter taps (K = 12), upsample-replicate-pad scratch, downsample-replicate-pad
  scratch, and per-thread output all live in threadgroup memory + registers.
- One read of `src` per threadgroup, one write of `dst`. No intermediate HBM.
- Hyperparameters baked in:
  `FILTER_SIZE = 12`, `UPSAMPLE_REPLICATION_PAD = 5`,
  `DOWNSAMPLE_REPLICATION_PAD_LEFT = 5`, `DOWNSAMPLE_REPLICATION_PAD_RIGHT = 6`,
  alpha/beta in log-scale (consume with `expf`).

This requires a new ggml op:

```c
// In ggml.h:
GGML_OP_AA_SNAKE_BETA,  // BigVGAN v2 anti-aliased SnakeBeta

GGML_API struct ggml_tensor * ggml_aa_snake_beta(
    struct ggml_context * ctx,
    struct ggml_tensor  * x,         // [T, C] F32
    struct ggml_tensor  * log_alpha, // [C]    F32
    struct ggml_tensor  * log_beta,  // [C]    F32
    struct ggml_tensor  * us_filter, // [K]    F32 (Kaiser-windowed sinc)
    struct ggml_tensor  * ds_filter);// [K]    F32
```

CPU forward: `ggml/src/ggml-cpu/ops.cpp` — port the existing
`aa_snake_beta_op` from CrispASR (`src/indextts_voc.cpp:189–321`); already
audited and shipping.

Metal forward: `ggml/src/ggml-metal/ggml-metal.metal` — translated upstream
`anti_alias_activation_forward<float, float, float>` kernel, registered as
`kernel_aa_snake_beta_f32` via the existing template-host-name machinery.

Dispatch: `ggml/src/ggml-metal/ggml-metal-ops.cpp` adds the `case
GGML_OP_AA_SNAKE_BETA` branch; `ggml-metal-device.cpp` returns true from
`supports_op` for F32 inputs with K=12.

CUDA/Vulkan: not implemented in the first PR; ggml-backend-sched falls back
to CPU on those backends until someone ports the upstream CUDA kernel.

## Verification plan

1. CPU forward bit-identity check against the CrispASR
   `aa_snake_beta_op` reference for a few synthetic inputs of varying
   `(T, C)` — same multiplies/adds in the same order, should match exactly.
2. Metal vs CPU forward: rmsdiff target ≤ 1e-5, max|Δ| ≤ 1e-4 (the same
   bound we currently observe between vDSP and scalar CPU paths in
   `src/indextts_voc.cpp`).
3. Round-trip: IndexTTS-1.5 voice clone, ASR-decode with parakeet-tdt-0.6b-v3,
   expect identical transcript on the JFK "quick brown fox" prompt across
   all three backends.
4. Click-detector: `np.diff` should show < 30 inter-sample jumps > 30 % FS
   on a 6.7 s synthesis (today's CPU AA path emits 0–27).

## Expected gain

Step A (auto-CPU when AA is on, May 2026) — vocoder ≈ 7.87 s on M1.

Step B-v2 (native ggml ops, opt-in `INDEXTTS_AA_BACKEND=native`,
May 2026) — vocoder ≈ 7.57 s CPU, ≈ 8.01 s GPU. CPU output is
bit-equivalent to Step A; GPU output drifts into noise floor but ASR
identical. Concat/reshape/scale graph overhead per AA site is what
keeps Metal from winning here — the fused-kernel route below collapses
those into one launch.

Step C-1 (vDSP-vectorised CPU SnakeBeta + downsample, May 2026) — vocoder
≈ 6.75 s avg of 3 runs (≈ 2–3 % over scalar; within noise but consistent
direction, bit-identical-modulo-fp-rounding output, free).

Step C-2 (this PR) — projected vocoder ≈ 1.5–2 s on M1 based on upstream's
CUDA speedup on A100. The big win is collapsing the per-AMP-block
Metal↔CPU round trip; secondary win is that the surrounding BigVGAN convs
can stay on Metal where they belong.

## Code provenance

The CPU forward is CrispASR's existing optimised op (commit `cd21faea`
plus the vDSP cleanups in the follow-up). The Metal kernel will be a
re-derivation of NVIDIA's CUDA kernel (Apache 2.0) — same algorithm, MSL
syntax, threadgroup memory instead of CUDA shared memory. Original CUDA
code referenced: `indextts/BigVGAN/alias_free_activation/cuda/anti_alias_activation_cuda.cu`.

## Notes for the maintainer

This is the first new-op PR from CrispASR — every previous patch (01–04)
modified existing ggml internals. Expect more review surface than a pure
perf patch. Suggest opening as `[RFC]` first to get an op-shape ack, then
filing the implementation. Cross-backend stubs (CUDA / Vulkan / SYCL
graceful-fallback to CPU forward via `ggml_backend_offload_op`) are part
of the PR scope.
