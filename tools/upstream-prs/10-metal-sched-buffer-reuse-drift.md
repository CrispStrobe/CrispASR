**Title:** `metal/ggml-alloc : long F32 GPU graphs accumulate drift sensitive to in-place buffer reuse pattern`

---

This is a bug report rather than a patch. It documents a precision
issue we hit running a 14-block diffusion UNet (the chatterbox-tts
S3Gen conditional flow-matching decoder) on Apple Silicon Metal, and
the bisect that points at `ggml-alloc`'s in-place buffer reuse rather
than any single op kernel. Filing because the root-cause investigation
needs ggml-internals knowledge beyond what an application can do.

## Repro shape

Pipeline: 1 down block + 12 mid blocks + 1 up block + final
projection. Each block: `causal_resnet_block` (CausalConv1d + LayerNorm
+ Mish + time-MLP add + CausalConv1d) + 4× BasicTransformerBlock
(LayerNorm + Q/K/V mul_mat + flash_attn_ext + output mul_mat + add +
LayerNorm + FFN up mul_mat + GELU + FFN down mul_mat + add). Weights
mostly Q8_0 with some F16. Activations all F32. Graph: ~396 mul_mats
per pass; run 10 times in a CFM Euler ODE solver.

Reference output dumped from a known-good CPU pass.

## Observed drift

CPU path (whole UNet computed on `ggml_backend_cpu`):
`cos_min = 1.000000`, max_abs vs ref = 0.

Pure GPU path (whole UNet on Metal):
`cos_min = 0.940`, max_abs vs ref = 14.4.

The GPU output is structurally similar (cos_mean = 0.976) but with
elements deviating by up to 14× the activation scale at the worst
positions. Downstream this produces unintelligible synthesised audio.

## Bisect findings — *not* a kernel-precision bug

We chased this through several rounds. Each is independent evidence
that no single op kernel is the source:

1. **Bit-match `mul_mat`.** We added a `GGML_PREC_F32` dispatch for
   Q8_0 × F32 that pre-quantises input to Q8_0 and runs an integer-dot
   kernel matching `ggml_vec_dot_q8_0_q8_0_generic` bit-for-bit (PR
   09). Confirmed bit-identical to CPU mul_mat output. With this
   kernel firing on all 350 prec-tagged UNet mul_mats, `cos_min` moves
   from 0.940 to **0.947** — essentially no change.

2. **Per-op pin bisect.** With `ggml_backend_sched_set_tensor_backend`
   used to pin a single op type to the CPU backend (and the rest of
   the graph on GPU), we measured `cos_min` per op type pinned:

   | Pin to CPU | cos_min | Frequency in UNet |
   | - | - | - |
   | `mul_mat` | 1.000 | high (~7/block) |
   | `norm`, `mul`, `add`, `flash_attn_ext`, `gelu` | 1.000 | high (≥1/block) |
   | `reshape`, `cont`, `concat`, `permute` | 1.000 | high (memory ops) |
   | `conv_1d`, `soft_max`, `mish`, `silu`, `scale` | 0.940 | low (≤1/block) |

   The clean correlation is **op frequency**, not op identity. Pinning
   any op that occurs frequently restores parity; pinning a sparse op
   doesn't. This points at sync-barrier density, not op correctness.

3. **`ggml_set_output` bisect.** `ggml_set_output` marks a tensor as a
   graph output, which makes `ggml_gallocr` skip the in-place buffer
   reuse path for it (see `ggml-alloc.c` around line 644). We tested:

   | What's marked output | cos_min |
   | - | - |
   | nothing extra (default) | 0.940 |
   | the first block's resnet output only | **0.879** (worse!) |
   | all 62 sub-block outputs | **1.000** |

   The single-checkpoint result is the smoking gun: a `set_output` on
   one specific tensor changes the allocator's reuse decisions
   downstream, and the new pattern produces *more* drift, not less.

4. **`GGML_NO_INPLACE=1` global knob** (added experimentally to
   `ggml_gallocr_allocate_node` to skip the in-place reuse path).
   Result: `cos_min = -0.97` (sign-flipped garbage), worse than
   baseline. Adding `GGML_METAL_CONCURRENCY_DISABLE=1` on top did
   nothing. So a clean "no in-place reuse" knob doesn't work as a
   drop-in fix — some downstream code expects in-place semantics in
   ways we couldn't trace within the bisect session.

5. **`kernel_norm` audit.** `kernel_norm_fuse_impl` uses `float sumf`
   accumulators end-to-end, F32 `simd_shuffle_xor` reduction, F32
   shmem. No silent downcast. Already has prior CrispASR patches for
   cross-simdgroup reduction (separate issue) and serial-reduction
   workaround. Not the source.

6. **`kernel_flash_attn_ext` audit.** The F32 K/V family
   (`FA_TYPES_F32`) keeps Q as `half` in shared memory (line 6430:
   `sq4[j*DK4 + i] = (q4_t) q4[i]` where `q4_t = half4`). Tried
   patching `FA_TYPES_F32`'s Q triple to
   `float, float4, simdgroup_float8x8` — `cos_min` went from 0.940
   to **0.860** (worse). So the F16 Q downcast IS happening, but
   "fixing" it changes the numerical ordering in a way that doesn't
   bit-match CPU either.

## What this means

The chatterbox UNet output on Metal GPU drifts in a path-dependent
way that depends on the `ggml_gallocr` buffer reuse pattern. No
single op kernel is "the" bug. Six independent fix attempts
(bit-match mul_mat, all 9 of the working PIN bisect ops to CPU,
set_output on 1 vs 14 vs 62 tensors, GGML_NO_INPLACE,
GGML_METAL_CONCURRENCY_DISABLE, F32 Q in flash_attn) either fail
outright or work in some calling contexts and fail in others.

The cleanest empirical fix — `set_output` on all 62 UNet sub-block
intermediates — restores bit-equivalence in the diff-harness call
context but NaNs in the smoke call context with the *exact same*
chatterbox model, same UNet graph topology, same CFM solver, same
seed. The diff-vs-smoke divergence is invariant to: random seed,
T_mel value, S3-tokenizer involvement, Metal concurrency on/off,
in-place reuse on/off. Something structural in `ggml-alloc`'s state
across multi-graph sched invocations differs between the two call
paths in a way our bisect couldn't isolate.

Our production workaround is to load the UNet's weight tensors on
the CPU backend (via `ggml_backend_sched`'s weight-residency
routing) so the entire UNet executes on CPU. That avoids the issue
entirely — homogeneous backend, no in-place-reuse interaction with
GPU command scheduling.

## Investigation pointers

The bisect points at `ggml-alloc.c` interaction with the Metal
scheduler:

- `ggml-alloc.c:622` `ggml_gallocr_allocate_node` — the in-place
  reuse decision is made per node based on `n_children == 1 &&
  n_views == 0`. The reuse choice depends on traversal order, so
  adding an output marker flips cascading reuse decisions
  downstream. The single-`set_output`-makes-things-WORSE finding
  confirms this is path-sensitive, not monotonic.

- `ggml-metal/ggml-metal-ops.cpp:159`
  `ggml_metal_op_concurrency_check` — the mem-range overlap check
  that adds Metal command-buffer barriers. Looks correct on
  inspection. Disabling concurrency entirely
  (`GGML_METAL_CONCURRENCY_DISABLE=1`) didn't change the drift, so
  this isn't the source either.

A "disable inplace reuse for graphs marked as F32-precision" knob
in `ggml-alloc` would let applications opt into bit-equivalent GPU
output at a memory cost, but we tested a naive global no-inplace
knob and it broke things — apparently some other downstream code
relies on in-place reuse semantics. A real fix would need to know
*which* downstream code depends on in-place and audit those.

## How to reproduce

Standalone repro requires the chatterbox model files (50 MB) and our
diff-harness scaffolding; we can extract a minimal `test-backend-ops`
case if helpful — flag this issue and we'll prepare one.
