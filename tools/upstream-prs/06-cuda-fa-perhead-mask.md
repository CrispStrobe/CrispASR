**Title:** `ggml-cuda : support per-head additive mask in FLASH_ATTN_EXT (MMA-F16 path)`

**Status:** design + minimal patch sketched (this doc), not yet
implemented in code, not yet bench-validated. Originally drafted
on the retired `issue81-phase1-uar-wip` branch; rewritten on main
2026-05-23 against current ggml master after closer reading of
the MMA-F16 kernel and discovering the launcher already plumbs
the needed strides — only the kernel body needs to consume them.

Tracking as the next concrete A1000 perf step after the fused
`norm_affine` + `siglu` win landed in `d758fe69` (which closed
target (b) of the issue #81 gap analysis). This PR targets the
**other** dominant CPU-fallback cost — target (a) of the same
analysis — which is `FLASH_ATTN_EXT` falling back to CPU on
per-head additive masks.

---

## The current rejection

`ggml/src/ggml-cuda/fattn.cu:423` unconditionally rejects FA
invocations whose mask is per-head:

```cuda
if (mask && mask->ne[2] != 1) {
    return BEST_FATTN_KERNEL_NONE;
}
```

This rules out **all** FA kernels for transformer-XL /
FastConformer style relative-position-bias attention where the
additive mask is per-head (`(T_kv, T_q, n_heads, 1)`). Conformer
/ Parakeet / Canary / FastConformer-CTC all use this
construction; on CUDA they all fall back to CPU, producing **24
CPU splits per chunk** (one per layer × `n_layers=24`) plus the
H↔D round-trip per split.

After `d758fe69` reduced CPU splits 144 → 72 on parakeet-tdt-0.6b-v3,
the remaining 72 are all from this guard. PERFORMANCE.md's gap
analysis attributes ~15-25 % wallclock to closing this.

## The clean fix is small (much smaller than the previous estimate)

Closer code reading shows the kernel signature **already** takes
the needed strides. Look at the MMA-F16 path:

- `ggml/src/ggml-cuda/fattn-mma-f16.cuh:1556` — the top-level
  `flash_attn_ext_f16` kernel signature:
  ```cuda
  const int32_t ne31, const int32_t ne32, const int32_t ne33,
  const int32_t nb31, const int32_t nb32, const int64_t nb33);
  ```
  All six mask dims/strides are already parameters.

- `ggml/src/ggml-cuda/fattn-mma-f16.cuh:1603` — uses `nb31` as
  the per-row stride:
  ```cuda
  const int stride_mask = nb31 / sizeof(half);
  ```

- `ggml/src/ggml-cuda/fattn-mma-f16.cuh:1635` and `:1681` — the
  per-iteration `mask_h` advance currently **only** offsets by
  sequence (dim 3), not by head (dim 2):
  ```cuda
  const half * mask_h = ncols2 == 1 && !mask ? nullptr :
      (const half *) (mask + nb33*(sequence % ne33));
  ```

The kernel **never reads `nb32` for the per-head offset.** That's
the entire bug. Compiler probably warns about an unused param;
the line at 1707 explicitly unuses it in the dead-code branch.

### Minimal patch for the easy case (`ncols2 == 1`)

When `ncols2 == 1`, the kernel processes exactly one Q head per
tile. The head index per tile is `zt_Q = z_KV * gqa_ratio + zt_gqa * ncols2`
(line 1630/1676). For per-head masks where `mask->ne[2] == n_heads`,
we add an offset by `nb32 * (zt_Q % ne32)`:

```diff
- const half * mask_h = ncols2 == 1 && !mask ? nullptr :
-     (const half *) (mask + nb33*(sequence % ne33));
+ const half * mask_h = ncols2 == 1 && !mask ? nullptr :
+     (const half *) (mask
+                     + nb33*(sequence % ne33)
+                     + nb32*(zt_Q % ne32));
```

Apply at both `:1635` and `:1681` (same expression, two locations
because of the kbc loop structure with main + tail iteration).

For `ne32 == 1` (the existing broadcast-mask case), `zt_Q % 1 == 0`
and the offset is zero — bit-identical behavior for the common
case. For `ne32 == n_heads`, each tile reads its own per-head
mask slice — the new behavior.

Total kernel-side change: **2 lines × 2 sites = ~4 LOC**.

### Relaxing the gate

`fattn.cu:423-425` needs to remain selective: only allow per-head
masks for the MMA-F16 path until the other kernels (VEC, TILE,
WMMA-F16) are similarly patched. Cleanest is a flag computed
once:

```diff
+ const bool mask_is_per_head = (mask && mask->ne[2] != 1);
- if (mask && mask->ne[2] != 1) {
-     return BEST_FATTN_KERNEL_NONE;
- }
```

then at each `return BEST_FATTN_KERNEL_{VEC,TILE,WMMA_F16};` add:

```diff
+ if (mask_is_per_head) {
+     // these kernels don't yet support per-head masks
+     break; // fall through to MMA-F16 dispatch
+ }
  return BEST_FATTN_KERNEL_VEC;
```

(Same pattern at each VEC/TILE/WMMA return site — there are ~5-7
of them in `ggml_cuda_get_best_fattn_kernel`.)

When the dispatch falls through, it will hit the MMA-F16 path
which is the patched one. We need to make sure MMA-F16 is reachable
on the relevant arches (sm_75+, AMD MFMA+) — which it is for any
post-Turing NVIDIA GPU (the A1000 Laptop is sm_86 so this
trivially applies).

### Other kernels deferred to future PRs

VEC, TILE, WMMA-F16: same kernel-body change pattern, but each
has its own per-head loop / tile structure. Defer to a follow-up
PR rather than bundling everything here. Per-arch breakdown:

- **Turing+ Ampere+** (sm_75+): MMA-F16 is the preferred kernel
  for everything except `Q->ne[1] == 1` decode-style queries —
  almost all conformer-style use cases hit MMA-F16. **Win for
  parakeet / canary / FastConformer-CTC on these arches.**
- **Volta** (sm_70): MMA-F16 is also reachable. Win.
- **Pascal / Maxwell / Kepler**: WMMA-F16 path or TILE. Falls
  through to CPU until those kernels are patched too — **no
  regression**, just no improvement for those arches.

### Hard case: `ncols2 > 1` (GQA folding)

When the kernel folds multiple Q heads into one tile (`ncols2 > 1`,
common for GQA models), each tile processes Q heads `zt_Q` through
`zt_Q + ncols2 - 1`. For per-head masks, each Q head in the tile
needs to read different mask rows. That's a much bigger change
inside `flash_attn_ext_f16_process_tile`:

- The mask load (`tile_mask` setup at lines 401-475) currently
  reads one row from `mask_h` and broadcasts across the heads in
  the tile.
- Per-head needs: read `ncols2` different mask rows, one per Q
  head in the tile.

**Defer this to a follow-up PR.** The current parakeet/canary case
is `gqa_ratio == 1` and `ncols2 == 1` (full Q heads = K/V heads,
no GQA folding), so the easy patch covers our target workload.

When this matters: GQA-conformer models if anyone ever builds
them (none in the current CrispASR catalog). The kernel still
falls back to CPU for those — same as today, no regression.

## Implementation scope

### Files touched (minimal patch)

| file | LOC | what |
|---|---:|---|
| `ggml/src/ggml-cuda/fattn.cu` | ~10 | gate relaxation: `mask_is_per_head` + per-return guards on VEC/TILE/WMMA |
| `ggml/src/ggml-cuda/fattn-mma-f16.cuh` | ~4 | mask_h offset at lines 1635 + 1681 |
| `tests/test-backend-ops.cpp` | ~30 | new FA per-head mask test case |

**Total: ~45 LOC.** Smaller than the previous estimate of 300-500
LOC because the strides are already plumbed through the launcher.

### Build gate

Same pattern as our other in-tree ggml-cuda patches: gate behind
a CMake flag default-OFF so upstream builds stay bit-identical:

```cmake
# ggml/CMakeLists.txt
option(GGML_CUDA_CRISPASR_FA_PERHEAD_MASK
       "CrispASR: allow per-head additive mask in CUDA FlashAttention (MMA-F16 path)"
       OFF)
```

Gate the four-line kernel-body change and the gate-relaxation in
`fattn.cu` behind `#ifdef GGML_CUDA_CRISPASR_FA_PERHEAD_MASK`.

When our internal builds want it on (after `test-backend-ops`
verification), flip the default ON in `release.yml`'s Windows-CUDA
matrix slot — same approach as `GGML_CUDA_GRAPHS=ON`.

### Validation

1. **`test-backend-ops` per-head mask case.** Mask shape
   `(T_kv, T_q, n_heads, 1)` with `n_heads ∈ {1, 2, 4, 8, 16}`,
   verified against the CPU reference. **Must pass on sm_75
   (Turing T4), sm_86 (Ampere A1000), sm_89 (Ada Lovelace, if
   available)** before merging.
2. **Parakeet WER regression check.** Run the standard JFK +
   60 s tiled clip; the transcript must be bit-identical to the
   current (CPU-fallback) path. Any drift → kernel has a bug.
3. **A1000 wallclock measurement.** Apply the WDDM-warmup
   protocol (see LEARNINGS.md "WDDM idle-clock-state hysteresis
   on consumer/laptop NVIDIA SKUs"); measure long-clip mean
   before/after. Expected: ~10-15 % wallclock improvement on top
   of the current `dll-postsiglu` baseline (3.03 s with warmup →
   target ~2.6 s, RTx ~24×).

### What we already learned (sources)

- `tools/upstream-prs/06-cuda-fa-perhead-mask.md` (this file's
  prior incarnation on the retired `issue81-phase1-uar-wip`
  branch) had the call-chain analysis but estimated ~100-200 LOC
  because it predated the closer read of the kernel signature.
- `PERFORMANCE.md` "Phase 0 / Phase 1 — root-causing the
  remaining 1.99× gap" identified this as target (a), with
  expected ~15-25 % wallclock impact.
- `LEARNINGS.md` "WDDM idle-clock-state hysteresis" — important
  for measuring the patch's actual impact (single-shot cold
  benches won't reveal it).

## Reference: prior failed approaches (do not repeat)

1. **`op_offload=true` in `ggml_backend_sched_new`** — +87 %
   regression (re-uploads weights per call).
2. **Folding BD into Q before FA** — semantic change to the model,
   would need re-training to validate.
3. **WIP branch `issue81-phase1-uar-wip`** — proposed loosening
   the *UNARY* contiguity gate. Superseded by `d758fe69`
   (different fix for the UNARY problem). The FA per-head mask
   work was design-only on that branch and is now restarted on
   `main` with the corrected kernel-level analysis.

The real fix has to land **inside the FA kernel**, not in the
client (model code) or in the dispatch (scheduler). The above
client-side workarounds perturb the graph in ways CUDA Graphs
and sched's allocator weren't tuned for.

## Effort estimate

- Phase 1 (this PR, MMA-F16 easy case + gate): **~1-2 days**
  including `test-backend-ops` and parakeet WER validation
- Phase 2 (VEC / TILE / WMMA-F16 patches): ~1-2 days each, only
  if a user reports needing those kernels for per-head-mask
  attention
- Phase 3 (`ncols2 > 1` GQA-folded case): ~3-4 days when a
  GQA-conformer model lands; not needed for current catalog

The Phase 1 work is the right next concrete A1000 perf step
after `d758fe69`.
