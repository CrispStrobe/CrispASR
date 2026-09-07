**Target:** ggml-org/ggml (standing via #1477) — Metal im2col flat dispatch.

**Status (2026-09-07):** Still needed. Re-ported in the CrispStrobe v0.23
consolidation to `kernels/conv.metal` and the current dispatch; repeat the
published M1 A/B suite before filing upstream.


**⚠ PROSE MUST BE HUMAN-AUTHORED.** llama.cpp/ggml's contribution policy
rejects AI-written PR text. Everything below is the fact sheet + patch for a
human to compose the PR from — do not paste it verbatim.

**Draft title:** `metal : flat-dispatch im2col for tiny-threadgroup and
conv_2d_dw shapes`

---

## The defect (two pathological shape classes, one cause)

`ggml_metal_op_im2col` launches `(N, KH, KW)`-thread threadgroups over an
`IC x OH x OW` grid. Two shape classes underfill Apple Silicon badly:

1. **Small `N*KH*KW`**: a 1x1 conv over a batch of 8 runs 8-thread
   threadgroups — a quarter of one simdgroup — across ~344k groups.
2. **The `ggml_conv_2d_dw` lowering** (`IC==1`, `N=C*batch`): each thread's
   `N/ntg0` loop strides a full plane on both src and dst → ~0.4 GB/s on M1.
   By per-op profile this was **70% of the CrispEmbed PP-OCRv6 medium
   recognizer graph** (batch-8 width groups, ~160 ms per ~50 MB depthwise
   materialization).

## The fix (patch: `23-metal-im2col-batch1-occupancy.patch`, fork commit `89a2039d`)

`kernel_im2col_flat`: one thread per dst element from a
`(ceil(OW*CHW/256), OH, N)` grid — dst writes contiguous across the
threadgroup, dw reads 2D-local in one channel plane, index math reduced to
32-bit divmods on the row-local index.

- **Preempt the "why not one flat index" review question**: a first cut
  decomposed a fully flat index with int64 divmods — **Apple GPUs emulate
  int64 division** and that cost more than the broken kernel. Grid-borne
  indices + row-local 32-bit divmods only.
- Selection is a shared predicate (`ggml_metal_im2col_use_flat`) used by both
  the pipeline getter and the encoder. **For upstream: drop the fork's
  `CRISPASR_METAL_IM2COL_FLAT` env opt-out** (3 references in the patch) and
  keep the pure shape predicate — auto-select, no env var.
- Bit-exact by construction: im2col is a gather-copy; each dst element is
  written exactly once with the same value (no FP reorder).

## Measured (Apple M1, interleaved pairs, outputs byte-identical every arm)

| workload | before | after | speedup |
|---|--:|--:|--:|
| PP-OCRv6 medium recognize (38-crop page) | 13.4-14.0 s | 5.8-6.2 s | **2.3x** |
| — its fused batch-8 width-group graphs | 2.6-3.1 s | 0.7-1.2 s | — |
| layout_detect (RT-DETR, N==1 convs) | 1.56-1.62 s | 0.95-0.99 s | **1.6x** |
| melotts HiFi-GAN decode (CrispASR, pin `89a2039d`) | 2.02 s | 1.10 s | **1.85x** |
| paraformer im2col nodes | 8.5-9.1 ms/node | 0.4-1.0 ms/node | 9-20x (wall-invisible) |

moonshine transcript byte-identical, RTF neutral-to-slightly-better; melotts
TTS→ASR round-trip clean. Recommend upstream runs `test-backend-ops` `IM2COL`
cases under Metal (incl. `N>1`) pre-merge.

## History / provenance

Successor to the pre-v0.17 `CRISPASR_METAL_IM2COL_OCC` OW-blocked variant
(this file's earlier draft; the v0.17 sync dropped it — see
`SYNC-v0.17-CONFLICTS.md`). The flat kernel covers that draft's batch-1 case
via the `N*KH*KW < 128` predicate arm AND the conv_2d_dw class the old draft
missed. CUDA is unaffected — `im2col_cuda` already parallelizes threads over
`IC*KH*KW`, not `N`.
