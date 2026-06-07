**Title:** `CUDA: tighten input-position loop in conv_transpose_1d`

---

## Background

`ggml_cuda_op_conv_transpose_1d` contains an inner loop that iterates
every input position `i ∈ [0, IL)` per output sample and filters with
an `if-continue`:

```c
for (int i = 0; i < src1_ne0; i++) {
    if (!(idx >= i*s0 && idx < i*s0 + src0_ne0)) continue;
    …
}
```

For a given output position `idx`, only positions `i` with
`i*s0 ≤ idx < i*s0 + K` contribute — i.e.
`i ∈ [⌈(idx − K + 1)/s0⌉, ⌊idx/s0⌋] ∩ [0, IL−1]`,
at most `⌈K/s0⌉` values (typically 2 for stride=K/2 transposed convs).

For representative TTS codec-decoder shapes (IL=400, K=16, s0=8) the
naive loop wastes `IL / ⌈K/s0⌉ ≈ 200×` iterations per output thread.
With 800 K output elements, the total kernel work is ~320 M conditional
iterations per dispatch — enough to trip the OS GPU watchdog (TDR /
`kGPUCommandBufferCallbackErrorTimeout`) on both AMD and NVIDIA.  This
is the root cause of the #155 crash reports from AMD RX 7900 XTX and
NVIDIA RTX 5060 Ti with SEANet-family codec decoders.

The identical bug was fixed for Metal in ggml#1477 (merged 2026-05-10).
This PR is the CUDA equivalent.

## Proposed change

Compute `i_min`, `i_max` analytically before the channel and inner
loops, iterate only `[i_min, i_max]`:

```
idx ≥ i*s0         →  i ≤ idx/s0            →  i_max = min(idx/s0, IL−1)
idx < i*s0 + K     →  i ≥ (idx−K+1)/s0 (⌈⌉) →  i_min = max(0, ⌈(idx−K+1)/s0⌉)
```

Integer ceiling for positive `a = idx − K + 1`: `(a + s0 − 1) / s0`.
If `a ≤ 0`, `i_min = 0`.

Move `idx` out of the channel loop (it is loop-invariant); wrap the
channel loop in `if (i_min ≤ i_max)`.

Output is **bit-identical** — same multiplies and adds in the same
order; zero wasted iterations.  Loop bound shrinks from IL to ≤ ⌈K/s0⌉.

Note: this PR is gated on #14 (F16 weight template) since it is applied
on top of that diff.  If the reviewer prefers, both can be squashed into
one commit — they touch overlapping lines in `conv-transpose-1d.cu`.

Patch: `17-cuda-conv-transpose-1d-loop.patch` (1 file, +14/−22 relative
to the #14-patched state).

## Why this and not "dispatch smaller tiles"

Tiling the dispatch (capping the output per kernel call) would also
avoid TDR but at the cost of added dispatch overhead and host-side
looping.  The analytical bound eliminates the wasted work entirely —
no extra launches, no host bookkeeping.

## Verification

Tested locally on Apple M1 Metal (the Metal equivalent is the merged
#1477 fix).  CUDA validation pending — the same derivation applies,
and the patch is a mechanical translation of the Metal fix.

Recommend running `test-backend-ops CONV_TRANSPOSE_1D` before and after
against the CUDA backend on at least two shapes:
- F32 × F32, s0=1 (basic correctness, regression for existing callers)
- F16 × F32, s0=8, K=16, IL=400 (TTS codec shape that triggered TDR)

The F32 path must produce bit-identical output (same multiply order,
same accumulator, no new FP rounding introduced).

## Risk

- **F32 callers**: bit-identical output; analytical bounds select the
  same `i` values the if-continue would have admitted.
- **F16 callers**: no change relative to the #14 patch.
- **Edge cases**: `i_min > i_max` (output position outside any kernel
  span) writes `accumulator = 0`; this is correct for positions beyond
  the valid output range.

## Files changed

- `ggml/src/ggml-cuda/conv-transpose-1d.cu` — +14 / −22

## AI-usage disclosure (for the actual PR body)

Per llama.cpp's CONTRIBUTING.md AI policy: the analytical bound
derivation and the structural kernel restructuring in this patch were
developed with mechanical AI assistance (Claude Code); the derivation
itself, the verification plan, and this PR description were written or
reviewed by the author.
