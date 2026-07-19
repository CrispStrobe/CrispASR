**Title:** `ggml : fix ggml_conv_1d output layout for batch N > 1`

---

`ggml_conv_1d` builds its result as

```c
result = ggml_mul_mat(ctx,
        ggml_reshape_2d(ctx, im2col, im2col->ne[0], im2col->ne[2]*im2col->ne[1]), // [N*OL, IC*K]
        ggml_reshape_2d(ctx, a, a->ne[0]*a->ne[1], a->ne[2]));                    // [OC, IC*K]

result = ggml_reshape_3d(ctx, result, im2col->ne[1], a->ne[2], im2col->ne[2]);    // claims [OL, OC, N]
```

The im2col is the **first** mul_mat argument, so the result's ne is
`[N*OL, OC]` — flat index `oc*(N*OL) + n*OL + ol`, i.e. **OC is the slowest
axis**. The final `ggml_reshape_3d` instead declares `[OL, OC, N]`, which means
flat `n*(OL*OC) + oc*OL + ol` — **N slowest**.

Those two expressions are identical when `N == 1` and differ otherwise. So the
returned tensor's declared shape contradicts its own contents for any batched
1-D convolution: a consumer indexing by `ne` reads transposed data.

**Repro** (`24-conv-1d-batch-reshape.repro.cpp`, standalone, CPU backend):
T=12, IC=3, OC=5, K=3, s=1, p=1, compared against a hand-rolled direct
convolution.

```
N=1  y.ne=(12,5,1)  cos=1.00000000  max_abs=4.8e-07  OK
N=2  y.ne=(12,5,2)  cos=0.41129104  max_abs=5.8e+00  MISMATCH
N=3  y.ne=(12,5,3)  cos=0.05935857  max_abs=7.1e+00  MISMATCH
```

**Fix.** Reshape to the true layout `[OL, N, OC]`, then permute to the declared
`[OL, OC, N]`. `N == 1` keeps the original zero-copy single-reshape path, so
every existing caller is bit-identical:

```c
if (im2col->ne[2] == 1) {
    result = ggml_reshape_3d(ctx, result, im2col->ne[1], a->ne[2], im2col->ne[2]);
} else {
    result = ggml_reshape_3d(ctx, result, im2col->ne[1], im2col->ne[2], a->ne[2]); // [OL, N, OC]
    result = ggml_cont(ctx, ggml_permute(ctx, result, 0, 2, 1, 3));                // [OL, OC, N]
}
```

After the fix all three cases report `cos=1.0`, `max_abs=4.8e-07`.

**Why this survived.** `tests/test-backend-ops.cpp` has **no** `conv_1d` cases at
all — it tests `IM2COL` and `MUL_MAT` as ops, but `ggml_conv_1d` is a composite
graph builder, so the reshape between them is never exercised. And inference
batch is 1 essentially everywhere (see the note in PR 23: batch is a poor
parallelization axis for inference), so no shipping caller hit it.

**Scope / risk.** The `N == 1` branch is the unmodified original statement, so
callers that pass batch 1 cannot change behaviour. The only code whose result
changes is code passing `N > 1`, which was reading a tensor whose `ne` lied. If
any such caller exists and had hand-compensated for the transpose, this would
break it — worth an audit before merge, though it is more likely that none
exist, which is why the bug went unnoticed.

**Companion in the same PR: `ggml_conv_1d_dw` batch support.** Different failure
mode, worth stating precisely. It reshapes `b [T,C,N]` to `[T,1,C,N]` and feeds
the 1-D im2col path, whose `GGML_ASSERT(b->ne[3] == 1)` then fires for any
`N > 1` — so batched depthwise conv **aborts** rather than miscomputing. (Its
trailing `ggml_reshape_3d(..., result->ne[2], 1)` also hardcodes `1` into
`ne[2]`.) That is a missing capability, not wrong output. Fix folds the batch
into the channel axis (`b -> [T, 1, C*N, 1]`, `cn = n*C + ch`) and tiles the
per-channel kernel with `ggml_repeat`, whose semantics give element `cn` the
kernel `cn mod C = ch`. The `[OL, 1, C*N]` result is bit-for-bit the
`[OL, C, N]` layout, so the final reshape is free. Verified at N = 1..4,
cos = 1.0, max_abs = 0.0 exactly (`24-conv-1d-batch-reshape.dw-repro.cpp`).

**Suggested companion:** add `conv_1d` and `conv_1d_dw` cases (including
`N > 1`) to `test-backend-ops` so both composites are covered.

Found while porting CREPE (a pitch model that batches frames for GPU
throughput) — the first ggml consumer we have that genuinely wants `N > 1`.
