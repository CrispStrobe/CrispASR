**Title:** `metal : fix cross-simdgroup reduction in kernel_norm / kernel_rms_norm / kernel_l2_norm`

---

`kernel_norm_fuse_impl`, `kernel_rms_norm_fuse_impl`, and
`kernel_l2_norm_impl` all use the same two-step reduction pattern to
sum across multiple simdgroups within a threadgroup:

```metal
if (tiisg == 0) {
    shmem_f32[sgitg] = sumf;      // each sg's lane 0 stores its partial
}
threadgroup_barrier(mem_flags::mem_threadgroup);

sumf = shmem_f32[tiisg];          // every lane reads shmem[lane_id]
sumf = simd_sum(sumf);            // simd_sum across the simdgroup
```

For configurations where the host-side `nth` doubles past `ne00_t`
and the last active simdgroup ends up with only 1-4 lanes that
entered the prior parallel-sum loop body (i.e. ne00_t = nth/2 + k
for small k), the cross-simdgroup `simd_sum(shmem[tiisg])` produces
incorrect totals on Apple Silicon. The per-row mean/variance comes
out wrong, leaking into the normalized output. CPU
`ggml_norm` / `rms_norm` / `l2_norm` are unaffected.

Empirically the affected `ne00_t` values (with C=512) are:

| ne00_t | nth | last sg active lanes | max_abs vs CPU |
| - | - | - | - |
| 33 | 64 | 1 | 0.273 |
| 65 | 128 | 1 | 1.20 |
| 66 | 128 | 2 | 0.128 |
| 97 | 128 | 1 | 1.66 |
| 129 | 256 | 1 | 2.93 |
| 130 | 256 | 2 | 0.93 |
| 131 | 256 | 3 | 0.45 |
| 132 | 256 | 4 | 0.088 |
| 257 | 512 | 1 | 3.95 |

All other ne00_t values verified bit-identical. The error scales
roughly with how few lanes contributed to the last simdgroup —
1-active-lane is the worst case.

The downstream impact in CrispASR was kokoro / StyleTTS2 AdaIN1d
producing garbage audio for short utterances ("hello world": 39000
samples, T_frames=65 in the F0Ntrain shared LSTM output) on Apple
Silicon Metal. CPU and long-input GPU work correctly. Bisect
captured in `tests/test_metal_norm_repro.cpp`.

## Fix

Replace the cross-simdgroup `simd_sum(shmem[tiisg])` with a serial
reduction by thread 0 of simdgroup 0. shmem[31] is unused by the
original pattern (the init zeroes it; no sg writes to it for nth
≤ 1024), so reuse it as a broadcast slot:

```metal
if (sgitg == 0 && tiisg == 0) {
    const uint n_sg = (ntg.x + 31) / 32;
    float total = 0.0f;
    for (uint sg = 0; sg < n_sg; sg++) {
        total += shmem_f32[sg];
    }
    shmem_f32[31] = total;
}
threadgroup_barrier(mem_flags::mem_threadgroup);
sumf = shmem_f32[31];
```

Applied at three sites:
- `kernel_norm_fuse_impl` mean reduction
- `kernel_norm_fuse_impl` variance reduction
- `kernel_rms_norm_fuse_impl` mean-square reduction
- `kernel_l2_norm_impl` squared-sum reduction

Patch also adds two small per-T helpers
(`crispasr_vec_sum`, `crispasr_vec_sqsum`) to replace the
`dot(scalar, scalar)` calls — `dot()` is only spec-defined for
vector types and the scalar instantiation should not rely on it.
This is independent of the reduction bug (didn't fix it on its own)
but worth folding in for spec-conformance.

Patch: `08-metal-norm-cross-simdgroup.patch` (1 file, +60/-12).

## Verification

`tests/test_metal_norm_repro.cpp` (standalone) sweeps T ∈ {32, 33,
64, 65, 66, 67, 97, 128-132, 256, 257, 320} with C=512 and asserts
`max_abs(CPU - Metal) == 0` per row. All pass with the patch; all
of the bug-pattern values diverge to max_abs > 0.08 without it.

End-to-end: kokoro short input ("hello world") on Apple-Silicon
Metal — parakeet roundtrip transcribes "Hello world!" (matches CPU)
post-patch; produces unintelligible "Mm-hmm." pre-patch. Long input
unchanged. Per-stage diff goes from 6 pass / 17 fail / 0 skip to
20 pass / 3 fail / 14 skip (the 3 remaining fails are Q8 quant
noise in BERT pooler/projector + duration LSTM, also present in
the CPU baseline).

## Why now

The pattern only manifests for backends where ggml_norm /
rms_norm / l2_norm is called with `ne00` in the affected range.
For typical transformer hidden-size-aligned LLMs (`ne00` = 512,
768, 1024, …) every value is divisible by 4 so the float4 kernel
variant runs with `ne00_t = ne00/4` — those happen to land on
"good" `ne00_t` values where the last simdgroup has many active
lanes. The scalar kernel variant (`kernel_norm_f32` with
`ne00 % 4 != 0`) is the path that exhibits the bug, and it's rare
in mainstream LLM workloads — but very common in audio backends
where the normalised axis is a per-frame length that varies with
input duration.

## Repro snippet

```cpp
ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 65, 512);
ggml_tensor * y = ggml_norm(ctx, x, 1e-5f);
// fill x with the same random values for CPU and Metal runs,
// then ggml_backend_tensor_get y and compare element-wise.
// Pre-patch: max_abs ≈ 1.2 (specifically the last element of
// each row is wildly off, ~75% of element error concentrated
// at y[64]). Post-patch: max_abs = 0.
```
