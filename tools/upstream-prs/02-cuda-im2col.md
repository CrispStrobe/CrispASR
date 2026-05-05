**Title:** `ggml-cuda : handle OW > 65535 in im2col`

---

`im2col_cuda` dispatches with `block_nums.y = OW`. CUDA caps grid Y at
65535. Conv1d encoders on raw 16 kHz audio with T > 65535 (≈ 4 s) trip
the limit — e.g. SEANet at 11 s lands at OW = 176000 — and the launch
returns `invalid configuration argument`.

Fix: clamp `block_nums.y` to `MIN(OW, MAX_GRIDDIM_Y)` and loop inside
the kernel with stride `MAX_GRIDDIM_Y`. Same in-kernel stride pattern
already used for the z axis in this kernel. Bit-identical for OW ≤
65535 (single iteration of the new outer loop).

Patch: `02-cuda-im2col.patch` (1 file, +20/-12). **At PR time, mirror
the same fix onto `im2col_3d_kernel` + `im2col_3d_cuda` dispatch
(added upstream since v0.10.0; same bug class — `OW` used as grid Y
unbounded).** See MASTER-AUDIT.md.

**Verification.** Tested on T4 / Jetson Orin via downstream consumer
(SEANet encoder at 11s/16kHz, codec graphs reaching T_out ≈ 176000).
Existing `test-backend-ops` im2col cases unchanged.
