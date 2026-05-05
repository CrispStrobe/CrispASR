**Title:** `ggml-cuda : tile cpy_scalar_transpose along grid_y`

---

`ggml_cpy_scalar_cuda`'s transposed branch asserts `grid_y < USHRT_MAX`.
With `CUDA_CPY_TILE_DIM_2D = 32` that caps `ne00n` at < 2,096,640.
Audio codec / vocoder graphs that emit `[T_pcm, 1, 1, 1]` past 2.1M
(e.g. 1500-frame talker × 1920 upsample = 2.88M PCM samples) abort
on `SIGABRT`.

Fix: tile the launch along grid_y. Add an `int y_block_offset`
parameter to `cpy_scalar_transpose`; the host loops in chunks of
`USHRT_MAX - 1` and the kernel splices `blockIdx.y` back onto the
offset. Single launch (and bit-identical) for `grid_y_total <
USHRT_MAX - 1`.

Multi-launch rather than in-kernel-stride because the kernel uses
`__shared__` tile buffers with a `cur_tile_buf` toggle that depends on
the launch's z-axis sweep — folding the chunk loop inside would require
careful interaction with shared-memory state.

`(int)y_off` cast guarded by `GGML_ASSERT(grid_y_total <= INT_MAX)`;
the kernel's existing int-typed `y` / `tx` indexing would overflow
first.

Patch: `03-cuda-cpy.patch` (1 file, +24/-16).

**Verification.** Tested on Jetson Orin AGX (sm_87, CUDA 12.8) via
downstream consumer (crispasr with qwen3-tts codec at T_pcm = 2.88M). `_q*_cuda`
overloads not affected — they use 1D `num_blocks < UINT_MAX` grids.
Recommend running existing `test-backend-ops` cpy cases (single chunk,
unchanged path).
