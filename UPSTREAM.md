# Upstream issues / patches we depend on

This file tracks fixes and features that this fork would benefit from
upstream (crispasr / ggml / NeMo / etc.). Each entry has the issue,
the impact on this fork, and the workaround we currently apply.

## crispasr — `examples/ffmpeg-transcode.cpp` mp4-container handling

**Status:** ⏳ pending upstream

**Issue.** When this fork is built with `-DCRISPASR_FFMPEG=ON`, all five CLIs
(`cohere-main`, `parakeet-main`, `canary-main`, `cohere-align`, `nfa-align`)
inherit `read_audio_data()`'s ffmpeg fallback path. That path correctly
decodes bare-codec files like `.opus` (verified, perfect transcript on
`samples/jfk.wav` transcoded to `.opus`), but it has known bugs on
mp4-family container formats:

- `.m4a` (AAC in mp4): crashes with `munmap_chunk(): invalid pointer` on
  the first audio chunk read
- `.webm` (Opus in WebM): hangs indefinitely after the libavformat headers
  are parsed

Both use the same `examples/ffmpeg-transcode.cpp` code path that loops
over `av_read_frame` + `avcodec_send_packet` + `avcodec_receive_frame`
and writes the resulting PCM into a memory buffer. The bug appears to be
in how that buffer is grown / freed for streams whose audio packets are
interleaved with other tracks (which is the mp4 family but not bare
opus / mp3 / flac).

**Impact on this fork.** The audio-formats section of the main README has
to recommend pre-conversion via `ffmpeg -i in.X -ar 16000 -ac 1 -c:a
pcm_s16le out.wav` for `.m4a` / `.mp4` / `.webm` / `.mov` even when the
`CRISPASR_FFMPEG=ON` build is used. The in-process path is only safe for
bare codecs.

**Workaround we apply.** Document the limitation in the README's
"Measured results" table and tell users to pre-convert. The
`CRISPASR_FFMPEG=ON` build is positioned as "in-process Opus support",
not as a complete substitute for pre-conversion.

**What an upstream fix would look like.** A patch to
`examples/ffmpeg-transcode.cpp` that:

1. Picks the correct stream index (`av_find_best_stream` for
   `AVMEDIA_TYPE_AUDIO`) instead of assuming stream 0
2. Properly resamples + grows the output buffer using `av_realloc` (the
   current code does an unchecked allocation that overflows on
   variable-bitrate AAC packets)
3. Handles the EOF / drain frames cleanly to avoid the `munmap_chunk`
   double-free signature

This needs an MR to ggml-org/crispasr. Once merged, this fork will
pick it up automatically on the next ggml subtree update.

**Reproduction:**

```bash
cmake -B build-ffmpeg -DCMAKE_BUILD_TYPE=Release -DCRISPASR_FFMPEG=ON
cmake --build build-ffmpeg -j --target parakeet-main

ffmpeg -y -i samples/jfk.wav -c:a aac -b:a 64k /tmp/jfk.m4a
./build-ffmpeg/bin/parakeet-main -m parakeet-q4_k.gguf -f /tmp/jfk.m4a
# → munmap_chunk(): invalid pointer ; aborted

ffmpeg -y -i samples/jfk.wav -c:a libopus -b:a 32k /tmp/jfk-only.webm
./build-ffmpeg/bin/parakeet-main -m parakeet-q4_k.gguf -f /tmp/jfk-only.webm
# → hangs indefinitely
```

vs the working bare-Opus path:

```bash
ffmpeg -y -i samples/jfk.wav -c:a libopus -b:a 32k /tmp/jfk.opus
./build-ffmpeg/bin/parakeet-main -m parakeet-q4_k.gguf -f /tmp/jfk.opus
# → "And so, my fellow Americans, ..."  (perfect transcript)
```

---

## ggml — VNNI Q8_0 dot product on x86 AVX-VNNI / AVX512-VNNI

**Status:** ⏳ design plan only, see [`ggml_plans.md`](ggml_plans.md)

**Issue.** ggml's `vec_dot_q8_0_q8_0` uses AVX2 `pmaddubsw` / `pmaddwd`,
which is ~2× slower than the AVX-VNNI `vpdpbusd` instruction available
on Cascade Lake / Ice Lake / Zen4. ONNX Runtime's MLAS already uses
VNNI for INT8 GEMM, so on x86 servers ONNX INT8 inference is ~5-6 s
faster than ggml Q8_0 on the same model.

**Impact on this fork.** Per the benchmark in `benchmark_cohere.md`,
ggml Q4_K hits ~15-17 s on a 5.4 s clip while ONNX INT4/INT8 hit ~10 s
inference (but with longer cold loads due to the `external_data` files).
A native VNNI Q8_0 dispatch would close that 5 s gap.

**Workaround.** None applied — Q4_K is already fast enough for the
common case, and the gap to ONNX only matters on x86 servers running
quantised CPU inference. Documented in `ggml_plans.md` as a potential
upstream contribution.

**What an upstream fix would look like.** Add `vec_dot_q8_0_q8_0_vnni`
in `ggml/src/ggml-cpu/arch/x86/quants.c`, dispatched via runtime
detection in `ggml-cpu/cpu-feats-x86.cpp`. The Q4_0_8_8 VNNI variant
already exists as a template; the work is mostly mechanical
(remove the unpack step since Q8 weights are already int8).

---

## NeMo Forced Aligner — official ONNX export of the auxiliary CTC model

**Status:** ⏳ wishlist (not blocking)

**Issue.** NVIDIA ships the auxiliary CTC alignment model bundled
inside `canary-1b-v2.nemo`'s tarball as
`timestamps_asr_model_weights.ckpt`. There is no standalone HuggingFace
release of just that aux model, and no ONNX/TensorRT/GGUF export.

**Impact on this fork.** We had to write `convert-canary-ctc-to-gguf.py`
to extract the aux checkpoint from inside the .nemo and convert it. If
NVIDIA shipped a standalone version (or an ONNX export) the conversion
script could be simpler and the dependency on tarball-internal layout
would go away.

**Workaround we apply.** Our converter handles it. Documented in
`hf_readmes/canary-ctc-aligner-GGUF.md` so users know where the model
came from.

---

## ggml — fork-local patches we already carry

These are not wishlist items — they are real CrispASR-local
modifications inside the vendored `ggml/` subtree, marked with
`// CrispASR patch` so a `git subtree pull` won't lose them silently.
Full root-cause / fix-shape per patch is in LEARNINGS.md
"ggml fork patches we carry". Reproducing the four-patch inventory
here so the upstream-PR question stays visible.

| # | File(s) | Symptom upstream | Status |
| - | --- | --- | --- |
| 1 | `ggml-cpu/{vec.cpp, vec.h, ggml-cpu.c, simd-mappings.h}` | `MUL_MAT(F16, F32)` quantises F32→F16 first; activations >65504 saturate to ±Inf and propagate NaN. Issue #38. | Carrying |
| 2 | `ggml-cuda/im2col.cu` | `OW > 65535` aborts CUDA dispatch (e.g. SEANet at 11s × 16kHz → OW=176000); applies to both 2D and 3D im2col kernels. | Carrying, filed upstream as [ggml-org/llama.cpp#22944](https://github.com/ggml-org/llama.cpp/pull/22944) (2026-05-11; ggml#1485 redirected per @CISC) |
| 3 | `ggml-cuda/cpy.cu` | `cpy_scalar_transpose` asserts `grid_y < USHRT_MAX`; qwen3-tts codec hits T_pcm=2.88M on CUDA. GH issue #65. | Carrying |
| 4 | `ggml-metal/ggml-metal.metal` | `kernel_conv_transpose_1d` iterates full IL per output, ~64× wasted work; trips macOS GPU watchdog on long qwen3-tts graphs. | ✅ merged upstream as [PR #1477](https://github.com/ggml-org/ggml/pull/1477) (2026-05-10); drop from local fork on next ggml bump |
| 5 | `ggml.c` (`ggml_conv_1d`, `ggml_conv_1d_dw`, `ggml_conv_2d`, `ggml_conv_2d_dw`) | After (1) sets `vec_dot_type=F32` for F16, conv graph builders that hardcode F16 im2col + F16 weight produce `MUL_MAT(F16, F16)` which the CPU backend rejects. Cast kernel to F32 when im2col is F32. | Carrying |
| 6 | `ggml-cuda/ggml-cuda.cu` (`ggml_cuda_op_mul_mat_cublas`, `use_fp16`) | CUDA counterpart of (1): `MUL_MAT(F16 weight, F32 act)` takes the fp16 cuBLAS path, quantising the F32 activation to F16 → ±65504 saturation → NaN → degenerate `!-loop` on GPU only (funasr SANM 70-layer encoder; CPU has (1), Metal has a native F16×F32 kernel). Exclude F16×F32 from `use_fp16` so it falls to the F32 `cublasSgemm` path. Quantized weights unaffected (MMQ/MMVQ). Found via the all-backends Kaggle P100 run 2026-05-31. | Carrying |

| 10 | `ggml-backend.cpp` (`ggml_backend_sched_split_graph`) | Sched mutates `node->src[j]` to internal `input_cpy` tensors; after `ggml_free(sched->ctx)` between calls, the next `alloc_graph` reads dangling pointers → silently skips creating cross-backend copies → GPU reads stale data. Chatterbox CFM solver, all platforms. | Carrying; `tools/upstream-prs/10-metal-sched-buffer-reuse-drift.md` |
| 15 | `ggml-backend.cpp` (sched backend routing) | Dual-backend `[CUDA,CPU]` sched produces Inf at LLM layer 2 → all-NaN by layer 3 in funasr Qwen2-0.6B. Same bug class as #11 (Metal NaN at large T). Workaround: `load_weights_split` to force LLM to CPU. Issue #125. | Carrying; `tools/upstream-prs/15-cuda-sched-nan-llm-decode.md` |

**Why these aren't upstream yet.** All were found while shipping a
specific CrispASR backend and were applied as the smallest local change
that unblocked us. None of them are CrispASR-specific in nature — any
project using ggml with similar workloads will hit them. Sending each
upstream is straightforward when we have time; until then they re-apply
on every ggml bump (we've already lost #2 once during the 0.9.8 → 0.10.0
subtree pull, and #5 surfaced as missed inventory during the master bump
test on 2026-05-05 because the original audit grep only matched
`CrispASR patch` and not `CrispASR fork`).

**(1) and (5) are coupled.** (1) sets `vec_dot_type=F32` for F16 weights
and (5) makes the conv graph builders cast their F16 weights to F32 to
match. Without (5), (1) crashes `kokoro --gpu-backend cpu` at
`ggml_backend_sched_split_graph`. Either send them as a single PR
upstream or design a single replacement that doesn't require splitting.

**Bump hygiene.** Before bumping ggml, snapshot `grep -rnE "CrispASR
(patch|fork)" ggml/`; after the bump, diff against the snapshot.
Anything missing is a patch upstream's master silently overwrote — find
the original commit, cherry-pick the hunk.

## Tracking

When any of these gets fixed upstream, drop a note here with the date
and the upstream commit/PR link, and remove the workaround if no longer
needed.

### Upstream progress log

- **2026-05-10** — Patch #4 (Metal conv_transpose_1d) merged upstream as
  [ggml-org/ggml#1477](https://github.com/ggml-org/ggml/pull/1477). Drop
  the `// CrispASR patch` hunk in `ggml-metal.metal` on the next ggml
  subtree bump.
- **2026-05-10** — Patch #2 (CUDA im2col OW > 65535) filed at
  ggml-org/ggml#1485; covers both `im2col_kernel` (2D) and
  `im2col_3d_kernel` (3D).
- **2026-05-11** — Patch #2 redirected per @CISC's comment on #1485:
  CUDA changes go to ggml-org/llama.cpp (development source-of-truth),
  not ggml-org/ggml (downstream sync target). Closed #1485, refiled as
  [ggml-org/llama.cpp#22944](https://github.com/ggml-org/llama.cpp/pull/22944).
  Routing rule recorded in `tools/upstream-prs/README.md` —
  `ggml-cuda/**` and `ggml-vulkan/**` future PRs file to llama.cpp;
  `ggml-cpu/**`, `ggml.c`, standalone-ggml stays at ggml-org/ggml;
  Metal is mixed (both work).
