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

## Tracking

When any of these gets fixed upstream, drop a note here with the date
and the upstream commit/PR link, and remove the workaround if no longer
needed.
