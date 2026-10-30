# OmniVoice M1 longer-input validation and exact recipe — 2026-10-23

This extends the [short-request generation audit](omnivoice-m1-startup-generation-2026-10-23.md)
with an 18-second target, the same 32 MaskGIT steps and a longer new text.
The user-facing switch/default table and ordinary marked CLI command are in
[the TTS guide](../tts.md#omnivoice-reproducible-parameters-and-m1-performance-switches).

## Source, weights and architecture

Runtime source is `50196b2cd6dec7576343a1cb01e2e021f0df358e`, ggml
`e2ae679a563014f46c546ce6c6448e8601e4dea6`. The comparator is unmodified
`ServeurpersoCom/omnivoice.cpp` at `1a468d8`, ggml `cc5df49a`.
Both are Release/Metal/Apple Accelerate on an eight-core, 16 GB Apple M1,
macOS 26. One heavy local model/build ran at a time; normal desktop workloads
were active. Individual wall/CPU times and load/memory preflights are retained.
This is a scoped interactive-host comparison, not a hardware-independent ranking.

| Artifact | Repository revision | SHA256 |
|---|---|---|
| `omnivoice-q8_0.gguf` | `cstr/omnivoice-GGUF` @ `92266739c4496b04a497e164cf8c2801a792551e` | `9d8835c80d50cce6ba66bed60c31e94836a7f990c40d2e22496c0b6710c98bb5` |
| `omnivoice-tokenizer-f16.gguf` | Same revision | `710ef610e1f2845c6b7333d5432376b24f2d20d2c54a8cec9bc118d183ecea63` |
| Nemotron3.5 RNNT Q4_K judge | `cstr/nemotron-3.5-asr-streaming-GGUF` @ `bbd95a9ca5fa0dfca3312a122dfc45a2b578b9c2` | `60a2b074cdf46e81d72bc15d3eb79505dafbb697135f558659d9068f651854a2` |
| Parakeet CTC Q4_K judge | `cstr/parakeet-ctc-0.6b-GGUF` @ `159612ab8741e595e95e992298acfc400dc502b7` | `8f8f00676e7b16e08e7f922e224af99bf1c02d6c58cd207b9613ef4c78d0d543` |

The main is Q8_0 with F16 text/audio embeddings and audio head. The backbone has
28 blocks, hidden width 1024, 16 query heads / 8 KV heads, head width 128 and FFN
width 3072. Eight codebooks use vocabulary 1025 (mask 1024), at 25 frames/second.
The supplied model metadata supplies the remaining architecture parameters.

The comparator uses the same local matched-active-value adaptations as the
[earlier engine comparison](omnivoice-m1-comparison-2026-10-23.md): main
SHA256 `b9613d128221c36713c0691b9758a32c5cc8567591e098e36759535d4a59ce64`,
companion `222072bd6781740db1dc9f1d6cd6d899fac76ac55579bf80035d30babf5ef93f`.
Its 36 F16 Snake alphas are widened exactly to F32 for its loader contract; other
active decoder values are preserved. Its unused template encoder is not
cloning-certified. No adapted weights are published. Equal seed numbers do not
make the engines' different RNG implementations generate identical speech.

## Held-constant generation parameters

| Parameter | Value |
|---|---|
| Language | Explicit English (`en`) |
| Seed |42, reset before every synthesis |
| MaskGIT steps |32 |
| Guidance |2.0 |
| Class temperature |0.0 |
| Position temperature |5.0 |
| Layer penalty |5.0 |
| Time shift |0.1 |
| Target duration |18 s = 450 frames = 3,600 codebook-major int32 codes |
| CPU threads |4 |
| GPU |Metal; flash-attention graph |
| Persistent graphs / fused embedding step |On / on |
| CFG interval / unified CFG |1 (exact) / off |
| Codec GPU / split / resident / decomposition / fastconv |0 /1 /1 /1 /1 |
| Reference cache |Off; these requests are unconditioned |
| Audio processing |Default preprocessing/cleanup, 0.1 s pad/fade |
| Timing |Exclude load and diagnostic-file writes; include lazy setup in warmup |
| Paired schedule |One warmup per arm; then 0,1,1,0,0,1 |
| Comparator schedule |One warmup, then three measured repetitions |

The exact text is:

> Before we leave for the station tomorrow morning, please close the windows,
> check the kitchen lights, and bring enough water and food for the long walk.
> We can meet our friends near the old bridge and decide which path to follow.

All model files are on the internal SSD, with already-warm filesystem/shader
caches. Startup is recorded separately, not included in warm request medians.
For the attention-view pair, packing is off. For the packing pair, attention
views are on in both arms. These are different campaigns; do not multiply their
ratios or compare one arm's absolute time across varying host load.

## Reproduction

Build the regular embedded-source backend (the measured experiments also use
precompiled Metal libraries, with exact output parity established separately):

```sh
cmake -S . -B "$BUILD" -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
  -DGGML_METAL=ON -DGGML_BLAS=ON -DGGML_METAL_EMBED_LIBRARY=ON \
  -DCRISPASR_BUILD_TESTS=OFF -DCRISPASR_BUILD_SERVER=OFF
cmake --build "$BUILD" --target omnivoice crispasr-core -j3

c++ -std=c++17 -O3 -Isrc -Iinclude -Iggml/include \
  tools/ci-heavy/omnivoice_engine_bench.cpp \
  "$BUILD/src/libomnivoice.a" "$BUILD/src/libcrispasr-core.a" \
  -L"$BUILD/ggml/src" -L"$BUILD/ggml/src/ggml-cpu" \
  -L"$BUILD/ggml/src/ggml-metal" -L"$BUILD/ggml/src/ggml-blas" \
  -Wl,-rpath,"$BUILD/ggml/src" -Wl,-rpath,"$BUILD/ggml/src/ggml-cpu" \
  -Wl,-rpath,"$BUILD/ggml/src/ggml-metal" -Wl,-rpath,"$BUILD/ggml/src/ggml-blas" \
  -lggml -lggml-base -lggml-cpu -lggml-metal -lggml-blas \
  -framework Accelerate -o "$PROOF/crisp-engine-bench"
```

Set `TEXT` to the exact paragraph above and `OV_MAIN`/`OV_CODEC` to the verified
model files. Use absolute `BUILD`/`PROOF` paths and create `PROOF` first. Pin all
runtime overrides, including inherited sampling variables:

```sh
export CRISPASR_ACCEPT_LICENSE=omnivoice-terms
export CRISPASR_OMNIVOICE_CODEC_GPU=0
export CRISPASR_OMNIVOICE_CODEC_SPLIT=1
export CRISPASR_OMNIVOICE_CODEC_RESIDENT=1
export CRISPASR_OMNIVOICE_CODEC_DECOMP=1
export CRISPASR_OMNIVOICE_CODEC_FASTCONV=1
export CRISPASR_OMNIVOICE_PERSISTENT_GRAPH=1
export CRISPASR_OMNIVOICE_FUSED_STEP=1
export CRISPASR_OMNIVOICE_AUDIO_LEGACY=0
export CRISPASR_OMNIVOICE_PREPROCESS_PROMPT=1
export CRISPASR_OMNIVOICE_POSTPROCESS_OUTPUT=1
export CRISPASR_OMNIVOICE_PAD_DURATION=0.1
export CRISPASR_OMNIVOICE_FADE_DURATION=0.1
export CRISPASR_OMNIVOICE_NUM_STEPS=32
export CRISPASR_OMNIVOICE_GUIDANCE=2
export CRISPASR_OMNIVOICE_CLASS_TEMP=0
export CRISPASR_OMNIVOICE_POS_TEMP=5
export CRISPASR_OMNIVOICE_UNIFIED_CFG=0
export CRISPASR_OMNIVOICE_CFG_INTERVAL=1
export CRISPASR_TTS_REF_CACHE=0
unset CRISPASR_METAL_PROFILE

export CRISPASR_OMNIVOICE_PACKED_PROJ=0
"$PROOF/crisp-engine-bench" "$OV_MAIN" "$OV_CODEC" "$TEXT" 18 8 \
  "$PROOF/views" --paired-attn

export CRISPASR_OMNIVOICE_ATTN_VIEWS=1
"$PROOF/crisp-engine-bench" "$OV_MAIN" "$OV_CODEC" "$TEXT" 18 8 \
  "$PROOF/packing" --paired-proj
```

The native helper fixes 32 steps, seed 42, language en and four threads internally.
It emits raw diagnostic PCM/code files; normal deliverables use the marked CLI
command in the TTS guide. License acceptance does not waive model obligations.
For precompiled CLI builds, use the complete resource/build instructions in
[the startup report](omnivoice-m1-startup-generation-2026-10-23.md#startup-and-usable-builds).

To reproduce the comparator, first run the benchmark-only adapter from the
previous comparison, then build the same public driver against its own API/libs:

```sh
c++ -std=c++17 -O3 -DCRISPASR_BENCH_OVCPP \
  -I"$OVCPP/src" -I"$OVCPP/ggml/include" \
  tools/ci-heavy/omnivoice_engine_bench.cpp "$OVCPP/build/libomnivoice-core.a" \
  -L"$OVCPP/build" -Wl,-rpath,"$OVCPP/build" \
  -lggml -lggml-base -lggml-cpu -lggml-metal -lggml-blas \
  -o "$PROOF/ovcpp-engine-bench"
"$PROOF/ovcpp-engine-bench" "$ADAPTED/same-weights-llm.gguf" \
  "$ADAPTED/same-weights-codec.gguf" "$TEXT" 18 4 "$PROOF/ovcpp"
```

Its backend log confirms four CPU threads on this M1; that comparator derives
its thread count from half the logical core count. Its model initialization
uses embedded-source shaders, whereas the measured CrispASR pair uses the
validated precompiled backend. Load is excluded, and earlier source/precompiled
CrispASR checks preserve exact output. This is not a matched-ggml-kernel study.

## Longer-input results

| Campaign / arm | Generation | Decode | Full request | Process CPU |
|---|---:|---:|---:|---:|
| Attention pair, original copies |26.711 s|1.558 s|28.298 s|2.641 s|
| Attention pair, views |24.446 s|1.544 s|26.105 s|3.759 s|
| Packing pair, views + separate projections |24.614 s|1.590 s|26.233 s|4.148 s|
| Packing pair, views + packed projections |24.453 s|1.512 s|25.965 s|4.001 s|
| omnivoice.cpp, matched active weights |—|—|35.284 s|3.593 s|

These are medians of three measured samples per arm, with each stage's median
computed independently. The larger CPU-time variation and individual wall times
show the host was contended. Attention views reduce request latency 7.7% in its
paired campaign. Packing's 1.0% difference again does not justify enabling an
additional 312 MB of weights by default.

All 16 CrispASR code arrays and float32 PCM arrays are byte-identical, including
both warmups and every timed repetition in both campaigns. Output is 406,296
samples at 24 kHz, 16.929 seconds. All four comparator outputs are byte-identical
to each other: 426,480 samples, 17.770 seconds. The observed whole-request gap
26.11 versus 35.28 seconds is about 26% on this text; engines ran sequentially
under changing desktop load, use different ggml revisions/RNGs and emit different
speech durations. It does not establish a general engine ranking. Their
request-time/emitted-duration ratios are about 1.54 and 1.99 respectively.

Both pinned ASR judges transcribe the entire paragraph with WER 0 for the views,
packed and comparator controls. Codes/PCM equality makes repeated ASR of all
identical CrispASR samples unnecessary. A real precompiled Release CLI also runs
the full paragraph in one request and emits exactly the same 3,600 codes, with
its normal watermark and C2PA chunk. Its built-in watermark detector reports
confidence 0.8113. Its single gen 21.53 s + decode 1.48 s observation is a surface
check, not an extra sample silently substituted into the paired timing medians.

Earlier fox-sentence voice-cloning failures remain open. This new unconditioned
paragraph validates longer-input content/parity, not every voice or precision.
No model, math, default or ggml revision changed in this continuation.

## Matrix probe: a next candidate, not a runtime improvement

`tools/ci-heavy/omnivoice_matmul_bench.cpp` reads the actual layer-0
`llm.blk.0.ffn_down.weight` (K 3072 × M 1024) from the pinned Q8 file. Arm 0 uses
its serialized quantized bytes. Arm 1 dequantizes on CPU to F32, rounds to F16
and uploads a second weight buffer. Both consume identical deterministic F32
inputs (`sin(index * 0.0017)`), at 150 and 450 token columns. Every input is
restored before reuse. Eight total warmups (four per arm) precede ten balanced
measured samples per arm. Wall and process CPU time exclude weight/input upload,
output readback and equality/norm checks.

It reports byte equality, maximum absolute error, cosine and both norms against
the Q8 result. All outputs in the completed probe have zero absolute error and
matching norms. Final hot-op medians are:

| Token columns | Q8 wall | F16 wall | Q8 CPU | F16 CPU |
|---|---:|---:|---:|---:|
|150|1.959 ms|1.810 ms|0.091 ms|0.111 ms|
|450|5.141 ms|4.682 ms|0.161 ms|0.162 ms|

The machine-readable receipt includes all 56 samples and the
medians; isolated millisecond timings are not full-generation speed predictions.
F16 storage grows this weight from 3,342,336 to 6,291,456 bytes (1.88×);
the full model's conversion,
allocation and all quality surfaces have not been tested. Keep this as a tuning
candidate; no F16 cache or new quantization artifact is shipped.

Compile this Metal-only helper with the same ggml link/rpath flags as above,
using `tools/ci-heavy/omnivoice_matmul_bench.cpp` instead of the engine driver;
it needs no CrispASR static archives. Pass only `"$OV_MAIN"` as its argument.
Do not label it a F16 source-checkpoint benchmark: its F16 values derive from
the already-quantized Q8 weights.

[Individual samples, parameter values, hashes, preflights and ASR transcripts](omnivoice-m1-long-input-2026-10-23.json).
