# OmniVoice reference-encoder A/B — 2026-10-23

Historical experiment below: subsequent listening approval and
[Metal default promotion](omnivoice-metal-defaults-2026-10-23.md) update the
current eligibility/defaults. The original raw measurements are retained.

All five encoder experiments are implemented, with independent switches and
retained baseline paths. The immutable-data cache now defaults on; the smaller
pointwise-convolution optimization defaults on only for Apple ARM CPU encoding.
Graph FC/RVQ, fused HuBERT and hybrid CPU/GPU encoding remain off by default.
The [raw receipt](omnivoice-encoder-2026-10-23.json) contains every measured
sample, hashes, code differences, cloned-audio results and ASR transcripts.

Follow-up: the optional [semantic F32 path](omnivoice-semantic-precision-2026-10-23.md)
removes the recorded station code difference on the tested M1 and seven-reference
suite. The unhinted measurements below remain historical controls; GPU defaults
stay off. The follow-up contains corrected-path synthesis and timing evidence.

## Measured decision

Apple M1, eight cores, 16 GB, macOS 26; four encoder CPU threads, Release,
Metal/Accelerate build, internal files and warm OS/shader caches. Ordinary
desktop activity introduces outliers. One warmup per arm is excluded; three
measurements per arm are interleaved. The final default trial reverses arm
order. Times cover fresh reference preprocessing and encoding, with disk
reference caching disabled; they exclude model load and output-file writes.

| Reference | Baseline median | New default | Wall reduction | CPU reduction | Codes |
|---|---:|---:|---:|---:|---|
| window |1.355 s|1.246 s|8.1%|7.9%|520/520 exact|
| station |1.253 s|1.172 s|6.4%|6.6%|488/488 exact|
| fox |1.338 s|1.268 s|5.2%|5.5%|Exact|

Every call in all three eight-call trials matches a separately compiled frozen
pre-experiment encoder. The baseline source SHA256 and experimental checkpoint
`48aa6d20a` are recorded in the receipt; the latter introduced all five paths
with defaults off. Final source/probe hashes identify the subsequent refinement.

These savings apply when encoding a new reference, not to every generation
step. A reference-code cache hit already avoids encoding. The generation model
and sampling recipe are unchanged, so no further whole-generation speedup or
new ranking against omnivoice.cpp is inferred from these numbers.

## Implemented paths and switches

All variables below start with `CRISPASR_OMNIVOICE_`. Explicit `0` selects the
old path. Cached buffers persist until tokenizer replacement, backend placement
change or destruction, even if a gate is subsequently disabled.

| Suffix | Unset | Implementation and cost |
|---|---|---|
| `ENCODER_CACHE` |On|Keep exact F32 tensor conversions, codebook squared norms and folded HuBERT positional weights. Same scalar accumulation order. About29.4 MB host payload on this tokenizer.|
| `ENCODER_FASTCONV` |`1` on Apple ARM CPU; `0` elsewhere|`1` bakes only F16 1×1 encoder weights into F32 and enables their matrix formulation; about26.2 MB. `2` also bakes wider convolution kernels, about280.9 MB. Decoder FASTCONV is a separate switch.|
| `ENCODER_GRAPH` |Off|One graph for FC and eight residual quantizers, F32 weights/accumulation and exact first-index tie selection. Extra projection/graph buffers; no convincing standalone CPU timing win.|
| `ENCODER_FUSED` |Off|Same HuBERT frontend/transformer builders in one graph, avoiding intermediate host readback. No convincing standalone CPU timing win.|
| `ENCODER_HYBRID` |Off|Keep DAC acoustic encoding on CPU, lazy-load a filtered GPU copy for HuBERT and the semantic bridge. With GRAPH, run FC/RVQ on that backend too. Filtered F16 weights add223.8 MB, plus caches/scratch. Requires GPU use; `--no-gpu` still wins.|

The default combination adds55,629,824 bytes of measured host-cache and
pointwise-weight payload; allocator overhead and graph scratch are additional.
Full baking was initially tested as FASTCONV=1; those prototype samples are
explicitly labeled in the receipt. In the final interface full baking is `2`.
Its larger memory cost did not justify promoting it. Cache allocation is lazy;
failed convolution/projection allocation publishes no partial tensor pointers.
Reference-cache identity includes the effective encoder policy and backend, so
experimental paths cannot silently reuse baseline codes.

The semantic-only loader admits canonical `sem.*`, `encoder_semantic.*`, FC
and active quantizer projection/codebook tensors. It excludes the acoustic
encoder, decoder and training averages. Failed GPU initialization falls back to
CPU for the context; the benchmark rejects that fallback when measuring GPU
performance. Tokenizer replacement permits another initialization attempt.

## Faster GPU experiments remain optional

Final-source paired medians, with each row compared against its own interleaved
all-encoder-gates-off control:

| Reference/path | Control | Enabled | Reference-code parity |
|---|---:|---:|---|
| window hybrid |1.271 s|0.777 s|Exact|
| station hybrid |1.285 s|0.782 s|4/488 differ|
| window all five |1.275 s|0.608 s|Exact|
| station all five |1.316 s|0.614 s|4/488 differ|

The actual semantic backend is `MTL0`, with its allocation checked. The station
changes affect one frame in later residual quantizers and repeat across GPU
arms. Full Session synthesis also changes generated codes and PCM for station.
Both frozen ASR judges transcribe the resulting new controls correctly, but
that cannot certify speaker identity, prosody or general quality. Consequently
GPU/fused/graph defaults stay off. Resolving this requires a CPU/GPU semantic
boundary and norm audit, followed by broader cloning acceptance. Whole-DAC Metal
encoder correctness remains a separate unresolved problem.

Two early GPU trials are rejected controls: one incorrectly required decoder
weights, another selected a comparator namespace rather than canonical `sem.*`.
They fell back to CPU and are excluded from GPU timing claims. The native probe
now exits unsuccessfully if a requested hybrid stage runs on CPU.

## Quality and regression evidence

The source architecture was checked against the installed Transformers Higgs
and DAC forward paths:24→16 kHz resampling and160-sample padding, HuBERT13-state
mean and temporal decimation, acoustic-first concatenation, FC, eight learned
project-in/project-out residual quantizers and first-centroid Euclidean ties.
Shared HuBERT builders retain the staged baseline. No Snake approximation,
quantization, resampling, sampler or licensed-weight change is included.

A freshly linked Python Session/C ABI run renders16 controlled clones across
window/station/fox references and baseline/default/hybrid/all variants. All
five baseline/default target pairs have byte-identical generated codes and
float PCM. Seven unique resulting waveforms are scored once each by both
frozen Nemotron RNNT and Parakeet CTC judges; exact duplicates inherit the
same result. Every new transcript has WER0. The two targets are:

- “The quick brown fox jumps over the lazy dog.”
- “We should bring enough water and food for the long walk home.”

Reference text, sample counts, PCM/code hashes, library hash, norms, transcripts
and judge model revisions/hashes are in the receipt. Main Q8_0 and tokenizer
F16 use `cstr/omnivoice-GGUF` revision
`92266739c4496b04a497e164cf8c2801a792551e`. Synthetic Piper references use
`cstr/crispasr-regression-fixtures` revision
`d0a7d7a8be318a5841dfdbe6ad37d3acf75523e3`, under
`index-echo-9b/roundtrip-piper`. Generation uses English, seed42 reset per call,
32 steps, duration6, guidance2, class temperature0, position temperature5,
exact CFG interval1; codec split/resident/decomposition and attention views on,
packing/unified CFG off. Preprocessing and cleanup remain enabled.

The actual-runtime unit covers F32/F16 scalar/cache/graph codes, first-index
ties, optional FC bias, failed actual backend allocation, default overrides,
pointwise/full-bake cache replacement, no-conversion models, semantic filtering
and teardown. Five focused CTest cases pass. ASan/UBSan passes; replacing the
tie selector with ordinary argmax causes the intended assertion failure.
Full CLI/shared-library builds and a normally marked cloned CLI run are checked
separately; the CLI's generated codes match the Session baseline.

These new short controls do not close earlier fox-cloning failures or hosted
Linux cloning acceptance. No new matched encoder benchmark against
omnivoice.cpp is claimed: the previous locally adapted companion's unused
encoder was not certified. The prior same-active-value unconditioned comparison
remains in the [engine report](omnivoice-m1-comparison-2026-10-23.md) and
[longer-input report](omnivoice-m1-long-input-2026-10-23.md).

## Reproducing the native A/B

Build the ordinary Release Metal library as documented in the TTS guide, then
compile the public probe against that build. `BUILD`, `OV_CODEC`, `REF` and
`OUT` are portable local aliases. Keep other heavy workloads idle and record
load/memory pressure with the samples.

```sh
c++ -std=c++17 -O3 -DGGML_USE_CPU -DGGML_SHARED -DGGML_BACKEND_SHARED \
  -Isrc -Iinclude -Iggml/include tools/ci-heavy/omnivoice_encoder_bench.cpp \
  "$BUILD/src/libcrispasr-core.a" \
  -L"$BUILD/ggml/src" -Wl,-rpath,"$BUILD/ggml/src" \
  -L"$BUILD/ggml/src/ggml-cpu" -Wl,-rpath,"$BUILD/ggml/src/ggml-cpu" \
  -L"$BUILD/ggml/src/ggml-metal" -Wl,-rpath,"$BUILD/ggml/src/ggml-metal" \
  -L"$BUILD/ggml/src/ggml-blas" -Wl,-rpath,"$BUILD/ggml/src/ggml-blas" \
  -lggml -lggml-base -lggml-cpu -lggml-metal -lggml-blas \
  -framework Accelerate -o encoder-bench
CRISPASR_OMNIVOICE_BENCH=1 \
CRISPASR_OMNIVOICE_ENCODER_BENCH_REVERSE=1 \
  ./encoder-bench "$OV_CODEC" "$REF" "$OUT" defaults
```

Modes are `cache`, `fastconv`, `graph`, `fused`, `hybrid`, `all`, `cpu`,
`defaults`, `full-bake`, and `baseline`. The paired modes explicitly disable
all five gates for control samples. `defaults` unsets them for candidate samples;
`cpu` enables cache and pointwise only; `full-bake` uses FASTCONV=2 only.
The first two samples are warmups, followed by three per arm. Each call writes
raw codebook-major int32 codes. `baseline` gives two all-off calls; for an
independent old-source build, extract the prior `src/omnivoice.cpp` as
`baseline.cpp`, add its directory to includes and define `OV_ENCODER_BASELINE`.

For production cloning, use the guide's ordinary CLI recipe, add `--voice`,
`--ref-text` and `--i-have-rights`, and retain license acceptance and marking.
Diagnostic raw Session artifacts in this audit have an explicit responsibility
attestation and remain identified local artifacts. No runtime policy or license
obligation is waived by these optimization switches.
