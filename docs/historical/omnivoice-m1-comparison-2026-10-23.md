# OmniVoice on Apple M1: engine comparison and resident codec

Baseline: CrispASR runtime `5acff9f21` / main `c34459b2e`, ggml
`e2ae679a563014f46c546ce6c6448e8601e4dea6`. Comparator:
`ServeurpersoCom/omnivoice.cpp` at `1a468d8`, ggml `cc5df49a`.
Both are Release builds with Metal and Accelerate, on an eight-core, 16 GB M1.
Only one heavy local build/model ran at a time. Other desktop activity varied;
these are scoped measurements on this host, not a hardware-independent ranking.

## What was held constant

The controlled request is “We should bring enough water and food for the long
walk home.” English is explicit, flash attention enabled, seed 42, 32 steps,
normal guidance/temperature defaults, and exactly 150 generated codec frames
(six seconds before output cleanup). Both engines use their public native APIs,
with four CPU threads. Model initialization and PCM/code file writes are outside
warm request timings. Audio cleanup remains enabled; retained output duration
can differ, so target duration is not mislabeled as emitted audio duration.

The supplied packages differ: omnivoice.cpp's main model quantizes its text and
audio embeddings/head to Q8; CrispASR keeps those three tables F16. The supplied
companions also differ. Initial default-package observations therefore do not
isolate engine overhead. A further source review found CrispASR's automatic
English conditioning versus the comparator's unspecified language. Those initial
observations are excluded from the controlled speed ratio.

`tools/ci-heavy/omnivoice_bench_adapt.py` makes local benchmark artifacts with
active LLM/decoder numeric values preserved. Main tensor names and metadata
use the comparator's schema. The standard vocabulary prefix and seven active
special-token IDs agree. Unused encoder weights retain the comparator template:
a precomputed HuBERT position-convolution weight differs structurally from
CrispASR's weight-norm parameters. This mixed companion is for unconditioned
benchmarking only; cloning with it is not certified. No adapted weights were
published.

An initial adapter preserved every active tensor byte, shape and dtype but
failed both ASR judges completely (empty transcripts, WER 1.0). Source review
found the comparator's DAC alpha loader supports F32/BF16 and interprets every
non-F32 alpha as BF16. CrispASR stores those alphas as F16. The corrected
adapter widens these alphas exactly to F32, checking numeric equality; all other
active tensor bytes remain unchanged. Rejected timings remain in the receipt
and cannot support a speed claim. The comparator source is unmodified.
The ggml pins and sampling implementations still differ; equal seeds do not
imply equal stochastic trajectories or identical speech between engines.

## Warm request measurements

| Engine / codec | Median warm request | Emitted audio | Both ASR WER |
| --- | ---: | ---: | ---: |
| CrispASR / CPU crop |8.84 s|4.01 s|0.0|
| CrispASR / resident Metal |8.08 s|4.01 s|0.0|
| omnivoice.cpp / Metal, matched active values |9.57 s before;10.01 s after|4.32 s|0.0|

Resident placement reduces CrispASR request latency by **8.6%** in the quieter
paired run. The comparator's six warm samples have a combined median 9.63 s;
CrispASR resident is about 16% lower latency for this case. Both comparator
campaigns return byte-identical PCM to the independently scored waveform.
A preceding more-contended paired run measured 21.11→19.61 s (7.1% reduction).
Absolute speed is highly load-dependent; the result supports this optimization,
not a general claim that one engine wins for every text or workload.

CrispASR's CPU/resident request A/B runs within one context: one warmup per arm,
then CPU/resident/resident/CPU/CPU/resident. Every synthesis resets the seed,
and saved codes check that codec placement cannot change generation. The
comparator has one warmup followed by three samples before and after the quieter
CrispASR A/B. Cross-engine measurements
are sequential rather than simultaneous; background load limits a fine-grained
ranking. All return codes, sample counts, finite values and nonzero energy are
checked. The JSON receipt retains individual observations and quality results.

## What changed

`CRISPASR_OMNIVOICE_CODEC_RESIDENT=1` adds a separately owned decoder-only
allocation to the existing split experiment. The full tokenizer stays on CPU;
only DAC decoder, fc2 and RVQ output/codebook tensors go into the GPU copy.
Training-only codebook averages and encoder projections are excluded. Both
sets of weights use the existing decode graph, permutation and F32-cache code.
The resident copy is prepared lazily, reused across syntheses and voice changes,
and released on tokenizer replacement or context destruction. Failed loads
clean their partial allocations. Thread-count setters also reach codec CPU
backends, including GPU initialization fallback.

On this F16 companion the resident allocation is about 124 MB including caches,
in addition to the existing CPU tokenizer. It avoids repeated full 403 MB
placement reloads. It is not a claimed 50 MB allocation.

For the existing four saved-code clones, three measured samples per arm follow
warmup, counterbalanced within each context. Timings include placement changes
needed before encoding another voice, but deliberately exclude encoder compute
and LLM generation:

| Saved-code case | CPU crop | Split with reload | Resident split |
| --- | ---: | ---: | ---: |
| window / fox |0.390 s|0.451 s|0.157 s|
| window / walk |0.495 s|0.499 s|0.223 s|
| station / fox |0.335 s|0.408 s|0.145 s|
| station / walk |0.475 s|0.456 s|0.207 s|

These are final-source timings after excluding 1,048,576 bytes of unused
codebook training averages. Resident weights are 43,147,424 bytes, F32 cache
40,862,592 bytes and permutations 40,026,112 bytes: 124,036,128 bytes total,
excluding graph scratch. Earlier contended CPU timings were 1.92–2.82 s and
resident 0.38–0.55 s. Report both campaigns rather than advertising the largest
4–7x ratio: the quieter final campaign measures about 2.2–2.5x codec speedup.
Final replay retains all 12 initial per-arm PCM hashes. Each campaign also has
16 byte-exact resident/old-Metal pairs. Fresh uncached encodes of both reference
voices retain identical codes after resident decoding and preserve the CPU
encoder allocation.

Generation remains the dominant cost, so codec speedup is not whole-request
speedup. Initialization varied from 13 to 145 seconds across these embedded Metal builds.
CrispASR reads the supplied external-volume weights while adapted comparator
weights are local; source compilation and storage differ. These load observations
do not support a cold-start ranking. Reusing an initialized session matters.

## Enable and reproduce

All three experiments remain **off by default**. On a GPU-enabled M1 build:

```sh
export CRISPASR_OMNIVOICE_CODEC_SPLIT=1
export CRISPASR_OMNIVOICE_CODEC_RESIDENT=1
export CRISPASR_OMNIVOICE_CODEC_DECOMP=1
```

Use the normal model-license/voice-rights acknowledgements required by your
calling surface. Resident alone has no effect without split. Disabling split
restores the ordinary codec placement; no public params struct changes.
GPU encoding is not enabled or certified. CUDA's existing placement default
is unchanged, and this M1 evidence does not certify a CUDA default.

Build `tools/ci-heavy/omnivoice_engine_bench.cpp` against each engine's own
`omnivoice.h` and libraries; define `CRISPASR_BENCH_OVCPP` for the comparator.
It takes `MODEL CODEC TEXT DURATION REPEATS OUTPUT_PREFIX`. CrispASR's
`--paired` final argument requires 8 repetitions and selects the balanced
CPU/resident comparison internally. Compile the fixed-code probe
`omnivoice_codec_resident_bench.cpp` against CrispASR core/ggml (it includes the
actual runtime; do not also link libomnivoice). Match the library build
definitions, including `GGML_MAX_NAME=128`; the private probes set that default. It takes the tokenizer and a
proof directory produced by `omnivoice_codec_ab.cpp`, or adds `--voice-check`
for two fresh uncached CPU encodes interleaved with resident GPU decoding.

For the unconditioned comparator adaptation, use:

```sh
python tools/ci-heavy/omnivoice_bench_adapt.py \
  --templates "$OVCPP_WEIGHTS" --crispasr-models "$CRISPASR_WEIGHTS" \
  --output "$PROOF"
```

## Validation and limits

Both pinned Nemotron RNNT and Parakeet CTC transcribe the whole-request
CrispASR CPU/resident outputs exactly (WER 0.0). On the four saved-code clones,
transcripts agree across all three decoder paths for both judges. `window/fox`
is 1/9 on each judge; `station/fox` is 2/9 on RNNT and 0 on CTC; both walk targets
are 0. These are unchanged baseline limitations, not relaxed acceptance gates.
All eight paired generations save identical code bytes. Final-source replay
has identical PCM to the already-scored per-arm outputs, so repeated ASR over
those exact bytes is unnecessary. The corrected comparator adapter also scores
WER 0.0 under both judges.

Hermetic coverage exercises the real private loader and public setter: selected
weight-byte preservation, encoder/training-state exclusion, source-file-free
reuse, replacement invalidation, truncated-load cleanup and GPU-to-CPU fallback.
The complete Release CLI and shared library also build and link. A fresh
Python Session through the C ABI returns code and PCM bytes identical to the
scored native baselines for both CPU and resident decoding, with 32 steps and
the same six-second target. Four selected CTest registrations pass. Native lifecycle code also passes
ASan/UBSan. These checks do not certify every backend or model quantization.

Earlier full Linux CPU speech campaign
[39861560201](https://github.com/CrispStrobe/CrispASR/actions/runs/39861560201)
failed its unchanged WER gate on three window-cloning surfaces (2/9); station
and unconditioned arms pass. This campaign predates the resident implementation.
The performance work neither waives that failure nor declares general cloning
acceptance. The new resident path stays gated while that broader question is open.

Individual timings, norms, hashes, transcripts and test results are in
[the machine-readable receipt](omnivoice-m1-comparison-2026-10-23.json).
