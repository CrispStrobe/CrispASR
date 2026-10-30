# Phonon-2

[Fermion Research's Phonon-2](https://huggingface.co/FermionResearch/Phonon-2)
is an English speech recognizer derived from NVIDIA Parakeet TDT 0.6B v3.
CrispASR uses its existing Parakeet backend for this model.

```sh
crispasr -m phonon2 -f recording.wav                      # Q8_0 default, ~674 MB
crispasr -m phonon2 --model-quant f16 -f recording.wav     # reference fidelity, ~1,255 MB
crispasr -m phonon2 --model-quant q4_k -f recording.wav    # smaller, ~402 MB
```

## Convert the upstream weights

Install the Parakeet converter's dependencies, plus `zstandard`:

```sh
pip install torch numpy gguf sentencepiece librosa pyyaml safetensors huggingface_hub zstandard
python models/convert-parakeet-to-gguf.py \
  --hf FermionResearch/Phonon-2 --output phonon2-f16.gguf
./build/bin/crispasr-quantize phonon2-f16.gguf phonon2-q8_0.gguf q8_0
./build/bin/crispasr-quantize phonon2-f16.gguf phonon2-q4_k.gguf q4_k
./build/bin/crispasr -m phonon2-q8_0.gguf -f samples/jfk.wav
```

`--hf` also accepts a downloaded Hugging Face snapshot or the extracted
directory containing `config.json` and `model.fermion`. Conversion checks the
upstream archive/container SHA-256, expands the five-value encoder weights
exactly (`0`, `±lo`, `±hi`, with two magnitudes per output row), and expands the
remaining integer tables with their stored row scales. It reads the model
configuration and vocabulary from the archive; no teacher weights or downloaded
Python code are loaded. The resulting GGUF retains `general.architecture =
parakeet`, so bindings auto-detect it through the C ABI even if the file is
renamed.

## Size, quality, and speed

The upstream **164 MB** figure describes its compressed transport file, not
an F16 or ordinarily quantized GGUF and not runtime RAM. Its custom five-value
packing differs from ggml's quantization formats. CrispASR expands that format
before GGUF conversion; Q8_0 and Q4_K apply another quantization step. Those
exports should not be assigned the upstream model's benchmark scores without
their own evaluations.

Local CPU validation on 21 clips (JFK, one upstream LibriSpeech sample, and
19 clips from the Hugging Face LibriSpeech test subset):

| Export | Download | Exact reference transcripts | Word edits versus reference* |
|---|---:|---:|---:|
| F16 | 1,255 MB | 21/21 | 0/298 |
| Q8_0 (default) | 674 MB | 19/21 | 1/298 |
| Q4_K (optional) | 402 MB | 15/21 | 7/298 |

\* Lowercase, ignore punctuation, retain apostrophes inside words; edit distance
against the independent model reference, **not WER against human ground truth**.
Q8's differences were one capitalization change and one proper-name spelling.
This is a small conversion/quantization check, not a reproduction of the seven
benchmark sets. F16 passed all 28 compared stages on both JFK and the upstream
LibriSpeech sample: minimum cosine 0.999996 / 0.999796, with tensor norm-ratio
error bounded by 0.065% / 0.259%. C ABI sessions auto-detected the backend and
produced these transcripts with an arbitrary model filename.

Fermion reports 5.21% average WER across seven English benchmark sets and 174×
realtime on an M5 MacBook Air using its MLX engine, excluding loading. These
figures are upstream measurements, not CrispASR performance claims. Phonon-2
is advertised and evaluated for English; its inherited multilingual vocabulary
does not establish multilingual accuracy after retraining.

## Reference validation

```sh
python tools/dump_reference.py --backend phonon2 \
  --model-dir FermionResearch/Phonon-2 --audio samples/jfk.wav \
  --output phonon2-jfk-ref.gguf
./build/bin/crispasr-diff phonon2 phonon2-f16.gguf phonon2-jfk-ref.gguf samples/jfk.wav
```

The reference uses the unmodified upstream container reader and stock
Transformers `ParakeetForTDT`, independently of the converter's unpacker.
Install a recent `transformers` release providing `ParakeetForTDT` to run it.
Only the teacher's small JSON configuration/tokenizer files are fetched.
`PHONON2_BASE_DIR` can point to a local copy of those files for offline runs.

Weights: **CC-BY-4.0**, derived from NVIDIA's Parakeet TDT v3 and retrained by
Fermion Research. Retain attribution and describe conversion/quantization
changes when redistributing. The vendored reference reader is Apache-2.0;
its license is in `tools/reference_backends/phonon2/LICENSE`. The upstream
[NOTICE](https://huggingface.co/FermionResearch/Phonon-2/blob/main/NOTICE)
documents the original model, changes, and training data.

## Integration checklist

Phonon-2 is a model variant of the existing Parakeet runtime. The contributing
checklist applies through that shared engine:

| Checklist point | Wiring |
|---|---|
| C runtime and stage timers | `src/parakeet.{h,cpp}`; `CRISPASR_PARAKEET_BENCH=1` |
| CLI adapter and factory | Shared Parakeet adapter; explicit `--backend phonon2` and model filename routing |
| CLI/library CMake linkage | Existing `parakeet` library in both CLI and shared C ABI; no duplicate runtime library |
| C ABI dispatch, lifecycle, setters | Explicit `phonon2` alias normalizes to `parakeet`; existing transcription, hotwords, beam, temperature, attention-context and cleanup paths |
| Architecture auto-detection | `general.architecture=parakeet`, shared `arch_backend_map.h`; works after renaming |
| Registry | `phonon2`, Q8 default, F16/Q4 alternatives and weight license |
| Quantization | Existing Parakeet rules, including pointwise matmuls; published exports validated |
| Reference and diff | Independent upstream reader + Transformers; existing Parakeet stage APIs and strict pinned nightly gate |
| Bindings | Generic C ABI reaches all bindings; Python/Go/Dart document the variant |
| Go static linkage | Existing Parakeet library; generator checks the unchanged cgo library list |
| Architecture/capability docs | `docs/architecture.md#phonon2`; generated feature matrix and library capability table |
| Tests and live environment | Container/registry tests, pinned regression, `test_phonon2_live.py`, `CRISPASR_MODEL_PHONON2` |

`Session(..., backend="phonon2")` explicitly opens this variant; `Session.backend`
reports the shared runtime name, `parakeet`. The CLI uses model metadata to select
English without automatically loading Whisper for LID. Non-English language
requests produce a warning; the model cannot honour them. Explicit language
identification remains available. Native punctuation is advertised. `--no-flash-attn` and the C ABI open flag
now reach the graph builder; the live guard checks the resulting node trace.
The raw Parakeet context default now reports flash enabled, matching the
previous graph behaviour; an explicit false selects manual attention.

## Reproducible profiling

Build both `crispasr-lib` and `crispasr-cli`. Set the model and library paths:

```sh
python tools/profile_phonon2.py --engine runtime \
  --model phonon2-q8_0.gguf --lib build/src/libcrispasr.so \
  --output runtime-q8.json
CRISPASR_SCHED_PROFILE=1 CRISPASR_PARAKEET_BENCH=1 \
CRISPASR_PARAKEET_ENC_PROBE=1 CRISPASR_PARAKEET_DECODE_TIMING=1 \
python tools/profile_phonon2.py --engine runtime --trace \
  --model phonon2-q8_0.gguf --lib build/src/libcrispasr.so \
  --output trace-q8.json
python tools/profile_phonon2.py --engine reference \
  --model /path/to/upstream/snapshot --output reference-cpu.json
```

The benchmark loads once, warms each 11/55-second shape, checks nonempty stable
transcripts and proportional word counts, then records three inference times
and their median. The trace is a separate run: the scheduler callback forces
node materialization and changes dispatch overhead. Compare timings only on
the same host, device, thread count and load. The Python reference is stock
Transformers on CPU, not the upstream MLX engine.

The manual **Phonon-2 integration and profile** GitHub workflow runs Linux CPU
and macOS Metal, checks wiring/live transcripts and F16 stage parity first,
then records F16/Q8/Q4 timings plus Q8 stage/node traces. Linux also measures
the independent Python reference on the same runner. Artifacts retain all raw
samples, transcripts, model checksums, dependency versions and host details.

## Runtime optimization coverage

The encoder, including subsampling and all 24 FastConformer blocks, is a ggml
scheduler graph, with CPU fallback for operations unsupported by the selected
backend. The scheduler currently registers GPU/CPU backends, not the separate
ggml BLAS backend. Linux encoder matmuls use ggml CPU kernels even when the
build enables OpenBLAS; OpenBLAS accelerates the shared mel filter projection.
Mel extraction and TDT token selection are CPU code. Decoder
execution depends on the device:

| Path | Predictor and joint |
|---|---|
| Linux CPU | Scalar C++ loops over cached F32 weights; the OpenBLAS build option does not accelerate these loops |
| Apple CPU / Metal | Apple Accelerate; Metal encoder with CPU decoder is the existing default |
| CUDA / Vulkan | ggml predictor/joint graphs, built and allocated once per decode call and reused across token steps |

There is no transformer KV cache in this TDT decoder: it retains the two LSTM
hidden/cell states and reuses the predictor output across blanks. Encoder
projections are computed ahead of the token loop; CUDA uses the measured GPU
projection path by default, Apple uses batched SGEMM, and Linux CPU currently
uses scalar per-frame projections. The CPU predictor/joint weight conversions
are initialized lazily and retained by the model context.

The encoder folds batch normalization into depthwise convolution weights and
fuses Q/K/V projections at load time. Quantized models use the shared pointwise
weight repacking rules. Attention uses ggml flash attention by default, with
relative-position scores still computed separately; `--no-flash-attn` selects
manual attention. CUDA also retains its backend-specific manual-attention
policy. These are shared Parakeet optimizations, not a new Phonon-specific
packed five-value kernel.

The scheduler/context persist, but the encoder graph is rebuilt for each call
and its buffers are allocated through the scheduler. The experimental
`CRISPASR_PARAKEET_ENC_CACHE` remains off: the existing implementation can reuse
stale tensor pointers and corrupt repeated-call output. The trace separates
build, allocation and compute costs; graph caching should only be reconsidered
if those first two costs are material. A scheduler trace materializes each
node and perturbs execution, so its absolute timings are diagnostic only.

## Measured CPU profile (2026-09-30)

[CI run 36784469150](https://github.com/CrispStrobe/CrispASR/actions/runs/36784469150)
measured runtime commit `9e9816631` and the independent Python reference on the
same Linux runner: AMD EPYC 9V74, 4 vCPUs, 4 inference threads, Release build,
OpenBLAS. The reference is stock Transformers `ParakeetForTDT` in F32 with
PyTorch 2.7.0 CPU, loading the original Fermion container through the upstream
reader. It is not the separate MLX implementation.

One model is loaded per process; each 11/55-second shape gets a warmup followed
by three timed calls. Loading, downloads, warmup and diagnostic callbacks are
excluded. The 55-second clip repeats JFK five times and stays on the ordinary
single-pass path. Every repeat produces stable nonempty output: 22/110 words.
F16 and Q8 transcripts match the Python reference exactly at both lengths;
Q4 matches normalized words, with a punctuation difference on the longer clip.
This is a throughput check, not an accuracy benchmark on natural long audio.

| Engine/export | 11 s audio: median / realtime | 55 s audio: median / realtime | Peak process RSS* |
|---|---:|---:|---:|
| Python reference F32 | 1.580 s / 6.96× | 8.042 s / 6.84× | 3,414 MiB |
| CrispASR F16 | 3.731 s / 2.95× | 19.222 s / 2.86× | 2,249 MiB |
| CrispASR Q8_0 | 1.760 s / 6.25× | 9.301 s / 5.91× | 1,614 MiB |
| CrispASR Q4_K | 2.043 s / 5.38× | 10.878 s / 5.06× | 1,320 MiB |

\* Peak RSS covers model load, warmup and inference in the isolated process;
it is not file size, tensor allocation size or GPU VRAM. Q8 takes 11–16% more
inference time than this Python reference and uses 53% less peak process RAM.
Q4 is smaller but slower than Q8 on this host; the default remains Q8.

All 28 compared frontend/encoder stages pass on this CPU runner: minimum cosine 0.999994 and tensor
norm-ratio error bounded by 0.066% (RMS error divided by reference RMS).
The archive covers mel, subsampling, 24 encoder layers and two encoder-output
checks. Predictor/joint numerical stage parity is not captured here; decoded
transcripts provide their end-to-end validation. The local shared-library run
also passes all 28 stages (minimum cosine 0.999996,
bound 0.065%) and repeated CTest/CLI/C ABI checks. The live guard includes an
explicit flash-off node trace.

A separate warmed Q8 diagnostic call reports mel 13.5 ms, encoder 1170.7 ms
and decoder 587.4 ms. Encoder graph build/allocation are only 0.51/0.47 ms.
The two FFN matmul shape groups account for 50.3% of traced encoder time;
fused Q/K/V accounts for 9.3%, flash attention 3.2%, and relative-position
matmul 2.2%. The Linux decoder remains scalar: its encoder projection alone
is 70.6 ms. The old trace called that path "cblas" incorrectly; the diagnostic
label is now corrected to "scalar" ("accelerate" on Apple builds).

These instrumented timings are for finding hotspots, not benchmark numbers.
The useful next experiments are optimized CPU predictor/joint matvecs and
encoder FFN kernels (including an explicit BLAS scheduler A/B), each with transcript and stage-parity A/B. Encoder graph
caching would save less than a millisecond in this trace and retains its known
correctness problem; it stays off. No new performance default was selected
from these measurements.

### macOS Metal CI coverage

The same successful run validates the Metal build, F16 stage parity and live
surfaces on an Apple M1 **virtual machine** (3 vCPUs, 7 GB). Its Apple Paravirtual
Metal device reports SIMD-group matrix multiplication unavailable. This is
useful Metal-path validation, not physical Apple GPU performance evidence.
With 4 inference threads, its medians are:

| Export | 11 s median / realtime | 55 s median / realtime | Peak process RSS |
|---|---:|---:|---:|
| F16 | 13.205 s / 0.83× | 35.467 s / 1.55× | 2,578 MiB |
| Q8_0 | 8.396 s / 1.31× | 15.167 s / 3.63× | 1,958 MiB |
| Q4_K | 8.404 s / 1.31× | 15.955 s / 3.45× | 1,653 MiB |

All 28 F16 stages pass (minimum cosine 0.999992, magnitude error bounded by
0.069%). Q8's warmed diagnostic uses Metal for the encoder and Accelerate for
the CPU decoder: encoder graph build/allocation 0.27/0.97 ms, versus 15.56 s
of instrumented encoder compute and 140 ms of decode. Callback overhead and
the virtual GPU prevent extrapolating these numbers to physical M1/M5 hardware
or comparing them against Fermion's 174× MLX result.

[The checked-in receipt](historical/phonon2-profile-2026-09-30.json) retains every raw
benchmark time, transcript, per-stage cosine/magnitude bound, model checksum
and pinned revision. The linked CI run additionally retains full wiring/live
logs, per-node traces, host details and Python dependency versions. Both Linux
and macOS jobs passed; the earlier run's missing Python `sentencepiece`
dependency was corrected before this measurement.

## Verified native CPU default (2026-10-01)

[Final CPU CI 36818985927](https://github.com/CrispStrobe/CrispASR/actions/runs/36818985927)
passes wiring/live guards, independent reference generation, all ten experimental
31-stage parity gates and warmed timing for the scalar baseline, selected default
and default plus single-thread encoder BLAS. AMD EPYC 7763, four vCPUs, four
inference threads; medians of three calls after shape-specific warmup. Loading
and instrumentation are excluded. The 55-second shape repeats JFK five times.

| Engine/export | Original, 11 s | New default, 11 s | Original, 55 s | New default, 55 s | New peak RSS |
|---|---:|---:|---:|---:|---:|
| F16 | 4.364 s | 3.704 s | 22.562 s | 19.393 s | 2,258 MiB |
| Q8_0 | 2.439 s | 1.782 s | 13.000 s | 9.856 s | 1,611 MiB |
| Q4_K | 2.500 s | 1.860 s | 13.287 s | 10.252 s | 1,315 MiB |
| Independent Python F32 | — | 1.798 s | — | 8.931 s | 3,406 MiB |

Q8 takes **27% less time on 11 seconds and 24% less on 55 seconds** than the
original scalar runtime (1.369×/1.319× speedup). It is approximately tied with
Python on the short shape and takes 10% more time on the longer shape, while
using 53% less peak process RAM. Q8 remains the recommended export: Q4 is
smaller but slower here. Every timed configuration's transcript matches its
same-quant scalar baseline at both lengths.

The selected default passes all 31 F16 rows against the independent Python F32
reference. Cosines below are printed to six decimal places; a value of 1.000000
does not imply bitwise equality. Magnitude bounds use relative RMS error with a
1% rounding margin and bound the global tensor norm, not every individual row.

| Stage group | Minimum cosine | Maximum global norm-error bound |
|---|---:|---:|
| Mel | 1.000000 | 0.00914% |
| Subsampling | 1.000000 | 0.02277% |
| 24 encoder layers | 0.999998 | 0.05417% |
| Encoder output (runtime/reference mel) | 0.999998 | 0.05789% |
| Full encoder projection | 1.000000 | 0.01531% |
| Production predictor SOS | 1.000000 | 0.00541% |
| Frame-zero joint logits | 1.000000 | 0.00532% |

Separate warmed Q8 traces report decoder 33.4 ms with backend projection,
versus 684.8 ms for the original scalar decoder; backend projection is 1.6 ms. These diagnostic
times are not benchmark medians. Encoder compute remains the main bottleneck;
The gated FFN experiments below target this bottleneck; encoder caching stays off.
Single-thread encoder BLAS regresses short Q8 to 2.940 s and long Q8 to 12.456 s
relative to the selected default. Its F16 long-clip improvement does not justify
a general default; retain the gated experiment and its corpus output changes.

[Cross-platform CI 36819484240](https://github.com/CrispStrobe/CrispASR/actions/runs/36819484240)
passes all 13 jobs, including 1,960 Linux unit tests, dynamic backend loading,
Vulkan, ASan, Windows, macOS, iOS and Android. Go/Rust binding checks, linkage
generator checks and the local auto/explicit/renamed/repeated/flash-off live guard
also pass. The [complete receipt](historical/phonon2-cpu-2026-10-01.json) retains raw repeated
times, all stages, exact transcripts, corpus comparisons, host details and model
checksums for both CPU sweeps.

## Gated FFN CPU experiments (2026-10-01)

Two alternatives are implemented, measured and **off by default**. Q8 remains
recommended. GGUF downloads are unchanged; the BLAS cache consumes runtime RAM.
Both experiments use the existing ggml graph and scheduler, with no new kernel
fork. Trace receipts verify actual FFN buffer types and backend placement.

The first sweep used an EPYC 7763 four-vCPU runner and four inference threads:

| Q4_K path | Warm 11 s | Warm 55 s | Peak RSS | Cold model load |
|---|---:|---:|---:|---:|
| Ordinary ggml | 1.850 s | 10.189 s | 1,316 MiB | 0.077 s |
| FFN CPU_REPACK | 1.508 s | 8.401 s | 1,318 MiB | 0.508 s |

Repacking reduces warmed inference time by **18.5% / 17.6%**, with approximately
the same RAM and an extra 0.43 s at load. At the pinned ggml revision, x86 Q8_0
has no CPU_REPACK kernel and F16 declines repacking; both fall back. Q8 at two
threads improves only about 5% / 3% over four, while F16/Q4 regress, so this does
not justify changing the global thread default. The
[first run](https://github.com/CrispStrobe/CrispASR/actions/runs/36830399630)
failed later in its prototype BLAS arm: only its completed CPU/repack arms are
used here. The scheduler correction keeps CPU last and explicitly pins all
non-FFN nodes to CPU and supported cached FFN matmuls to BLAS.

The corrected BLAS sweep used a different EPYC 9V74 four-vCPU runner. Compare
within this table, not across the two hosts. Public inference and BLAS thread
counts are both four:

| Export/path | Warm 11 s | Warm 55 s | Peak RSS | Cold model load |
|---|---:|---:|---:|---:|
| F16 ordinary | 3.966 s | 20.733 s | 2,247 MiB | 0.178 s |
| F16 cached FFN BLAS | 6.157 s | 11.918 s | 3,823 MiB | 3.880 s |
| Q8_0 ordinary | 1.746 s | 9.685 s | 1,609 MiB | 0.109 s |
| Q8_0 cached FFN BLAS | 5.964 s | 11.912 s | 3,190 MiB | 1.279 s |
| Q4_K ordinary | 1.856 s | 10.232 s | 1,316 MiB | 0.071 s |
| Q4_K cached FFN BLAS | 6.855 s | 12.081 s | 2,901 MiB | 1.224 s |
| Independent Python F32 | 1.921 s | 9.650 s | 3,419 MiB | — |

Cached BLAS cuts long F16 time by **42.5%**, but regresses short F16 and both
Q8/Q4 shapes. It caches 96 F32 matrices (1,536 MiB of added weight storage).
One- and two-thread BLAS alternatives also regress Q8/Q4. Fast isolated FFN
GEMMs therefore do not establish a whole-model win; the regression's cause has
not been isolated. These are three-call shape-warmed medians, excluding load
and instrumentation. The 55 s shape repeats JFK five times.

All three BLAS thread settings pass the strict 31-stage F16 reference gate:
minimum cosine at least 0.999997, maximum global norm-error bound 0.0569%.
The repack F16 gate exercises fallback, so it cannot validate Q4 repacking.
Against the same Q4 baseline, local repacking's final encoder cosine is 0.995941,
relative RMS error 3.54%, and maximum frame norm difference 1.90%; the strict
same-quant stage gate fails. Ordinary Q4 repeats are bit-exact, while isolated
repack GEMM differences are only about 4e-7 relative RMS. Those observations do
not prove the cause of the larger model-level difference.

For context, the 25 encoder stages captured with independent reference mel
also compare directly with Python F32. The following are diagnostics, not a
claim that quantized models meet the F16 acceptance thresholds:

| Path | Minimum stage cosine | Maximum relative RMS error |
|---|---:|---:|
| Ordinary Q8_0 | 0.999211 | 1.672% |
| Cached FFN BLAS Q8_0, one thread | 0.999962 | 0.422% |
| Ordinary Q4_K | 0.987905 | 7.083% |
| Cached FFN BLAS Q4_K, one thread | 0.993321 | 6.281% |

The local C ABI corpus covers 21 clips per export. Cached BLAS at one thread
preserves 61/63 original transcripts; both changes move toward Python F32.
Reference-exact counts for F16/Q8/Q4 become 21/20/16, versus 21/19/15. Human
word errors on the 19 labelled clips remain 13/272 for every export/path;
this is not a measured human accuracy improvement. Q4 repacking preserves all
words on 21 clips, with two punctuation changes and 19/21 exact transcripts.
Its CI JFK punctuation also changes. BLAS at two/four threads preserves both
JFK shape transcripts but has not received the full local corpus sweep.
Short encoder batches below the BLAS minimum correctly use cached F32 on CPU.
These checks do not establish all autoregressive states or word-timing parity.

[Corrected profiling CI](https://github.com/CrispStrobe/CrispASR/actions/runs/36834441110),
[all 13 cross-platform jobs](https://github.com/CrispStrobe/CrispASR/actions/runs/36834689343),
[lint](https://github.com/CrispStrobe/CrispASR/actions/runs/36834688811) and
[cross-ISA kernel probes](https://github.com/CrispStrobe/CrispASR/actions/runs/36834375261)
pass. The [complete receipt](historical/phonon2-ffn-cpu-2026-10-01.json) retains host/model
identities, raw timings, stage rows, corpus text and labelled word comparisons,
including rejected numerical experiments. These are CPU measurements; no GPU
or Apple performance improvement is claimed.

## CPU optimization controls

On Phonon-2 CPU builds with AVX2 and F16C, prediction and joint decoding use
persistent ggml graphs by default, including one bulk encoder-to-joint projection.
Apple retains Accelerate; other CPU models and instruction sets retain their
previous paths. `CRISPASR_PARAKEET_GGML_DECODE=0` restores the scalar decoder
on Linux. `CRISPASR_RNNT_GPU_ENC_PROJ=0` independently restores the CPU
projection. These controls also allow A/B comparisons against the original.

To select the experimental OpenBLAS decoder, set both
`CRISPASR_PARAKEET_GGML_DECODE=0` and `CRISPASR_PARAKEET_CPU_BLAS=1`.
It uses the cached F32 predictor LSTM and joint matrices, and batches the
invariant encoder projection. OpenBLAS development files must be present at
configure time and `CRISPASR_MEL_BLAS` enabled. `CRISPASR_PARAKEET_FORCE_SCALAR`
disables automatic CPU ggml selection and CPU BLAS, preserving the original
scalar path; an explicit `GGML_DECODE=1` still overrides automatic selection.
Set BLAS threading before starting the process; the decoder does not change the
process-wide OpenBLAS thread count for small matvecs.

`CRISPASR_PARAKEET_ENCODER_BLAS=1` registers the ggml BLAS backend before CPU
in the encoder scheduler. Unsupported operations keep CPU kernels. This
experiment requires a built/loaded ggml BLAS backend and stays off by default.
Quantized matmuls can pay extra dequantization costs. The public thread count
now reaches both CPU backend instances and optional encoder BLAS. The earlier
four-thread receipt already matched ggml's default of four; other requested
counts now take effect. `CRISPASR_PARAKEET_ENCODER_BLAS_THREADS=1` overrides
BLAS threads within the public thread limit. The ggml BLAS backend changes
OpenBLAS threading process-wide, also affecting a BLAS decoder; profiling
records the actual count after inference. Encoder caching remains off.

`CRISPASR_PARAKEET_FFN=repack` selects CPU_REPACK only for the four FFN
matmul weights in each encoder layer. Other weights use ordinary CPU buffers.
The loader checks type/shape/ISA support and falls back when no kernel exists;
at the pinned ggml revision, x86 accepts Q4_K but declines Q8_0 and F16. This
experiment stays off. Repacking changes in-memory layout, not the GGUF file.

`CRISPASR_PARAKEET_FFN=blas` converts the same FFN weights to F32 once per load
and explicitly assigns only their supported matmuls to the ggml BLAS backend.
Other encoder nodes stay on CPU; the scheduler retains its required final CPU
backend. Phonon-2 caches 96 matrices, adding 1,536 MiB of weight storage. Missing
BLAS falls back to ordinary CPU weights. Small batches unsupported by BLAS use
the cached F32 weights on CPU. This experiment also stays off. It requires a
built/loaded ggml BLAS backend and defaults to one BLAS thread;
`CRISPASR_PARAKEET_FFN_BLAS_THREADS` selects 1 through the public thread limit.
The BLAS thread setting affects process-wide OpenBLAS threading. FFN modes are
mutually exclusive; scoped FFN BLAS takes precedence over broad encoder BLAS.
Unset `CRISPASR_PARAKEET_FFN`, or use `ggml`, to retain ordinary encoder weights.

`CRISPASR_PARAKEET_FFN_TRACE=1` reports actual weight type, shape, buffer,
repack traits and scheduled backend for the FFN nodes. Use it in a separate
instrumented process, not in warmed timing. `CRISPASR_PARAKEET_DIFF_THREADS`
sets the stage harness thread count for thread A/B; inference uses the public
session/CLI thread setting as usual.

The manual `.github/workflows/phonon2-ffn-ab.yml` measures exact FFN shapes and
1/2/4-thread settings, independent Python timing, strict 31-stage F16 reference
gates, same-quant encoder diagnostics, warmed inference, load time and peak RSS.
Its `blas` sweep compares scoped BLAS against the ordinary four-thread default;
`full` adds CPU thread and repack configurations. Every alternative runs in its
own process, and traces prove the intended kernel/backend was selected. The
same-quant checker requires cosine, relative RMS and per-frame norm agreement
by default; the experimental sweep explicitly uses `--report-only` to retain
failed diagnostics without accepting those paths as a new default.

The CPU A/B workflow validates **31** frontend/encoder/transducer rows against
an independent Transformers F32 dump: the original 28 rows plus all encoder
projections, the raw predictor output after the production one-blank SOS, and
joint logits at frame zero using reference encoder activations. It checks cosine
and relative RMS error, which bounds global tensor norm-ratio error. The legacy
NeMo two-zero probe remains available separately. These probes do not capture
every autoregressive state; decoded-output checks remain required.

`.github/workflows/phonon2-cpu-ab.yml` offers a manual selected-default or full
experimental sweep. Each configuration runs in a separate process with warmed
11/55-second shapes and three timed repeats; diagnostic traces run separately.
The first full sweep is retained in
[CI run 36813752349](https://github.com/CrispStrobe/CrispASR/actions/runs/36813752349):
AMD EPYC 7763, four vCPUs, four inference threads. This is a different CPU from
the September 30 receipt, so compare paths within each run.

| Decoder / encoder configuration | Q8, 11 s median | Q8, 55 s median |
|---|---:|---:|
| Original scalar / ggml CPU | 2.440 s | 13.010 s |
| OpenBLAS, one thread / ggml CPU | 1.885 s | 10.307 s |
| OpenBLAS, four threads / ggml CPU | 1.852 s | 10.007 s |
| Persistent ggml, scalar projection / ggml CPU | 1.869 s | 10.304 s |
| Scalar / encoder BLAS, four threads | 6.826 s | 14.815 s |
| OpenBLAS / encoder BLAS, four threads | 6.705 s | 11.881 s |

All first-sweep configurations pass 31-stage parity and unchanged transcripts
for all three exports at both lengths. The OpenBLAS decoder cuts Q8 time by
24%. The ggml decoder is similarly fast but the separate trace still spends
82 ms in the scalar encoder projection. This motivates the bulk backend
projection used by the selected native CPU path. Both OpenBLAS and the ggml
path with backend projection also preserve all **63** original corpus outputs
(21 clips × F16/Q8/Q4). This is equality with the original runtime; original
Python transcript agreement is 21/21, 19/21 and 15/21 respectively.

The four-thread encoder BLAS experiment regresses short clips substantially
and stays opt-in. Its long F16 improvement does not justify a general default.
The dedicated final comparison additionally checks the native default and a
one-thread encoder BLAS experiment on the same host as their scalar baseline.

The single-thread encoder BLAS corpus experiment preserves 61/63 original
outputs: one Q8 name spelling and one Q4 comma change, both toward the Python
reference. Exact Python agreement becomes 21/21 F16, 20/21 Q8 and 16/21 Q4.
These are recorded as output changes, not accuracy regressions; the experimental
path remains separate from the default that preserves every original output.
