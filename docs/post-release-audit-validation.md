# Post-v0.8.49 optimization qualification

This procedure qualifies the repairs based on `26945b343b855dda9b0bce5e653cd5832c60bf7a`.
The local repair tests establish bounded CPU correctness. They do not establish
real-model quality, CUDA/Metal/Vulkan execution, or a performance ranking across
machines. The workflow and model driver below are prepared validation jobs;
their existence is not a successful external run.

## Platform tests ready to run

The manual workflow is
[post-release-audit.yml](../.github/workflows/post-release-audit.yml). It has no
push or pull-request trigger, does not publish artifacts to a release, and does
not download model weights.

| Profile | What it exercises |
|---|---|
| Linux scalar | Host kernels without CBLAS or OpenMP; std::thread fallback and one-thread paths |
| Linux CBLAS/OpenMP | The actual linked CBLAS iSTFT path, bounded BLAS scratch, and OpenMP worker budgeting |
| macOS Accelerate | Accelerate feature definitions/linking and numerical oracles; CPU execution, with Metal disabled |
| Windows MSVC/OpenMP | MSVC compilation, portable environment controls, and OpenMP numerical behavior |
| Linux shared CLI/C ABI | Haas aliases through the real CLI and public C ABI into the intended model loader, using a model-free fixture |

The numerical jobs build the changed runtime tests, RVQ, iSTFT, CQT,
diarization, worker-budget, LFR/CTC tests, eight A/B suites, and the allocation
probes. CBLAS/OpenMP jobs explicitly build their extra executables, so a missing
optional feature cannot silently qualify as a scalar-only success.

`tools/check_performance_test_discovery.py` compares the executable inventories
with actual CTest registrations. It requires the previously omitted LFR/CTC
cases and the heterogeneous A/B suites. The `performance-parity` label runs
numerical checks; the optional repeated timing tier remains separate. There is
no machine-independent minimum-speedup assertion in the correctness gate.

Run the workflow on a reviewed branch containing the repair commits, for example:

```bash
gh workflow run post-release-audit.yml --ref REVIEWED_BRANCH
```

The `gh` command is an optional convenience for a maintainer with repository
access; the same workflow can be selected in GitHub Actions. Nothing is
dispatched by the local repair package. Retain the source SHA, CMake cache,
discovery inventory, CTest log, compiler version, and failed-test logs from each
matrix leg. A failed platform leg must remain failed until its cause is fixed.

## Real models need a reviewed fixture bundle

The repository's regression manifest pins fixture revision
`70853acda3351e3456f7ed53bd5845a221a6de69`, but the audited manifest has **no
ArkTTS, Haas, or MeanVC2 backend entry**. Its note mentioning MeanVC2 does not
provide a model pin, expected output, or an acceptance gate for that backend.
Do not interpret the existing nightly suite as qualifying these models.

The existing stage runners also have narrower scope than complete inference:

- ArkTTS `crispasr-diff` checks prompt/prefill and isolated vocoder stages. It
  does not exercise several stateful autoregressive decoding steps.
- Haas's initial predictor/joint stages do not establish recurrent correctness
  over a full utterance or successful routing through every public entrypoint.
- MeanVC2's diff injects reference BNF/interpolation and reference mel into later
  stages. Passing those stages does not establish the composed pipeline using
  native BNF, native interpolation, real conditioning, and generated waveform.

The new
[model gate](../tools/ci-heavy/post_release_model_gate.py) runs those stage
checks and then requires additional raw outputs from a reviewed capture adapter.
It rejects missing files, mismatched hashes, missing stage names, zero CUDA
kernels, missing output captures, stale output directories, non-finite values,
shape mismatches, and failed transcript or numerical gates.

Start from
[manifest.template.json](../tools/ci-heavy/post-release-audit/manifest.template.json).
It is deliberately **unqualified and rejected by the validator**: the source
SHA, file hashes, reference revisions, and acceptance bounds are placeholders.
No real-model captures or accepted thresholds are fabricated in the template.

### What the bundle must contain

Put the manifest beside a `fixtures/` directory, or set `fixture_root` to a
relative directory beneath the manifest's parent. Every referenced path must
remain within that directory. The inventory pins every file by full SHA-256.
The model, reference archive, and original implementation each need a full
40-character immutable revision. Supply a pinned reference-environment lockfile
and the capture-adapter source itself.

| Backend | Required additional evidence |
|---|---|
| ArkTTS | Prompt K/V; at least three incremental logits, K, and V captures; complete generated waveform; independent ASR transcript of that waveform |
| Haas | At least three decoder and joint-network steps; complete utterance transcripts through CLI `haas`, `haas-ja`, `haas-asr`, and C ABI automatic/each explicit alias dispatch |
| MeanVC2 | Native BNF, its actual fourfold aligned interpolation, mel produced from those native features, complete converted waveform, independent speaker embedding of the generated waveform, and independent ASR transcript |

Use a short deterministic F16 profile first, then add longer/context-boundary
and quantized profiles as separate cases with separate pins and acceptance
bounds. Include multiple languages/speakers or longer audio where the intended
supported scope requires them. One accepted short case only qualifies that case.

### Capture adapter contract

Adapters are fixture-specific and must be implemented/reviewed alongside the
original-model captures. The template names `capture.py` files but does not
supply them: there are no accepted real-model fixtures here from which to infer
their original-model versions, prompts, noise, or expected outputs. This is a
remaining external qualification prerequisite, not a completed integration test.

The adapter is executed as an argument array without a shell. The first argument
must be the pinned executable adapter, or `{python}` followed by the pinned
Python adapter. Supported substitutions occupy whole arguments:

| Placeholder | Meaning |
|---|---|
| `{python}` | The validator's Python interpreter |
| `{file:ID}` | A verified file in the immutable inventory |
| `{bin:NAME}` | A built native executable under `build/bin` or `build/bin/Release` |
| `{library}` | The source-matched Linux `libcrispasr.so` |
| `{out}` | A fresh, case-specific output directory |

Adapters must invoke the candidate native runtime for the arrays and waveform
under test. Do not substitute reference tensors into a later native stage when
claiming complete-pipeline acceptance. Reference arrays are used by the
validator only after native capture completes. Preserve all generated waveform
bytes and intermediate arrays on the worker for independent review.

For ArkTTS, drive the actual slow-transformer prefill and incremental builders
with a fixed prompt/token history, capturing K/V after every step. The internal
test pattern in `tests/test-arktts-cache.cpp` demonstrates how to access those
builders without creating copied kernels. Also synthesize an entire utterance
through the public entrypoint. Include positions that exercise cached history;
comparing only the initial logits cannot detect the audited allocator bug.

For Haas, use a neutrally named model file for automatic C ABI detection and
every alias. Export successful transcripts from actual calls; an initialization
failure reaching the right loader is covered by the separate model-free test
and is insufficient here. Retain decoder state across at least three steps.

For MeanVC2, obtain the 256-float conditioning embedding from the original
WavLM/ECAPA speaker encoder and pin the exact bytes. The native runtime currently
has no reference-audio speaker encoder. `meanvc2_set_speaker_audio` returns `-2`,
and `meanvc2_convert` requires an explicitly set embedding. The standalone
harness now accepts it directly:

```bash
build/bin/test-meanvc2-convert model.gguf speaker-embedding.f32le source.wav converted.wav
```

The harness defaults to the first 25,600 source samples. Set
`CRISPASR_MEANVC2_ALL=1` when qualifying the complete source recording. Use mono
PCM16 at 16 kHz for this harness. The adapter should retain the public conversion
output in addition to any diagnostic stage replay.
Capture the interpolation from the native production helper; a separately
rewritten Python interpolation would not test the C++ composition being repaired.

The C++ and PyTorch normal RNGs do not become identical merely by using seed 42.
For strict DiT/intermediate comparisons, replay identical frozen noise. For the
public conversion waveform, use explicitly approved duration/amplitude bounds
plus independent intelligibility and speaker-similarity gates when the random
draws differ. Do not present those quality gates as samplewise waveform parity.

The gate accepts numerical arrays as `.npy` with object/pickle loading disabled,
and waveform captures as mono PCM16 `.wav` or numeric `.npy`. Numerical gates
require matching shapes, finite values, explicit maximum absolute error,
relative L2, and cosine thresholds. A nonzero speaker embedding can use a cosine-only
gate. Waveform comparisons cannot use cosine alone, which would miss amplitude
regressions. Waveform quality mode requires positive duration/RMS ratio ranges and a
finite peak bound; it also requires the separate transcript/speaker evidence
specified for the backend. Expected transcripts are pinned UTF-8 files;
comparison normalizes Unicode NFC and whitespace only.

Thresholds must be decided from the original implementation and intended
precision before inspecting candidate failures. Do not relax them merely to
make a new run green. Record the rationale in the fixture review.

## CUDA/model job

Provision a Linux x86-64 worker with label `crispasr-audit-gpu`, CUDA development
tools, CMake, a compiler, Python 3.10 or newer with NumPy, NVIDIA Nsight Systems (`nsys`), sufficient
RAM/VRAM/disk, and the reviewed fixture bundle. Provision the adapter's locked
reference/evaluator dependencies separately. The workflow does not install or
execute unpinned remote model code.

The model driver requires a clean tracked checkout whose commit exactly matches
`source_sha`. It verifies the fixture hashes before execution and again after
execution. Preflight is distinguishable from model acceptance:

```bash
python3 tools/ci-heavy/post_release_model_gate.py \
  --manifest "$FIXTURE_ROOT/manifest.json" --preflight-only --out model-audit
```

This produces `inputs_verified_only`; it does not claim any model ran. On the
worker, after building the candidate, the full command is:

```bash
python3 tools/ci-heavy/post_release_model_gate.py \
  --manifest "$FIXTURE_ROOT/manifest.json" \
  --build-dir build-model-audit --out model-audit
```

To dispatch the prepared job, set `run_models=true` and provide the absolute
worker-side manifest path. The model job waits for all numerical platforms and
the model-free Haas dispatch job. A missing manifest or prerequisite is an error.

ArkTTS and Haas stage/adapter commands run under Nsight Systems. The driver
requires a CUDA kernel summary with a positive instance count for each command;
GPU availability, a linked CUDA library, and allocated GPU memory are
insufficient. Keep the native inference and independent quality-evaluator runs
distinguishable in the adapter: evaluator CUDA work must not serve as evidence
that the target model ran on CUDA. Review the retained `.nsys-rep` process/kernel
trace to confirm attribution to native inference. The workflow uploads the
small summary CSVs; full traces remain on the worker.

MeanVC2 is explicitly recorded as **CPU host-kernel execution**, even on this
worker. Its GPU-selected weight allocation does not establish GPU inference.
There is no claim of a completed CUDA MeanVC2 port.

Each successful supplied fixture gate records native binary hashes, manifest
and input hashes, stage names, raw numerical metrics, transcript-gate names,
process timing boundaries, and CUDA kernel counts where required. Process
seconds include initialization and profiling overhead; they are not warm
steady-state inference latency. Raw captures remain available on the worker;
the workflow uploads receipts/logs without automatically publishing audio or
speaker embeddings.

The driver's model-free negative controls can be run independently:

```bash
python3 tests/test_post_release_model_gate.py -v
```

They validate the gate's rejection behavior. They are not substitutes for any
of the model, CUDA, macOS, or Windows runs described above.
