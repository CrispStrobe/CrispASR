# OmniVoice runtime and acceptance audit — 2026-10-22

Audit inputs: remote main `22e1b6ada4cc8eced45f30bf967f614f19f360fe`
and codec branch `f76d4b190a109ba17b6854a71bcd0e3ae6dc4b22`.
PR #521 was already merged on main; the four codec optimization commits were
still on `perf/omnivoice-codec-decomp`. Older green branch checks do not certify
its latest source.

## Findings and disposition

- **Split-off changed ordinary cloning placement.** The voice-prompt cache-miss
  path unconditionally rebound the tokenizer to CPU. Only explicit split mode
  now changes placement; CUDA retains its existing ordinary GPU-codec policy.
- **Reference cache could hide backend differences.** Its key did not distinguish
  CPU/GPU encoder policy. The cache namespace is versioned and includes encoder
  backend placement, so old unclassified entries cannot certify a CPU/GPU A/B.
  Diagnostic comparisons must also disable the cache to exercise fresh encoding.
- **Repeated and failed loads leaked or retained invalid state.** Replacement
  loads now release previous allocations, failed/partial loads release their
  resources, and cleanup clears all borrowed encoder/decoder tensor pointers.
  A failed replacement leaves the tokenizer unloaded; it does not preserve the
  previous tokenizer. The same cleanup handles destruction and experimental
  placement changes. GPU-request policy is recorded separately from actual
  backend identity so CPU fallback does not cause repeated reloads.
- **Optional pre-permutation had an unsafe allocation-failure fallback.** The
  shared helper published tensors before buffer allocation and freed their
  context on failure. It now publishes only after success and refuses unsupported
  input dtypes. The fallback cannot consume dangling permutation pointers.
- **Decomposition retained an unused 40,026,112-byte F32 copy.** Prepare the
  permuted kernels first, then bake crop kernels only where permutation is
  absent. Failure still gets the established crop fallback. Decoder source
  tensors are about 44.2 MB, but current F32 caches add about 80.9 MB. A resident
  decoder therefore needs roughly 125 MB before graph scratch, not 50 MB.
- **The default promotion was not supported by consistent latency results.**
  Decomposition remains off on every backend unless explicitly opted in with
  `CRISPASR_OMNIVOICE_CODEC_DECOMP=1`. The counterbalanced CPU probe reduced CPU
  work but did not establish a consistent wall-time win. CUDA/Metal/Vulkan also
  need their own validation. Split remains off by default.
- **The cloning campaign selected a passing reference after observing output.**
  `station` met the declared `min(reference WER) <= 0.2` rule but was excluded
  because its clone failed. The corrected harness tests every eligible reference
  against both cross-sentence targets. Historical `window` results remain valid
  measurements of that subset, not acceptance of the complete campaign.
- **Several conclusions exceeded the evidence.** Six matching printed digits
  do not prove array identity. A bisect whose CPU intermediates disagree with
  the reference does not isolate a Metal kernel. ASR agreement raises a fixture
  concern without proving pronunciation causality. Two cold/warm timing pairs
  do not establish a fixed break-even count. Documentation now states these
  limits. ASR intelligibility also does not measure speaker similarity.

## Verification scope

The fixture pin resolves in the **model repository**
`cstr/crispasr-regression-fixtures`, revision
`d0a7d7a8be318a5841dfdbe6ad37d3acf75523e3`; the dataset repository with the same
name is different. Local fixture bytes match the historical manifest hashes.
Local model bytes match the public metadata at OmniVoice revision
`92266739c4496b04a497e164cf8c2801a792551e`:

| File | SHA256 |
| --- | --- |
| `omnivoice-q8_0.gguf` | `9d8835c80d50cce6ba66bed60c31e94836a7f990c40d2e22496c0b6710c98bb5` |
| `omnivoice-tokenizer-f16.gguf` | `710ef610e1f2845c6b7333d5432376b24f2d20d2c54a8cec9bc118d183ecea63` |

The native regression includes the actual runtime and uses tiny synthetic GGUF
weights. It covers replacement, missing/truncated input, recovery, the split-off
placement guard, CPU decomposition controls and a forced backend allocation
failure. An independent run with AddressSanitizer and UBSan passed these paths.
The existing output-allocation test passed five normal/OOM/gain cases and its
four historical/mutated failure controls. The new placement-guard mutation and
the literal historical permutation helper both fail the native regression as
expected. CTest executes four selected tests (17 permutation assertions across
two cases, native lifecycle, and four Python acceptance-policy cases).
Historical acceptance is marked rejected while retaining measured per-arm data.
Completed roundtrip scoring writes an aggregate receipt, including failed WER
gates, before exiting nonzero.

## Fresh live evidence

Two references (`window`, `station`) each generated two cross-sentence targets
at seed 42 / 32 steps. Each target's codes were generated once and decoded with
both CPU upsamplers. Across all four pairs, PCM lengths agree, maximum error is
one PCM16 step (1/32768), and both pinned ASR judges produce identical transcripts
between crop and decomposition. Both `window` crop PCM hashes exactly match the
historical receipt, independently preserving that baseline.

| Reference / target | Nemotron WER, both paths | Parakeet CTC WER, both paths |
| --- | --- | --- |
| window / fox | 0.111 | 0.111 |
| window / walk | 0.000 | 0.000 |
| station / fox | **0.222** | 0.000 |
| station / walk | 0.000 | 0.000 |

The established Nemotron gate therefore remains failed on `station`. CTC
agreement with the target is important contrary evidence: this does not, by
itself, establish a synthesis defect or justify changing the judge/threshold.
A matched original/native comparison on eligible references remains needed;
intelligibility and speaker similarity are separate questions.

A second decoder-only probe used separate correctly prepared contexts, one
warmup per arm and three measurements per arm in ABBAAB order for each clip.
All 32 outputs were byte-exact to their respective first-run baselines after
the cache reduction. The cache totals are 80,888,704 bytes for crop, or
40,862,592 baked + 40,026,112 permuted bytes for decomposition.

| Clip | Crop median wall ms | Decomp median wall ms | Crop / decomp median CPU ms |
| --- | ---: | ---: | ---: |
| window / fox | 2821 | 2876 | 7091 / 6727 |
| window / walk | 3184 | 3443 | 8173 / 8056 |
| station / fox | 1698 | 1968 | 3673 / 3536 |
| station / walk | 2269 | 2018 | 4645 / 4075 |

The shared machine remains noisy. These observations support lower CPU work
but **do not support a consistent latency win**, so decomposition stays off by
default on CPU too. Keeping it opt-in avoids turning an uncertain performance
claim into user-visible default behavior.

The first live timings are single fixed-order pairs and are illustrative.
Reload cost is separate from decode, which includes output DSP. Detailed norms,
PCM/code/model hashes, transcripts and timing boundaries are in
[the live receipt](omnivoice-audit-live-2026-10-22.json).

The POSIX driver `tools/ci-heavy/omnivoice_codec_ab.cpp` takes main GGUF,
tokenizer GGUF, reference WAV (or `none`), reference text and output prefix.
It saves each code array once and both float32 PCM outputs. Build it against
the current `omnivoice`, `crispasr-core` and ggml libraries. For a regular
shared CPU build (set `BUILD` to its directory):

```sh
cmake --build "$BUILD" --target omnivoice
c++ -std=c++17 -O3 -Isrc -Iggml/include tools/ci-heavy/omnivoice_codec_ab.cpp \
  -L"$BUILD/src" -lomnivoice -lcrispasr-core -L"$BUILD/ggml/src" \
  -lggml -lggml-cpu -lggml-base -Wl,-rpath,"$BUILD/ggml/src" -o codec-ab
CRISPASR_OMNIVOICE_CODEC_GPU=0 CRISPASR_OMNIVOICE_CODEC_SPLIT=0 \
  CRISPASR_TTS_REF_CACHE=0 ./codec-ab MAIN.gguf TOKENIZER.gguf window.wav \
  "Please close the window before you leave the room." out/window
```

Create the output directory first. GPU-library builds may require their own
backend link flags. These probes do not establish general cloning acceptance,
CUDA/Vulkan decomposition parity, or removal of every onset transient.

The decoder-only replay is `tools/ci-heavy/omnivoice_codec_decode_bench.cpp`.
It takes the tokenizer and a proof directory containing both reference WAVs,
code arrays and expected PCM files from the first driver. It includes the actual
runtime and needs `crispasr-core` and ggml libraries (do not also link
`omnivoice`). It checks byte equality on every warmup and measured decode.

## Hosted CPU follow-up

The first repair campaign on Linux CPU completed at source `b608bcdd1`:
[run 39857487336](https://github.com/CrispStrobe/CrispASR/actions/runs/39857487336).
Its independently pinned utility oracle passes and the actual raw-session DSP
comparison has maximum error 0.0. The complete nine-arm speech gate **fails**:
`window` primary, marked clone and CLI clone score 2/9 ("Quick brown ... lazy
dot"), while both `station` targets score zero. This differs from the M1 pattern;
fixture selection cannot establish general acceptance or isolate the cause.

Artifact `12459969252` was downloaded and checked, ZIP SHA256
`ce4aa4c1461de0304fcb3367d04601c2ced6043cdfc57945d5e5edefe6e0d7fa`.
The live receipt retains its transcripts and component results. It predates the
cache/default corrections and does not certify current runtime source `5acff9f21`.
The current-source full campaign is
[run 39861560201](https://github.com/CrispStrobe/CrispASR/actions/runs/39861560201),
queued at harness source `f87af58aa` (runtime code `5acff9f21`). The earlier
queued39860613583 was cancelled after adding per-surface code capture.
CI/lint for the latest main tip remain pending.

The current harness retains generated code arrays and hashes for every CLI/
session arm, including separate marked outputs. Marking-only comparisons must
preserve codes. This lets follow-up work separate generation drift from decoder,
DSP and judge differences instead of inferring the cause from a transcript.

To reproduce the reported decomposed GPU experiment, explicitly set both
`CRISPASR_OMNIVOICE_CODEC_SPLIT=1` and
`CRISPASR_OMNIVOICE_CODEC_DECOMP=1`. Split alone changes placement and retains
the crop upsampler; it is not a promise of faster decode or total latency.
