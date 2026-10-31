# Post-v0.8.49 audit repairs

This repair batch starts from `26945b343b855dda9b0bce5e653cd5832c60bf7a`, the
audited post-release head. The release comparison is against **v0.8.49**,
`d94264379dada437aa4ea9163eb34ffe4554422a`. It repairs demonstrated correctness,
memory, scheduling, dispatch, and test-discovery failures while retaining the
useful optimization work. It does not establish that every backend is free of
regressions on every model or device.

## Repairs and regression coverage

| Area | Repair | Production code and regression test |
|---|---|---|
| ArkTTS cached decoding | Mark K/V and hidden tensors read after graph execution as outputs before graph allocation. The allocator may otherwise reuse their storage after the last graph consumer. | [arktts.cpp](../src/arktts.cpp), [test-arktts-cache.cpp](../tests/test-arktts-cache.cpp) |
| VoxCPM2 convolution | Partition output time as well as channels, restoring useful parallelism for the final one-channel convolution. Each output retains the release's reduction order. Tile arithmetic also handles large dimensions without signed overflow. | [voxcpm2_cpu_conv.h](../src/core/voxcpm2_cpu_conv.h), [test-voxcpm2-cpu-conv.cpp](../tests/test-voxcpm2-cpu-conv.cpp) |
| CPU worker budgets | Require enough independent work per worker, honor context budgets, and use OpenMP where available. Cheap layer normalization, CQT/decimation, and related operations avoid creating unnecessary teams. Nested diarization candidates reuse the outer budget. | [parallel_for.h](../src/core/parallel_for.h), [test-parallel-for.cpp](../tests/test-parallel-for.cpp), [test-cpu-ops-ab-bench.cpp](../tests/test-cpu-ops-ab-bench.cpp) |
| iSTFT memory | Restore frame-at-a-time scalar processing and use bounded parallel/CBLAS tiles with the original overlap-add order. Frame-result/BLAS tiles use at most 256 KiB, or one frame if larger; scalar workers also own O(`n_fft`) spectrum scratch. Required output buffers still grow with audio length. | [istft.h](../src/core/istft.h), [test-core-istft.cpp](../tests/test-core-istft.cpp), [test-core-istft-memory.cpp](../tests/test-core-istft-memory.cpp) |
| RVQ near ties | Preserve the original F32 norm accumulation order. Keep each frame on one worker through all codebook stages, avoiding a new team per stage. | [rvq.cpp](../src/core/rvq.cpp), [test-core-rvq.cpp](../tests/test-core-rvq.cpp) |
| Diarization similarity | Compute each symmetric pair once and use the actual self-dot on the diagonal. Zero and tiny rows must not be assigned a fabricated similarity of one. | [spectral_diarize.cpp](../src/core/spectral_diarize.cpp), [test-spectral-diarize.cpp](../tests/test-spectral-diarize.cpp) |
| Haas aliases | Canonicalize the supported aliases consistently in registry lookup, CLI dispatch, and C ABI entrypoints/capability lookup. The model-free entrypoint test requires the intended loader's diagnostic, rather than accepting any failed load. | [backend_aliases.h](../src/core/backend_aliases.h), [test-registry.cpp](../tests/test-registry.cpp), [test_haas_dispatch.py](../tests/test_haas_dispatch.py) |
| MeanVC2 conditioning | Match `align_corners=True` BNF interpolation. Unsupported reference-audio conditioning returns `-2` and preserves an existing embedding; conversion requires an explicitly supplied finite embedding. The conversion harness accepts that embedding. | [meanvc2.cpp](../src/meanvc2.cpp), [meanvc2.h](../src/meanvc2.h), [test-meanvc2-conditioning.cpp](../tests/test-meanvc2-conditioning.cpp), [test-meanvc2-convert.cpp](../tests/test-meanvc2-convert.cpp) |
| Portable CBLAS | Supply headers, feature definitions, and linked CBLAS symbols through one interface target. Build separate scalar and actual-CBLAS iSTFT oracles. A scalar-only success does not qualify CBLAS. | [CrispASRCPUBlas.cmake](../cmake/CrispASRCPUBlas.cmake), [src/CMakeLists.txt](../src/CMakeLists.txt) |

Two additional bounded correctness fixes were found during implementation:
MeanVC2 now releases both owned weight partitions during destruction and on
initialization failure, and iSTFT includes the highest positive-frequency bin
for odd FFT lengths. The latter is a pre-existing general-helper defect, not a
new post-release optimization regression. The independent full-DFT oracle now
covers odd lengths 3, 5, and 21 as well as even production sizes. Invalid iSTFT
dimensions, output-length overflow, and negative SAME-padding widths return no
PCM.

## Test and benchmark repairs

[tests/CMakeLists.txt](../tests/CMakeLists.txt) registers numerical checks in the
`unit` and `performance-parity` tiers and repeated timing in the separate
`bench` tier. Six heterogeneous suites previously used adjacent Catch tags
that intersected unrelated test families and discovered no cases. The omitted
LFR and CTC cases are now included. List-valued CTest labels are applied after
Catch discovery by
[performance-test-properties.cmake](../tests/performance-test-properties.cmake),
avoiding the pinned Catch 3.7.1 helper's semicolon forwarding problem.

The [discovery guard](../tools/check_performance_test_discovery.py) compares
Catch's executable inventories with actual CTest selections. It requires all
eight A/B executables, the newly added runtime tests, LFR/CTC cases, and optional
CBLAS/OpenMP cases when requested. A matching executable name or empty CTest
selection is insufficient. Numerical tests explicitly reject nonfinite output;
the CQT checks include both real and imaginary results. FireRed/F5 tests now
call shared production kernels instead of copied implementations.

Repeated timing is enabled only by `CRISPASR_RUN_PERF=1`; machine-speed ratios do
not determine unit-test success. The repaired
[model benchmark](../tools/bench_mel_ab.py) accepts portable fixture paths,
requires nonempty cases and expected transcripts, checks every run, explicitly
removes inherited serial controls from its accelerated arm, and fails missing
or mismatched output. It distinguishes process timing, which includes model
loading, from a reported inference timer. Its rejection behavior is covered by
[test_bench_mel_ab.py](../tests/test_bench_mel_ab.py).

## Optimizations retained

The batch retains the useful sparse/precomputed frontend work, zero-copy LFR
assembly, hashed CTC prefix lookup, attention/kernel locality improvements,
weight and Fourier-basis caches, graph allocation, and CBLAS inverse transforms.
The changes constrain scheduling and memory use and restore required numerical
or lifetime semantics. They do not roll back the entire post-release series.
The RVQ norm reassociation is deliberately removed because its changed rounding
can change discrete codes; parallelism across independent entries/frames is
retained.

Context budgets now reach the repaired helpers from their production callers,
including Granite, FireRed/F5, VoxCPM2, SenseVoice/Paraformer/Wav2Vec2,
Kokoro/Roformer, Kyutai/Mimo/Omnivoice, BasicPitch/TabCNN/BTC, and the affected
TTS iSTFT callers. These budgets govern the runtime's own workers. Vendor BLAS
thread pools remain separately controlled by that library's settings. A general
reusable worker pool and broad weight-prepacking redesign remain separate work.

## Bounded local evidence

The raw repair receipts distinguish component checks from final integration.
These are synthetic CPU observations, not real-model or GPU acceptance:

- ArkTTS: **17,936 assertions** passed across four head/prompt configurations,
  each with four incremental decode steps. The unchanged audited head fails
  the same stateful negative control. Checks include logits, hidden state, and
  every retained prefix K/V value against full-prefix recomputation.
- VoxCPM2: for F32 `Cout=1`, `Cin=32`, `K=7`, `T=48000`, four OpenMP workers,
  medians were **2.837732 ms** at the release, **7.331098 ms** at the audited
  head, and **2.717691 ms** after repair. The 15 measured calls per arm rotated
  order after two warmups; layout transposes and their allocations are included.
  All compared outputs were finite and exactly equal. This is one isolated
  production kernel on a shared CPU worker.
- iSTFT memory: at `N=64`, `T=5000`, the forced-serial audited implementation
  used **1,920,904** peak live C++ allocation bytes and failed the bound. The
  repair used **960,576** bytes and produced the same **79,984-sample** length.
  The probe warms the immutable Fourier basis and counts C++ allocation
  payloads; these figures are not process RSS or BLAS-private allocation totals.
- The repaired scalar and Linux CBLAS iSTFT suites each passed **254,026
  assertions**, including independent full-DFT checks, odd lengths, trim
  boundaries, and multi-tile output. Separate RVQ and spectral edge tests pass.
- **738 release-to-candidate parity cases** passed exactly: alignment 90, CTC
  beam search 216, Kaldi fbank 48, LFR 192, mel 96, and resampling 96.
- A separate three-way LayerNorm diagnostic used three warmups and 21 rotated
  measurements per arm. For 16 frames of 1,024 floats without OpenMP, medians
  were **28.983 us** at the release, **166.120 us** at the audited head, and
  **8.593 us** after repair with a four-worker budget. Cheap repaired work stays
  serial. Across 2/16/64/512 frames in portable and OpenMP profiles, all outputs
  were finite and the largest release difference was **1.1920929e-6**. These
  shared-worker timings demonstrate the tested scheduling effect, not a
  universal end-to-end speedup.

### Integrated build and test status

The implementation and test sources are recorded in local commit
`1b5a16c40118a3b15d53c7dd6bbc66c7458bc203`. The subsequent documentation commit
and the final source-tree identity are recorded in the downloadable bundle's
manifest. No upstream branch or external workflow was published or launched.

The final Linux CPU configuration uses GCC 13.3, the pinned ggml submodule,
OpenMP, and the available Netlib CBLAS implementation. It completed:

- The shared library, actual CLI, diagnostic CLI, MeanVC2 conversion harness,
  and all five affected CBLAS-enabled backend library targets.
- **131/131 selected CTest cases passed**, with zero failures or skips. The
  actual discovery guard separately required **47 cases in 13 executables**,
  including all eight A/B suites and the CBLAS/OpenMP variants.
- Actual Haas entrypoints: **14 CLI/C ABI dispatch checks** and **7 capability
  lookups** passed. The full registry executable passed **7,280 assertions in
  73 cases**. The metadata-only dispatch fixture does not qualify inference.
- **11 benchmark control tests** and **15 external model-gate control tests**
  passed inside that integrated CTest run.
- The Go cgo dependency list matched a fresh static CPU/Opus dependency graph.
  This checks generated linkage consistency, not a Go runtime execution.
- Clang-format 18 accepted all 62 changed C/C++ files; whitespace checks passed.

The source bundle retains the final JUnit XML, native CTest log, discovery
inventory, commands, return codes, and build-product hashes under
`evidence/final-integration/`. Component negative controls and raw diagnostic
results are retained separately. The initial parallel full-build attempts hit
truncated assembler output; the completed full build used one compile worker.
The backend UBSan checks passed. An earlier ASan/UBSan test body passed, but its
process failed during LeakSanitizer's inaccessible `/proc` task scan; this is
not reported as a complete ASan qualification.

## Reproduce the numerical gate

From a checkout containing this repair batch, initialize the pinned submodules
and configure a CPU scalar profile. Install the repository's usual C++/CMake
build prerequisites; Catch2 3.7.1 is fetched at configure time unless supplied
with `FETCHCONTENT_SOURCE_DIR_CATCH2`.

```bash
git submodule update --init --recursive
cmake -S . -B build-audit -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=OFF -DCRISPASR_BUILD_TESTS=ON \
  -DCRISPASR_BUILD_SERVER=OFF -DGGML_NATIVE=OFF \
  -DGGML_CUDA=OFF -DGGML_METAL=OFF -DGGML_BLAS=OFF \
  -DGGML_OPENMP=OFF -DCMAKE_DISABLE_FIND_PACKAGE_OpenMP=TRUE \
  -DCRISPASR_CPU_BLAS=OFF -DCRISPASR_MEL_BLAS=OFF
cmake --build build-audit --config Release --parallel 2 --target \
  test-arktts-cache test-meanvc2-conditioning test-meanvc2-convert \
  test-voxcpm2-cpu-conv test-core-istft test-core-istft-memory \
  test-core-rvq test-core-cqt test-spectral-diarize test-diarize-smooth \
  test-parallel-for test-kaldi-fbank test-ctc-stitch \
  test-align-ab-bench test-conformer-ab-bench test-cpu-ops-ab-bench \
  test-firered-ab-bench test-voxcpm2-ab-bench test-kokoro-diarize-ab-bench \
  test-cqt-diarize-ab-bench test-rvq-istft-ab-bench
python3 tools/check_performance_test_discovery.py --build-dir build-audit
ctest --test-dir build-audit -C Release -L performance-parity \
  --output-on-failure --timeout 300 --no-tests=error
```

Use a **separate build directory** for CBLAS/OpenMP. Set `CRISPASR_CPU_BLAS`,
`CRISPASR_MEL_BLAS`, and `GGML_OPENMP` to `ON` and omit the
`CMAKE_DISABLE_FIND_PACKAGE_OpenMP` override. Supply a CBLAS development package
(or Accelerate on macOS); a successful configure with scalar fallback is not a
CBLAS result. In addition to the target list above, build
`test-core-istft-blas`, `test-core-istft-memory-blas`, and
`test-parallel-for-openmp`. Then require those implementations in discovery:

```bash
python3 tools/check_performance_test_discovery.py --build-dir build-audit-blas \
  --require-blas --require-openmp
ctest --test-dir build-audit-blas -C Release -L performance-parity \
  --output-on-failure --timeout 300 --no-tests=error
```

The standard macOS profile uses Accelerate with OpenMP disabled; omit its
OpenMP target and `--require-openmp`. For the actual Haas entrypoint test,
configure a Linux build in `build-audit-entrypoints` with
`BUILD_SHARED_LIBS=ON` and `CRISPASR_BUILD_EXAMPLES=ON`, then run:

```bash
cmake --build build-audit-entrypoints --config Release --parallel 2 \
  --target crispasr-cli crispasr-lib
ctest --test-dir build-audit-entrypoints -C Release \
  -R '^haas-dispatch-entrypoints$' --output-on-failure --no-tests=error
```

Its metadata-only fixture proves dispatch to the intended loader; it
intentionally cannot prove successful model loading or transcription.

## External qualification remains required

Follow [post-release-audit-validation.md](post-release-audit-validation.md) and
the prepared manual
[qualification workflow](../.github/workflows/post-release-audit.yml) for macOS,
Windows, real CUDA execution, and pinned model fixtures. Workflow preparation
and validator self-tests are distinct from successful external execution.

Real-model acceptance still needs stateful ArkTTS generation, complete Haas
utterances through public entrypoints, and composed MeanVC2 conversion with
native BNF/interpolation/mel plus an independently derived speaker embedding.
MeanVC2 reference-audio speaker extraction is unsupported, and its current
host-kernel execution is not a GPU port. Broader WER/CER, perceptual speech,
speaker similarity, long-context behavior, quantization, and cross-hardware
performance require representative fixtures and their own recorded gates.
