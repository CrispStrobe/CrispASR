# SCNet and SCNet-large

The `scnet` backend separates a stereo mixture into `drums`, `bass`, `other`,
and `vocals` at 44,100 Hz. Both official model sizes use the GGUF architecture
name `scnet`; the model metadata identifies the size. A concrete converted
model path is required.

## Implementation and qualification status

The native graph, converter, original-reference capture, quantizer policy,
CLI, HTTP separation route, and Session C ABI are implemented. The isolated
native library builds, and actual F16/Q8_0/Q4_K synthetic-weight graphs pass
finite-output, geometry, repeated-call, changed-input, and result-lifetime
checks. These controls establish execution and state handling. They do not
establish checkpoint fidelity, source-separation quality, or speed.

The hosted qualification checks the author's original checkpoints against
the original Python implementation. Until its recorded numerical, production
surface, and performance gates pass, this backend remains a candidate. GPU
qualification is separate from the CPU campaign. No unqualified checkpoint
is registered for automatic download.

The source is pinned to
[`starrytong/SCNet@5d95bf96b19c3eede63248d171efeca8e3abb948`](https://github.com/starrytong/SCNet/tree/5d95bf96b19c3eede63248d171efeca8e3abb948).
The converter and original-reference loader verify the source-file hashes
before importing upstream code. The original `SCNet.forward` and
`Seperator.separate_music_file` methods execute unchanged during reference
capture. Hooks and a line observer collect actual intermediate tensors and
normalization statistics; they do not substitute reference calculations.

### Current measured baseline

Both model sizes in [run 41403856227](https://github.com/CrispStrobe/CrispASR/actions/runs/41403856227)
at source `94ea189e4a95bbe966f9a3410a83c5a4e82f4c6b` pass all 35 complete-pipeline
stages, every timed waveform check, and production-surface checks in each
precision. Both fail performance, using two threads for each implementation
and the same 24-second overlapping-chunk fixture. Small ran on AMD EPYC 7763;
large ran on Intel Xeon Platinum 8370C, so comparisons are within each worker:

| Model | Precision | Native median | Original median | Original/native speed ratio |
|---|---|---:|---:|---:|
| SCNet | F16 | 46.308 s | 16.427 s | 0.355× |
| SCNet | Q8_0 | 46.534 s | 16.154 s | 0.347× |
| SCNet | Q4_K | 46.640 s | 15.696 s | 0.337× |
| SCNet-large | F16 | 146.481 s | 49.519 s | 0.338× |
| SCNet-large | Q8_0 | 146.895 s | 50.140 s | 0.341× |
| SCNet-large | Q4_K | 147.120 s | 50.377 s | 0.342× |

These are three alternating timing pairs after one warmup. The original F32
checkpoint is the reference in each comparison. This source predates the four
final-matrix quantization repair; the result does not qualify the current policy.
The [small baseline](historical/scnet-runtime-baseline-2026-11-02.json) and
[large baseline](historical/scnet-large-runtime-baseline-2026-11-02.json) bind the
source, models, fixture, metrics, artifact and diagnostic logs. Fresh full
qualification is required after runtime changes.

A [focused diagnostic](https://github.com/CrispStrobe/CrispASR/actions/runs/41411681073)
at `b51801b` produces the exact same full F16 output PCM as the small baseline.
It attributes 18.225 seconds to recurrent gates, 3.981 seconds to recurrent
GEMMs, 6.288 seconds to other BLAS matrices, and substantial remaining time to
copy/layout/broadcast operations. The [profile receipt](historical/scnet-runtime-profile-2026-11-02.json)
retains nested timer scopes and build flags. It is a synchronized diagnostic,
not a replacement throughput qualification.

The subsequent SIMD and CPU layout changes pass all 35 stages against the
unchanged original on the 0.25-second fixture for both sizes and all three
formats. With scalar gate math selected, all 35 native F16 payloads also match
the earlier captures byte-for-byte in both sizes. The
[optimization controls](historical/scnet-runtime-optimization-controls-2026-11-02.json)
separate those layout identity checks from SIMD numerical checks and retain the
short-input limitation. Hosted full-pipeline timing and qualification remain
required for the changed runtime.

## Inference behavior

Default inference follows the published helper's 20-second segments, 50%
overlap, triangular blending, waveform normalization, and one independently
drawn random shift per overlapping chunk. Shift padding includes the real
preceding context from the full recording. CrispASR resets a Python-compatible
random sequence for each separation call, using seed zero by default; the
CLI's `--seed` selects another repeatable sequence.

The raw model path preserves the original normalized rectangular STFT,
reflective centering, parity-dependent right padding, unbiased standard
deviations, recurrent gate ordering, and orthonormal internal FFTs. The output
has exactly the input frame count. The direct native `no_segment` option is a
diagnostic call to the raw model: it bypasses the outer waveform normalization,
chunking, shifts, and overlap assembly. It is not the default performance
comparison.

Convolutions, learned projections, normalization, and transposed-convolution
scatter use ggml operations. Exact-length FFTs and sequential recurrent loops
use native CPU graph nodes. On a build with CBLAS, recurrent matrix operations
use that provider; the portable implementation remains available. Backend
reporting includes `CPU-LSTM-FFT`, including when an accelerator handles other
operations. A selected GPU does not imply that the entire model runs there.

On CPU, layout nodes use row copies, blocked permutations, and disjoint
convolution-column blocks. Already contiguous inputs are reused where an
operation only requires layout; diagnostic captures still retain independent
copies. Recurrent gate updates select an AVX2/FMA or AArch64 NEON F32 kernel
when available and otherwise use the scalar implementation. The reported
compute description identifies the selected gate kernel. These changes retain
F32 state and the same numerical gates; platform qualification remains limited
to the hardware in the accepted receipts.

## Build

Use the normal CrispASR build for the CLI and Session library. The optional
probe provides the same production backend in a smaller comparison executable
and shared library:

```bash
cmake -S . -B build-scnet -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=ON -DCRISPASR_BUILD_SCNET_PROBE=ON \
  -DCRISPASR_BUILD_TESTS=OFF -DGGML_BLAS=ON \
  -DGGML_BLAS_VENDOR=OpenBLAS -DCRISPASR_CPU_BLAS=ON
cmake --build build-scnet --parallel 2 \
  --target crispasr-cli crispasr-lib crispasr-quantize scnet-probe scnet-validation
```

The opt-in probe enables position-independent dependencies for static builds
as well. An unavailable CBLAS provider selects the portable recurrent path;
timing receipts must identify the actual provider and thread budget.

## Convert the author's checkpoint

Install a platform-appropriate `torch==2.8.0` and the packages in
`tools/requirements-scnet.txt`. Obtain the pinned upstream source and the exact
checkpoint linked in its README:

| Variant | Author checkpoint | Configuration |
|---|---|---|
| SCNet | [Official download](https://drive.google.com/file/d/1CdEIIqsoRfHn1SJ7rccPfyYioW3BlXcW/view) | Pinned `conf/config.yaml` |
| SCNet-large | [Official download](https://drive.google.com/file/d/1s7QvQwn8ag9oVstGDBQ6KZvacJkvyK7t/view) | [Author's large configuration](https://drive.google.com/file/d/1qxK7SZx6-Gsp1s3wCrj98X7--UcI4O3K/view) |

The variants also differ in frequency bands. SCNet uses
`[0.175, 0.392, 0.433]`, with encoder frequency counts `2049 → 616 → 186 → 57`.
The author's large configuration uses `[0.225, 0.372, 0.403]`, giving
`2049 → 705 → 243 → 85`. The native loader validates these variant contracts
and uses the original double-precision split calculation.

The author-linked downloads observed in
[run 41397985492](https://github.com/CrispStrobe/CrispASR/actions/runs/41397985492)
have these SHA-256 digests:

| Artifact | Recorded SHA-256 |
|---|---|
| SCNet checkpoint | `f8aff3bc8dd816aad392800d9d69519add719a2fbd22550af6f8cb2138d25ff5` |
| SCNet-large checkpoint | `719e5abb8ed920305dad546ac3cd6fb0b1e9c3092d14ce21827bfc0423af3070` |
| Large configuration | `629a4901184bf1d3a75b0b13904f35974785aa042cad3c010fd576248cdce3f0` |

Record the SHA-256 of that download before conversion and supply it explicitly:

```bash
python models/convert-scnet-to-gguf.py \
  --source-dir /path/to/pinned-SCNet \
  --checkpoint /path/to/scnet-author.th \
  --checkpoint-sha256 RECORDED_SHA256 \
  --variant scnet --output scnet-f16.gguf

build-scnet/bin/crispasr-quantize scnet-f16.gguf scnet-q8_0.gguf q8_0
build-scnet/bin/crispasr-quantize scnet-f16.gguf scnet-q4_k.gguf q4_k
```

For the large model, use `--variant scnet-large` and pass its downloaded
configuration with `--config`. The author has not published a checkpoint
checksum. A digest discovered on the first author-linked download identifies
the tested bytes; it must not be described as independently author-attested.

The converter requires all 372 tensors in the checkpoint's `best_state`,
preserves original names, rejects unsupported shapes and nonfinite values,
and uses PyTorch's restricted tensor loader. Matrix storage is F16; biases and
normalization vectors remain F32. Convolution weights are flattened in original
kernel order. Transposed-convolution kernels are rearranged to the native
projection-and-scatter layout. Layout version 1 is mandatory in the loader.

Q4_K-ineligible rows use Q8_0 when their width permits it. Rows narrower than
32 retain source precision. SCNet never silently substitutes Q4_0 for Q4_K.
Both quantized formats retain the final spectral decoder's four matrices in
F16: `decoder.2.0.conv.weight` and `decoder.2.1.convtrs.{0,1,2}.weight`.
Quantizing these reconstruction matrices caused the quiet bass stem to fail
the unchanged global gates in the initial short-input experiment. Restoring
only these matrices passes all 35 stages for both formats while leaving every
other tensor byte-identical. For small SCNet this preserves 48,640 parameters,
adds 45,600 bytes, and retains target-type matrix coverage of 99.4469% Q8_0 and
76.5695% Q4_K. See the [quantization diagnosis](historical/scnet-quantization-diagnosis-2026-11-02.json)
for failed defaults and the bounded repair; full qualification remains a
separate requirement.
The validator reports actual tensor and element coverage; a filename alone
does not establish a quantization type. Precision exceptions require measured
fidelity evidence before qualification.

## CLI and Session

```bash
build-scnet/bin/crispasr --separate \
  -m scnet-f16.gguf -f mixture.wav --seed 0 --sep-output-dir separated

build-scnet/bin/crispasr --separate \
  -m scnet-q8_0.gguf -f mixture.wav --stems vocals --sep-output-dir vocals
```

The shared separation dispatcher decodes and resamples to the model rate.
It writes stereo PCM WAV files with the selected stem names. The GGUF
architecture detects SCNet even if the file is named `weights.gguf`.

The existing `Session.separate()` surface is used by bindings; no
transcription method or synthetic transcript is involved. The C ABI exposes
the same source order, frame count, and sample rate. Native result views live
until replacement or session close; language wrappers that return owned arrays
retain their own data. See [the shared separation surface](source-separation-surface.md)
for the HTTP endpoint and wrapper contract.

## Numerical and performance gates

`tools/dump_scnet_reference.py` captures exactly 35 stages from the original
implementation: input and waveform statistics; the first real shifted chunk;
STFT and spectral normalization; every encoder, skip, dual-path, feature-FFT,
and decoder output; raw chunk stems; and complete assembled stems.

`tools/validate_scnet.py` independently parses the native and original captures
and requires exact shapes, complete stage sets, finite values, and amplitude
agreement. The full-tensor gates were fixed before the pretrained-model run:

| Precision | Global cosine | Norm ratio | Relative L2 | F16 row gate |
|---|---:|---:|---:|---|
| F16 | >0.999 | 0.99–1.01 | ≤0.05 | Cosine >0.999 or the bounded numerical floor below |
| Q8_0 | >0.98 | 0.95–1.05 | ≤0.20 | — |
| Q4_K | >0.94 | 0.80–1.20 | ≤0.40 | — |

For every learned/output F16 row that misses the angle threshold, both its RMS
error and its maximum absolute error must be at most `2^-10` times the original
whole stage's RMS and maximum magnitude, respectively. This uses binary16
machine epsilon, approximately 0.0977%, as a declared engineering tolerance
scaled exclusively from the original reference; it is not a bound on network
roundoff error.
An all-zero reference stage permits no added signal. Raw minimum row cosine,
raw cosine-failure counts, rows accepted by the numerical floor, and remaining
failures are all retained. The policy is
`cosine_or_binary16_reference_floor_v1`.

This additional row policy was corrected after the first checkpoint probe.
The original PyTorch model with F16-rounded matrix weights reproduced all
3,368 decoder-row failures from native F16, while native versus original with
identically rounded weights passed every decoder row. Separately, the same
original PyTorch GELU inputs in different tensor layouts differed by about
`1.7e-7` near zero and failed the former angle-only rule even without F16
storage. The [precision diagnosis](historical/scnet-precision-diagnosis-2026-11-02.json)
is explicitly nonqualifying. The previous run remains failed under its
previous policy; a fresh full run must satisfy the declared correction.
The original F32 checkpoint remains the only qualification reference.

Analytic STFT rows can have exactly zero imaginary values. Their row statistics
are reported separately while complete-stage amplitude, relative error, shape,
and finite-value gates still apply. No fitted gain, time alignment, missing
stage substitution, or lower-precision threshold fallback is permitted.
At least half of matrix elements must use the declared target storage type.

Full CPU qualification uses a contiguous 24-second music excerpt with the
default inference recipe and two threads for both implementations. Each
precision runs one warmup followed by three alternating original/native
pairs. Loading and decoding are outside the clocks; the complete separation
method is inside, including output assembly and host transfer. Output checks
run outside the timing interval. Every completed timing is retained, and the
native median must be no slower than the original median.

The hosted workflow `.github/workflows/scnet-qualification.yml` builds the
actual CLI, Session library, probe, and quantizer, verifies generated capability
data, runs original observer controls, and requires all three precisions to
pass an independently checked 0.25-second raw probe as well as the complete
overlapping-chunk, timing, and production-surface gates. It retains source snapshots, commands,
hashes, logs, and incremental JSON receipts. Failed runs remain failed;
diagnostic or stages-only runs cannot report full acceptance. The fixture is
an implementation-parity control, not a held-out quality benchmark.

After a model size passes every gate, its workflow retains a separate
`qualified-<variant>-GGUF-<commit>` artifact containing the three tested GGUFs,
the original MIT notice, conversion and quantizer provenance, exact hashes,
and accepted numerical/performance receipts. Packaging rechecks their source,
binary, model, and receipt identities before copying the explicit model files.
These bundles use concrete local model paths; registry publication is optional.
Music, author checkpoints, raw activation captures, and diagnostic F32 models
are excluded from the model bundle.

## License

Upstream code is MIT licensed. The [author's explicit confirmation](https://github.com/starrytong/SCNet/issues/35)
also covers the author's released SCNet and SCNet-large pretrained weights and
format-converted redistribution with attribution. The complete notice is
retained in `models/licenses/SCNet-MIT.txt`, embedded in converted GGUF metadata,
and copied alongside each converted file. This evidence does not establish
the license or byte identity of unrelated third-party retrainings.
