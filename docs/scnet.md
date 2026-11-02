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
agreement. The gates were fixed before the pretrained-model run:

| Precision | Global cosine | Norm ratio | Relative L2 | F16 row gate |
|---|---:|---:|---:|---|
| F16 | >0.999 | 0.99–1.01 | ≤0.05 | Every learned-stage and waveform-channel row >0.999 |
| Q8_0 | >0.98 | 0.95–1.05 | ≤0.20 | — |
| Q4_K | >0.94 | 0.80–1.20 | ≤0.40 | — |

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
data, runs original observer controls, and tests each precision through both
numerical and production-surface gates. It retains source snapshots, commands,
hashes, logs, and incremental JSON receipts. Failed runs remain failed;
diagnostic or stages-only runs cannot report full acceptance. The fixture is
an implementation-parity control, not a held-out quality benchmark.

## License

Upstream code is MIT licensed. The [author's explicit confirmation](https://github.com/starrytong/SCNet/issues/35)
also covers the author's released SCNet and SCNet-large pretrained weights and
format-converted redistribution with attribution. The complete notice is
retained in `models/licenses/SCNet-MIT.txt`, embedded in converted GGUF metadata,
and copied alongside each converted file. This evidence does not establish
the license or byte identity of unrelated third-party retrainings.
