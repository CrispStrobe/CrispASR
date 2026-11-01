# Mini BS RoFormer V2.5: quantization and qualification

The backend is `mini-bs-roformer`; the model is
[`HiDolen/Mini-BS-RoFormer-V2.5-46.2M`](https://huggingface.co/HiDolen/Mini-BS-RoFormer-V2.5-46.2M/tree/38dafc78834607aec57d8c0cb3568523f9ea5563).
The converter pins revision `38dafc78834607aec57d8c0cb3568523f9ea5563` and
retains the original tensor names. This is the V2.5 architecture with temporal
grouping, grouped-query axial attention, register tokens, stem attention
pooling and a shared complex-mask decoder. It is distinct from the existing
Mel-Band RoFormer backend.

## Storage precision

The original checkpoint is BF16. BF16 parameter values can be represented
exactly in F32; the converter writes matrices as F16 and keeps arithmetic
operands and transform windows as F32. Conversion, numerical comparison and
speed comparison are separate operations.

| Tensor family | F16 artifact | Q8_0 artifact | Q4_K artifact |
|---|---|---|---|
| Axial transformer matrix weights | F16 | Q8_0 | Q4_K |
| Temporal grouping / ungrouping SwiGLU matrices | F16 | Q8_0 | Q8_0 |
| Frequency and stem pooling matrices | F16 | Q8_0 | Q8_0 |
| Band input projections | F16 | F16 | F16 |
| Shared mask and sigmoid gate projections | F16 | F16 | F16 |
| DeltaRMSNorm deltas, learned probes, register tokens, periodic Hann windows | F32 | F32 | F32 |

The masks are shared by four stems, while pooling supplies the stem-specific
conditioning. The policy preserves these boundaries and keeps most encoder
weights quantized. The policy alone is not evidence that a checkpoint passes
the quality gate; the actual model must pass the reference comparisons.
Explicit per-tensor overrides remain available for separately measured
calibration experiments.

The pinned checkpoint produces the following storage inventory with the
default recipes. Q4_K deliberately contains Q8_0 boundary matrices; it is a
mixed precision artifact. These counts include 6,144 deterministic F32 Hann
window elements in addition to the checkpoint's 46,173,188 learned parameters.

| Artifact | File bytes | F32 elements | F16 elements | Q8_0 elements | Q4_K elements |
|---|---:|---:|---:|---:|---:|
| F16 | 92,488,448 | 39,428 | 46,139,904 | 0 | 0 |
| Q8_0 | 52,675,360 | 39,428 | 3,672,576 | 42,467,328 | 0 |
| Q4_K | 38,519,584 | 39,428 | 3,672,576 | 14,155,776 | 28,311,552 |

The default artifacts do not require an importance matrix. The optional
`--imatrix` runner argument is available for explicit calibration experiments;
it checks input-column coverage for every Q4_K matrix and records the
calibration hash. An importance matrix must not silently substitute a different
fixture or weaken a comparison threshold.

Create the base and the two quantized variants:

```sh
python models/convert-mini-bs-roformer-to-gguf.py \
  --model /path/to/pinned-model --output mini-bs-roformer-f16.gguf
build/bin/crispasr-quantize mini-bs-roformer-f16.gguf mini-bs-roformer-q8_0.gguf q8_0
build/bin/crispasr-quantize mini-bs-roformer-f16.gguf mini-bs-roformer-q4_k.gguf q4_k
```

The precision policy is architecture-specific in `crispasr-quantize`.
`tests/test-mini-bs-quant-policy.py` runs the real tool on model-free GGUFs and
checks byte preservation, the Q8 floors, re-quantization, user overrides and
the unchanged policy for other architectures.

## Using the genuine CrispEmbed tools

Both projects use ggml quantization. CrispEmbed's
[`tools/quantize.cpp`](https://github.com/CrispStrobe/CrispEmbed/blob/1607d38833e83aebc3f4d852c8ee4e34fa96712e/tools/quantize.cpp)
states that it was adapted from CrispASR, and its
[`src/crispembed_diff.h`](https://github.com/CrispStrobe/CrispEmbed/blob/1607d38833e83aebc3f4d852c8ee4e34fa96712e/src/crispembed_diff.h)
provides the shared ground-truth comparison contract. Their command-line
override syntax differs: CrispASR uses partial regular expressions with the
first match winning; CrispEmbed uses full-name globs with the last match
winning. A regex recipe must not be copied into the glob tool.

`tools/validate_mini_bs_roformer.py --quantizer-kind crispembed` invokes the
actual supplied `crispembed-quantize` executable with matching glob overrides.
It explicitly preserves F32 probes and register tokens, which the generic
CrispEmbed quantizer would otherwise treat as eligible matrices. The runner
checks the resulting tensor types and records tensor/parameter counts by type.

The equivalent direct commands for the genuine executable are:

```sh
crispembed-quantize mini-bs-roformer-f16.gguf mini-bs-roformer-q8_0.gguf q8_0 \
  --tensor-type 'freq_domain_model.mask_model.band_split.*.weight=f16' \
  --tensor-type 'freq_domain_model.mask_model.mask_estimator.*.weight=f16' \
  --tensor-type '*.probe=f32' \
  --tensor-type '*.register_tokens=f32'

crispembed-quantize mini-bs-roformer-f16.gguf mini-bs-roformer-q4_k.gguf q4_k \
  --tensor-type 'freq_domain_model.mask_model.band_split.*.weight=f16' \
  --tensor-type 'freq_domain_model.mask_model.mask_estimator.*.weight=f16' \
  --tensor-type '*.probe=f32' \
  --tensor-type '*.register_tokens=f32' \
  --tensor-type 'freq_domain_model.freq_pooling.*.weight=q8_0' \
  --tensor-type 'freq_domain_model.time_pooling.*.weight=q8_0' \
  --tensor-type 'freq_domain_model.mask_model.time_conv.*.weight=q8_0' \
  --tensor-type 'freq_domain_model.mask_model.time_deconv.*.weight=q8_0'
```

With the pinned base GGUF and ggml revision, these recipes produced files
byte-identical to the architecture-aware `crispasr-quantize` defaults across
all 458 tensors. The Q8_0 artifact SHA256 is
`273e58a716224b38b722cd85000a8a533e2f2281ef5939cfced6bb513e713aa3`;
the Q4_K artifact SHA256 is
`595021e32765a0b4adb158c3d33368c5751e25af7fbfb4597a22d4bc305b8741`.
This establishes quantizer equivalence; the numerical gates still execute the
model and compare it with the original blueprint.

For the actual CrispEmbed diff implementation, compile this repository's
small capture adapter against an external pinned CrispEmbed checkout:

```sh
c++ -O3 -std=c++17 -I/path/to/CrispEmbed/src \
  tools/mini_bs_crispembed_diff.cpp -o crispembed-diff
```

This adapter includes the real `crispembed_diff.h`. It does not rename the
CrispASR executable or substitute a different cosine implementation. It adds
exact stage-set and shape checks, finiteness checks and the fixed strict gates
around the external comparator. It rejects missing stages instead of relying
on the comparator's historical minimum-length behavior.

## Numerical gates

Generate reference activations by executing the pinned original Python model:

```sh
python tools/dump_mini_bs_roformer_reference.py \
  --model-dir /path/to/pinned-model --audio music-8s.wav \
  --output reference.gguf --threads 2
```

The fixture must be native 44,100 Hz stereo. The reference stores its exact
input waveform; the native diff starts with those same samples. This removes
decoder and resampler differences from the model comparison.

Twenty-five boundaries are mandatory for the stage qualification: input, both STFTs, band projection,
temporal compression, both axes of every transformer layer, final encoder,
both attention pools, masks, and the complete bass/drums/other/vocals
waveforms. Capture and compare a variant directly:

```sh
CRISPASR_MINI_BS_THREADS=2 CRISPASR_MINI_BS_GPU=0 \
CRISPASR_MINI_BS_CAPTURE_GGUF=native-q8.gguf \
  build/bin/crispasr-diff mini-bs-roformer \
  mini-bs-roformer-q8_0.gguf reference.gguf music-8s.wav

./crispembed-diff reference.gguf native-q8.gguf q8_0
```

The artifact gates are strict: Q8_0 whole-tensor cosine must be **greater
than 0.98** and Q4_K must be **greater than 0.94** at every required stage,
including each separated stem. F16 requires whole-tensor cosine greater
than 0.999. The CrispEmbed adapter also requires F16 feature-row and waveform
channel cosine greater than 0.999. It reports the STFT output's per-bin minimum
without using that minimum as a gate: a row there is only one complex Re/Im
pair, whose angle is ill-conditioned in near-zero spectral bins. The whole
STFT tensor remains gated. Quantized artifacts report row minima but use
global cosine, following CrispEmbed's distinction between port correctness
and quantization damage.

The adapter independently requires finite values, exact shapes, and a norm
ratio between 0.80 and 1.20, matching the native runtime's original amplitude
contract. An exploratory external 0.90–1.10 check was stricter; its earlier
failed results remain in the calibration logs. The fixed user cosine gates
are unchanged. A scaled or silent output cannot pass merely by
having an acceptable cosine. `CRISPEMBED_DIFF_COS_THRESHOLD` cannot lower the
adapter's fixed acceptance thresholds.

## Paired performance comparison

The qualification runner creates both quants, inventories them, runs all
stage comparisons and optionally runs the external CrispEmbed adapter. It
then loads the original model and each native variant once, performs warmups,
alternates engine order for at least three measured repetitions, and compares
every stem after every repetition.

Build the production library and tools with the
[Linux CPU recipe](mini-bs-roformer.md#build-the-cpu-runtime) before running the
commands below. CPU timing results use a Release build with native CPU
instructions and the optional ggml OpenBLAS backend. A build without BLAS
requires its own speed measurement; numerical compatibility alone does not
establish equivalent performance.

The primary qualification reference contains all 25 stages for one complete
352,800-frame stereo chunk: the original model's trained eight-second forward
geometry. Short raw-forward captures remain useful structural diagnostics,
but their low-energy stem behavior can differ substantially from complete
chunks. The runner permits them with `--stages-only` and cannot claim overall
acceptance from them. Failed short diagnostics remain part of the evidence.

An additional output reference may be passed with `--full-reference`; it must
contain `input_audio` and all four stems. This supplements the primary
25-stage check. The receipt records both input geometries. Direct native
comparison uses `CRISPASR_MINI_BS_OUTPUTS_ONLY=1`, and the external adapter
uses its explicit `--outputs-only` option. Each refuses a missing stem.
For constrained storage, `--discard-native-captures` removes each native
archive after its hash and actual external comparison results are recorded.
It does not skip or reduce any comparison.

```sh
python tools/validate_mini_bs_roformer.py \
  --model-dir /path/to/pinned-model \
  --f16 mini-bs-roformer-f16.gguf \
  --reference reference.gguf --audio music-8s.wav \
  --quantizer /path/to/crispembed-quantize --quantizer-kind crispembed \
  --diff build/bin/crispasr-diff --crispembed-diff ./crispembed-diff \
  --library build/src/libcrispasr.so \
  --output-dir qualification-8s --threads 2 --device cpu \
  --warmups 1 --repetitions 3 --mode separate --python-dtype bfloat16
```

`--mode separate` compares the actual overlap/add APIs with the original
method's default batch size of four. The model-card examples use a batch size
of two; both settings process the eight- and twelve-second fixtures in one
Python batch because those fixtures contain one and two chunks respectively.
`--python-batch-size 1` is an explicit optional
memory control, recorded in the receipt; its timings must be distinguished
from the published default. Use a complete trained eight-second chunk and a
longer fixture crossing chunk boundaries. `--mode
forward` is available for short diagnostic and graph-only comparisons; it
must not be reported as long-audio separation performance.

The fixed numerical oracle uses the original model in F32. The default
performance profile, `--python-dtype bfloat16`, uses the original class's
published `from_pretrained` loading path without a dtype override. This is
necessary because the pinned config selects BF16 and the pinned Transformers
version honors that default. Casting a model constructed in F32 to BF16 can
initialize its nonpersistent transform windows differently. The receipt
therefore records the actual parameter dtype and each original buffer's dtype
and byte hash, as well as differences between the timed Python output and the
F32 oracle. Native output quality remains gated against the same F32 oracle
after every timed repetition.

`--python-dtype float32` is a separate explicit comparison profile. Passing
its speed gate does not establish performance against the default BF16
blueprint. Native matrix storage is recorded separately for each artifact,
and the native compute-precision getter records the actual matrix execution
profile separately from its F32 output dtype. Both sides include result allocation
and transfer to host, and exclude model loading and audio decoding.

The receipt distinguishes `profile_acceptance` from overall `acceptance`.
Final acceptance requires the published BF16 loader, `--mode separate`, and
the method's default batch size of four, together with the genuine CrispEmbed
quantizer and external comparator. F32, raw-forward, reduced-batch and local
tool-only experiments can pass their selected profile while leaving overall
acceptance false. This prevents a diagnostic result from becoming the final
qualification claim.

Per-model `quality_acceptance` is native output versus the fixed F32 oracle.
The separate `native_vs_timed_python_acceptance` reports output agreement with
the actual timed Python profile, while
`timed_python_matches_f32_port_tolerance` reports whether that Python profile
itself meets the stricter F32 port tolerance. These are distinct measurements;
speed comparison with BF16 does not imply identical BF16 and F32 output.

The performance gate requires the native median to be no slower than the
original median on the same workload, with passing output gates. GPU runs
also require native backend identity evidence; a requested GPU is insufficient
if execution fell back to CPU. The default CPU run explicitly disables native
GPU use. The receipt records the requested device and observed backend,
the immutable PCM digest, CPU model and instruction flags, container CPU and
memory limits, library hash, adjacent CMake build options and BLAS identity
when available. Both engines' inputs are checked for mutation after timing.

`validation.json`, raw command logs, tensor inventories and native captures
remain in the output directory. A missing stage, failed command, skipped
benchmark, threshold boundary score, or slower native median cannot produce
overall acceptance. `--stages-only` intentionally leaves performance and
overall acceptance false.

An implementation-parity fixture is not a source-separation quality dataset.
Claims about held-out music quality or other hardware require their own
inputs and measurements; no model or speed claim follows from a successful
build or from a reference archive compared with itself.


## Integration validation

The CLI, HTTP separation endpoint and Session C ABI share the new runtime.
The public source order is `bass`, `drums`, `other`, `vocals`; audio is
interleaved stereo at 44,100 Hz, with exactly the input number of frames.
Architecture detection must also work after renaming the model file.

For quick development, the optional targets below link the production runtime
without the unrelated speech backends. The validation shared library exposes
the same `mini_bs_roformer_*` functions used by the complete library.

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=ON -DGGML_NATIVE=ON -DGGML_OPENMP=ON \
  -DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS \
  -DGGML_CUDA=OFF -DGGML_METAL=OFF -DGGML_VULKAN=OFF \
  -DGGML_SYCL=OFF -DGGML_OPENCL=OFF \
  -DCRISPASR_BUILD_MINI_BSR_PROBE=ON -DCRISPASR_BUILD_TESTS=ON
cmake --build build --parallel 2 --target \
  mini-bs-roformer-probe mini-bs-roformer-validation test-mini-bs-roformer
ctest --test-dir build --output-on-failure -R '^Mini BS RoFormer '
```

`mini-bs-roformer-probe mini-bs-roformer MODEL REFERENCE [AUDIO]` accepts the
same model-specific arguments as `crispasr-diff`; it calls the production diff
function. These targets establish a focused build and runtime test boundary,
not full CLI/binding acceptance.

After the current CLI and shared library are built, run the actual-surface
fixture explicitly. It checks neutral-filename Session auto-detection, all four
stem names, exact stereo length/rate, getter bounds, repeated result ownership,
and agreement between Session arrays and selected CLI WAV outputs. It also
compares repeated and changed PCM inputs against independent cold calls, then
returns to the first input to detect stale cached graph or decoder inputs.

```sh
CRISPASR_MINI_BS_ROFORMER_MODEL=/path/to/mini-bs-roformer-f16.gguf \
  ctest --test-dir build --output-on-failure -R '^mini-bs-roformer-surface$'

python tests/test-mini-bs-roformer-surface.py \
  --crispasr build/bin/crispasr --library build/src/libcrispasr.so \
  --model /path/to/mini-bs-roformer-f16.gguf --frames 352800
```

The surface test returns skip code 77 when the model variable is absent. A
supplied model that cannot load or separate fails the test. Repeat with Q8_0
and Q4_K artifacts after their numerical gates pass. The independent C ABI
architecture test (`test-arch-backend-map`) covers the converter's canonical
architecture using metadata-only GGUFs. Backend-name aliases do not introduce
alternate checkpoint architecture profiles: metadata must use `mini-bs-roformer`.

Before integration, regenerate and check the capability catalogue and feature
matrix/documentation counts from the current CLI build. Regenerate Go's
dependency list separately with `tools/sync_go_cgo_ldflags.py`, which constructs
the standard static dependency graph rather than reusing optional local libraries.
The capability catalogue is generated with the complete optional ONNX roster,
matching the repository's `linux-unit` CI configuration. A native-only binary
must not accidentally delete those existing ONNX rows. CI additionally runs
cross-platform builds, the unit tier, formatting/header guards, secret checks,
and binding validation. Successful focused model tests do not establish those
separate checks.
