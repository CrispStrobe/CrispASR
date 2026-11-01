# Mini-BS-RoFormer V2.5 source separation

The `mini-bs-roformer` backend implements
[`HiDolen/Mini-BS-RoFormer-V2.5-46.2M`](https://huggingface.co/HiDolen/Mini-BS-RoFormer-V2.5-46.2M/tree/38dafc78834607aec57d8c0cb3568523f9ea5563).
It returns four stereo stems in the checkpoint's order: **bass, drums, other,
vocals**, at 44,100 Hz. Inference is native C++ and ggml; Python is required
only for conversion and original-reference validation.

The upstream revision is pinned to
`38dafc78834607aec57d8c0cb3568523f9ea5563`. The
[original configuration](https://huggingface.co/HiDolen/Mini-BS-RoFormer-V2.5-46.2M/blob/38dafc78834607aec57d8c0cb3568523f9ea5563/config.json)
and
[original Python blueprint](https://huggingface.co/HiDolen/Mini-BS-RoFormer-V2.5-46.2M/blob/38dafc78834607aec57d8c0cb3568523f9ea5563/modeling_bs_roformer.py)
define this implementation. The model card identifies the source as MIT.
This architecture has temporal grouping, grouped-query attention and a shared
stem-conditioned decoder; its weights cannot be loaded by the existing
Mel-Band RoFormer backend.

## Build the CPU runtime

From the repository root, initialize its submodules and install the project
[build prerequisites](install.md), including OpenBLAS development headers and
libraries. The Linux CPU qualification uses a Release build with OpenBLAS,
OpenMP and instructions enabled for the build machine:

```sh
git submodule update --init --recursive
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=ON \
  -DCRISPASR_USE_SYSTEM_GGML=OFF -DCRISPASR_PORTABLE_CPU=OFF \
  -DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS \
  -DGGML_NATIVE=ON -DGGML_OPENMP=ON \
  -DGGML_CUDA=OFF -DGGML_METAL=OFF -DGGML_VULKAN=OFF \
  -DGGML_SYCL=OFF -DGGML_OPENCL=OFF \
  -DCRISPASR_BUILD_EXAMPLES=ON -DCRISPASR_BUILD_TESTS=ON \
  -DCRISPASR_BUILD_SERVER=OFF
cmake --build build --parallel 2 --target \
  crispasr-cli crispasr-quantize crispasr-diff
```

This produces `build/bin/crispasr`, both quantization/comparison executables,
and `build/src/libcrispasr.so`. Build and measure on the intended CPU:
`GGML_NATIVE=ON` enables that machine's instruction set. The paired CPU
qualification uses two threads for each engine; it does not establish GPU
fidelity or performance.

## Convert and run

Install the reference/conversion dependencies, then produce the F16 artifact:

```sh
python -m pip install -r tools/reference_envs/mini-bs-roformer/requirements.txt
python models/convert-mini-bs-roformer-to-gguf.py \
  --model HiDolen/Mini-BS-RoFormer-V2.5-46.2M \
  --output mini-bs-roformer-f16.gguf

build/bin/crispasr-quantize mini-bs-roformer-f16.gguf mini-bs-roformer-q8_0.gguf q8_0
build/bin/crispasr-quantize mini-bs-roformer-f16.gguf mini-bs-roformer-q4_k.gguf q4_k

build/bin/crispasr --separate -m mini-bs-roformer-q8_0.gguf -f song.wav -t 2 --no-gpu
mkdir -p stems-q4k
build/bin/crispasr --separate -m mini-bs-roformer-q4_k.gguf -f song.wav -t 2 --no-gpu \
  --stems vocals --sep-output-dir stems-q4k
```

The CLI reads audio at the model's sample rate, preserves stereo and writes one WAV per
selected stem using the shared [source-separation surface](source-separation-surface.md).
The GGUF architecture selects the backend; a concrete GGUF path is required.

`--model` also accepts a local snapshot containing `config.json` and
`model.safetensors`. The converter reads safetensors directly without executing
the model's Python code. It validates every expected tensor name and shape
before writing a model. The checkpoint contains 456 learned tensors and
46,173,188 parameters. `--dtype f32` is available for diagnostics.

The converter preserves checkpoint names and matrix layouts. It records
source/configuration hashes, the available original-code hashes, and source
revision metadata. A local directory without a revision marker or a Hugging
Face snapshot commit directory is explicitly marked `local-unverified`.

For the genuine `crispembed-quantize` overrides, the external `crispembed-diff`
adapter, fixed numerical gates and paired speed measurement, see
[quantization and qualification](mini-bs-roformer-validation.md). A precision
policy is not a substitute for measured fidelity or performance.

## Native API and Python binding

The public C ABI uses the existing session separation functions in
`include/crispasr_session.h`:

```c
crispasr_session *s = crispasr_session_open("mini-bs-roformer-q8_0.gguf", 4);
/* Check s before use. pcm holds interleaved float32 stereo at 44100 Hz.
   n_frames is the sample count PER CHANNEL, not the number of floats. */
int count = crispasr_session_separate(s, pcm, n_frames);
/* On success count == 4. For each index 0..3: */
int frames = 0;
const char *name = crispasr_session_separate_stem_name(s, 0);
const float *bass = crispasr_session_separate_stem(s, 0, &frames);
/* Consume or copy 2 * frames floats before another call or session close. */
crispasr_session_close(s);
```

The direct backend interface in `src/mini_bs_roformer.h` additionally exposes
`mini_bs_roformer_default_params`, `mini_bs_roformer_init_from_file`,
`mini_bs_roformer_separate`, `mini_bs_roformer_result_free`, and
`mini_bs_roformer_free`. Direct results own their audio until explicitly freed.
The native separation call can duplicate mono into the stereo model; the
shared session ABI expects stereo.

For the Python example, install the source package with
`python -m pip install -e ./python`. Select the newly built Linux shared
library explicitly so an older installed library cannot take precedence:

```python
import soundfile as sf
from crispasr import Session

audio, sample_rate = sf.read("song-44100-stereo.wav", dtype="float32", always_2d=True)
assert sample_rate == 44100 and audio.shape[1] == 2
with Session("mini-bs-roformer-q8_0.gguf", lib_path="build/src/libcrispasr.so",
             n_threads=2, use_gpu=False) as session:
    stems = session.separate(audio)
for name, interleaved in stems.items():
    sf.write(f"song_{name}.wav", interleaved.reshape(-1, 2), 44100)
```

The Python arrays own their storage and survive subsequent calls. The session
does not resample raw PCM; callers supply the model's native sample rate.

## Blueprint and tensor layouts

The neural graph comprises the following original operations. Shapes below
are row-major Python shapes with the singleton batch axis omitted; GGML's
dimension order is reversed.

| Stage | Shape or operation |
|---|---|
| Input spectrum | `[T, 8196]`: 2049 frequency bins × 2 channels × Re/Im |
| Band split | 80 independent DeltaRMSNorm + linear projections → `[T, 80, 256]` |
| Temporal grouping | Pad `T` to a multiple of 4; normalize then group feature/time as `(d, t_c)` |
| Grouping SwiGLU | 1024 → 2048 → 512, producing `[Tc, 80, 512]` |
| Registers | Pad four time rows and four band columns; initialize only their 4×4 intersection from learned tokens |
| Axial encoder | Six time-attention/frequency-attention pairs, each with a SwiGLU residual block |
| Attention | 8 query heads, 4 key/value heads, 64 channels per head; noncausal attention, split-half RoPE |
| Encoder output | Remove register rows/columns, then DeltaRMSNorm → `[Tc, 80, 512]` |
| Frequency pooling | One learned probe attends over bands independently for each compressed frame |
| Time pooling | Four learned probes attend over the pooled temporal sequence → `[4, 512]` |
| Stem conditioning | Add each stem embedding to every encoder position |
| Shared decoder | SwiGLU 512 → 2048 → 1024; unpack `(d, t_c)`, normalize, remove temporal padding |
| Complex masks | Per-band linear projection multiplied by a sigmoid gate, producing `[4, T, 4100]` |
| Waveforms | Complex mask × 1025-bin output spectrum, then iSTFT → `[4, 2, samples]` |

For the trained 352,800-sample chunk, `T = 690`; two zero frames are appended
before grouping, `Tc = 173`, and the axial grid includes 177 time positions
and 84 bands.

Several details are necessary for original parity:

- **DeltaRMSNorm stores a delta.** Its scale is `1 + parameter`, with epsilon
  `1e-6`. Treating `*.scale` as an ordinary norm multiplier is incorrect.
- **Temporal grouping is feature-major.** The packed axis is `(d, t_c)`, with
  time position varying fastest. The decoder reverses precisely that layout.
- **Registers occupy a corner.** The new time-only and band-only regions start
  at zero; only the bottom-right 4×4 corner contains learned values. All
  register positions have identity RoPE, equivalent to position zero.
- **RoPE uses split halves.** Its base is 10000; it is not adjacent-pair
  rotation. Each key/value head serves two consecutive query heads.
- **Pooling has no query residual or RoPE.** Cross-attention produces a value
  which then receives its own normalized SwiGLU residual. No learned positional
  embedding is present in the checkpoint's pooling layers.
- **One gate covers a complex pair.** Repeat each gate value twice, in Re/Im
  order; do not duplicate the complete gate vector.
- **The decoder is shared by all stems.** Stem-specific conditioning comes
  from the pooling probes. The native runtime reuses a one-stem decoder graph
  to bound activation memory, while preserving all four original outputs.

Learned operations remain in ggml graphs, including band projections,
normalization, grouped attention, pooling, temporal SwiGLUs and mask/gate
projections. The backend's cached radix-2 FFT and overlap/add operate in F32. Reused
graphs avoid rebuilding the model for every chunk; reusable inputs must
remain protected from allocator aliasing across stems and calls.

## STFT, reconstruction and recording boundaries

The input transform uses a 4096-point FFT; the mixture transform and iSTFT
use 2048 points. Both use hop 512, `center=True`, reflect padding, periodic
Hann windows and no transform normalization. The converter stores the exact
F32 windows produced by `torch.hann_window`, including the model's two
different window sizes. Frequency packing is frequency-major,
channel-next, Re/Im-fastest.

The backend creates immutable FFT plans for the supported sizes 2048 and
4096. Each plan caches bit-reversal swaps and stage twiddles computed
independently with double-precision sine/cosine, then stored as F32. F32
butterflies use these direct values rather than repeatedly updating a complex
twiddle, whose accumulated rounding error is amplified by normalization in
quiet frequency bands. Workers retain separate F32 scratch arrays, so the
plans can be shared safely across concurrent frame/channel transforms. The
2048 plan also serves inverse transforms through conjugation and division by
the FFT size. This implementation is private to Mini-BS-RoFormer; the shared
FFT implementation used by other backends is unchanged.

Whole-recording separation follows the original `separate` default: 8-second
chunks, 4-second stride, right-zero-padding of the last chunk, and a linear
fade over the first/last 10% of each chunk. The weighted sum is divided by
the accumulated window with a minimum denominator of `1e-8`, then cropped
to the exact input length. The endpoints of the original fade window are
zero; the native path preserves that behavior.

The direct `no_segment` parameter requests raw whole-buffer forward for
diagnostics. It does not perform the recording-level padding or fades.
Reference fixtures for raw forward must be longer than half the 4096-point
FFT, as required by the original reflection padding.

## Precision and reference provenance

The supplied checkpoint stores BF16 weights. The F16 converter preserves all
DeltaRMSNorm deltas, learned pooling probes, register tokens and transform
windows as F32; other matrix weights are stored as F16. Quantization protects
the small band projections and mask/gate output projections and applies
additional precision floors to the shared conditioning/temporal boundaries.
The exact Q8_0/Q4_K policy and inventory checks are documented in the
[qualification guide](mini-bs-roformer-validation.md#storage-precision).

`tools/reference_backends/mini_bs_roformer.py` imports the original classes
from the pinned snapshot, verifies the Python files, configuration and checkpoint
against their SHA-256 hashes, and loads the checkpoint strictly. It uses forward hooks and wrappers around the
original transform methods; it does not substitute a Python implementation of
the native graph. Captures are either owned arrays or synchronously serialized
before the original forward continues, protecting against later in-place changes.
Archives record the original source identity, actual Torch/Transformers
versions, reference dtype, exact input PCM hash and sample geometry.

Numerical captures retain the explicit F32 oracle. The separate published
benchmark loader calls the same verified original class's `from_pretrained`
with its pinned configuration and no dtype override, then moves only its
device. This preserves the framework's original parameter dtype selection
and nonpersistent-buffer initialization. It checks loading information for
missing, unexpected or mismatched weights; casting an already constructed
F32 oracle is not used as a substitute for that initialization path.

Use the dedicated stereo entry point:

```sh
python tools/dump_mini_bs_roformer_reference.py \
  --model-dir /path/to/pinned-model --audio song-44100-stereo.wav \
  --samples 22050 --output reference-short.gguf --threads 2

python tools/dump_mini_bs_roformer_reference.py \
  --model-dir /path/to/pinned-model --audio song-44100-stereo.wav \
  --samples 352800 --output reference-8s-all25.gguf --threads 2
```

The dedicated dumper streams all 25 boundaries by default. It declares the
exact tensor shapes first, then writes each hook result directly into its
aligned GGUF extent in blocks of at most 4 MiB. This keeps serialization
memory bounded without changing the original forward. A complete 8-second
archive has 572,284,864 bytes of F32 tensor data, plus a small header; that
storage and the original model's own forward peak are still required. On a
RAM filesystem, archive storage also consumes system memory. The output path
is published atomically only after all requested stages have been captured
once with the expected shapes and finite values. `--in-memory` retains the
previous capture strategy for diagnosis. A short fixture is useful for
debugging, but cannot replace trained-length qualification.

The generic `tools/dump_reference.py --backend mini-bs-roformer` remains
available for its 16-kHz-mono harness. It resamples that input and stores the
exact resulting stereo PCM in the archive. Native-rate stereo input is
preferred for music qualification.

### Original-input importance statistics

`tools/collect_mini_bs_roformer_imatrix.py` collects calibration statistics
from the inputs actually supplied to all 344 original `torch.nn.Linear`
modules. It retains per-input-column float64 sums of squares and row counts,
with bounded reduction blocks and no intermediate-stage retention:

```sh
python tools/collect_mini_bs_roformer_imatrix.py \
  --model-dir /path/to/pinned-model \
  --audio diagnostic-stereo-0p5s.wav music-8s.wav --samples 22050 \
  --output mini-bs-roformer-imatrix.gguf --threads 2
```

The explicit sample count selects the first 0.5 seconds of each input. The
GGUF stores F32 **sums of squares**, named exactly like the original weights,
and `UINT64 count.<weight-name>` metadata. The genuine quantizer computes
the mean energy by dividing each vector by its count. The sidecar JSON
records exact calibration PCM hashes, source hashes, versions, sample counts,
column counts and activation-row counts. No calibration input or tensor is
silently omitted.

Pass the resulting file with `--imatrix` to the genuine quantizer, together
with the precision-preserving overrides in the
[qualification guide](mini-bs-roformer-validation.md). GGML Q4_K can use
these weights to minimize activation-weighted quantization error; its Q8_0
implementation ignores the importance argument. Calibration does not establish
acceptance, and inputs used for calibration must be distinguished from
independent qualification inputs. The numerical and performance gates remain
unchanged.

The development fixtures include a deterministic stereo diagnostic and fixed
8/12-second excerpts of the
[TorchAudio source-separation tutorial sample](https://docs.pytorch.org/audio/main/tutorials/hybrid_demucs_tutorial.html),
which the tutorial identifies as **A Classic Education — NightOwl**, licensed
CC BY-NC-SA 4.0. Those excerpts are internal educational parity inputs and are
not distributed with this implementation. They are not claimed as a training
holdout: the model card says training used MUSDB18-HQ train and test splits.
Parity and timing receipts must identify the actual fixture and execution
scope; implementation fidelity does not establish held-out separation quality.
