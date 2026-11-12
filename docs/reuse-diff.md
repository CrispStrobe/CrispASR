# RE-USE intermediate comparison

The RE-USE CPU implementation is experimental research and is not a product
CLI, server or session backend. Its standalone build shares CrispASR's existing
comparison machinery without linking the reference loader into `libcrispasr`.
See the [reconciliation record](reuse-reconciliation-20261112.md). To diagnose it against the pinned original checkpoint, create a
16 kHz, mono, 16-bit PCM WAV with at least 400 samples and a length divisible
by 80. The reference dumper rejects inputs over 16,000 samples; select each
short, non-silent speech excerpt deliberately. One-second comparisons are
substantially slower and create larger reference archives.

```sh
git submodule update --init --recursive
# BUILD_DIR is a caller-selected build directory outside the source tree.
cmake -S tools/ci-heavy/issue579_reuse -B "$BUILD_DIR" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache
cmake --build "$BUILD_DIR" -j2
ctest --test-dir "$BUILD_DIR" --output-on-failure
python tools/dump_reference.py --backend reuse --model-dir /path/to/pinned-model \
  --audio excerpt.wav --output reuse-ref.gguf
"$BUILD_DIR/reuse-research-diff" reuse-f32.gguf reuse-ref.gguf
# Optional: write unconditioned native-endian float32 waveform after both gates pass.
"$BUILD_DIR/reuse-research-diff" reuse-f32.gguf reuse-ref.gguf native-waveform.f32
```

The dumper checks the pinned checkpoint and configuration hashes and uses the
separately audited, non-fused CPU selective scan. The reference archive embeds
the exact input samples, not model weights. The native run consumes that input
from the archive. The standalone command takes only the model and reference
archive; the input WAV is used by the dumper.
Keep reference archives outside the repository.

The report compares raw frontend magnitude and phase, all six dense encoder
boundaries, the temporal and final boundary of each of the 30 Mamba blocks,
the two decoder outputs, and the reconstructed waveform. The `native/` rows
use the native frontend. The `conditioned/` rows repeat the forward pass with
reference phase values, isolating later model stages. Both groups must pass;
the conditioned run is diagnostic and cannot substitute for frontend parity.
The gate checks raw relative RMS, per-row cosine and amplitude ratio, and reports
maximum error. This matters because +π and −π describe the same
complex value but are different inputs to this model.

New reference archives also capture the exact upstream Hann window and the
STFT's real and imaginary values. Optional `diagnostic/` rows compare these
before phase calculation; they do not replace or relax the existing gates.
Older reference archives remain usable and simply omit those rows. For a
cross-environment first-frame investigation, run
`tools/ci-heavy/issue579_reuse/frontend_stability.py --help`: it captures
byte-checked input, window and spectrum, and can replay a second environment's
window. Its output is diagnostic and never declares model parity.

`tools/ci-heavy/issue579_reuse/chunked.py` is a research-only bounded
overlap-add diagnostic. It pads a short or partial final window and returns the
original length. The default one-second window fits the standard comparator.
To experiment with five-second windows, explicitly pass `window_samples=80000`,
`hop_samples=40000` and `max_window_samples=80000`; the callback must support
that length. These settings do not establish a production or streaming policy. The pinned
[NVIDIA example](https://huggingface.co/nvidia/NVIDIA-NemotronLabs-RE-USE/blob/31f82f2efad1095034492f236e8dac9239ba713b/inference_chunk.sh)
uses five-second windows and 50% overlap; its
[implementation](https://huggingface.co/nvidia/NVIDIA-NemotronLabs-RE-USE/blob/31f82f2efad1095034492f236e8dac9239ba713b/inference_chunk.py)
ends its final window at the end of the recording. The shorter diagnostic
windows and padding rules must be evaluated separately before product use.

On one five-second speech proxy with seeded synthetic noise, nine one-second
windows with 50% overlap each passed all 71 unconditioned and 71 conditioned
native stage checks. The stitched native and Python outputs differed by
5.3e-7 relative RMS, while the stitched and whole-recording Python outputs
differed by 0.111 relative RMS. Whole versus chunked reference SI-SDR was
15.46 versus 13.67 dB and STOI was 0.691 versus 0.683 on that sample. These
are diagnostic observations, not a quality threshold or streaming-equivalence
claim. See the [run receipt](historical/reuse-chunked-five-second-20261112.json);
its per-window reference GGUF archives were not retained.

The [ten-second run](historical/reuse-ten-second-20261112.json) passed all 142
stage checks in each of three overlapping five-second windows and recovered
the three reference archives. The native stitched waveform differs from the
CPU reference by 8.54e-7 relative RMS. On this one noisy speech proxy, however,
SI-SDR decreases from 5.00 to 3.48 dB while STOI increases from 0.631 to 0.715.
The bulk of the error occurs in the latter five seconds. This is a quality
concern requiring upstream fused-reference and broader corpus investigation,
not evidence that a product or streaming implementation is ready.
An independent [saved-window audit](historical/reuse-saved-waveform-quality-20261112.json)
found strong context sensitivity: on the same 5–7.5-second speech region,
the middle and final reference windows score 6.39 and −9.03 dB SI-SDR,
respectively. Both estimates concern the same single speech proxy.

An independent [CPU corpus receipt](historical/reuse-quality-cpu-20261112.json)
records 16 cases from four LibriSpeech test-clean speakers. SI-SDR improved on
all 12 synthetic-noise cases; clean inputs were also altered. These are short
read-speech diagnostics, not a listening test or a native-parity result. The
separate 72 [reference fingerprints per case](historical/reuse-quality-cpu-stage-fingerprints-20261112.json)
record intermediate magnitudes and hashes but do not compare native stages.
An additional [CPU ASR proxy](historical/reuse-quality-cpu-asr-proxy-20261112.json)
compares each condition with the same recognizer's clean-audio transcript. Across
48 reference words per condition, disagreement changed from 18 to 17 edits for
white 0 dB, 15 to 8 for white 10 dB and 15 to 10 for pink 5 dB; clean output
introduced 2 edits. These are recognizer-agreement diagnostics, not human WER.

The `reuse` GGUF architecture is intentionally not registered as a product
backend. The published v0.8.51 Linux shared library rejected the pinned real RE-USE
GGUF through both C ABI detection and session opening in three repeated cycles;
a Whisper metadata positive control and a null-argument control passed. The
relevant detector and session-open function bodies and shared architecture and
Parakeet schema tables are byte-identical to this candidate. The shared architecture-map and C ABI unit suite also passed against this
released library: 630 assertions across seven cases, including the added RE-USE
detection and session-open controls. The
[interface receipt](historical/reuse-c-abi-boundary-20261112.json) records the
asset and source hashes. This is a released-library control, not execution of
a newly built candidate library.

A separate CPU shared-library build of source `529e073f8` subsequently passed the
same real-GGUF, Whisper metadata and null controls. Its unchanged architecture-map
tests passed 630 assertions across seven cases, plus 11 alternate-mapping
assertions across two cases. The [rebuilt-candidate receipt](historical/reuse-c-abi-candidate-20261112.json)
pins source, library, model and log hashes. This qualifies the research-model
rejection boundary; RE-USE product inference and representative quality remain open.

For aligned five-second qualification artifacts, `quality_metrics.py` reports
SNR, scale-invariant SDR and STOI for the noisy input, whole Python reference,
chunked Python reference and chunked native output. Install `soundfile` and
`pystoi` before running it. Treat the provided speech-plus-synthetic-noise case
as a diagnostic, not a listening test or a representative quality benchmark.

To reproduce the released-library interface control after verifying and
extracting the pinned v0.8.51 Linux library archive:

```sh
LD_LIBRARY_PATH="$RELEASE_LIB_DIR" python tools/ci-heavy/issue579_reuse/c_abi_boundary.py \
  --library "$RELEASE_LIB_DIR/libcrispasr.so.0.8.51" --model reuse-f32.gguf
```

For a resource-intensive whole-window investigation, the separate
`reuse-long-diff` research executable admits up to 80,000 samples (five seconds
at 16 kHz). The standard executable and reference dumper remain limited to
16,000 samples by default; the Python adapter requires an explicit
`max_samples=80000` for this diagnostic. This larger bound is experimental and
has not been established as a production memory, quality or interface policy.
The experimental cuFFT frontend still rejects inputs over one second; the
five-second retry uses the explicitly qualified AVX2 oneMKL frontend.

The first five-second full-window attempt captured CPU and non-fused CUDA
Python references but the native comparator produced zero stage rows: the
oneMKL frontend still rejected inputs over 16,000 samples. See the
[failed-run and reference-quality receipt](historical/reuse-full-window-v1-20261112.json).
On one synthetic-noise JFK speech proxy, the CPU reference increased SNR from
5.00 dB to 13.62 dB and STOI from 0.577 to 0.691. These are narrow objective
diagnostics, not a listening test or native-model qualification.

After fixing the oneMKL input bound, the [five-second retry receipt](historical/reuse-full-window-v2-20261112.json)
records all 71 unconditioned and 71 conditioned stages passing on the same
speech proxy. Its native waveform differs from the CPU reference by 6.16e-7
relative RMS. The separately run non-fused T4 Python reference is a cross-device
observation, not qualification of a native GPU or fused model backend.
