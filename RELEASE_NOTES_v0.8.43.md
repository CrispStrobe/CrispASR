# v0.8.43 — audio correctness, frontend parity and measured opt-in optimizations

This release covers merged PRs #519, #520, #522, #523, #524 and #525. Experimental quantization candidates and unfinished hardware
validation are excluded from the released behavior described below.

## VoxCPM2: correct session audio rate and configurable CFM steps

VoxCPM2 session synthesis now returns native 48 kHz mono PCM, matching the
sample-rate getter and CLI. The previous session path halved the sample count
while advertising 48 kHz, causing clients that trusted the getter to play or
transcribe the result at double speed. Binding documentation now directs callers
to the backend output-rate getter when saving or playing PCM.

The existing session `set_tts_steps` setter and explicit CLI `--tts-steps`
override now reach VoxCPM2's CFM solver and apply to the next synthesis call.
Ten steps remain the default; lower counts can reduce speech quality. There is
no new eight-step quality claim or Intel B390 performance claim.

PR #524 passed all 45 applicable CI checks, including its model regression,
with one intentional skip. Integrated native files and VoxCPM2 C ABI changes
match the validated CUDA/Vulkan source. Original NVIDIA/F16 recognition reads
all 22 fixed VoxCPM2 controls correctly. This is decoded-output evidence,
not a claim of full Python/native waveform equality.

Evidence: [session contract and proof pins](docs/voxcpm2-session-contract-2026-10-09.json).

## Nemotron 3.5: correct terminal frontend masking and boundary diagnostics

The frontend now zeroes frames beyond floor(samples/hop), matching the
pinned original NVIDIA attention mask while retaining the original STFT
shape. The old terminal frame could carry log-mel values into preencode.
The diff tool now checks actual mel, preencode and encoder shapes and stages;
an additive C diagnostic API exposes the preencode output.

PR #525 passed all 44 applicable checks, with one intentional skip. Exact
source passed all three unchanged F16 stage gates on CPU and actual dual T4s.
On T4, encoder cosine minimum is 0.99999088 and relative L2 is 0.00028004;
all 26 original transcript controls agree in fresh and reverse-reused
sessions. Long-turn streaming presets 0/2/3 pass packet-size token equality,
byte-exact reset/repeat confidence and idempotent final flush. These streaming
checks do not claim parity with the original Python streaming implementation.

Evidence: [CUDA stages and streaming proof](docs/nemotron-frontend-gpu-2026-10-09.json),
[CPU original transcript controls](docs/nemotron-frontend-controls-2026-10-09.json).

## Python: band-limited array resampling and contiguous input buffers

Five Python transcription entry points now use the existing native Kaiser
polyphase resampler for array inputs at rates other than 16 kHz. This addresses
aliasing from the former linear interpolation path. An additive C helper
exposes in-memory resampling; older native libraries use a bounded NumPy
compatibility implementation without requiring SciPy.

Inputs become contiguous float32 arrays, including strided inputs. Exact
16 kHz input values are preserved. The independent file-loading path still
uses miniaudio; this change concerns Python array inputs.

PR #522 passed all 50 applicable CI checks, with one intentional skip. Eight
checks against the actual shared library cover alias rejection, agreement
between native and fallback output within 1e-7, input/rate/error contracts,
contiguity and independent allocation/free ownership.

Details: [Python audio resampling](docs/python-audio-resampling.md).

## MiMo-ASR: opt-in cached GPU decode

`CRISPASR_MIMO_ASR_GPU_STEP_GRAPH=1` enables a cached text-only step graph while
retaining the existing prefill path. The cache is invalidated before every
transcription. The optimization remains OFF by default.

Actual T4 ABBA measurements improve warm full-call medians by 2.4–2.7%, with
sampled GPU peak memory increasing by 18 MiB. All 64 EN/ZH speech calls,
eight CLI/session pairs and 114 full-vocabulary comparisons remain byte-exact:
cosine 1, relative L2 zero, matching norms and argmax. This does not establish
CANN, P100 or new quantization acceptance. PR #519 passed applicable CI.

## Opt-in parallel mel projection

`CRISPASR_MEL_PROJECTION_PARALLEL=1` enables parallel scalar mel projection;
it remains OFF by default. The implementation retains the serial path below
64 frames or without OpenMP and leaves the BLAS path unchanged.

All 144 hosted ABBA cases are bit-exact. Four-thread projection medians improve
3.30–3.58× at 64 frames and 3.05–3.50× at 3000 frames on the hosted runner.
A contended local VPS regressed at 64 frames, so these component timings are
not an end-to-end ASR speed claim. Qwen3 stage gates and CLI transcripts pass.
PR #520 retains KokerZhou's independently authored mel implementation from
PR #492 and passed applicable CI and selected model regressions.

## Index-Echo: decoder activation calibration support

Opt-in activation collection now composes with decoder stage capture.
Without calibration environment variables, the existing callback remains in
use. A native CPU unit verifies actual GGUF activation sums and row counts.

Actual dual-T4 collection OFF/ON passes three clips × 76 stage comparisons
in each arm, complete file translations, CLI/anonymous-model C ABI calls and
TTS-ASR roundtrips. Decoded receipts match. Collection covers 249 decoder
matrices, including valid widths and counts for all 96 FFN matrices. PR #523
passed all 42 applicable CI checks, with one intentional skip.

The held-out smoke statistics are excluded from quantization. A separately
prepared 48-clip CC0 EN/ZH corpus (259.176 seconds) is PCM-disjoint from the
acceptance inputs; fresh dual-T4 calibration also collected valid statistics
for all 96 FFN matrices. This establishes calibration coverage, not Q4 model
acceptance.

Evidence: [callback smoke proof](docs/index-echo-calibration-smoke-2026-10-09.json),
[independent corpus and GPU calibration](docs/index-echo-calibration-corpus-2026-10-09.json).

## Model and validation status

No production model files, registry quantization pins or quality thresholds
change with these merged PRs. Actual dual-T4 calibrated Echo Q4 testing passed
its full F16 control, but all three candidates failed stage/magnitude and
complete translation gates. No candidate is promoted; the terminal proof is
pinned in `docs/index-echo-q4-imatrix-validation-2026-10-09.json`.
Nemotron RNNT/prompt guards recover all 26 fixed transcripts, but Q4 encoder
parity remains rejected. Further FFN projection isolation also rejects both
926 MB mixed-precision candidates (encoder cosine 0.977366 / 0.882196);
no quantization default is changed. The merged frontend length-mask correction passes
all three F16 stage gates and all 26 original transcript controls with no
fresh/reused differences. Actual dual-T4 validation also passes all three stage gates, the 26 original
transcripts, fresh/reused equality and native streaming packet/reset/flush
checks for presets 0/2/3. PR #525 is merged; further Q4 encoder precision work is unfinished. Full proof:
`docs/nemotron-frontend-gpu-2026-10-09.json`.
OmniVoice PR #521 remains a draft because clone acceptance is rejected.
The pinned original F32 model also fails the same fixed clone control at 2/9
word errors with and without cleanup. This upstream result does not waive
native acceptance; no cleanup default is promoted.
The remaining CANN and hardware-specific work in PR #492 is not included.

Before publication, finish integration CI for the chosen release commit and
verify the produced packages and their runtime dependencies. The merged PR
checks above do not substitute for final release-artifact validation.

The subsequent Nemotron attention isolation also rejects all three source-guarded
Q4 arms. FFN+QKV source precision is closest at encoder cosine 0.998536 versus
the unchanged 0.999 gate, with a 1,143,237,536-byte model and only 28,311,552
bytes of actual Q4 tensor payload. FFN+attention-out and FFN+position-source
arms fail at 0.996296 and 0.987063. All seven baseline arrays match the prior
run exactly, F16 passes all three stages, and all 24 original frame-layer
controls pass for F16. These mostly-source diagnostic hybrids are excluded
from production quantization and release performance claims. See [nemotron-encoder-attention-2026-10-09.json](docs/nemotron-encoder-attention-2026-10-09.json).
