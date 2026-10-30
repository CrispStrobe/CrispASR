# OpenVoice2 reference and cloning validation

Status (2026-10-28): the bounded three-case conditioned-cloning test is accepted
on T4 CUDA: 27/27, 27/27 and 28/28 original-reference stages, raw C API,
exact-repeat/cache and clip-mean controls, WER <=0.2 and independent speaker
improvement >0.01. [Accepted quality receipt](historical/openvoice-conditioning-accepted-2026-10-28.json).
The separate current-runtime replay also passes all 27 stages on CPU and CUDA
against the v0.8.47 SDK and matching ggml headers. The real Python Session/C ABI
composition also passes both sentences through
raw and normal marked output, with architecture detection, exact repeats,
native rates, actual CUDA and unchanged quality gates.
[Session receipt](historical/openvoice-session-accepted-2026-10-28.json). Actual CLI and full rebuilt-current-library acceptance, and model publication,
are recorded in the current update below. Broader voices/languages remain separate. The older attempts below are historical failed controls,
not instructions to repeat a passing campaign. See [the handoff](maintenance-handoff.md).

**Current update (2026-10-29):** The full current CLI and Session/C ABI
passed the declared F16 profile on CPU and T4 CUDA. The qualified generic
converter is published at [revision `61741391da62c278148a2c75b6c0c5611d447149`](https://huggingface.co/cstr/openvoice2-tcc-GGUF/tree/61741391da62c278148a2c75b6c0c5611d447149), with
the original MIT notice and pinned provenance. All four uploaded files pass
anonymous full-byte readback; validation recordings, target embeddings and
reference captures remain private. Broader languages, natural references,
other devices and voice identity rights remain separate. See the
[publication receipt](historical/openvoice2-tcc-publication-2026-10-29.json) and
[maintainer handoff](maintenance-handoff.md).

## Recovered execution state — 2026-10-28

The previous bake is terminal **ERROR**. Its uploaded worker identifies itself
as `2026-10-19.openvoice-bake.2`, native source `71e329c`. The recovered native
log reports **no CPU ggml backend registered**, then aborts at the backend
buffer assertion while loading MeloTTS. It supplies no pretrained numerical,
GPU, conditioning or speech-quality acceptance. The old source and terminal
logs are preserved privately before any replacement submission.

The already prepared worker adds `GGML_BACKEND_DL` and module discovery from
the audited bundle directory. Its conditioning mode uses source `6be9fda36`,
with the same ggml revision as the frozen runtime. No model implementation
changed since that source. New worker versions add retained load, available
memory, scratch and GPU-memory checks before compilation and every reference
or native inference phase; insufficient resources stop before work. The two
host C++ delta compilations remain sequential and use the shared harness
heartbeat. No ggml/CUDA SDK rebuild or Torch reinstall is needed.

Conditioning kernel version **1**, worker
`2026-10-28.openvoice-conditioning.2`, was admitted after a bounded capacity
check. It received two T4s, passed both C++ delta compilations and conversion,
and retained five successful resource checks. It then failed with HTTP 403
while trying to create a Hugging Face staging model repository, **before model
loading or inference**. GPU allocation is not GPU execution. The receipt has
`complete: false`, `model_acceptance: false`, `gpu_executed: false`.
[Portable failed-attempt receipt](historical/openvoice-conditioning-attempt-2026-10-28.json).
The uploaded source, terminal log and output are preserved before replacement.

Worker `2026-10-28.openvoice-conditioning.3` separates validation from publication.
By default it saves the converted model, original references, full MIT notice,
attribution and hash/size checkpoints in kernel output without requiring Hub
write permissions. `CRISPASR_OV2_PUBLISH=1` explicitly enables publication and
requires write access; a requested publication failure remains a failure.
`publication_requested` and `publication_complete` are separate from numerical
`model_acceptance`. No reference, stage, waveform, content or speaker gate is
relaxed. Bounded model-free tests prove that read-only validation makes no Hub
write calls and that denied requested writes preserve evidence and fail.

Do not poll the entire historical kernel list in a burst: an initial roster
query hit rate limiting. Use a bounded current roster after backoff, respect
the two-session cap, and preserve the previous terminal artifacts before a
software retry. Never infer a running kernel's failure from missing logs.

## First pretrained numerical failure and bounded repair

Conditioning kernel version **2**, worker `.3`, passed the loader and read-only
checkpoint path, generated a capture-neutral original reference, and executed
the native CUDA vocoder on T4. It failed **11/27 required stages**. Both PCM
inputs and frontend/reference encoder match closely; the first material
posterior WaveNet difference is cosine 0.994142 and norm ratio 0.980296.
Posterior statistics fail the unchanged magnitude gate; forward-flow cosine
then drops to 0.133117, and waveform cosine is -0.030380.
[Full stage receipt](historical/openvoice-original-parity-failure-2026-10-28.json).
The producer's `gpu_executed` flag is set only after every case; its false flag
in this failed first case does not negate the scheduler's actual CUDA splits
and completed vocoder output. Numerical/model acceptance remains false.

Reading the pinned original inference path identifies two recipe defects:

- `zero_g` passes a **zero tensor** through the posterior conditioning affine
  layer. The native path skipped that layer entirely, dropping its learned
  bias. The actual F32 checkpoint has 6,144 nonzero bias values (norm 11.9577).
- Forward residual flow is **coupling then channel flip**; inverse flow visits
  the flip first. Native forward flow incorrectly flipped before coupling too.

Tiny tests compile the actual production CPU bodies against biased affine
fixtures. Both fail before repair (bias output 0.0572535 versus 0.2013839;
flow first value 4 versus 3.5), then pass after the two corrections, including
one/two-block forward expectations and inverse reconstruction. Existing owned
conditioning/noise/clip-average controls still pass. No large local model runs.
These tests justify the bounded fixes; they do not qualify pretrained speech.

Next replay the **identical retained model and original reference bytes** on
Kaggle CPU/CUDA, retaining every original gate. Keep real model/reference data
separate from tooling caches and carry the full MIT notice and synthetic-audio
attribution. Full three-case cloning, shared-noise, independent ASR/speaker and
higher-level Session/CLI acceptance remain separate after structural replay.
Do not replace this failed reference with native output or loosen tolerances.

## Frozen replay v1 — two repairs verified, final activation isolated

The exact same model/reference hashes were replayed on CPU and T4 CUDA with
model CPP `581234dfb` and the original audited runtime/header ABI. Both now pass
**26/27** unchanged gates. Posterior WaveNet and forward/reverse flows reach
cosine 1.0 and near-unit magnitudes. This verifies the zero-input bias and flow
ordering corrections on the actual pretrained case.
[Both full stage tables](historical/openvoice-frozen-replay-2026-10-28.json).

Only raw `dec_output` still fails: norm ratio 1.11145, despite cosine 0.999420.
The normalized waveform passes, illustrating why raw magnitude is mandatory.
The original Generator uses explicit slope 0.1 inside upsample blocks, then
`F.leaky_relu(x)` with its **default 0.01** before `conv_post`; native code used
0.1 for both. Only that last slope is corrected.

A small test executes both real graph-builder activation calls on signed inputs;
it fails before correction (-0.1 versus -0.01), then passes while preserving the
upsample slope. Separately, an owned-input replay of just the last activation,
224-weight convolution and tanh uses frozen original `dec_stage3` plus unchanged
F16 model weights. Correcting the slope changes norm ratio 1.111307 to 0.999915
and relative L2 0.116960 to 0.000125; cosine becomes 0.999999996.
[Owned-input evidence](historical/openvoice-final-activation-owned-input-2026-10-28.json).
This subsecond boundary check used one CPU thread after a resource check, without
running the full model locally. Whole-model CPU/CUDA replay of the final slope
and the full cloning/quality gates remain required. No acceptance bar changed.

## Frozen replay v2 — 27/27 on CPU and T4 CUDA

With model CPP `1b874a104`, the same model/reference bytes now pass **all 27**
unchanged stage and magnitude gates on CPU and actual T4 CUDA. Raw waveform
cosine is 0.99999973 and norm ratio is 1.000042 (relative L2 about0.000735).
All input/source/hash, device-work and resource evidence was independently
checked. [Accepted structural receipt](historical/openvoice-frozen-replay-accepted-2026-10-28.json).
`replay_acceptance` is true; `model_acceptance` intentionally remains false.

The next proof is the original three-case conditioning/quality worker with the
same three corrections: tau0/0.3, shared original noise, exact cache reuse,
uninstrumented model C API waveform/embedding, unequal clip means and mean-
conditioned waveform, then independent ASR and speaker improvement. The current
worker pins the audited old runtime/header ABI; whole-current-ggml and Session/
CLI acceptance remain separate. No default model is promoted from replay.

## Recipe and license boundary

OpenVoice V2 separates a base TTS from tone-color conversion. The native
MeloTTS path uses its own English dictionary/neural G2P; conversion uses raw
reference PCM, native STFT, six Conv2d layers, GRU, a 256-dimensional speaker
projection, posterior encoder, source/target-conditioned flow and vocoder.
Neither inference path imports whisper-timestamped or Unidecode. The OpenVoice
converter reads checkpoints directly; the reference imports original
`openvoice.models` and `openvoice.mel_processing`, not `se_extractor` or the
text/watermark frontend. This is not an exhaustive dependency-license audit of
an arbitrary Python environment installed by a user.

Upstream source pin: `74a1d147b17a8c3092dd5430504bd83ef6c7eb23`.
[OpenVoice code](https://github.com/myshell-ai/OpenVoice/blob/74a1d147b17a8c3092dd5430504bd83ef6c7eb23/LICENSE)
and [official V2 weights](https://huggingface.co/myshell-ai/OpenVoiceV2) declare
MIT; retain notices. [MeloTTS](https://github.com/myshell-ai/MeloTTS/blob/main/LICENSE)
is MIT too. Their complete Python install has separate copyleft dependencies:
pinned whisper-timestamped1.14.2 advertises GPLv3 (current upstream is AGPLv3),
and Unidecode1.3.7 is GPLv2-or-later. Those helpers are not vendored here.

Upstream `se_extractor.get_se(vad=True)` runs Silero VAD, concatenates speech,
partitions approximately ten-second clips and averages clip embeddings. Our
current raw-reference API computes one embedding over the provided PCM. A
clean, single-utterance raw reference matches upstream `extract_se([clip])`,
not the entire automatic VAD frontend. Do not call this VAD parity, multilingual
text-cleaner parity, or exact librosa resampling parity. The explicit `openvoice2_extract_speaker_embedding_segments` API now averages
caller-selected ordered speech clips with equal weights, and
`openvoice2_set_target_embedding` copies that prepared identity into conversion.
Automatic VAD selection remains a separate policy; never describe supplied
spans as an automatic upstream frontend. Unequal-length tiny-clip probes guard
equal weighting, invalid spans and unchanged outputs after failure. Actual
multi-clip model acceptance remains pending.

The upstream demo selects the saved source embedding by actual MeloTTS speaker
key. Both native CLI and shared session now resolve that key from model
metadata on each synthesis call. Missing keys use embedding extraction from
actual source PCM, never an unrelated first/default saved speaker.

## Reproduce a deterministic bake

Use the pinned upstream checkout and original checkpoint root containing
`converter/config.json`, `converter/checkpoint.pth` and `base_speakers/ses/`.
Choose licensed source/reference WAVs; record their hashes and attribution.

```sh
export CRISPASR_OV2_UPSTREAM="$OPENVOICE_SOURCE"
export CRISPASR_OV2_REFERENCE_WAV="$REFERENCE_WAV"
export CRISPASR_OV2_SOURCE_SPEAKER=en-us
export CRISPASR_OV2_DEVICE=cpu
export CRISPASR_OV2_TAU=0
python tools/dump_reference.py --backend openvoice2 --model-dir "$CHECKPOINT_ROOT" \
  --audio "$SOURCE_WAV" --output "$ARTIFACTS/ref.gguf"
build/bin/crispasr-diff openvoice2 "$MODEL_GGUF" "$ARTIFACTS/ref.gguf" "$SOURCE_WAV"
```

The Python forward calls the original model's `voice_conversion`, with owned
hook snapshots. Reference GGUFs contain the exact native-rate source/reference
PCM used by the model, so the C++ comparison isolates model math from resampler
implementations. The recipe and source identity are required archive metadata.
Keep deterministic tau=0 acceptance separate from upstream tau=0.3 stochastic
quality. The dumper can now record the actual original posterior noise at0.3;
`openvoice2_set_posterior_noise` owns it and the driver requires its28th stage.
A shared seed alone is insufficient across different random samplers. The
original waveform must also be byte-identical after removing capture hooks and
replaying the same seed. These newer paths are not covered by the earlier
source71e329ccf Kaggle run. The header now documents the actual long-standing native default0.
Original watermarking is disabled for model-boundary parity; converted output
also exposes our existing peak normalization as a separate stage.

Twenty-seven deterministic stages (28 with posterior-noise replay) are required, without optional skips: both PCM inputs and
spectrograms, reference normalization, six reference convolutions, GRU,
projected target/source embeddings, posterior pre/WaveNet/stats/latent,
forward/reverse flow latents, vocoder pre/four MRF stages, raw decoder waveform
and final normalized PCM. The driver prints cosine, both norms, norm ratio,
relative L2 and maximum absolute error. Gate: cosine>=0.99, norm ratio within2%,
relative L2<=0.15; the two PCM inputs must be exact. These are aggregate stage
metrics, not per-row minimum cosine. Missing/wrong-size/nonfinite stages fail.

Diagnostic captures bypass the reference cache so stages cannot disappear on
a hit. Ordinary cache identities now include reference-encoder coefficients,
preventing embeddings from another precision/checkpoint being reused. Snapshot
callbacks are synchronous and must not re-enter or free the context.

## GPU and quality acceptance

GPU execution is opt-in via `CRISPASR_OV2_GPU=1` and the caller's `use_gpu=true`.
For the diff command also set `CRISPASR_DIFF_USE_GPU=1`. Reference encoder and
flows remain CPU; only the vocoder graph can execute on CUDA. A device listed
by the runtime or weights in GPU memory alone does not establish execution.
Do not flip the gate without recorded numerical and decoded-output proof.

`tools/kaggle/openvoice-bake/` uses the independently audited CUDA12.4 sm60/sm75
runtime from source191815511 and its exact ggml revision. It compiles only the
changed model files and strict driver against those prebuilt libraries, not
ggml, CUDA kernels or the full SDK. Record both base and delta identities.
The separate full-CMake check39255636270 failed during configuration because
the parent wrapper project changed CMAKE_SOURCE_DIR used by existing C2PA/llama
paths. The worker now configures the repository root and injects only its
target declarations. Run39258859196/source6be9fda36 compiled both models and
the driver, then failed executable linking because the toolkit lacked the
libcuda.so.1 alias. The link-only alias repair stays outside the runtime bundle.
This build check is not model acceptance.

The kernel preserves the host NVIDIA loader paths and runs Torch phases in
separate processes from the packaged native CUDA stack. Version1 failed at
post-reexec nvidia-smi before inference; version2 repairs this preflight.
Submitted v2 is still queued and lacks the later GGML_BACKEND_DL compile and
module-discovery repair required by the audited base; it must be superseded
after retaining logs. The prepared worker matches this ABI, searches modules
from the native bundle cwd, and uses a different cwd/host loader environment
for Torch subprocesses. The intended matrix records original/native stage comparison,
uninstrumented public model C API waveform/embedding parity and deterministic
cache reuse, with CPU Torch fallback on sm60 while the native vocoder genuinely
uses CUDA. It then checks independent ASR WER<=0.2 for baseline/original/clone
and independent Resemblyzer similarity improvement>0.01 over baseline. Preserve
failed originals as controls; do not relax thresholds after seeing a result.
This is a synthetic closed-loop voice test, not general human-speaker accuracy.
The prepared conditioning mode (`CRISPASR_OV2_CONDITIONING=1`) freezes newer
source6be9fda36, adds a third tau0.3 replay and unequal-length original/native
clip means plus mean-conditioned waveform comparison. Its entry version and
source are distinct from submitted v2; it has not been run. Its standalone
immutable bootstrap is `tools/kaggle/openvoice-conditioning/`. Converted GGUFs and
producer versions are retained in newer receipts; v2 lacks those extra fields.

Remaining acceptance scope: raw mixed-rate preprocessing vs original librosa;
automatic Silero selection and actual supplied-clip averaging; actual tau0.3 shared-noise replay; quant variants
with sensitive tensors retained; full shared Session/CLI model execution;
additional languages, natural references and supported device families.
Public reference archives belong in `cstr/crispasr-regression-fixtures`, with
immutable revisions, source/model/file hashes and licenses. Keep private
operational paths and account details in ignored maintainer notes.

## Checkpoint-free numerical encoder probe

[The synthetic receipt](historical/openvoice2-refenc-synthetic-2026-10-19.json) compares
the unmodified original ReferenceEncoder class to the literal native CPU encoder
body, using deterministic F32 tensors and synthetic coefficients. Input129x513
exercises all six convolutions and three GRU steps. All nine boundaries pass:
minimum cosine0.9999999999998275; maximum norm-ratio deviation3.02e-8; maximum
relative L2 error5.88e-7. Restoring the historical H/W-axis swap yields first
convolution cosine0.405004 and final embedding cosine0.971432, with final norm
ratio1.0663; the probe rejects it.

```sh
python tools/validation/openvoice_refenc_toy.py --upstream "$OPENVOICE_SOURCE"   --out "$ARTIFACTS/toy-refenc"
```

No pretrained checkpoint, GGUF decoding, STFT, full SDK, CUDA or waveform is
involved in this small arithmetic check. Coefficients are read as F32 vectors.
It establishes the encoder's CPU math/layout and a rejecting structural control;
it does not replace the pending actual-model/GPU/quality matrix.

## Producer checkpoint and publication policy

The repaired conditioning worker (`.6`, kernel version 4) passes all three
cases. Clone WER is 0.1111, 0 and 0.1111; independent speaker cosine improves
by 0.1833, 0.2373 and 0.1494 over the baseline. Original and native clips pass
the unchanged content gate. All artifact hashes and reported WER values were
independently checked after download. This closes the bounded raw C API
conditioning/quality proof on the frozen runtime, without promoting a default.
The current-runtime 27-stage CPU/CUDA proof is retained separately.
No model was published: validation was explicitly read-only. Earlier Hub
repository creation returned 403; publishing needs an authorized destination
and retained notices, followed by full anonymous byte readback.

Conditioning kernel version 3 passed all three original-reference CUDA cases
(27/27, 27/27, 28/28), exact repeats, uninstrumented raw C API conversion and
clip-mean conditioning. Its independent quality process failed in PyAV decoding
before ASR results; this historical failure is superseded by accepted version 4. The repair reads the
generated WAVs with SoundFile, averages channels and resamples to mono float32
at 16 kHz before passing arrays to the unchanged Whisper decoder. It preserves
the WER <=0.2 and independent speaker-improvement >0.01 gates.
[Structural receipt](historical/openvoice-conditioning-structural-2026-10-28.json).

The separate `tools/kaggle/openvoice-current-replay` worker qualifies the same
frozen inputs against the shipped v0.8.47 CUDA SDK and matching `e2ae679a` ggml
headers. It verifies archive bytes, dependency resolution and the owning shared
library for model symbols before running all 27 stages on CPU and CUDA. Only
the corrected OpenVoice/Melo translation units and driver compile. No SDK or
Torch rebuild is involved. This structural profile keeps `model_acceptance`
false and does not replace full speech, Session or CLI qualification. Kernel
version 1 has now passed all 27 stages on CPU and T4 CUDA, with resolved
dependencies and verified overlay symbol owners. Raw waveform norm ratio is
1.000042. [Current-runtime receipt](historical/openvoice-current-runtime-replay-2026-10-28.json).

The shared worker checkpoints the converted GGUF under an explicitly unvalidated
staging path and each original-module reference before native execution. Each
receipt records hashes and immutable publication revisions; attribution and MIT
notices accompany the artifacts. A later native/quality failure must preserve
these producer outputs. Staging is not accepted model publication. Canonical
model publication is gated by the bounded parity/quality matrix, and its receipt
still needs independent audit. No actual pretrained/GPU acceptance is claimed
by the prepared worker or by a successful package build.
