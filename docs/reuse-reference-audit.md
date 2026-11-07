# Issue #579: NemotronLabs RE-USE reference audit

This is reference work for a future CrispASR backend, not a claim that RE-USE
is currently supported by CrispASR. The requested [NVIDIA checkpoint](https://huggingface.co/nvidia/NVIDIA-NemotronLabs-RE-USE)
is pinned to `31f82f2efad1095034492f236e8dac9239ba713b`; its `model.safetensors`
SHA256 is `06badaf4029b297f97f7fe48d62990b48d5482608210df577354e0d4e0ed70e1`.
The published model is under OpenMDW-1.1. Distinguish it from the separate,
noncommercial original RE-USE checkpoint.

[CPU run 42312957754](https://github.com/CrispStrobe/CrispASR/actions/runs/42312957754)
ran the official network with a non-fused, CPU-only implementation of the
Mamba-1 selective-scan equations. That adapter has independent small analytic
scan controls; it is not the official CUDA kernel. The audit pinned audio.cpp
at `04a587779625f334b666b26f2ec92827f4cbfad7`, retargeted its converter's
original-checkpoint digest guard to the requested pinned digest, and verified
all 1,416 converted F32 tensors against their original bytes and shapes.

For one 0.5-second, mono 8 kHz excerpt from the official microphone fixture,
the CPU reference and audio.cpp output were compared without alignment or
normalization. Cosine similarity was 0.99999999985 and relative RMSE was
0.00001734. The run also saved encoder and selected Mamba-layer activations as
an artifact. The manually dispatched `issue579 RE-USE CPU reference audit`
workflow reproduces this bounded comparison.

This does not validate a CrispASR implementation, long recordings, additional
sample rates, stereo, quantization, or GPU execution. A separate Kaggle GPU
reference kernel is included under `tools/kaggle/issue579-reuse-reference`;
the initial attempt could not start because the weekly GPU quota was exhausted.
Before adding a backend, compare a native CrispASR port with these stages and
the upstream implementation, validate waveform and speech-quality gates on a
representative corpus, and exercise its CLI, C ABI and bindings as applicable.

## Native CPU port investigation

A separate, experimental CrispASR CPU reference is under development. Its
[original-checkpoint audit 42378662267](https://github.com/CrispStrobe/CrispASR/actions/runs/42378662267)
passed isolated original-weight encoder, selective-scan and decoder comparisons,
and validated the strict GGUF loader including malformed-file controls. The
full 0.1-second public-audio comparison failed at the encoder output with
3.61% relative error, exceeding the 0.2% stage gate; this is not a passing
end-to-end result.

A local comparison on the same excerpt found raw phase differences near the
+π/−π branch cut, particularly in the first reflected FFT frame. Although those
angles represent nearly identical complex values, the model consumes the raw
phase as an input, so complex-spectrum equivalence alone is insufficient. The
native port remains experimental while the source of the difference and its
impact on subsequent stages and waveform are investigated.

The independent Kaggle CPU run `issue579-reuse-cpu-diff-v3` tested source
`b6cd39102f5cfbd0aa30e727f0715dfc1f408c22` and ggml
`1b972bb754aa5f83e2d872ccf37a335717f6658d` on a 0.1-second 16 kHz public
speech excerpt. `crispasr-diff` compared the frontend, all six encoder
boundaries, both temporal and frequency boundaries in all 30 blocks, decoder
outputs and waveform. Native frontend magnitude passed, but native raw phase
failed with 24.8% relative RMS; the resulting waveform failed with 1.03%
relative RMS. All phase-conditioned stages passed, including the waveform at
0.000186% relative RMS. Conditioning is diagnostic and does not make the
native end-to-end comparison a pass. The pinned original model and source
hashes were checked in the worker; the CPU kernel reported its failed gate.

A follow-up CPU probe (`issue579-reuse-fft-cpu-v2`) compared the same
byte-identical 1,600-sample input on Kaggle and the local reference worker.
Both used PyTorch 2.11, but different builds and CPU environments. Their raw
STFT phases differed by approximately 2π in 47 bins, all in the first
reflected frame. The input SHA256 was
`12e4f02454a088fedba2fb0876101d79211e8b1b6ae27c62968e501e4fe8f283`.
This demonstrates that this upstream raw-phase boundary itself varies between
these environments; it does not establish a portable native parity fix. The
unconditioned original-model and waveform gates remain open.

The two captured phases were then fed through the same pinned original model
and the same input on both the local CPU and Kaggle CPU (`issue579-reuse-phase-ab-v1`).
Both A/B runs measured a 1.48% relative waveform difference between the two
upstream phase inputs. With a 0.5% relative waveform gate, the triangle
inequality rules out a single output passing against both references. A
first-frame candidate that maps values near either ±π branch to +π was also
measured locally: it changed the original waveforms by 6.66% and 7.35%,
respectively, on this excerpt. That candidate is rejected; it is not a change
to the reference or a passing parity result. Broader audio and hardware work
is needed before choosing any production behavior.

The Kaggle CPU follow-up `issue579-reuse-multi-phase-v1` repeated the A/B on
five separate 0.1-second excerpts from the same public recording. Each had
54–73 first-frame branch flips and no flips in later frames. Feeding both phase
variants through the same pinned original CPU model gave 1.19–3.79% relative
waveform differences; none of the five pairs can both satisfy the 0.5% gate
against a single output. The [receipt](historical/reuse-phase-multi-20261107.json)
records input hashes and each result. These are five excerpts from one
recording, not a representative quality corpus.

`tools/ci-heavy/issue579_reuse/phase_stability.py` reproduces the single-pair
A/B and candidate check from an NPZ containing `audio`, `local_phase`, and
`kaggle_phase`; the caller must establish that both phase arrays came from the
same audio. It checks the pinned source hashes and reports the input hash,
the compatibility bound and waveform errors. The candidate is diagnostic only.

### Scope correction and independent implementations

The 0.5% waveform threshold above was selected for this investigation; it is
not an NVIDIA specification. The five 100 ms excerpts all come from one
recording. The incompatibility shown is between those particular short-clip
references under that threshold, not a proof that a portable RE-USE runtime is
impossible or that longer audio has the same error. The upstream script runs
CUDA and the CPU audit substitutes a non-fused Mamba adapter. The adapter uses
the pinned upstream SEMamba and STFT modules and exact checkpoint, but is not a
direct run of NVIDIA's fused CUDA Mamba kernels. Default whole-file and
five-second overlapping chunked inference still need direct qualification.

Independent implementations include [speech.cpp](https://github.com/nyosegawa/speech.cpp/blob/main/docs/models/semamba.md),
whose authors report comparisons against original CUDA reference dumps;
[audio.cpp](https://huggingface.co/audio-cpp/RE-USE-GGUF), which publishes a
conversion of the same revision and waveform comparisons; and
[reuse-rs](https://github.com/altunenes/re-use-rs), whose offline Rust path
loads this same checkpoint. [RE-USE Audio](https://github.com/quangng2000/reuse-audio)
also supplies an ONNX Runtime Android implementation and independent export
and reference checks for the pinned checkpoint. Their reported results are
useful independent evidence, not proof of CrispASR parity.

A later independent check exported the pinned weights with the
[RE-USE Audio ONNX exporter](https://github.com/quangng2000/reuse-audio/tree/1dafc11fc057d0453300a5d8863ec558e1d6387b/tools/model-export)
and ran its CPU graph on the exact same 16 kHz, 100 ms input spectra as the
unfused Python reference. The encoder, first and last blocks, magnitude and
phase outputs all agreed within 0.0002% relative RMS; the reconstructed
waveform differed by 0.000103%. This checks an independently written whole
network with shared spectra, not the STFT or NVIDIA's fused CUDA Mamba path.

On that same short PCM input, the released speech.cpp 0.10.1 CPU binary with
its separately converted pinned checkpoint produced 16-bit PCM differing by
2.70% relative RMS from the local Python reference (2.96% versus the other
CPU phase reference). Its developers also report much greater drift for an
especially short 30 ms input. See the [receipt](historical/reuse-independent-runtime-20261107.json)
for hashes, methods and stage metrics. A short-input difference does not
establish the quality or error of either implementation on normal recordings.

To check the adapter against the Mamba blueprint itself, Kaggle run
`issue579-official-gpu-v2` used the pinned NVIDIA SEMamba and checkpoint with
[Mamba v2.2.4's original forward and `selective_scan_ref`](https://github.com/state-spaces/mamba/tree/v2.2.4/mamba_ssm),
executed as PyTorch CUDA operations on a Tesla T4. It forced the **unfused**
reference and convolution fallback because the fused extensions were unavailable.
With the same GPU spectra, our adapter agreed at all six encoder layers, all
30 block outputs, and magnitude output; the worst relative RMS was 0.000069%.
This is evidence for the adapter and network math, not validation of Mamba's
fused CUDA kernels. On the 100 ms input, the full unconditioned CPU-adapter
waveform differed from this GPU reference by 1.87%, and speech.cpp's 16-bit
output by 2.04%. The [receipt](historical/reuse-official-unfused-gpu-20261107.json)
records source hashes and all 37 adapter comparison results.

The 72-stage unfused CUDA capture was also converted into a reference archive
and compared to CrispASR with `crispasr-diff`. All 71 conditioned comparisons
passed, with a 0.000147% relative waveform difference. The unconditioned
native run still failed, with 2.10% waveform error on this same 100 ms input.
The GPU capture remains unfused and the test remains a short-clip diagnostic;
neither result establishes fused-kernel parity or general speech quality.

A follow-up ran speech.cpp 0.10.1 on a one-second, 16 kHz PCM excerpt from
the same public recording and compared its 16-bit output with the pinned
Python network and CPU adapter. Relative RMS difference was 0.103%, below the
investigative 0.5% threshold. The [independent-runtime receipt](historical/reuse-independent-runtime-20261107.json)
includes the input hash. This is one normal-duration sample, and is not a fused
CUDA comparison, a quality evaluation or a guarantee for other audio.

The [independent port and license audit](reuse-port-audit.md) records pinned
source licenses, model provenance, implementation differences and follow-up
validation requirements.


The one-second Kaggle CPU follow-up `issue579-reuse-cpu-diff-v4` ran the
experimental source at `f169cc3a3b4718739febffc67afd9c13dcbc67d6` against
the pinned original network with its unfused CPU Mamba adapter. Native
frontend magnitude passed, but raw phase failed (8.48% relative RMS, with a
maximum difference of 2π), and intermediate gates failed. The native waveform
passed the investigative 0.5% gate at 0.0702% relative RMS. All 71
phase-conditioned comparisons passed, with 0.0000735% relative waveform error.
The overall run failed because intermediate comparisons remain required;
conditioning is diagnostic. The [receipt](historical/reuse-one-second-diff-20261107.json)
records every comparison. This is one sample, not a quality corpus or a fused
CUDA reference.


A [window and FFT investigation](historical/reuse-window-fft-diagnosis-20261107.json)
replayed the 100 ms input on local and Kaggle CPU PyTorch 2.11 builds. Their
Hann windows were byte-identical, yet 47 first-frame raw-phase bins differed
by more than one radian; replaying the Kaggle window locally left the same 47
flips. The native window differed in 27 of 640 float values. Replacing it
with the exact PyTorch window in a diagnostic native FFT changed its branch
mismatches from 36 to 38 against local PyTorch and from 35 to 31 against Kaggle.
This isolates a backend or environment effect between those PyTorch runs; it
does not exclude separate effects from native window rounding. No window or
phase convention change was accepted as a parity fix.

An external oneMKL 2025 CPU frontend prototype independently generates the Hann
window with high-accuracy vector cosine and runs the public DFTI real FFT. On
the local x86-64 environment it matched the local PyTorch window and complex
FFT byte-for-byte on one 100 ms and one one-second input; raw phase had no
branch flips and differed by at most 2.39e-7. Replaying its phase with the
native model's own magnitude passed all 71 existing native intermediate and
waveform checks against the local 100 ms reference, including 9.93e-7 relative
waveform RMS. The independently generated window produced the same bytes as
the captured-window prototype used for that replay. It still disagreed with
the Kaggle phase reference at 47 first-frame bins; its waveform was 1.49%
different from that reference, above the 0.5% gate. The
[receipt](historical/reuse-onemkl-prototype-20261107.json) records the full
stage results. This is a pinned-profile result, not portable or fused-CUDA
parity. The local profile loaded roughly 220 MiB of external MKL/OpenMP shared
objects; none are bundled. An opt-in research integration is being evaluated.

The experimental branch `work/issue579-reuse-decoder` now has an opt-in oneMKL
research frontend at `4830b8e22`. On the same local 100 ms reference its
integrated unconditioned run independently computed both magnitude and phase,
and passed all 71 intermediate and waveform checks; all 71 conditioned checks
also passed. Unconditioned waveform relative RMS was 6.60e-7. The
[integrated receipt](historical/reuse-onemkl-integrated-01s-20261107.json)
includes source and executable hashes and every stage. The one-second
integrated run also passed all 71 unconditioned and 71 conditioned checks, with 6.72e-7 relative waveform RMS; the
[receipt](historical/reuse-onemkl-integrated-1s-20261107.json) records all
142 rows and input, model and executable hashes. The default research
executable does not link MKL, and requesting an unavailable profile fails explicitly. Isolated SDK and
default research CTests pass; a full product rebuild was not performed. The
MKL profile remains experimental and does not resolve the Kaggle-reference
phase discrepancy or establish GPU, long-recording, quality or fused-kernel
parity.

### CPU FFT dispatch and cross-machine parity

The independent Kaggle CPU capture uses PyTorch 2.11 CPU with oneMKL 2024.2
and AVX2 dispatch; the local machine normally dispatches AVX-512. On five
100 ms excerpts from the same recording, independently computed oneMKL 2025
AVX2 complex spectra matched Kaggle's float32 real and imaginary **bits**,
including signed zeros, with no raw phase branch flips. The AVX-512 control
flipped 54–73 branches per excerpt. The [five-clip receipt](historical/reuse-avx2-multi-fft-20261107.json)
records the comparisons. This is a bounded x86 CPU result, not a guarantee for
other versions, CPUs or recordings.

The explicit `mkl-avx2` research profile at branch commit `3f43acce9` requires
`MKL_ENABLE_INSTRUCTIONS=AVX2` at process startup, no `MKL_CBWR`, an external
oneMKL SDK and an AVX2 host. It passed all 71 unconditioned and all 71
conditioned comparisons against a separately generated Kaggle original-network
CPU reference, with unconditioned waveform relative RMS 1.06e-6. See the
[cross-machine receipt](historical/reuse-avx2-kaggle-integrated-20261107.json).
The reference uses an unfused Mamba adapter. Environment variables set a
oneMKL instruction ceiling; users must qualify actual dispatch and version,
not assume an environment string alone guarantees identical results. The
standard `native` and `mkl` profiles remain unchanged.

A separate Kaggle T4 experiment used Torch-prepared GPU frames and the same
window to compare public cuFFT with Torch's GPU STFT. On one 100 ms input,
complex output was bit-identical and raw phase had no branch flips; it also
matched the prior unfused CUDA reference phase within 2.39e-7. The
[receipt](historical/reuse-cufft-fft-probe-20261107.json) does not validate
native preprocessing, a native CUDA frontend, the full model or fused Mamba.
The independently computed CUDA research frontend and model comparison were
subsequently tested as described below.


The CUDA research profile at `1d889456a` independently generates the Hann
window, prepares the input frames and computes magnitude and phase through
cuFFT. Kaggle run `issue579-reuse-cuda-profile-v2` passed all six isolated
CTest targets, including physical-GPU frontend tests at 400, 1600 and 16000
samples. Its integrated 1600-sample native run passed all 71 unconditioned
and 71 conditioned comparisons against the retained original CUDA stages;
the unconditioned waveform relative RMS was 1.0843e-6. The
[receipt](historical/reuse-cuda-integrated-20261107.json) records every
stage, the GPU and source/model hashes. The reference uses the unfused Mamba
CUDA path; the native model graph and inverse FFT remain CPU implementations.
This is one short model input on a T4, not full GPU backend, fused-kernel,
long-recording, other-device or speech-quality qualification.

For the original fused Mamba comparison, Kaggle compiler preflight `issue579-reuse-fused-source-v1` passed on a T4 with the installed
PyTorch 2.11/CUDA 13 and pinned CCCL 2.8.2 headers. It checked the original
Mamba 2.2.4 and causal-convolution C++ bindings and legacy CUB helpers. Only
build configuration was adjusted for the GPU architecture and headers; the
package and kernel source manifests remained unchanged. The
[preflight receipt](historical/reuse-fused-preflight-20261107.json) records
pins and checks. The subsequent build and capture are documented below;
preflight alone does not establish fused-model parity.

The pinned original Mamba and causal-convolution CUDA extensions subsequently
built successfully in `issue579-reuse-fused-source-v2`, with their package and
kernel source manifests unchanged. The attempted model capture stopped before
inference because Kaggle's current Transformers package lacks an import used
by Mamba 2.2.4. The [build receipt](historical/reuse-fused-build-20261107.json)
records the wheel hashes and failure. A capture-only retry with compatible
Transformers dependencies is documented below.

The one-second follow-up `issue579-reuse-original-stages-v2` independently
captured the original network on Kaggle CPU. The local `mkl-avx2` research
profile passed all 71 unconditioned and 71 conditioned comparisons against
that reference; unconditioned waveform relative RMS was 5.31e-7. The
[receipt](historical/reuse-avx2-kaggle-1s-20261107.json) records every stage,
input and model hashes. This is one excerpt, using the unfused CPU adapter;
it does not establish fused CUDA, other hardware or speech-quality parity.


The capture-only retry `issue579-reuse-fused-capture-v2` loaded the pinned
built wheels with compatible isolated Transformers dependencies. It ran the
original fused Mamba 2.2.4 model on a T4 and captured 72 tensors; the worker
counted 120 calls each to the fused selective-scan and causal-convolution CUDA
extensions. The [capture receipt](historical/reuse-fused-capture-20261107.json)
records input, model and wheel hashes and dependency versions. The native
cuFFT comparison against this capture is documented below; the capture alone
is not a native parity pass.

The subsequent native comparison `issue579-reuse-cuda-fused-v1` used that
original fused capture. The independently computed cuFFT frontend and native
CPU model graph passed all 71 unconditioned and 71 conditioned comparisons;
unconditioned waveform relative RMS was 1.1407e-6. All six physical-GPU
CTest targets passed. The [fused parity receipt](historical/reuse-cuda-fused-parity-20261107.json)
records every stage and the original capture, model and payload hashes.
This result is for one 100 ms clip on a T4; it does not qualify a full GPU
model backend, long recordings, other devices or speech quality.


The one-second original fused capture `issue579-reuse-fused-capture-1s-v1`
also completed on a T4, with 120 calls to each original fused extension.
The [receipt](historical/reuse-fused-capture-1s-20261107.json) records input,
model and output-stage hashes. The native comparison is documented below;
the capture itself is not a parity pass.

The one-second native follow-up `issue579-reuse-cuda-fused-1s-v1` also passed
all 71 unconditioned and 71 conditioned checks against the original fused
CUDA capture. Unconditioned waveform relative RMS was 6.5531e-7, and all six
physical-GPU CTest targets passed. The [one-second fused receipt](historical/reuse-cuda-fused-1s-parity-20261107.json)
records every stage and the capture, model and payload hashes. It covers one
one-second clip on a T4, not long recordings, other devices, a GPU model graph
or speech quality.
