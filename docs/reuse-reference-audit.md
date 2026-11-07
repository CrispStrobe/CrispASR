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
