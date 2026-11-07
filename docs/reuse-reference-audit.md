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
