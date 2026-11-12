# RE-USE research reconciliation — 2026-11-12

This candidate reconciles frozen research source `2abc00b29` with main
`30d03c393`. It retains the scalar reference, converter, fixtures, original-weight
oracles and optional oneMKL/cuFFT frontend probes as research tooling. It adds
no model registry entry, product backend, CLI/server/session operation or
binding API. Metal remains deferred.

The standalone project under `tools/ci-heavy/issue579_reuse` builds
`reuse-research-diff` with the existing `crispasr_diff::Ref` comparison machinery.
Its strict model loader is linked into the standalone research library only.
Production root, source, CLI and test CMake files and the product diff dispatcher
are unchanged. See [the build instructions](reuse-diff.md).

The branch's runtime/reference headers, converter, comparison gates and numerical
fixtures are preserved exactly. The import keeps main's existing model notices,
license, reference adapter, phase-stability test, independent-port audit and all
19 historical RE-USE JSON receipts. Those receipts retain their original source,
input and environment scope; this reconciliation does not relabel them as tests
of the new combined candidate. Exact file hashes are recorded in the
[reconciliation receipt](historical/reuse-reconciliation-20261112.json).

The standalone CTest suite runs independent PyTorch-generated frontend, encoder,
selective-scan, Mamba, bidirectional/time-frequency Mamba and decoder fixtures;
the pinned 1,416-tensor inventory; synthetic network and loader rejection
controls; and converter, observer, phase, numerical-gate and missing-SDK
controls. A new small real-GGUF test calls the actual unchanged comparison gate
and rejects errors hidden by aggregate cosine: a tiny-energy row with the wrong
direction, 1% scale drift, a raw phase branch flip, missing/wrong-shaped tensors
and NaN. The separate original-weight oracle executables are built but require
their independently captured model inputs to run.

Local validation passed all 19 unique model-free CTests (18 existing controls
plus the new GGUF gate), and three focused oneMKL profile/probe CTests. A fresh
candidate build also replayed the existing 100 ms independent Kaggle CPU
reference with `mkl-avx2`: all 71 unconditioned and 71 conditioned rows passed.
The unconditioned waveform relative RMS was `1.0561246e-6`, with cosine and
amplitude ratio both 1. The receipt includes the model, reference, executable,
log and installed SDK component hashes. No reference was regenerated and no
new GPU or speech-quality claim follows from this bounded replay.

The model-free workflow is scoped to these research files. The manually
dispatched original-checkpoint audit uses the standalone loader/oracle targets;
its existing pinned-source, intermediate and waveform gates are unchanged.
Neither workflow has been dispatched for this candidate.

Remaining acceptance work includes representative recordings and speech-quality
evaluation, normal/long durations, chunk transitions, additional sample rates,
float/PCM behavior, silence/impulse boundaries and precision policies. The
historical 100 ms and one-second CPU/T4 captures do not establish a full GPU
model backend. Any future product integration also needs CLI, server, C ABI,
bindings, lifecycle and fresh/reused-call coverage. Default native FFT phase
still fails some cross-environment intermediate gates; the qualified AVX2
oneMKL/cuFFT profiles remain explicit experimental choices.
