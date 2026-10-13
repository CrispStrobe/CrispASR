# Dia CUDA validation package

This experimental package composes current accepted main with the opt-in Dia
cache candidate. The four neural/sampler/diff source files must match CPU
control source `2205d920d879108b73651ebb9f3eb9499da4e5f5` exactly. Three small
contracts accept those files and reject changed cache precision or a missing
sampler header before a hosted build starts.

`dia-cuda-speech-validation.yml` builds on GitHub with CUDA 12.4.1 for SM60
and SM75, Ninja and ccache. It packages CLI, server, shared C ABI, diff,
quantizer and same-source Python bindings; SONAME aliases and relative RPATH
are preserved. Dependency closure is checked, user-space CUDA libraries are
bundled, and the driver remains external. Source files, actual compiler
version, library hashes and archive size/SHA are retained in the artifact.

A successful build establishes compilation and packaging only. CPU cache
control [38357576078](https://github.com/CrispStrobe/CrispASR/actions/runs/38357576078)
and the composed interface remain separate acceptance requirements.
Per the user’s explicit steering, actual GPU cache triage can run while CPU
controls execute; broader model acceptance still follows their results. The CUDA hardware step must prove real native device
allocation and retain strict original cosine/magnitude plus actual CLI/C-ABI
ASR results. Profiling follows correctness. Kaggle is reserved for real GPU
work; CPU validation remains on GitHub. Production model pins/defaults and
Q4 acceptance stay unchanged.

The local guard command is:

```bash
TMPDIR=/mnt/storage/dia-scratch python3 tests/test_dia_cuda_bundle.py -v
```

Use an existing writable scratch directory. The guard loads no weights and
builds no native runtime. The workflow runs it in the runner’s scratch area.

The `dia-cuda-cache-control` Kaggle driver uses the shared harness and existing
Torch installation. It runs native CUDA cache OFF/ON/OFF stages and real
CLI/C-ABI speech, requiring at least 6,000 MiB observed allocation for the
full-F32 model and actual CUDA scheduler splits. The model and sampler/ASR
gates remain fixed. It records compiler/physical-device incompatibility before
large downloads and does not retry to select hardware. The push helper renders
an immutable source pin only from clean, pushed source.

## Explicit replay device correction

The first actual T4 run uses source 3595fa25b. Its diff dispatcher clears the
GPU environment request and forces CPU; the worker's allocation/scheduler
checks must reject that path. Complete v1 terminal logs/output must be saved
before resubmitting. It cannot establish CUDA model execution.

The replacement captures the explicit GPU request before the replay guard,
applies the existing force-GPU gate and context flag, and restores the parent
environment. Default replay remains CPU. Source-compiled controls cover request
aliases, precedence, malformed settings, force handoff and restoration. Neural,
cache and sampler implementation files remain identical to the fixed CPU control;
the changed diff dispatcher is independently hashed in package provenance.

The kernel keeps the same fixed model/ref/prompt/seed/filter controls and checks
actual process GPU memory and CUDA scheduler splits. It warms and refreshes the
existing HF compiler-cache dataset per the maintainer guide; cache artifacts
are separate from public model proofs. Audited source-weight/template blobs
inside this run's ephemeral scratch may be released after the export child
exits, with size/hash receipts, to leave space for native models and outputs.
No external/shared cache path is deleted. No Torch reinstall is performed.
