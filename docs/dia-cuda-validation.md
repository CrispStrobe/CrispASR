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
and the composed interface require actual speech acceptance before physical
GPU model tests. The CUDA hardware step must prove real native device
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
