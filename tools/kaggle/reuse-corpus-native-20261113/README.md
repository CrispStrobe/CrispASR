# RE-USE corpus intermediate diagnostic

CPU kernel `crispasr-reuse-corpus-native-20261113` version 3 was submitted on
2026-11-13 and completed successfully. Both cases pass all 71 unconditioned and
71 conditioned gates; independent archive/waveform checks confirm native/reference
relRMS of `4.67e-7` and `5.58e-7`. See the [bounded result](../../../docs/historical/reuse-corpus-native-20261113.json).
No GPU, CI, dataset upload or
concurrency-policy change is involved.

The job captures new full original-reference GGUF archives for
`speaker-61-white-10db` and `speaker-121-white-0db` from the retained quality
producer output. These are data-informed contrasting diagnostics, selected
after observing quality metrics; they do not establish corpus-wide parity.
Each five-second input must match both its retained WAV hash and raw F32 sample
hash. Noise is never regenerated. The original upstream model and audited
non-fused CPU Mamba adapter remain source/hash pinned. The native comparator
is the unchanged `reuse-long-diff` entry point from `crispasr-diff` machinery.
All 71 unconditioned and 71 conditioned numerical gates remain mandatory.

CPU workers have no internet. The generated `code_file` therefore contains a
compressed CPU/x86 source payload, the MIT-licensed `gguf` Python package and
three small native frontend tools. Existing model data and quality-worker
output are attached directly; no tooling/cache dataset or model upload is
needed. Kaggle rejected the initial 6.7 MB script because source must be below
1 MB. The submitted 809,784-byte script removes unused GPU/other-architecture
and unrelated source files and uses LZMA compression. Numerical code is unchanged.
`prepare.py` produces the single-file script from the readable tracked runner.
Version 1 passed all nine actual-worker frontend controls but failed to compile
because the payload omitted a transitive speaker-identity header. Version 2
added that existing header and passed the actual-worker native build and diff
controls, then exposed a missing inherited Python path in the converter child.
Version 3 supplies the bundled package path to child processes. A focused local
test imports the real bundled `gguf` package in a child; all 13 controls pass
with the wheel fixture configured. The version 3 model comparisons now pass.

Only the optional frontend is precompiled. Build the unchanged
`frontend_mkl.cpp` as `libreuse-frontend.so` using GCC 11, C++17, `-O2 -shared
-fPIC`, official installed oneMKL headers and the public DFTI/VML functions
exported by the installed `libtorch_cpu.so`. Build the unchanged
`frontend_mkl_probe.cpp` and `tests/test-reuse-frontend-profile.cpp` against
that library; define `CRISPASR_REUSE_HAVE_MKL=1` for the profile control.
Neither Intel headers nor its runtime are redistributed. The ELF dependency
floor and exact binary/source hashes are recorded in `build-receipt.json`.
The prepared payload uses the default libstdc++ C++11 ABI; the frontend requires
GLIBC 2.35 and GLIBCXX 3.4.21. The worker runs all nine existing oneMKL controls,
including full five-second batch/single complex spectra and AVX2 startup
rejection, against its own Torch library before building or loading the model.

The retained producer used Torch `2.11.0+cpu`; the worker's actual library hash
and build configuration are recorded and functional controls establish
compatibility. The model/diff code
and pinned ggml are compiled on the worker with the existing standalone CMake
project and an imported optional frontend target. The normal one-second
comparator and product runtime are untouched. The offline runner discovers
preinstalled build tools instead of calling the harness's apt/pip bootstrap;
it still uses its cache/link flags, safe build-job policy and heartbeat.

Fresh full archives are copied and verified before native comparison. A
later failed gate leaves earlier archives, full comparison logs and inputs
available for recovery. Success retains input/reference/native WAVs, raw
native samples, full 142-row logs, provenance, the two approximately 1.05 GB
archives and per-case results. Old stage hashes and norms are environment
diagnostics only; fresh numerical archives decide parity. Fresh versus
retained original waveform differences are also recorded numerically.
Use the existing streaming recovery tool for references above 1 GB.
Subprocess errors preserve completed archives and logs. CPU model calls have
no artificial wall-time or RSS admission limit.

Local preparation checks:

```sh
python tests/test_reuse_corpus_native_kernel.py
REUSE_MKL_FRONTEND_PROBE=<native>/reuse-frontend-mkl-probe \
REUSE_MKL_PROFILE_CONTROL=<native>/reuse-profile-mkl \
MKL_ENABLE_INSTRUCTIONS=AVX2 LD_LIBRARY_PATH=<native>:<torch>/lib \
python tests/test_reuse_mkl_probe.py
python tools/kaggle/reuse-corpus-native-20261113/prepare.py \
  --native <native> --gguf-wheel <gguf-0.18.0-py3-none-any.whl> \
  --output <staging>/qualification.py
```

Submit the generated script with the adjacent metadata. Read the actual
comparison logs before claiming native parity; normal failed-run diagnosis and
retries are supported. Success on these two cases remains a bounded diagnostic.
