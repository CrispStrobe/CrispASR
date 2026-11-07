# Optional oneMKL frontend experiment

`frontend_mkl_probe.cpp` is an independent CPU STFT experiment, outside the
default runtime and build. Its shared `frontend_mkl.cpp` implementation generates the periodic Hann window with float32
angles and oneMKL VML `vmsCos(..., VML_HA)`, computes the reflected-padding
640-point real FFT through the public DFTI header interface, then returns raw
`atan2` phase. It reads no reference arrays, model weights, or PyTorch library.
It does not canonicalize phase.

Install an external Intel oneMKL developer package supplying `mkl.h` and the
matching runtime. The tested profile uses Linux x86_64, oneMKL 2025.0 build
20241009 and Intel OpenMP 2025.0; other library versions, CPUs and platforms
need qualification. Intel permits royalty-free redistribution under the
[Intel Simplified Software License](https://www.intel.com/content/www/us/en/developer/articles/tool/onemkl-license-faq.html).
The local Conda packages identify the license as
`LicenseRef-IntelSimplifiedSoftwareOct2022`. The experiment does not bundle them.

From the repository root, with `MKLROOT` set to that external SDK:

```sh
c++ -O2 -std=c++17 -I"$MKLROOT/include" \
  tools/ci-heavy/issue579_reuse/frontend_mkl_probe.cpp \
  tools/ci-heavy/issue579_reuse/frontend_mkl.cpp \
  -L"$MKLROOT/lib" -Wl,-rpath,"$MKLROOT/lib" -lmkl_rt \
  -o "$RUN_DIR/frontend-mkl-probe"
"$RUN_DIR/frontend-mkl-probe" --self-test
REUSE_MKL_FRONTEND_PROBE="$RUN_DIR/frontend-mkl-probe" \
  python tests/test_reuse_mkl_probe.py
"$RUN_DIR/frontend-mkl-probe" input.wav "$RUN_DIR/native.f32" ha batch
python tools/ci-heavy/issue579_reuse/frontend_stability.py \
  --audio input.wav --output-npz "$RUN_DIR/torch.npz" \
  --native-f32 "$RUN_DIR/native.f32" --report "$RUN_DIR/comparison.json"
```

`RUN_DIR` is a caller-selected existing artifact directory. Some SDK layouts
put libraries under `lib/intel64`; adjust the linker search path accordingly.
Compile without fast-math. The accepted input is mono PCM16 at 16 kHz,
400..16000 samples, divisible by 80. The output is little-endian float32:
`window[640]`, then `real`, `imaginary`, `log1p_magnitude` and raw `phase`, each
time-major `[1 + samples/80, 321]`.

The executable tests check 84 exact raw DC/Nyquist branch and amplitude
invariants, silence, deterministic batch/single results, finite output, binary
layout and rejected inputs. `la`, `ep` and `libm` are diagnostic window controls;
they are not accepted substitutes for the HA profile. Compare their raw phases
with the same Torch capture to measure the effect of window arithmetic.

On the retained 0.1-second and 1-second inputs, the independently generated HA
window matched all 640 Torch Hann float32 values. Its DFTI real/imaginary arrays
matched local Torch 2.11 CPU STFT byte-for-byte; raw phase had zero branch flips
and maximum error 2.39e-7. This is same-environment evidence on two inputs.
The earlier captured-window experiment used an exported Torch array; that
dependency has been removed from this executable.

The integrated research diff, using both independently computed magnitude and
phase, passed all 71 existing unconditioned intermediate/waveform gates on the
0.1-second local reference. Its waveform relative RMS error was 6.60e-7
(maximum sample error 1.61e-7); all 71 phase-conditioned comparisons also passed.
This is a bounded CPU research profile, not a product backend qualification.

The local and retained Kaggle Torch windows were already byte-identical, yet
their FFTs chose opposite raw phase branches in 47 first-frame bins. This MKL
profile reproduces the local reference, retaining those 47 differences from
Kaggle. Its waveform differs from the Kaggle-phase original by about 1.49%,
exceeding the unchanged 0.5% waveform gate. One waveform cannot meet both
incompatible reference gates. An optional MKL implementation can qualify a
pinned CPU reference profile; these results do not establish portable parity,
GPU support, or a reason to replace the default frontend.

The tested runtime mapped MKL/OpenMP shared objects totaling approximately
220 MiB on disk; this is not a measurement of resident memory.

The same implementation is explicitly opt-in in the research diff. Configure
the main repository with `-DCRISPASR_REUSE_MKL_FRONTEND=ON` and an external
`MKLROOT`, then request `CRISPASR_REUSE_FRONTEND=mkl` when running
`crispasr-diff reuse`. The default profile is `native`, including in an
MKL-enabled build. An unknown profile or requesting MKL in a default build
fails clearly before loading the reference or weights. The optional library
links only to research targets, not the product library or CLI backend.

An isolated build avoids compiling unrelated model backends:

```sh
cmake -S tools/ci-heavy/issue579_reuse -B "$RUN_DIR/build" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DCRISPASR_REUSE_MKL_FRONTEND=ON
cmake --build "$RUN_DIR/build" -j2
ctest --test-dir "$RUN_DIR/build" --output-on-failure
CRISPASR_REUSE_FRONTEND=mkl "$RUN_DIR/build/reuse-research-diff" model.gguf ref.gguf
```

This uses the repository's recursive ggml checkout by default. An explicitly
selected pinned checkout can be supplied with `CRISPASR_REUSE_GGML_SOURCE_DIR`.
The isolated driver calls the same `reuse_diff::run` as `crispasr-diff`.
The `mkl/` rows use independently computed magnitude **and** raw phase. The
`mkl-conditioned/` rows remain diagnostic. All existing 71 intermediate and
waveform acceptance gates are unchanged; a failed group makes the run fail.
The optional CTests exercise both unavailable-profile rejection without an SDK
and enabled-profile routing/invalid-input ownership with the SDK, plus the
standalone executable controls. No product backend or GPU path is enabled.
The SDK control also checks that the caller's local MKL thread setting survives
both successful execution and rejected nonfinite input. All DFTI statuses are
checked; errors fail the run without substituting another frontend or changing
the existing comparison thresholds.
