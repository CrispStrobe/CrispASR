# CrispASR's own `src/` ships baseline x86-64 on every release leg

**Status: finding, not yet acted on. Needs a decision — deliberate or
oversight?** Surfaced while optimising the Basic Pitch convolutions
(`docs/music-transcription/BASIC_PITCH_CONV_PERF.md`); it is a much larger
result than that optimisation and does not belong buried inside it.

## The claim

Release builds give the vendored **ggml** an explicit instruction set, or full
runtime ISA dispatch. They give **CrispASR's own hand-written kernels under
`src/` nothing at all.** Every one of them compiles for generic x86-64 — SSE2,
4-wide — on every platform leg we ship, including the legs that deliberately
hand ggml AVX2+FMA.

## The evidence

Counted in `.github/workflows/release.yml` at `fed73fb4`:

| | count |
|---|---|
| legs passing `-DGGML_AVX2=ON -DGGML_FMA=ON -DGGML_F16C=ON` | **15** |
| legs passing `-DGGML_BACKEND_DL=ON -DGGML_CPU_ALL_VARIANTS=ON` (runtime multi-variant dispatch, §355/#405) | **2** |
| legs passing `-DCRISPASR_PORTABLE_CPU=ON` (deliberate full baseline) | **3** |
| occurrences of `CMAKE_CXX_FLAGS` or `CMAKE_C_FLAGS` anywhere in the file | **0** |

And in a local default build (`build/CMakeCache.txt`, `GGML_NATIVE:BOOL=ON`):

| object | `%ymm` refs | `%xmm` refs | `vfmadd` |
|---|---|---|---|
| `ggml/src/libggml-cpu.so` | **19886** | — | 5427 |
| `src/CMakeFiles/basic-pitch.dir/basic_pitch.cpp.o` | **0** | 1460 | 0 |
| every object in `src/CMakeFiles/crispasr-core.dir/core/*.o` | **0** | — | — |

The ninja rule for a `src/` TU is, in full:

```
FLAGS = -O3 -DNDEBUG -fPIC -Wmissing-declarations -Wmissing-noreturn -Wall
        -Wextra -Wpedantic -Wcast-qual -Wno-unused-function -O3 -fopenmp
```

No `-march`, no `-mtune`, nothing.

⚠ `GGML_AVX2:BOOL=OFF` in a local `CMakeCache.txt` does **not** mean ggml is
baseline — `GGML_NATIVE=ON` supersedes those switches. Reading that cache entry
as "nothing in this tree gets AVX2" is wrong, and was the first conclusion
reached here before the objdump.

## Why it is probably not deliberate

There *is* an explicit portable-CPU policy, and it is well argued —
`CMakeLists.txt` around the `CRISPASR_PORTABLE_CPU` option, citing #261/#302:
GPU inference initialises ggml-cpu before selecting CUDA/Vulkan, so a CPU
helper compiled for AVX2/BMI2 raises `SIGILL` at model load on older
workstations, before the runtime ISA diagnostic can print anything. The option
forces fifteen `GGML_*` ISA knobs off together so a release job cannot omit one.

That policy is about **ggml**, and it is applied to ggml with care: three legs
opt into the full baseline, fifteen opt into AVX2+FMA+F16C, two ship every
variant and pick at load time. All three strategies are deliberate and
documented.

`src/` is in none of them. It is not baseline *by policy* — it is baseline
because nobody ever gave it a flag. The tell is that the legs which
deliberately hand ggml AVX2 do not hand it to `src/` either, which no policy
would ask for: on those artifacts we have already decided AVX2 is acceptable.

## What it is worth

Measured on the one kernel actually tried (Basic Pitch `contour_conv`, 340 of
the network's 484 MMAC per window): **+45% on the convolutions, +32% end-to-end
on a whole file**, from 8-wide instead of 4-wide, byte-identical output. Any
other conv-heavy or DSP-heavy hand-written kernel under `src/` is in the same
position. Non-exhaustively, `src/core/` alone holds `mel.cpp`, `kaldi_fbank.h`,
`fft.h`, `istft.h`, `cqt*.h`, `hifigan.h`, `seanet_decoder.h`,
`dac_decoder.h`, `audio_resample.cpp`, `cpu_ops.h`, `cpu_attention.h` — all of
them float-heavy, all of them currently 4-wide.

Note also what it is *not* worth: the same measurement showed FMA buying
nothing and AVX-512 being no better than AVX2 on that kernel, because it is
load/dependency-bound rather than FP-bound. So this is "roughly one extra lane
doubling, sometimes", not a free 4×. It still beats per-kernel hand-porting.

## The three options

1. **Runtime dispatch per kernel** — what
   `src/core/cpu_packed_conv1d.h`, `src/core/basic_pitch_conv.h` and the
   `*_simdconv.h` adapters already do: `__attribute__((target(...)))` clones
   plus `__builtin_cpu_supports`. Correct on every leg including the portable
   ones, no build change, no SIGILL risk. Cost: per-kernel work, and it is only
   done for a handful of kernels today.
2. **Per-leg `CMAKE_CXX_FLAGS`, mirroring the ggml flags already chosen for
   that leg.** Cheapest by far, a few lines in `release.yml`, and it makes
   `src/` consistent with an existing decision rather than making a new one.
   Must NOT be applied to the `CRISPASR_PORTABLE_CPU` or `CPU_ALL_VARIANTS`
   legs, which exist precisely to avoid it.
3. **Extend `CRISPASR_PORTABLE_CPU` to cover `src/` explicitly** (it currently
   only loops over `GGML_*`), so at least the portable contract is stated for
   our own code instead of being true by accident. Worth doing regardless of
   1 or 2, because today a future `-march` added to `src/` would silently
   defeat the portable legs.

Suggested order: (3) now, since it costs nothing and closes a real trap;
then decide between (1) and (2) with a measurement on a second kernel.

## Open questions for whoever picks this up

- Is there a reason `src/` was left out that is not recorded in CMake? The
  `CRISPASR_PORTABLE_CPU` comment block is detailed about ggml and silent about
  our own code, which reads like scope rather than intent — but that is an
  inference.
- Do the `CPU_ALL_VARIANTS` legs need a `src/` story at all? Those artifacts
  pick a ggml CPU module at load time; our `src/` code in the same binary
  cannot. Runtime dispatch (option 1) is the only answer that works there.
- `bindings/go/whisper.go`'s cgo `LDFLAGS` are CI-enforced against the CMake
  graph; is there an equivalent place to assert "no `src/` TU carries `-march`"
  so the portable contract is guarded rather than assumed?
