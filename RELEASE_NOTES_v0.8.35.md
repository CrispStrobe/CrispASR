# CrispASR v0.8.35

A performance release for the Basic Pitch convolutions, and one capability
that was being computed and then thrown away.

The theme is the same one v0.8.34 named: **claims that were only claims**.
Two of them here. MT3's whole reason for existing is that it is
multi-instrument, and the ABI dropped the instrument before any caller could
see it. And CrispASR's own `src/` compiles baseline x86-64 on every release
leg while the ggml it vendors gets AVX2 on fifteen — which is documented
here rather than fixed, because it is a decision rather than a bug.

---

## Basic Pitch convolutions: SIMD and threading

Runtime-dispatched AVX2 / AVX-512 kernels plus `core_parallel::for_each_chunk`
over the disjoint `(oc, h)` output rows. Reuses `core_cpu_conv1d`'s existing
`Isa` / `isa_available` dispatch — no new scheme, no `-march=native`, so the
shipped binary stays portable.

Hermetic CI A/B, one arm per process, cold run discarded, median of at least
three, within-run variance 0.05%:

| | reference | SIMD, 1 thread | 4 threads |
|---|---|---|---|
| ubuntu-24.04, 4 cores | 104.8 ms | 1.66x | **3.59x** |
| macos-14 (M1), 3 cores | 94.8 ms | 1.01x | **2.67x** |

Output is **byte-identical** on both platforms at 1 and 4 threads, so the
gate defaults **on**; `CRISPASR_BASIC_PITCH_FASTCONV=0` is the way back and
`bp_conv2d_ref` is kept verbatim. `n_threads` now reaches all six convolution
call sites — previously it reached only the GGUF loader.

Three results worth more than the speedup:

- **A contended box understates the *faster* kernel.** SIMD-only measures
  1.45x on a loaded shared machine and 1.66x clean: the faster kernel has
  less slack to hide a stall in.
- **arm64 gains nothing from SIMD (1.01x).** NEON is already baseline there,
  so its entire 2.67x is threading. Reading the x86 number and planning for a
  phone would plan wrong.
- **FMA buys 0-3%**, inside run-to-run spread. This kernel has too little
  arithmetic reuse to be FP-throughput bound — `contour_conv` has 8 output
  channels, so each input element is reused ~8 times against 936 in a real
  GEMM.

**Caveat shipped with the flip:** `n_threads` defaults to 4, and at 4 threads
the per-call `std::thread` spawn (six convs per window, ~84 spawns per 22 s
file) costs ~75% more CPU than 2 threads for ~9% less wall. Batch and server
callers should pass 2 until this routes through `core/worker_pool.h`.

## MT3's per-note instrument, which was being discarded

`mt3_note_event` carries `program`, `instrument` and `is_drum`. All three were
dropped at the ABI boundary, because the note record is a flat
`[start_ms, end_ms, midi, velocity]` and widening it would break every
existing reader.

It is not widened. `crispasr_session_piano_note_programs` hands out a
**parallel** int array; a caller that does not ask is unaffected.

```
0-127   General MIDI program
128     percussion (GM channel 10, which carries no meaningful program)
-1      this model does not identify an instrument
```

`-1` rather than `0` because `0` is *Acoustic Grand Piano* and would be
indistinguishable from a real answer. `piano-transcription` and `basic-pitch`
report `-1` throughout.

Mirrored in the **Dart** and **C#** bindings. Both degrade rather than probe:
against a library predating the export they report `-1` for every note, so a
caller reads the sentinel instead of checking for the symbol. Go and Rust
expose no piano API and are unchanged.

**Why it matters:** benchmarked against MusicNet's test split by
`mir_eval.transcription` rules and validated against the official reference
implementations to 0.2 F1 points, MT3 scores **76.5% note-level F1 against
Basic Pitch's 44.2%**, at 0.26x real time. Multi-instrument transcription is
what earns that, and it was arriving flattened to a single part.

## Two guard defects around the piano accessors

- `piano_last_notes` was nested inside **both** `CA_HAVE_CREPE` and
  `CA_HAVE_PIANO_TRANSCRIPTION`, so note storage accidentally depended on
  CREPE being compiled in — a build with basic-pitch or MT3 but without CREPE
  would not have compiled the very branches that push into it.
- `crispasr_session_piano_notes` was guarded by `CA_HAVE_PIANO_TRANSCRIPTION`
  alone while `crispasr_session_piano_n_notes` accepted all three note-event
  backends. A build with MT3 or basic-pitch but without piano-transcription
  reported a note count and then handed back `nullptr`.

Both now guard on all three.

## Raised, not fixed: `docs/improvements/SRC_ISA_GAP.md`

Release builds give vendored **ggml** an explicit instruction set or full
runtime dispatch, and give **CrispASR's own hand-written kernels under `src/`
nothing at all**. In `release.yml`: **15** legs pass
`-DGGML_AVX2=ON -DGGML_FMA=ON -DGGML_F16C=ON`, **2** ship
`GGML_BACKEND_DL + GGML_CPU_ALL_VARIANTS`, **3** pass
`CRISPASR_PORTABLE_CPU=ON`, and **zero** mention `CMAKE_CXX_FLAGS`.
Confirmed on a local default build: `libggml-cpu.so` carries **19,886**
`%ymm` references, the whole 23 MB `libcrispasr.so` carries **56**.

There *is* a considered ISA policy — `CMakeLists.txt` documents why
(#261/#302: ggml-cpu initialises before CUDA/Vulkan is selected, so an AVX2
CPU helper raises `SIGILL` at model load before the runtime ISA diagnostic
can print). It is applied to ggml three different ways. `src/` is in none of
them, and the tell that this is scope rather than intent is that the legs
which *deliberately* hand ggml AVX2 do not hand it to `src/` either.

The doc lays out evidence, three options and open questions. It does not
decide. Cheapest first step regardless: `CRISPASR_PORTABLE_CPU` loops over
`GGML_*` only, so a future `-march` in `src/` would silently defeat the
portable legs.

## A green tick that measured nothing

Flipping the conv gate's default silently broke the A/B workflow that proved
it: the reference arm passed *no* env var, correct only while the gate
defaulted off, so one run reported **1.00x for every arm and passed**. Fixed
by pinning `=0` explicitly *and* asserting the arm name the harness prints
matches the one requested.

A gate's default is part of every harness that reads it, and "no difference"
is the result a broken benchmark produces most convincingly.
