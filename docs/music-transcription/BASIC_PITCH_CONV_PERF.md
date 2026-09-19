# Basic Pitch — convolution path A/B

Branch `perf/basic-pitch-conv`. Status: **fast path implemented, byte-identical,
measurably faster, default still OFF.** See the verdict at the bottom for what
would justify flipping it.

## 1. The premise in the file header was wrong

`src/basic_pitch.cpp` opened with:

> The whole network is six **small** convolutions … The expensive part is the
> CQT front end (core/cqt2010v2.h), not the network.

Both halves are false, and the second one had shaped the file (it is the stated
reason the network runs in plain C++ loops with no attention paid to them).

Counted from the six call sites in `bp_forward_window`, one 43844-sample window
(T = 172):

| layer | shape | stride_w | out | MMAC | share |
|---|---|---|---|---|---|
| `contour_conv` | 8→8, 3×39 | 1 | 172×264 | **340.0** | 70.2% |
| `onset_conv`   | 8→32, 5×5 | 3 | 172×88  | 96.9 | 20.0% |
| `note_conv`    | 1→32, 7×7 | 3 | 172×88  | 23.7 | 4.9% |
| `note_out`     | 32→1, 7×3 | 1 | 172×88  | 10.2 | 2.1% |
| `contour_out`  | 8→1, 5×5  | 1 | 172×264 | 9.1 | 1.9% |
| `onset_out`    | 33→1, 3×3 | 1 | 172×88  | 4.5 | 0.9% |
| **total** | | | | **484.4** | |

The CQT is roughly 6 MMAC against that. Measured with
`CRISPASR_BASIC_PITCH_TIMING=1` on the 22.3 s GuitarSet clip (14 windows), the
per-window split is `cqt ≈ 50–80 ms`, `hstack ≈ 3–6 ms`, `conv ≈ 310–1470 ms`,
`activations ≈ 4–16 ms`. Even at the contended end of that range the
convolutions are 5–20× the front end. `contour_conv` alone is 70% of the
network. The header comment has been corrected in place, and
`tests/test-basic-pitch-conv.cpp` now asserts the MMAC budget so a shape edit
cannot silently invalidate it.

## 2. What was actually left on the table

Two things, and the first is the bigger surprise:

1. **The build ships baseline x86-64.** `CMAKE_CXX_FLAGS` is empty in
   `build/CMakeCache.txt` and `GGML_AVX/AVX2/AVX512` are all `OFF`, so
   `basic_pitch.cpp` compiles with `-O3 -DNDEBUG -fPIC … -fopenmp` and nothing
   else. GCC auto-vectorised the contiguous `stride_w == 1` inner loop to 4-wide
   SSE (`mulps`/`addps` on `%xmm`) and could go no wider. This is not specific
   to this backend — it is the whole tree.
2. **`n_threads` never reached the convolutions.** It appears twice in
   `basic_pitch.cpp`: a default, and `core_cpu_backend::set_n_threads`, which in
   this backend governs GGUF loading only. The conv loops were single-threaded,
   which is why 2 and 4 threads measured the same.

## 3. What was built

`src/core/basic_pitch_conv.h` (new, weight-free, no ggml) holds both paths:

- `bp_conv2d_ref` — the original loop, untouched, still the default.
- `bp_conv_fast` — per-function `__attribute__((target(…)))` AVX2 / AVX-512 /
  AVX2+FMA kernels selected at runtime through `core_cpu_conv1d::isa_available`
  (the existing house dispatch from `core/cpu_packed_conv1d.h`; no new scheme),
  plus `core_parallel::for_each_chunk` over the `(oc, h)` index space.

Gates:

| variable | effect |
|---|---|
| `CRISPASR_BASIC_PITCH_FASTCONV=1` | select the fast path (default: off, reference loop) |
| `CRISPASR_BASIC_PITCH_CONV_ISA=scalar\|avx2\|avx2fma\|avx512` | override kernel dispatch for A/B |
| `CRISPASR_BASIC_PITCH_TIMING=1` | per-window cqt/hstack/conv/activation split to stderr |

`-march=native` was NOT added anywhere; the shipped binary stays portable and
the ISA choice is a runtime `__builtin_cpu_supports` decision.

### Bit-identity is by construction, not by tolerance

Per output element the accumulation order is unchanged — `ic`, then `kh`, then
`kw` ascending, same zero-weight skip, same in-bounds predicate. Vectorising
across `wo` cannot reorder anything, because distinct `wo` are distinct
accumulators; the kernel only holds the running `orow[wo]` in a register across
the `kw` loop instead of round-tripping it through L1 once per tap. The default
AVX2 kernel uses separate `mul` + `add` and its target has no FMA, so it is
byte-for-byte equal to the reference. `avx2fma` and `avx512` are **not**
bit-identical (an FMA rounds once where mul+add rounds twice, and GCC contracts
inside an `avx512f` target clone because AVX-512F carries FMA); they are never
auto-selected and exist only to price that rounding.

## 4. Two mistakes worth recording

Both were caught by measurement, and both would have shipped as "wins".

- **`__attribute__((optimize("fp-contract=off")))` cost 3.7×.** Copied from
  `core/cpu_packed_conv1d.h`, where it is correct. On GCC the `optimize`
  attribute *replaces* the function's optimisation options rather than adding to
  them, so the annotated tap lost `-O3`: 226 ms → 1230 ms for the six layers.
  Contraction is instead kept off by construction (no `-mfma` in the TU, and
  `avx2` has no FMA of its own).
- **The first kernel was 3.7× SLOWER than the loop it replaced.** It put `kw` on
  the inside, accumulating one output element in a register. That reads well and
  cannot auto-vectorise — a branchy reduction into a scalar. The reference's own
  shape (`kw` outer, `wo` inner and contiguous) is the fast one; the fallback
  now *is* that shape, restricted to a range.

Also: `core/cpu_packed_conv1d.h` `#undef`s `CRISPASR_CPU_PACKED_CONV1D_X86` at
the end of the header, so `#if CRISPASR_CPU_PACKED_CONV1D_X86` in a consumer
silently evaluates to 0 and compiles every AVX kernel out. The first "AVX2"
measurements were pure scalar. Consumers must carry their own probe.

## 5. Numbers

Machine: 4-core Xeon Skylake (avx2 + avx512f + fma), **heavily contended** —
load average 15–26 throughout, 11 concurrent agent sessions, 8.8 GB swap in use.
Wall-clock on this box swings 2–4× run to run and fabricates both wins and
losses, so **CPU time (`CLOCK_PROCESS_CPUTIME_ID`, min of ≥8 in-process
iterations, ≥3 separate processes per arm, cold process discarded) is the
primary metric**; it reproduced to ±2%. Wall is reported alongside and should be
read as a lower bound on the benefit, not as the result.

### Convolutions only (`484.4 MMAC` per window)

| arm | CPU ms/window | GMAC/s | vs reference |
|---|---|---|---|
| reference (SSE, 1 thread) | **237** | 2.04 | 1.00× |
| fast, portable fallback, 1 thread | 233 | 2.08 | 1.02× |
| fast, **avx2**, 1 thread | **163** | 2.96 | **1.45×** |
| fast, avx512f, 1 thread | 167 | 2.90 | 1.42× |
| fast, avx2+fma, 1 thread | 163 | 2.98 | 1.46× |

Per layer (CPU ms, reference → avx2): `contour_conv` 148.3 → **90.5** (1.64×),
`contour_out` 3.80 → 2.35, `note_out` 6.38 → 6.46, `onset_out` 2.81 → 2.53,
and the two `stride_w == 3` layers unchanged by design — `note_conv` 13.7 → 13.2
and `onset_conv` 56.1 → 52.1. The entire win is `contour_conv`.

Two results worth stating plainly because they contradict the obvious guesses:

- **FMA buys nothing here** (163 vs 163 ms). The kernel is not FP-throughput
  bound. So bit-identity is free, which is why the default kernel is the
  bit-identical one.
- **AVX-512 is no better than AVX2** (167 vs 163 ms) — expected on Skylake-SP,
  where 512-bit FP downclocks. Widening past 8 lanes is not the lever.

### Whole file, end-to-end (22.3 s GuitarSet clip, 14 windows, f16 GGUF)

| arm | CPU s | wall s (contended, lower bound) |
|---|---|---|
| reference, 1 thread | **4.36** | 9.76 |
| fast avx2, 1 thread | **3.31** (1.32×) | 7.87 |
| fast avx2, 2 threads | 3.31 | 7.21 |
| fast avx2, 4 threads | 3.37 | 6.89 |

1.45× on the convolutions dilutes to 1.32× on the file because the CQT, WAV
read, resample and note-creation are untouched.

### Threading: honestly unmeasurable on this box

Total CPU rises only 162.4 → 167.7 ms (+3%) from 1 to 4 threads, so the split is
close to free. But achieved parallelism (`cpu_min / wall_min`) never exceeded
**0.94×** at any thread count — the machine has no spare cores, and even the
1-thread arm measured 0.71×. **The wall-clock threading win is therefore NOT
demonstrated.** On an idle 4-core box the arithmetic implies ~43 ms/window, but
that is a projection, not a measurement, and it is not being claimed.

### Proof the work actually happened

Not an exit code (HARD RULE 8 — an `echo BUILT` after a failed link printed
green during this work, and was caught only by `ls` on the binary):

- `n_notes = 153` with the full event list on every arm, and per-window
  `CRISPASR_BASIC_PITCH_TIMING` lines ×14, matching the 14 windows
  `bp_run_file` derives from hop = 43844 − 30×256 = 36164.
- The arms differ in time while producing identical output, so neither is
  short-circuiting.
- `tests/test-basic-pitch-conv.cpp` reports `39 assertions in 2 test cases`.

## 6. Output equality

The strongest available proof, and stronger than a cosine: **byte equality**.

Hermetic unit test, on the six real shapes at the production window, 1 and 4
threads, including a forced zero weight to exercise the shared skip:

```
All tests passed (39 assertions in 2 test cases)
```

The guard was checked to FAIL rather than merely pass — forcing
`CRISPASR_BASIC_PITCH_CONV_ISA=avx2fma` (a kernel known to differ) produces
`REQUIRE( std::memcmp(...) == 0 )` … `-6 == 0` on `contour_conv`.

End-to-end on `/mnt/storage/tuner-bench/datasets/audio/00_BN1-129-Eb_comp_mic.wav`
through the real `basic_pitch_transcribe`, comparing the **raw heads** (before
peak-picking, per the dev-harness preference) by FNV-1a over the raw float bits
*and* the L2 norm (HARD RULE 2b — cosine is scale-blind), plus every note event:

```
contour:    n=510048 fnv=e12d82dc76b68767 |x|=82.767633814
note_head:  n=170016 fnv=166f457729d5e528 |x|=61.967697523
onset_head: n=170016 fnv=1f8da0da28088393 |x|=50.228157125
n_notes=153
```

| arm | result |
|---|---|
| fast avx2, 1 thread | **byte-identical** (heads + all 153 events) |
| fast avx2, 4 threads | **byte-identical** |
| fast portable fallback, 3 threads | **byte-identical** |
| fast avx2+fma, 1 thread | differs: same 153 notes, same midi/start/end/velocity, `amplitude` differs in the 8th decimal; head norms agree to 9 digits |
| fast avx512f, 1 thread | differs, same character |

### A correction to the brief's baseline

The brief's expected baseline was **151** note events. This build (repo `main`
at `fed73fb4`, `basic-pitch-f16.gguf`) produces **153** on both arms. The
earliest event matches exactly — midi 51 at 34.8 ms, velocity 74 — as does the
distinct-pitch set. The 151 came from the Dart harness against the **published
pub 0.8.33 `.so`**, i.e. a different build, and quite possibly a different
resampler path into 22.05 kHz. The discrepancy is *not* caused by this change:
the reference arm of this binary, with the gate off, also gives 153. Whoever
owns that harness should re-baseline against a locally built `.so` before
treating 151 as ground truth.

## 7. Verdict

### Should the default flip?

**The evidence supports it, and it has deliberately not been flipped here.**
The dev guide's bar is "wins on speed AND quality". Quality is not merely
non-regressed, it is byte-identical, so no regression is possible. Speed is
+45% on the convolutions and +32% on the file, on a stable metric, in separate
processes. That clears the bar.

Two reasons it is left gated, per the brief and rule 3a:

1. The brief asked for the old path to stay default until proven, and half the
   design — threading — is **not** proven on this box.
2. Every measurement here is from one contended 4-core Skylake VPS. A clean run
   (CI, or a Kaggle box) confirming the AVX2 CPU-time win and showing real
   wall-clock scaling at `n_threads = 4` is a cheap way to remove all doubt.

Flipping is then a one-line change to `bp_fastconv_on()` (to
`!crispasr_env::explicitly_off(...)`), with the gate kept as the way back.

### `ggml_conv_2d` or direct SIMD, for *this* backend?

**Direct SIMD. `ggml_conv_2d` is the wrong answer here, and the memory figure is
only half the reason.**

The im2col matrix for `contour_conv` is `(H·W_out) × (IC·KH·KW)` =
`45408 × 936` floats = **170.0 MB per window** (85 MB at F16), against a current
largest activation of 363k floats = **1.45 MB**. That is a 117× working-set
blow-up, on a box that is already OOM-killing sessions.

But the decisive argument is that the blow-up buys nothing. im2col earns its
memory by amortising the materialised patch matrix across many output channels.
`contour_conv` has **OC = 8**. The resulting GEMM is `M=45408, K=936, N=8` — so
skinny in N that every element of the 170 MB matrix is read essentially once.
That is the textbook worst case for im2col: pay 170 MB of writes and 170 MB of
reads to save nothing. Add the per-conv graph-build + `sched_alloc` overhead the
dev guide's "inverse-default regime" names, ×6 convs ×14 windows per file, and
it is a clear loss for the 70% of the work that matters.

The nuance: **`onset_conv` is the one place `ggml_conv_2d` could plausibly
win.** Its im2col is `15136 × 200` = 12.1 MB — tractable — and with OC = 32 there
is real amortisation. It is 96.9 MMAC (20% of the network), currently runs at
1.86 GMAC/s, and is untouched by this change because `stride_w == 3` gives a
gathered inner loop. If someone wants the next 20%, that is where to look, and
routing *only that layer* through `ggml_conv_2d` (or a stride-3 gather kernel)
is the experiment — not a wholesale port.

### Does this generalise to mt3 / piano-transcription / crepe / beat-this?

A recommendation with its evidence, not work done:

1. **The `CMAKE_CXX_FLAGS`-is-empty finding is tree-wide and is the cheapest
   lever in this document.** Nothing in this build gets AVX2 — not this backend
   and not ggml (`GGML_AVX2:BOOL=OFF`). Any conv-heavy or DSP-heavy backend in
   this tree is running 4-wide. Before porting anything to ggml, someone should
   establish whether that is deliberate (portable-binary policy, in which case
   runtime dispatch like this one is the only route) or an oversight. This is
   worth a targeted audit; `linux-isa-fallback-verify.yml` already exists as the
   place to hang it.
2. **The im2col arithmetic is the screening test, and it is one multiplication.**
   Compute `(H·W_out)·(IC·KH·KW)·4` bytes and compare with `OC`. Large matrix and
   small `OC` ⇒ direct SIMD. Modest matrix and `OC ≥ 32` ⇒ `ggml_conv_2d` is
   worth measuring. CREPE (1-D convs, wide channels) and piano-transcription
   (2-D CNN into GRUs, wide channels) are likely on the favourable side of that
   line; a Basic-Pitch-shaped 8→8 layer is not. mt3 is a T5 transformer and is
   attention-bound, so it is precisely the case the "inverse-default regime"
   warns regresses — it should be screened by a profile, not ported on principle.
3. **Whatever is done, do it behind this shape of gate.** The byte-equality
   harness here (weight-free header + hermetic memcmp test + raw-head FNV +
   norms end-to-end) cost very little and caught two of my own regressions. It
   generalises to any of these backends unchanged.

## Reproducing

```bash
export CRISPASR_MODELS_DIR=/mnt/storage/gguf-models
ctest --test-dir build -R test-basic-pitch-conv --output-on-failure   # byte-equality
CRISPASR_BASIC_PITCH_TIMING=1 build/bin/crispasr --backend basic-pitch <wav>   # stage split
CRISPASR_BASIC_PITCH_FASTCONV=1 build/bin/crispasr --backend basic-pitch <wav> # fast path
```

Run each arm as a **separate process** and judge by CPU time, not wall, unless
the box is idle.
