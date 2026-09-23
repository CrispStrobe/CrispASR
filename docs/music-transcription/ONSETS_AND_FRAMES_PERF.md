# Onsets & Frames — the ggml lifecycle work, measured

**What this is.** The A/B record for the four-step plan
`docs/ggml-optimisation-playbook.md` §7 sets out for `src/onsets_and_frames.cpp`,
executed in the playbook's order, with a number for each step separately so it
is visible which one paid. Two of the four paid, one was already done, and one
does not pay and was not adopted.

**What it is not.** A claim that these numbers transfer. See "The measurement
box" below — this VPS is not a measurement instrument and the playbook says so
(§7, last paragraph). Every number here is a within-box A/B against a baseline
taken the same way on the same box in the same session, which is the only thing
it can honestly support.

---

## Summary

| step | what | verdict |
| --- | --- | --- |
| 1 | hoist the three allocators | **paid** — −7.5% at 10 s, −10.3% marginal |
| 2 | apply `n_threads` | **already done** — this file never had the §6.4 bug. No change. |
| 3 | fuse the eight per-forward computes | **does not pay** — measured at 0.005% of runtime. Not adopted. |
| 4 | the BiLSTM | **paid, the largest one** — −17.4% at 10 s, −23.6% marginal. Opt-in. |

Cumulative, single-threaded, CPU-seconds per audio-second:

| | baseline | after 1 | after 1+4 | change |
| --- | --- | --- | --- | --- |
| 10 s clip | 0.6938 | 0.6416 | **0.5301** | −23.6% |
| 60 s clip | 0.6218 | 0.5611 | **0.4352** | −30.0% |
| marginal | 0.6073 | 0.5450 | **0.4163** | −31.5% |

*Marginal* is `(cpu₆₀ − cpu₁₀) / 50 s` — the per-audio-second cost with the fixed
model-load and front-end cost removed, i.e. the number that scales with clip
length. The gap to native ONNX Runtime's 0.103 goes from **5.9× to 4.0×**
marginal. It is not closed.

> **On Apple Silicon, and on quantisation.** `PIANO_METAL_AB.md` §4 measures
> O&F on a GitHub `macos-14` runner (chip "Apple M1 (Virtual)", 3 cores,
> 3 threads) at **0.161× real time at f32** — about 6× faster than real time —
> and finds **no measurable speed difference between f32, q8_0 and q4_0** —
> two independent runs disagree by more than any effect, so the first run's
> apparent 8% q4_0 penalty is noise and is retracted there. That null is
> consistent with a model whose cost is convolution and a host-side LSTM
> recurrence rather than weight bandwidth. What q8_0/q4_0 do buy, and this
> reproduces, is memory: peak RSS 300 → ~220 MiB on a 30 s clip. That doc also
> records why there is no GPU column.

---

## The measurement box

Hetzner VPS, 4 vCPU Skylake-SP (AVX-512F, **no** AVX-512 VNNI), 7 GB RAM, 11 GB
swap, `/mnt/volume1` at 97%. Heavily contended: several agent sessions run
concurrently, and wall-clock swings 2–4× run to run — a 4 s clip that costs
2.8 CPU-seconds took 19 s of wall time while the F1 job was running. **CPU time
is the primary metric throughout and wall is reported only as a lower bound.**

Method, per `BASIC_PITCH_CONV_PERF.md` §5 and the dev guide's A/B rules: one arm
per process, minimum of 4 separate processes per arm, `MKL_NUM_THREADS=1` pinned
on every run, `-t 1` unless stated. Audio is MusicNet test piece 2303 resampled
to 16 kHz mono 16-bit, clipped to 10 s and 60 s from the same 5 s offset.

**`MKL_NUM_THREADS=1` is not optional here.** CrispASR issue **#453**: Debian's
threaded MKL `sgemm`, linked alongside `libgomp`, silently multiplies
`core_mel`'s upper mel bins by the thread count. `tools/oaf_parity.py:84-102`
already pins it and calls it "a host defect, not a model one". Without it the
reference and the runtime disagree for reasons unrelated to any change here.

---

## Before anything: the per-layer dumper

O&F was the one model of the six with **no per-layer parity path at all**
(playbook §5.2). What it had — `tests/oaf_parity_dump.cpp` plus
`tools/oaf_parity.py` — compares a mel and five post-sigmoid heads against
native onnxruntime. That is end-of-pipeline: a regression inside a ConvStack or
a BiLSTM reads as "the onset head moved", not as a layer, and the script has no
gate (it exits 0 unless the dump subprocess itself fails).

So the dumper was built first, on the shape §5.3 documents:

```bash
PY=/mnt/volume1/miniconda/bin/python
export MKL_NUM_THREADS=1

# 1. the mel this runtime actually computes
build/bin/oaf-parity-dump model.gguf audio16k.wav /mnt/storage/oaf-parity/ggml 1

# 2. the per-stage reference, run on THAT mel
$PY tools/reference_backends/onsets_and_frames.py \
    --onnx   /mnt/storage/tuner-bench/onnx/onsets_and_frames.onnx \
    --mel    /mnt/storage/oaf-parity/ggml.mel.f32 \
    --output /mnt/storage/oaf-parity/ref.gguf

# 3. per-stage cosine, gated at COS_THRESHOLD = 0.999
build/bin/crispasr-diff onsets-and-frames model.gguf \
    /mnt/storage/oaf-parity/ref.gguf audio16k.wav
```

26 stages: `mel`; `{onset,offset,activation,velocity}_{conv0,conv1,conv2,fc}`;
`{onset,offset,frame}_bilstm`; `combined_input`; and the five `*_logits`.

Three design points worth keeping:

- **It is a standalone script, not the `dump_reference.py` `dump()` contract** —
  the same exception `basic_pitch.py` takes, and for a concrete reason. The ONNX
  graph's input is a log-mel, not audio, so a `dump(model_dir, audio, ...)`
  would have to reimplement `core_mel` in Python. That risks a front-end
  mismatch of its own *and* conflates the front end with the model. It is handed
  the C++ runtime's own mel instead, which is `oaf_parity.py:5-11`'s two-stage
  design extended per layer — and the `mel` stage is still compared first, so a
  front-end change surfaces as itself.
- **Every captured tensor gets `ggml_set_name` AND `ggml_set_output`.** Without
  the latter gallocr reuses the memory and a later layer overwrites the value
  (§5.0, §6.6).
- **Capture is opt-in behind a null sink**, so the production path constructs no
  extra graph nodes. ggml executes every graph output (§6.8), so a baseline that
  still builds them is not a baseline.

**Cosine is scale-blind**, so the report prints `|mine|` and `|ref|` RMS columns
next to it. HARD RULE 2b: a CQT missing librosa's `scale=True` once passed
correlation 0.9999 while every bin was low by up to 152×.

Baseline, 4 s of MusicNet 2303, f32: **all 26 stages PASS at cos = 1.0000000**,
max_abs ≤ 1.9e-05, every `|mine|`/`|ref|` pair identical to six figures.

⚠ **Be clear about what the `mel` stage in this harness does and does not
prove.** Because the reference is dumped from this runtime's own mel, that stage
compares `core_mel` against itself and reports `max_abs = 0.000e+00` — it
catches a *change* to the front end between the dump and the diff, and nothing
else. The real front-end check is still `tools/oaf_parity.py`, which compares
against torchaudio and where the honest number is `max 2.946e-02, cos
1.00000000` — not bit-identical, as it should not be. Run both; they answer
different questions.

---

## Step 1 — hoist the three allocators

Three `ggml_gallocr_new` sites, nine `ggml_gallocr_free` sites, zero
`ggml_gallocr_reserve` calls, one of the news **inside** the per-chunk loop. A
60 s clip paid new/free 43 times per forward: once per conv-stack chunk (8
chunks × 4 stacks), twice per BiLSTM direction, once per head.

Fixed on the `src/hft_transformer.cpp:108-115` pattern — whose own comment names
this file's fresh-allocator-per-chunk as the reason it exists. One allocator per
graph shape as a context member (`alloc_conv`, `alloc_proj`, `alloc_head`),
created lazily, freed only at teardown.

Deliberately a `gallocr` and not a scheduler, and the cgraph is still **rebuilt
every call** over the reused metadata arena rather than cached: caching a cgraph
across calls leaves the previous allocation's `buffer`/`data` on the cached
tensors and the second call onward reads stale memory
(`core/step_graph_cache.h:66-70`, #208).

| clip | CPU-s/audio-s before | after | Δ | wall before | wall after |
| --- | --- | --- | --- | --- | --- |
| 10 s | 0.6938 | 0.6416 | **−7.5%** | 8.34 s | 7.54 s |
| 60 s | 0.6218 | 0.5611 | **−9.8%** | 55.71 s | 43.46 s |
| marginal | 0.6073 | 0.5450 | **−10.3%** | | |

Output unchanged: frames 313/1875 and notes 89/538 identical on both clips, 26/26
stages still at cos 1.0000000.

---

## Step 2 — `n_threads` was already applied

§6.4's bug is a context that records `n_threads` and never calls
`ggml_backend_cpu_set_n_threads`, so the model silently runs at
`GGML_DEFAULT_N_THREADS = 4` regardless of `-t`. Three backends in this tree have
shipped it (`wespeaker.cpp:566`, chatterbox HISTORY §210/§212, and a variant in
Basic Pitch).

**This file does not have it.** `oaf_nthreads()` reads `ctx->params.n_threads`
and `core_cpu_backend::set_n_threads` is called immediately before all three
`ggml_backend_graph_compute` sites. Measured rather than asserted, 10 s clip:

| `-t` | wall | CPU |
| --- | --- | --- |
| 1 | 7.81 s | 6.38 s |
| 2 | 5.51 s | 6.71 s |
| 4 | 5.79 s | 8.03 s |

Wall falls and CPU rises — thread scaling, not a fixed default. **No change
made.**

Incidental finding, out of scope and unfixed: **`-t 4` costs 25% more CPU than
`-t 2` for no wall gain on this box.** That is the same shape as the hFT
thread-scaling result (4.84 → 7.35 CPU-s per audio-s between one thread and
four, `HFT_TRANSFORMER.md`). `-t 2` is the better operating point here.

---

## Step 3 — fusing the computes does not pay, and was not adopted

The playbook's rationale: "Eight `gallocr_alloc` + compute round trips per
forward is a lot of dispatch for a small model", citing voxcpm2, where folding
many tiny graphs into one gave **2.3× on CPU before any other change**
(`LEARNINGS.md` L9079).

**That precedent does not transfer, because O&F's graphs are not tiny.** Measured
directly, by timing graph build + `ggml_gallocr_alloc_graph` separately from
`ggml_backend_graph_compute` in the conv-stack chunk, 10 s clip, `-t 1`:

```
T= 259 nodes= 37   build+alloc 0.41 ms   compute 1579.70 ms
T=  60 nodes= 37   build+alloc 0.02 ms   compute  125.87 ms
T= 259 nodes= 37   build+alloc 0.06 ms   compute 1771.97 ms
T=  60 nodes= 37   build+alloc 0.02 ms   compute  138.66 ms
```

**0.02–0.41 ms of dispatch against 126–1772 ms of arithmetic — 0.005%.** Each
conv-stack chunk is over a second of work in 37 nodes. Fusing all eight computes
into three would save on the order of 1 ms out of 6400 ms.

The allocator hoist is part of why: with a persistent gallocr,
`ggml_gallocr_alloc_graph` reuses the existing buffer and costs almost nothing,
so step 1 removed most of what step 3 was aimed at.

A second measurement points the same way. Raising `OAF_CONV_CHUNK` from 256 to
512 — which *halves* the number of conv-stack computes for a 10 s clip — made it
**worse**, 0.6416 → 0.6867 CPU-s/audio-s (**+7%**), because the im2col working
set grows from ~101 MB to ~124 MB per chunk and falls further out of cache.
Bigger graphs are not free here; they are the wrong direction.

**Not adopted.** No code was written for step 3.

---

## Step 4 — the BiLSTM, screened before it was written

§8 item 2 lists "whether a ggml-graph BiLSTM beats a scalar recurrence at O&F's
sequence length" as explicitly **not established**, and §1b gives the reason to
doubt it: `core/lstm.h` unrolls ~12 nodes per timestep, so at T ≈ 3000 that is a
~200k-node graph across three BiLSTMs, against a per-step 384×1536 mat-vec small
enough that node overhead could dominate.

### Where the time actually goes

`ONSETS_AND_FRAMES_BENCH=1`, 10 s clip, `-t 1`, after step 1:

| stage | cost | share |
| --- | --- | --- |
| mel | ~50 ms | 1% |
| conv_stack × 4 | ~1210 ms each, 4850 ms | 72% |
| bilstm × 3 | ~600 ms each, 1810 ms | 27% |
| heads × 5 | below the noise floor | — |

And inside one BiLSTM at T = 313:

| part | cost |
| --- | --- |
| input projection, forward (a ggml GEMM) | 19.1 ms |
| input projection, reverse | 18.0 ms |
| **scalar recurrence, forward** | **272.1 ms** |
| **scalar recurrence, reverse** | **295.7 ms** |

**91% of the BiLSTM is the scalar recurrence**, and the recurrence is 25% of the
whole forward. This also disposes of half of §1b on its own: `core/lstm.h`'s
stated advantage is hoisting the input projection out of the timestep loop, and
**this file already does that** — `oaf_lstm_input_proj` is exactly that hoist,
and it costs 20 ms. There was nothing left to win on that axis.

### The screen

A standalone probe, one direction, this box, `-t 1`, building the recurrence as
one unrolled graph and timing build, alloc and compute separately against the
scalar loop copied verbatim from the source:

| T | scalar (whole cell) | unrolled graph (build + alloc + compute) | ratio |
| --- | --- | --- | --- |
| 313 | 289 ms — 0.924 ms/step | 2.1 + 4.2 + 54.0 = **60.2 ms** — 0.192 ms/step | 0.21× |
| 1875 | 1704 ms — 0.909 ms/step | 10.1 + 30.6 + 321.7 = **362.4 ms** — 0.193 ms/step | 0.21× |
| 3000 | 2903 ms — 0.968 ms/step | 13.3 + 48.1 + 517.8 = **579.1 ms** — 0.193 ms/step | 0.20× |

**§1b's node-count concern does not materialise.** The graph is ~4.7× cheaper and
its per-step cost is *flat* in T — 0.192, 0.193, 0.193 ms/step across a 10×
range. Build and alloc together stay near 10% of the total even at 48,000 nodes.

The gap is not mysterious. `docs/improvements/SRC_ISA_GAP.md` establishes that
everything under `src/` ships baseline x86-64 with no `-march`, so the scalar
loop runs SSE2 4-wide at ~1.3 GFLOP/s, while ggml's `mul_mat` reaches its
AVX2/FMA kernel through runtime dispatch.

### What was built

The recurrence unrolled into a graph directly, **not** by adopting
`core/lstm.h` — which is a port rather than a drop-in (§6.2), would not have fit
the ONNX `iofc` gate order without changes, and whose one real advantage this
file already had. T is chunked at 512 steps with a carried `(h, c)`, so node
count and the metadata arena are bounded rather than linear in clip length; that
is **exact**, because the recurrence is exact.

**It is not bit-identical to the scalar path and cannot be.** ggml's `mul_mat`
reduces in a different order, and this is a feedback path, so a last-ulp
difference at step 0 propagates. Hence `CRISPASR_OAF_GRAPH_LSTM=1`: the scalar
path stays the default and is not deleted (A/B rule 1, §5.7 item 3), and the
graph path falls back to it on any failure, so the gate can never make the model
stop working — only slower.

| clip | scalar | graph | Δ | wall scalar | wall graph |
| --- | --- | --- | --- | --- | --- |
| 10 s | 0.6416 | 0.5301 | **−17.4%** | 7.54 s | 6.89 s |
| 60 s | 0.5611 | 0.4352 | **−22.4%** | 43.46 s | 28.45 s |
| marginal | 0.5450 | 0.4163 | **−23.6%** | | |

Decoded output unchanged: frames 313/1875, notes 89/538, identical to the
baseline on both clips.

---

## Parity and quality after

**Per-stage, 4 s of MusicNet 2303, f32.** All 26 stages PASS at cos = 1.0000000
in *both* arms. The graph arm is in fact marginally **closer** to onnxruntime
than the scalar one:

| stage | scalar max_abs | graph max_abs |
| --- | --- | --- |
| `onset_bilstm` | 3.353e-06 | **2.891e-06** |
| `offset_bilstm` | 3.323e-06 | **2.697e-06** |
| `frame_bilstm` | 1.872e-05 | **1.532e-05** |

Which is what you would expect: ORT's LSTM also computes the recurrent term as a
GEMM, so the graph path is structurally nearer the reference than the serial
loop was.

**End-to-end against native onnxruntime**, `tools/oaf_parity.py`, 10 s clip,
`-t 1`, both runtimes handed the same mel:

| head | scalar max_abs | graph max_abs |
| --- | --- | --- |
| onset | 6.099e-07 | **5.503e-07** |
| offset | 6.854e-07 | **5.960e-07** |
| frame | 1.062e-06 | 1.395e-06 |
| activation | 1.957e-06 | 1.957e-06 |
| velocity | 8.899e-08 | 8.899e-08 |

`cos = 1.00000000` on every head in both arms. Front end: `max 2.946e-02, cos
1.00000000` against torchaudio, unchanged.

Two things worth reading off that table. **`activation` and `velocity` are
bit-identical between the two arms** — which they must be, because neither head
passes through a BiLSTM, and that is a free structural check that the gate is
touching only what it claims to. And the three heads that *do* pass through one
move by well under a part in a million.

**Decision agreement at the shipped thresholds**, which is the number that
matters for a transcriber rather than the RMS over 27,544 mostly-near-zero
cells:

| arm | onset | frame |
| --- | --- | --- |
| scalar | **100.0000%** (90 above vs 90 in the reference) | **100.0000%** (217 vs 217) |
| graph | **100.0000%** (90 above vs 90 in the reference) | **100.0000%** (217 vs 217) |

**Repeated-call validation** (§5.7 item 5, §6.7). A persistent gallocr may alias
an input tensor's slot with a later intermediate, and the symptom is run 0
correct and run 1 onward quietly wrong — invisible to a one-shot CLI invocation,
which is how this class of bug has shipped before (#208; `LEARNINGS.md` L14367).
`oaf-parity-dump` now takes an `[n_repeats]` argument that transcribes the same
audio N times on the **same context** and compares the full head tensors
bitwise. Three calls, both arms: **bitwise identical to run 0** every time.

---

### Note-level F1 on MusicNet — the task-level gate

`tools/oaf_musicnet_f1.py`, all ten MusicNet test pieces, `-t 1`,
`MKL_NUM_THREADS=1`, both arms through the same decoder.

**The graph arm against the native-onnxruntime reference, per piece:**

| piece | P | R | **F1** | F1+off | notes est |
| --- | --- | --- | --- | --- | --- |
| 1759 | 63.0 | 57.0 | 59.8 | 18.4 → **18.5** | 1559 |
| 1819 | 47.0 | 36.9 | 41.4 | 5.5 | 1038 |
| 2106 | 40.9 | 20.3 | 27.1 | 1.5 | 993 |
| 2191 | 53.9 | 32.5 | 40.5 | 4.1 | 332 |
| 2298 | 61.6 | 44.8 | 51.9 | 24.4 | 703 |
| 2303 | 88.2 | 85.0 | 86.5 | 32.5 | 692 |
| 2382 | 46.3 | 12.8 | 20.1 | 12.6 | 542 |
| 2416 | 44.7 | 43.4 | 44.0 | 11.9 | 1344 |
| 2556 | 74.6 | 67.5 | 70.9 | 22.1 | 1308 |
| 2628 | 73.6 | 55.1 | 63.1 | 10.6 | 1137 |

**Note-level precision, recall, F1 and the estimated note count are identical to
onnxruntime on all ten pieces.** Diffed programmatically, not by eye. The single
deviation anywhere in the table is `F1+off` on 1759, 18.4 → 18.5 — the
offset-sensitive variant, which requires the note *end* to agree within a
tolerance, so a sub-frame drift on a handful of long notes moves it by a tenth
while the onset/pitch decision does not move at all. That is the expected
signature of a non-bit-identical recurrence, and it is the right size.

Throughput on the same run is steady at **0.431–0.449 CPU-s per audio-second**
across all ten pieces, which corroborates the 60 s clip figure (0.435) on real
full-length material rather than a hand-cut excerpt.

**Aggregate over all ten pieces, f32 and q8_0, both with the graph BiLSTM:**

| arm | P | R | **F1** | F1+off | **solo piano** | rest | xRT | cpu-s/s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `f32graph` | 59.7% | 42.4% | **49.6%** | 13.8% | **69.0%** | 40.5% | 0.756 | 0.442 |
| `q8graph` | 59.7% | 42.4% | **49.6%** | 13.9% | 68.9% | 40.4% | 0.739 | 0.425 |

**49.6% overall and 69.0% solo piano is exactly the figure this port is
documented at.** The graph BiLSTM reproduces it to the last printed digit at
f32, and q8_0 lands on the same overall F1 with solo piano 0.1 point lower —
which is q8_0 quantisation, not the recurrence: the per-piece q8 deltas
(41.1 vs 41.4, 27.2 vs 27.1, 52.0 vs 51.9, 43.9 vs 44.0 …) scatter in *both*
directions around the f32 values, which is what rounding noise looks like and
not what a systematic error looks like.

One incidental result worth flagging: **q8_0 and f32 are indistinguishable
here**, 0.425 vs 0.442 CPU-s per audio-second, where §4 of the playbook reports
hFT measuring q8_0 at 29% *more* CPU than f32.

⚠ This originally read "q8_0 is slightly FASTER than f32 here", and that was
over-reading a 3.8% difference on a box whose spread for a single arm is
20–50%. `docs/ggml-repack-buft-evaluation.md` §4 settles it: hFT's 29% is a
real kernel effect, but this model's q8_0 GGUF leaves **every convolution
weight in F32** — 72% of the forward by the table above — and its LSTM `R`
matrices are dequantised at load, so only about 2% of the forward is exposed
to the quantised GEMM path at all. The largest effect it could show is under
a percent. See "What is still open" item 3.

**The default (scalar) arm's F1 needs no re-measurement**, and this is worth
stating rather than leaving implicit: its decoded output is unchanged by
construction — identical note counts on both clips, 26/26 stages at cos
1.0000000, and bitwise-identical repeated calls — so its F1 is still the
documented 49.6% overall / 69.0% solo piano. A run of it was started and killed
partway to give the machine to the arm whose number was actually unknown.

### The in-tree test suite

`ctest -R "onsets|btc"`, with `CRISPASR_MODEL_ONSETS_AND_FRAMES` pointed at a
real GGUF and `MKL_NUM_THREADS=1`: **18/18 pass**, including the three O&F
*live* tests (`init and sample rate`, `mel geometry`, `note events are well
formed`). Run across all four cells:

| | scalar (default) | `CRISPASR_OAF_GRAPH_LSTM=1` |
| --- | --- | --- |
| f32 | 3/3 pass | 3/3 pass |
| q8_0 | 3/3 pass | 3/3 pass |

The three `btc-chords` live tests skip for want of a model, as they do on any
box without one — see the open item below.

This was run locally on purpose. GitHub CI could not be used as the gate for
this work: `main` took three pushes from other sessions while these commits were
landing, and the workflow concurrency group cancelled every in-flight run before
the main `CI` job finished. `Lint`, `pages build and deployment`, `Build WASM`
and `Docker Smoke` did each complete **successfully** on SHAs carrying this
code, so the compile and style gates are covered; the unit-test gate is covered
by the table above rather than by a green tick.

---

## What is still open

1. **The 4.0× gap to ONNX Runtime is not closed.** After both changes the
   conv stacks are ~80% of the remaining time. That is `ggml_conv_2d` im2col
   arithmetic, and nothing here touched it. `BASIC_PITCH_CONV_PERF.md` §7's
   screening arithmetic is the place to start: O&F's convs are 48→48→96 channels
   at 3×3, which is the *favourable* side of the im2col line, so the fix is more
   likely a kernel or ISA question than a graph-shape one. `SRC_ISA_GAP.md` is
   the other half of it.
2. **`-t 4` costs 25% more CPU than `-t 2` for no wall gain.** Unexplained here.
3. ~~**q8_0 measured slightly FASTER than f32 here**~~ — **RESOLVED, and the
   right reading is "indistinguishable", not "faster".**
   `docs/ggml-repack-buft-evaluation.md` §4 has the working. hFT's 29% is real
   and has now been reproduced at the kernel level with no model involved:
   generic-path q8_0 `MUL_MAT` costs 1.08–1.31× f32 for transformer-shaped
   GEMMs, and hFT is 83.5% weight GEMM with all 63 of its quantised tensors on
   that path.

   This model is not, and the GGUF header says so. `onsets-and-frames-q8_0.gguf`
   has **19 Q8_0 tensors and 43 F32** — and **every convolution weight is
   F32**, which is 72% of the forward by the table above. The LSTM `R`
   matrices are Q8_0 in the file but `onsets_and_frames.cpp:76` dequantises
   them once at load, so the scalar recurrence — 91% of the BiLSTM — never
   sees a quantised tensor either. What is actually exposed to ggml's
   quantised GEMM path is the LSTM input projections (19.1 + 18.0 ms of ~600
   ms per BiLSTM, ≈1.7% of the forward) and the fc/head GEMMs, which this
   profile records as below the noise floor.

   ≈2% exposure at a 1.1–1.3× kernel penalty predicts about **+0.6%** overall.
   The measured difference was 3.8%, in the other direction, on a box whose
   spread for a *single* arm is 20–50%.

   **Re-measured on a clean runner it is 0.99×** — 1.57 vs 1.58 CPU-s, one
   process per arm, interleaved, median of 3, runner load ~1.0 (CI run
   35819605846). Indistinguishable, exactly as the GGUF's tensor types predict.
   The 0.96× on the VPS was contention, not a speedup. The caution in the original entry —
   "one box, one model, a small margin" — was exactly right; the margin was
   below the measurement floor.

   **The rule this yields is sharper than "measure":** open the GGUF and look
   at which tensors are actually quantised before predicting anything from a
   quantisation A/B. A converter that leaves the convolutions in F32 has
   already decided the A/B cannot move.

   The generalisable lesson: **the op mix tells you in advance how far a
   quantisation A/B can possibly move.** A model that spends 81% of its time
   on ops quantisation does not touch cannot show a large effect either way,
   so a small measured difference there is a null result, not a finding.
4. **`src/btc_chords.cpp`'s allocator hoist is not validated at runtime.**
   Neither a btc-chords GGUF, nor the BTC PyTorch checkpoint, nor the
   BTC-ISMIR19 tree exists on this box, so `tools/btc_torch_parity.py` and
   `crispasr-diff btc` could not be run. The 12 btc unit tests pass but never
   build the graph; the 3 live tests skip for want of a model. The change is
   mechanically identical to the O&F one, and the §6.6 hazard was closed by
   inspection — `btc_forward_block` has exactly five `ggml_set_input` tensors and
   re-sets all five after every alloc — but it wants a parity run from someone
   with the checkpoint.
5. **None of these numbers was taken on a clean machine.** They are within-box
   A/Bs and should be reproduced on a CI runner before being quoted anywhere.
