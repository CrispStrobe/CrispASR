# hFT-Transformer in ggml

`src/hft_transformer.{h,cpp}` — a port of Toyama et al.'s hierarchical
frequency-time transformer for piano transcription (ISMIR 2023,
`sony/hFT-Transformer`, MIT) to CrispASR's ggml runtime, with a GGUF converter
that quantises it.

**Read the Cost section before choosing this over `onsets-and-frames`.** The
accuracy is real and it is the smallest model that reaches it. The throughput
is not competitive on this hardware, and the measurement says why.

## Why this model, and what was being tested

CrispTuner's benchmark (`CrispStrobe/flutter_tuner`, `bench/REPORT.md` §35–36)
scored six transcribers on MusicNet's test split under
`mir_eval.transcription`. On the three solo-piano pieces:

| transcriber | solo-piano note F1 | size |
| --- | --- | --- |
| Kong / piano-transcription | 71.2% | 154 MB ONNX / 77 MB F16 GGUF |
| **hFT-Transformer** | **70.5%** | 22 MB ONNX |
| Onsets & Frames | 69.1% | 106 MB F32 ONNX |
| Basic Pitch | 57.5% | 110 KB |

hFT matches Kong at one seventh of the size and beats Onsets & Frames by 1.4
points with a fifth of its parameters. §36.3 of that report recommended the
O&F port first and put this one second, **"and measured before it is
believed"**, because the case for it rested on an argument rather than a
number:

> 83.5% of its MatMul is weight GEMM, so int8 kernels apply to the dominant
> cost rather than to a rounding error, and on CPU a good q8 GEMM is typically
> two to three times fp32. That would take 1.36× real time to somewhere near
> 0.5×.

That is the bet this port tested. **It does not pay off on this hardware**, and
the Cost section below is the measurement, the reason, and the condition under
which the reasoning would still hold.

## What is here

| file | what |
| --- | --- |
| `models/convert-hft-transformer-to-gguf.py` | ONNX → GGUF, F32 / F16 / Q8_0 / Q4_0 |
| `src/hft_transformer.{h,cpp}` | the runtime |
| `examples/cli/crispasr_backend_hft_transformer.cpp` | legacy `transcribe()` adapter |
| `examples/cli/crispasr_piano_cli.cpp` | the `--piano` arm (the real surface) |
| `tests/test_hft_transformer_live.cpp` | contract + invariants (`[hft-transformer]`) |
| `tests/hft_parity_dump.cpp` | dumps mel + heads for the ONNX diff |
| `tools/hft_parity.py` | numeric agreement vs native onnxruntime |
| `tools/hft_musicnet_f1.py` | note-level F1 on MusicNet's test split |
| `.github/workflows/piano-metal-ab.yml` | CPU-vs-GPU A/B on macOS; see `PIANO_METAL_AB.md` |

> **Speed, on Apple Silicon rather than the VPS.** `PIANO_METAL_AB.md` §4
> measures hFT at **0.926× real time at f32, 0.621× at q8_0 and 0.563× at
> q4_0** on a GitHub `macos-14` runner (chip "Apple M1 (Virtual)", 3 cores,
> 3 threads, 30 s clip) — i.e. **under real time at every quantisation, on the
> CPU alone**, against the 2.14× this document's own figures record on the
> contended x86 VPS. The same doc explains why no GPU number accompanies it:
> no hosted GitHub macOS image exposes a GPU that can run a dense-GEMM model.


```
python models/convert-hft-transformer-to-gguf.py \
    --input /path/hft_transformer.pruned.onnx \
    --output hft-transformer-q8_0.gguf --quant q8_0 \
    --check-mel --verify-fusion

crispasr --piano -m hft-transformer-q8_0.gguf -f piano.wav --piano-format midi
```

**Use the pruned graph.** The unpruned export keeps all fifteen forward
outputs, two of which nobody decodes — `enc_vector` at `[1, 128, 4, 88, 256]`
is 11.5 M floats by itself — and could not be loaded and run at all on the box
these numbers came from. `bench/tool/prune_hft.py` in `CrispStrobe/flutter_tuner`
cuts the graph to `onset_B`, `offset_B`, `mpe_B`, `velocity_B` (1,621 → 1,517
nodes). The converter only ever reads initializers, so the weights are the
same either way; the pruning matters for the ONNX arm of the harnesses.

## Architecture

5.48 M parameters, three stages, all of them transformer:

```
log-mel [T, 256 bins]
  │   32 margin frames of log(1e-8) each side; tail padded to a multiple of 128
  │
  ├─ ENCODER, one sequence per answered frame (batch 128 × 256 freq tokens)
  │     per (frame, bin): the 65-frame window → Conv2d(1,4,(1,5)) → [4·61=244]
  │                       → Linear(244, 256)        ... FUSED, see below
  │     × √256, + pos_embedding_freq[0:256]
  │     3 × { x = LN(x + SelfAttn(x)) ; x = LN(x + FF(x)) }
  │
  ├─ DECODER-FREQ, 88 pitch queries cross-attending to those 256 tokens
  │     x = pos_embedding_freq[0:88]                 (no √256 on the query)
  │     layer_zero : LN(x + CrossAttn(x, enc)) ; LN(x + FF(x))
  │     2 ×        : LN(x + SelfAttn(x)) ; LN(x + CrossAttn(x, enc)) ; LN(x + FF(x))
  │     → [128 frames, 88 pitches, 256] → transpose → [88, 128, 256]
  │
  └─ DECODER-TIME, 128 time tokens (batch 88 pitches)
        x = x·√256 + pos_embedding_time[0:128]
        3 × { x = LN(x + SelfAttn(x)) ; x = LN(x + FF(x)) }
        onset / offset / mpe = Linear(256, 1)   velocity = Linear(256, 128)
```

Attention is 4 heads × 64 with scale 1/8 everywhere. Every head emits
**logits**; the reference `infer.py` thresholds them at 0.5 as though they were
probabilities (really a sigmoid threshold of 0.62), so this runtime applies the
sigmoid and thresholds at 0.5, and `onset_threshold` here means what it says.

### The four things that are wrong-but-runnable

**1. Each layer has ONE LayerNorm, applied two or three times.** The checkpoint
carries a single `layer_norm.weight`/`.bias` per module and the graph reuses it
after every residual add. A port that allocated a set of gains per application
would run, and would be wrong by however much the applications differ. The
converter reads the module, not the graph node, so there is nothing to get
backwards.

**2. The convolution and the token embedding are fused, exactly.** Both are
linear in the 65-tap window and there is no nonlinearity between them, so they
collapse into a single `Linear(65, 256)`:

```
K[m, d] = Σ_c Σ_{j : 0≤j≤60, 0≤m-j≤4}  W_tok[c·61 + j, d] · W_conv[c, m-j]
b[d]    = b_tok[d] + Σ_c b_conv[c] · Σ_j W_tok[c·61 + j, d]
```

16,640 weights where the graph carries 62,464, and two `im2col` passes removed
from the runtime. `--verify-fusion` checks the collapsed form against an
explicit conv + 244×256 matmul on random input and asserts the residual is at
f32 rounding level (**7.1e-07 relative**). The GGUF records
`hft.front_end = "fused-conv-tok-embedding"` and the runtime refuses a file
that declares anything else, so a future unfused converter cannot silently feed
a 244-wide embedding to a front end that has no conv.

**3. The front end has three non-default settings and they are shipped, not
rebuilt.** 256 mel bins, n_fft 2048, hop 256, f_min 0, f_max 8000, at 16 kHz,
**power 2.0**, **constant (zero)** centre padding, then `log(mel + 1e-8)` — an
*add*, not a clamp — with a **periodic** Hann window, the **HTK** mel scale and
**slaney** filter normalisation. `core_mel` can express all of it, but neither
`build_htk_fb` nor `build_slaney_fb` is this combination, so the converter
computes the filterbank and the window, checks them against librosa
(`--check-mel`: **1.5e-08** and **3.0e-08**) and writes them into the GGUF as
`hft.mel_fb` / `hft.window`. The runtime refuses a GGUF without them rather
than guessing.

Note also that hFT does **not** drop a sample before the STFT the way Onsets &
Frames does, so `T = n / hop + 1` here against O&F's `(n-1) / hop + 1`.

**4. `mode_velocity='ignore_zero'` is not plumbing — it is the model's only
working filter.** The reference decoder drops any note whose velocity head
reads zero at the onset frame. §35.4 of the flutter_tuner report swept the
onset threshold from 0.2 to 0.7 and found hFT **completely flat at 52.2% F1**,
which is not saturation (0.3% of the onset head's values sit in [0.2, 0.5)) but
the gate removing exactly the candidates a lower threshold admits. Turning the
gate off and sweeping instead gives 45.1 / 47.0 / 49.8 / 52.1 — so the gate
reaches a *higher* F1 than the best thresholded arm without it, while answering
more often. It is exposed here as `ignore_zero_velocity` (default true) so it
can be measured, not so it can be tuned.

## How it is computed

Unlike `src/onsets_and_frames.cpp` there is no hand-rolled recurrence: the
whole model is ggml graphs. Two graphs per 192-frame window:

* **encoder + frequency decoder**, in chunks of `frame_chunk` frames (default
  32). Nothing in either stage mixes across frames — the encoder reads one
  65-frame window per answered frame and the frequency decoder's queries are
  the same 88 positional embeddings every time — so a chunk is **bit-identical**
  to the unchunked result, and `tests/test_hft_transformer_live.cpp` asserts
  that across a factor of eight in chunk size. What the chunk bounds is the
  attention score tensor: `[256, 256, 4, chunk]` is 16.8 MiB at 32 frames and
  would be 134 MiB for a whole window.
* **time decoder + the four heads**, once per window, on `[256, 128, 88]`.

Both allocators are created once and kept for the life of the context. §36.4 of
the flutter_tuner report named "a fresh ggml allocator per convolution chunk"
as one of the two unfixed causes of the Onsets & Frames arm's throughput gap;
this arm runs four encoder chunks and a time decoder for every 2.048 s of
audio, so it was not worth repeating.

Quantisation covers every 2-D matrix consumed by `ggml_mul_mat` whose
contraction axis is a multiple of 32: 63 tensors — all the attention
projections, all the feed-forward matrices and the velocity head. The
positional embeddings are *added*, not multiplied, so they stay F32; the fused
front end's contraction axis is 65 and stays F32; the three scalar heads are
`[1, 256]` and stay F32 because quantising 256 floats saves nothing and they
sit directly under a sigmoid.

## Numbers

### Sizes

| file | size | what is quantised |
| --- | --- | --- |
| `hft_transformer.pruned.onnx` (the source) | 21.8 MiB | — |
| `hft-transformer-f32.gguf` | 21.8 MiB | nothing |
| `hft-transformer-q8_0.gguf` | **7.0 MiB** | 63 of 157 tensors |
| `hft-transformer-q4_0.gguf` | **4.5 MiB** | the same 63 |

### Agreement with the ONNX export

`tools/hft_parity.py`, 5 s of MusicNet `2191.wav` at 16 kHz, both runtimes fed
the **same** mel so that only the model differs. onset / offset / mpe are
post-sigmoid, so an absolute error is directly a probability error; velocity is
the argmax bin, so its "error" is in bin numbers.

| head | f32 max abs | f32 cos | q8_0 max abs | q8_0 cos | q4_0 max abs | q4_0 cos |
| --- | --- | --- | --- | --- | --- | --- |
| onset | 2.5e-06 | 1.00000000 | 2.0e-02 | 0.99993485 | 2.8e-01 | 0.99350368 |
| offset | 4.4e-06 | 1.00000000 | 1.6e-02 | 0.99988704 | 1.9e-01 | 0.99168242 |
| mpe | 5.8e-06 | 1.00000000 | 3.9e-02 | 0.99995091 | 3.4e-01 | 0.99613684 |
| velocity (argmax bin) | **0** | 1.00000000 | **0** | 1.00000000 | 92 bins | 0.87038436 |

**At f32 the port is exact** — 2.5e-06 on the onset head is float32 rounding
over a 5.5 M-parameter graph, and the velocity argmax is identical in every
one of the 33,792 cells.

Decisions, which is what a transcriber lives on rather than the RMS over
mostly-near-zero cells. hFT has two: the thresholds, and the velocity gate.

| arm | onset cells | offset cells | mpe cells | ignore_zero gate | exact velocity bin |
| --- | --- | --- | --- | --- | --- |
| f32 | **100.0000%** | **100.0000%** | **100.0000%** | **100.0000%** | **100.0000%** |
| q8_0 | 99.9911% | **100.0000%** | 99.9911% | **100.0000%** | **100.0000%** |
| q4_0 | 99.9379% | 99.9822% | 99.8728% | 99.9645% | 99.8609% |

**q4_0's damage lands on the velocity head**, whose cosine drops to 0.870 and
whose argmax moves by up to 92 bins. That is the head the `ignore_zero` gate
reads, so q4_0 is perturbing the one filter this model has — which is a more
specific prediction than "q4_0 is a bit worse", and the F1 table below is where
it shows up.

The front end is checked in the same run, against librosa on the same PCM: max
abs **6.2e-03**, mean **1.4e-05**, RMS **7.4e-05**, cos 1.00000000 in the log
domain. The max sits on near-floor bins, where `log(x + 1e-8)` turns a tiny
linear difference into a large log one.

### Note-level F1 on MusicNet's test split

`tools/hft_musicnet_f1.py`, all ten MusicNet test pieces (24.7 minutes, 13,589
reference notes), `mir_eval.transcription` with the default 50 ms onset and 50
cent tolerances, `offset_ratio=None` for the headline number. Every arm —
including the ONNX export under native onnxruntime — goes through the same
decoder and the same metric. Counts are POOLED across pieces, not averaged per
piece, because that is what §35.3 of the flutter_tuner report did.

**The ONNX row reproduces that report to the digit: 57.2% / 48.0% / 52.2%
overall and 70.5% on solo piano**, and per piece it lands within 0.2 points on
all ten. The harness is calibrated before anything is claimed for the port.

| arm | size | P | R | **F1** | F1 w/ offsets | solo piano | everything else |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ONNX export, onnxruntime | 21.8 MiB | 57.2% | 48.0% | **52.21%** | 18.49% | **70.52%** | 44.52% |
| ggml **f32** | 21.8 MiB | 57.2% | 48.0% | **52.21%** | 18.49% | **70.52%** | 44.52% |
| ggml **q8_0** | 7.0 MiB | 57.3% | 48.0% | **52.23%** | 18.50% | **70.51%** | 44.54% |
| ggml **q4_0** | 4.5 MiB | 57.5% | 48.4% | **52.55%** | 18.77% | **70.71%** | 44.90% |

Per piece:

| piece | 1759 | 1819 | 2106 | 2191 | 2298 | 2303 | 2382 | 2416 | 2556 | 2628 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ONNX | 59.9 | 44.2 | 31.4 | 42.0 | 58.3 | 88.6 | 21.6 | 52.9 | 73.3 | 66.7 |
| ggml f32 | 59.9 | 44.2 | 31.4 | 42.0 | 58.3 | 88.6 | 21.6 | 52.9 | 73.3 | 66.7 |
| ggml q8_0 | 59.8 | 44.3 | 31.3 | 42.0 | 58.3 | 88.6 | 21.6 | 52.9 | 73.4 | 66.8 |
| ggml q4_0 | 60.3 | 44.5 | 31.7 | 43.7 | 58.7 | 88.6 | 22.0 | 52.8 | 73.4 | 67.0 |

**At f32 the port is not approximately the model, it is the model**: identical
F1, identical F1-with-offsets, identical estimated-note counts, on every one of
the ten pieces and on every column. That also closes the loop on the decoder,
which the activation diff cannot see — the script decodes the heads in Python
and separately checks the C++ runtime's own note list against that decode, note
for note. Over a 195-second piece the two agree on all 1,457 notes with a worst
disagreement of **68 µs** on an onset and 65 µs on an offset, which is float32
against float64 in the sub-frame refinement and nothing else.

**q8_0 is free.** 52.23% against 52.21%, 70.51% solo piano against 70.52%,
18.50% with offsets against 18.49% — inside rounding on a 13,589-note corpus,
for a third of the size. Per piece the largest disagreement anywhere is 0.1
points.

**q4_0 does not cost accuracy here; it very slightly gains.** 52.55% against
52.21%, and up on eight pieces of ten. That is 0.34 points on 13,589 notes and
should be read as noise rather than as an improvement — but it is worth saying
why it is *not* the loss the Onsets & Frames port measured, because the
mechanism is specific and it was predicted by the head-by-head diff above.

O&F's q4_0 cost 0.5 points of F1-with-offsets because the head it perturbed
most was the **frame** head, and the frame head sets note durations. Here the
head q4_0 damages most is **velocity** — cosine 0.870, argmax moving by up to
92 bins — and velocity does not set a duration. It feeds the `ignore_zero`
gate. So q4_0's errors change *which* notes survive rather than how long they
last, and on this corpus that comes out marginally in q4_0's favour: it emits
more notes than f32 on nine pieces (1,472 against 1,457 on 1759; 1,310 against
1,288 on 1819) and both precision and recall move up together, which is what a
**better-calibrated filter** looks like rather than a noisier one.

That is a result about this corpus and this gate, not a recommendation to
prefer 4-bit weights. The honest reading is that hFT's accuracy is
**insensitive to weight precision down to 4 bits**, which is unusual and is
consistent with a model whose output is dominated by a discrete gate rather
than by a regression.

**So the accuracy half of the question is answered: q8_0 holds hFT's
solo-piano F1 exactly, and so does q4_0.** The cost half is answered below,
and it is where the port does not deliver.

### Cost

Intel Xeon (Skylake-SP, IBRS, no TSX), 4 vCPU, Hetzner CX-class VPS, carrying a
background load average of 3–7 from other tenants throughout. Both sides of the
comparison are a **whole process doing the whole job** — front end, every
window of inference, and the note decoder — because §36.4 of the flutter_tuner
report records what happens otherwise: the Onsets & Frames ggml arm's whole
process was compared against ORT's inference call alone, and CPU-seconds
against wall-clock, and the gap was published as an order of magnitude when it
was 5.5–7×. `tools/hft_ort_cost.py` is the ORT half.

Same 30 s clip of MusicNet `2191.wav`, onnxruntime 1.23.2 against the pruned
graph:

| arm | threads | wall × real time | CPU-s per audio-s | peak RSS |
| --- | --- | --- | --- | --- |
| ggml **f32** | 1 | 4.85× | **4.84** | **237 MiB** |
| ggml q8_0 | 1 | 6.30× | 6.25 | 232 MiB |
| ggml q4_0 | 1 | 6.38× | 6.26 | 230 MiB |
| ggml **f32** | 4 | **2.14×** | 7.35 | **237 MiB** |
| ggml q8_0 | 4 | 2.73× | 9.17 | 232 MiB |
| native ORT | 1 | 3.46× | 3.09 | 1001 MiB |
| native ORT | 4 | **1.28×** | 3.77 | 1004 MiB |

At 5 s rather than 30 s the ggml f32 arm costs 29.19 CPU-s against 145.06 — 3
windows against 15 — so **9.66 CPU-seconds per 2.048 s window** and a fixed
cost of about **0.2 CPU-seconds**. Startup and the 22 MiB model load are not
what is expensive here; the arithmetic is.

Three things to read out of that table.

**1. Quantisation makes this model SLOWER, not faster.** q8_0 costs 29% more
CPU than f32 (6.25 against 4.84 single-threaded) and q4_0 the same again. §36.3
predicted "two to three times fp32" from int8 kernels applying to 83.5% of the
arithmetic. The arithmetic share was right; the kernel assumption was not, for
two nameable reasons. ggml's fast repacked int8 GEMM lives in
`ggml/src/ggml-cpu/repack.cpp` and is reached only through the CPU device's
**extra** buffer types, via the registry proc address
`ggml_backend_dev_get_extra_bufts`. `GGML_CPU_REPACK` is ON in this build, so
the code is compiled in — but `core_gguf::load_weights` never asks for an
extra buffer type, so every backend that loads through it (which is all six
transcription models) takes ggml's generic route, which re-quantises the
activation operand to Q8_0 on every GEMM and then runs a `vec_dot`.

**One correction worth carrying, because it was got wrong here first:** the
repack path is *not* unreached everywhere in the tree. `src/crispasr.cpp`
requests it, llama.cpp-style, for the whisper backend. It is unreachable for
everything that loads through `core_gguf`, which is a different and narrower
claim. `docs/ggml-optimisation-playbook.md` §4 is the authority, has the line
numbers, and prices the three obstacles to fixing it — per-op routing (the
extra buffer types support only `MUL_MAT` and `GET_ROWS`), a likely conflict
with the zero-copy mmap path, and the fact that this box is Skylake-SP with
AVX-512F but **no AVX-512 VNNI**, so there is no int8 dot-product instruction
for the repacked path to reach even once it is selected.

**Update — this has now been tested.** `docs/ggml-repack-buft-evaluation.md`
is the authority. Three things it changes here:

- The 29% is **real on the box it was measured on** and reproduced there at
  the kernel level with no model involved: generic-path q8_0 `MUL_MAT` measures
  1.08–1.31× the cost of f32 for transformer-shaped GEMMs on that Skylake-SP
  VPS. At 83.5% weight GEMM that predicts 1.08–1.26× whole-model, and 1.29×
  was measured. Not noise.
- ⚠ **But it does not generalise, and the same A/B on clean runners shows how
  far it does not.** Whole-model, one process per arm, interleaved, median of
  3: on GitHub's AMD EPYC 7763 hFT q8_0 is **1.09×** f32 (14.05 vs 12.88
  cpu-s), and on `ubuntu-24.04-arm` it is **0.50×** — the model runs **twice as
  fast** quantised. Skylake-SP is the worst case because its AVX-512 makes the
  *f32* GEMM unusually fast, not because the quantised path is slow in absolute
  terms. **Quote the 29% as an AVX-512 x86 figure or not at all.**
- **q4_0 through the repack buffer type is now the fastest arm of all**: 9.3
  cpu-s on the EPYC, **0.72× f32** and **1.69×** the same GGUF without
  repacking; 35.7 cpu-s on arm64, **0.35× f32**. Peak RSS is unchanged at
  233–234 MiB, so giving up zero-copy mmap costs nothing at this model size.
- The repack buffer type **is** offered on this box, and it does pay — 1.4–5.6×
  over the generic quantised path — **without needing VNNI**. The AVX2 kernels
  do not use it; the interleaved layout pays on its own.
- But it **cannot rescue this model's q8_0 GGUF on x86**: ggml has no repacked
  q8_0 kernel for x86 at all (it is gated on NEON+dotprod/i8mm or RISC-V).
  Reaching the fast path on x86 means q4_0 or q4_K, which is an accuracy
  question this document has not asked. On arm64 the table inverts and q8_0
  does get a kernel — so the same GGUF may behave differently on a phone.

`src/hft_transformer.cpp` now loads through `core_gguf::load_weights_repack()`,
which is a no-op (and keeps the zero-copy mmap) for the q8_0 and f32 files on
x86, and takes the fast path for q4_0/q4_K.

**2. The gap to ORT is 1.6×, not an order of magnitude.** CPU-seconds per
audio-second at one thread: 4.84 against 3.09, a factor of **1.56**. Wall at
four threads: 2.14× against 1.28×, a factor of **1.67**. For comparison the
Onsets & Frames ggml arm sits at 5.5–7× its ORT equivalent (§36.4) and 0.671
CPU-s per audio-second single-threaded. hFT is a far heavier model — 249 GFLOP
of MatMul per 2.048 s window, where O&F's whole 30 s pass is a fraction of that
— but the runtime is much closer to ORT on it, because this port is nothing but
GEMM in one graph, where the O&F port is a scalar C++ recurrence and a fresh
allocator per chunk.

At 9.66 CPU-seconds per 248.6 GFLOP window, the ggml arm is turning **25.7
GFLOP per CPU-second** at f32 single-threaded. ORT reaches roughly 31 per core
on the same box. That is the honest summary of the GEMM comparison: ggml is
within about 20% of MLAS per core here, and the rest of the gap is thread
scaling — ggml's f32 arm goes 4.84 → 7.35 CPU-seconds per audio-second between
one thread and four, because its threadpool spin-waits at barriers on a box
whose four cores are already shared.

**3. Memory is where this port wins outright: 237 MiB against ORT's 1001 MiB**
on the 30 s clip, a 4.2× reduction, and against the **3.63 GB** at which
`onnx_runtime_dart` was OOM-killed on a single window (§35.5). The frame
chunking is why — the attention score tensor is `[256, 256, 4, 32]` rather than
`[256, 256, 4, 128]` — and it costs nothing, because the chunked result is
bit-identical.

It is not flat in the length of the audio, though, and the doc should say so:
over the MusicNet split the peak was **799 MiB at f32 and 714 MiB quantised**
on pieces of 1.5–3.8 minutes, against ORT's 1001 MiB, so the honest headline is
a 1.25–1.4× reduction on a real piece and 4.2× on a short one. The part that
grows is the runtime's own buffers — the padded log-mel and the four head
arrays are linear in the clip — not the graph arena, which the chunking pins.

### So: does q8_0 bring hFT near real time, and does it hold its F1?

**No, and no.** Quantisation costs 29% of the throughput rather than buying
2–3×, and the fastest arm — f32 — is still **2.14× real time on four shared
cores**, against native ORT's 1.28× and Onsets & Frames' 0.44×. There is no
configuration of this model in this runtime, on this hardware, that is under
real time.

What the port does deliver is **the same accuracy as the ONNX export at f32,
in 237 MiB instead of 1001 MiB, from a 21.8 MiB file, with no onnxruntime
linked** — on every platform CrispASR builds for. Whether that is worth 1.4
points of solo-piano F1 over `onsets-and-frames`, which runs at 0.44× real
time from a 30.8 MiB q8_0 file, is a judgement about the application. For
offline transcription of a piece it is affordable; for anything interactive it
is not, and **`onsets-and-frames` remains the right default piano arm.**

## A note on what was NOT done

* **The repack buffer type.** See point 1 above. It is the single measurement
  that would either rescue or bury the int8 argument, and it is a change to
  `core_gguf::load_weights` that every backend in the tree would inherit, so it
  wants its own change and its own regression pass rather than being smuggled
  in here.
* **No GPU backend.** `use_gpu` is accepted and ignored; the context always
  initialises the CPU backend, as `onsets_and_frames.cpp` does.
* **No per-stage reference dumper.** `tools/hft_parity.py` compares the mel
  and all four heads against native onnxruntime, which is end-of-pipeline: a
  regression introduced inside the encoder shows up as "the onset head moved",
  not as a layer. `tools/reference_backends/onsets_and_frames.py` plus the
  `crispasr-diff onsets-and-frames` arm is what the per-layer half looks like,
  and hFT has no equivalent — `tools/check-backend-wiring.py` reports it as an
  advisory gap, correctly. A file in `tools/reference_backends/` with no
  `crispasr-diff` arm to consume it would silence the checker without adding a
  gate, so it was not written.
* **No streaming.** The window is a fixed 192 frames and the model needs 32
  frames of lookahead, so a streaming arm is possible and is not implemented.
* **The HF GGUF repo is not uploaded**, so `-m auto` will 404 for this
  backend. Build the file locally with the converter.
