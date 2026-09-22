# Onsets & Frames in ggml

`src/onsets_and_frames.{h,cpp}` — a port of Hawthorne et al.'s piano
transcriber (ISMIR 2018; the MIT reimplementation everyone actually uses) to
CrispASR's ggml runtime, with a GGUF converter that quantises it.

## Why this model

CrispTuner's benchmark (`CrispStrobe/flutter_tuner`, `bench/REPORT.md` §35–36)
scored six transcribers on MusicNet's test split under
`mir_eval.transcription`. On the three solo-piano pieces:

| transcriber | solo-piano note F1 | size |
| --- | --- | --- |
| Kong / piano-transcription | 71.2% | 154 MB ONNX / 77 MB F16 GGUF |
| hFT-Transformer | 70.5% | 22 MB |
| **Onsets & Frames** | **69.1%** | 106 MB F32 ONNX |
| Basic Pitch | 57.5% | 110 KB |

Speed was never the objection: under native ONNX Runtime O&F ran at **0.055×
real time**, eighteen times faster than the audio. The objection was 106 MB of
download. Profiling said **94% of its cost is weight-bearing** — Conv 46%, LSTM
29%, MatMul 19%, and no attention at all, every MatMul carrying a weight matrix
— so quantisation should keep the arithmetic and lose the size. §36.3 of that
report recommends exactly this port, at q4_0, as recommendation #1.

It came out smaller than that recommendation predicted: **18.6 MiB at q4_0**
against the estimated 27 MB.

## What is here

| file | what |
| --- | --- |
| `models/convert-onsets-and-frames-to-gguf.py` | ONNX → GGUF, F32 / F16 / Q8_0 / Q4_0 |
| `src/onsets_and_frames.{h,cpp}` | the runtime |
| `examples/cli/crispasr_backend_onsets_and_frames.cpp` | legacy `transcribe()` adapter |
| `examples/cli/crispasr_piano_cli.cpp` | the `--piano` arm (the real surface) |
| `tests/test_onsets_and_frames_live.cpp` | contract + invariants (`[onsets-and-frames]`) |
| `tests/oaf_parity_dump.cpp` | dumps mel + heads for the ONNX diff |
| `tools/oaf_parity.py` | numeric agreement vs native onnxruntime |
| `tools/oaf_musicnet_f1.py` | note-level F1 on MusicNet's test split |

```
python models/convert-onsets-and-frames-to-gguf.py \
    --input /path/onsets_and_frames.onnx \
    --output onsets-and-frames-q4_0.gguf --quant q4_0 --check-mel

crispasr --piano -m onsets-and-frames-q4_0.gguf -f piano.wav --piano-format midi
```

## Architecture, and the three things that are wrong-but-runnable

The graph is four ConvStacks over one shared log-mel, two of them followed by a
BiLSTM, and a fifth BiLSTM over the concatenation of three of the heads:

```
mel [T, 229] ──┬─ ConvStack ─ BiLSTM(768→384) ─ Linear(768,88) ─→ onset
               ├─ ConvStack ─ BiLSTM(768→384) ─ Linear(768,88) ─→ offset
               ├─ ConvStack ───────────────────  Linear(768,88) ─→ activation
               └─ ConvStack ───────────────────  Linear(768,88) ─→ velocity

  cat(onset, offset, activation) [264] ─ BiLSTM(264→384) ─ Linear(768,88) ─→ frame
```

ConvStack is `Conv2d(1→48,3×3)+ReLU, Conv2d(48→48,3×3)+ReLU, MaxPool(1,2),
Conv2d(48→96,3×3)+ReLU, MaxPool(1,2)`, then a transpose to `[B, T, C, F]`, a
flatten to `96 × (229//4) = 5472`, and `Linear(5472, 768)`. 26.49 M parameters.

**1. The ONNX export's output names are shifted by one.** `torch.onnx.export`
was handed four `output_names` for a forward that returns five tensors, so
every name slid down a slot: the graph output called `frame` is the
pre-combination *activation*, the one called `velocity` is the real *frame*
head, and the unnamed fifth output `679` is the velocity. Believing the names
costs **9.1% → 5.6%** note F1 with offsets required, and nothing raises.

The converter establishes the mapping from the wiring, not the names: it finds
the `Concat` whose three inputs are all graph outputs, follows it forward to
whichever graph output it reaches, and names that one `frame`. It writes the
GGUF under the *meanings*, so the runtime never has to know the export existed.

**2. ONNX LSTM gate order is `iofc`.** PyTorch's is `ifgo`. The weights come
out of the ONNX initializers, so they are in ONNX order; the converter stores
`oaf.lstm_gate_order` in the GGUF and `onsets_and_frames_init_from_file`
refuses a file whose declared order it does not implement. A silent gate
permutation produces a model that runs and emits plausible nonsense.

**3. The front end has four non-default settings and they are shipped, not
rebuilt.** 229 mel bins, n_fft 2048, hop 512, f_min 30, f_max 8000, at 16 kHz,
with a **periodic** Hann window, **power 1.0** (magnitude, not power),
the **HTK** mel scale and **slaney** filter normalisation, then
`log(clamp(·, 1e-5))`. `core_mel` can express all of that, but neither
`build_htk_fb` (HTK scale, no normalisation) nor `build_slaney_fb` (Slaney
scale *and* slaney normalisation) is the combination this checkpoint wants. So
the converter computes the filterbank and the window, checks them against
`torchaudio` (`--check-mel`: max abs diff **6.6e-07** and **2.1e-07**), and
writes them into the GGUF as `oaf.mel_fb` and `oaf.window`. The runtime refuses
a GGUF without them rather than guessing. This is
`crispasr-crispembed-dev.md` rule 2b's advice taken literally.

BatchNorm needs no handling at all: `torch.onnx.export` folded all twelve of
them into the convolutions in eval mode, so the graph carries biased Convs.
That is one whole class of bug — see `piano_transcription.cpp`'s
`PIANO_BN2D_EPS` comment for what it costs when it goes wrong — that the
exporter removed for free.

## How it is computed

Following `src/piano_transcription.cpp`: the convolutional and dense parts run
in a ggml computation graph, and the recurrent layers are computed **by hand
outside it**, because ggml has no fused RNN op and unrolling thousands of
timesteps would exhaust the graph's node budget.

Two things are done differently from that file, both for memory:

* **The conv stack is evaluated in 256-frame chunks with a 3-frame halo.** A
  3×3 kernel with pad 1 reaches exactly one frame per layer and `MaxPool(1,2)`
  does not mix time at all, so a chunk is bit-identical to the unchunked result
  everywhere except the true sequence edges, where the real zero padding
  applies and no halo exists. Without it, `ggml_im2col` on a three-minute piece
  wants gigabytes; with it the working set is flat in the length of the audio.
* **The LSTM input projection `W·x + b` is batched into one `ggml_mul_mat` over
  all timesteps**, leaving only `R·h` in the per-step loop. That is what lets
  the input-projection weights stay quantised: they are the larger half of the
  LSTM and ggml's quantised GEMM kernels consume them directly. `R` is
  dequantised to F32 once at load, because a per-timestep ggml graph would cost
  more in graph overhead than the arithmetic saves.

Quantisation covers every 2-D weight whose contraction axis is a multiple of
32 — the four `Linear(5472, 768)` (63% of all parameters on their own) and the
LSTM `W` and `R`. Convolution kernels, biases, the filterbank, the window and
the combined stack's input projection (264 is not a multiple of 32) stay F32.

## Numbers

### Sizes

| file | size | what is quantised |
| --- | --- | --- |
| `onsets_and_frames.onnx` (the source) | 101.1 MiB | — |
| `onsets-and-frames-f32.gguf` | 101.9 MiB | nothing |
| `onsets-and-frames-q8_0.gguf` | **30.8 MiB** | 19 tensors (the fc, the LSTM W and R) |
| `onsets-and-frames-q4_0.gguf` | **18.6 MiB** | the same 19 |

19 of the 62 tensors carry 25.9 M of the 26.49 M parameters, which is why
quantising only those still gets a 5.5× reduction. §36.3 of the flutter_tuner
report estimated "roughly 27 MB" for q4_0; the measured figure is 18.6 MiB
because the LSTM weights quantise too.

### Agreement with the ONNX export

`tools/oaf_parity.py`, 30 s of MusicNet `2303.wav` at 16 kHz, both runtimes fed
the **same** mel so that only the model differs. Activations are post-sigmoid,
so an absolute error is directly a probability error.

| head | f32 max abs | f32 cos | q8_0 max abs | q8_0 cos | q4_0 max abs | q4_0 cos |
| --- | --- | --- | --- | --- | --- | --- |
| onset | 5.3e-07 | 1.00000000 | 9.6e-03 | 0.99999483 | 1.6e-01 | 0.99894381 |
| offset | 1.1e-06 | 1.00000000 | 1.5e-02 | 0.99997040 | 2.4e-01 | 0.99527969 |
| activation | 3.1e-06 | 1.00000000 | 2.1e-02 | 0.99999532 | 1.6e-01 | 0.99967197 |
| **frame** | 1.3e-06 | 1.00000000 | 2.1e-02 | 0.99997450 | 2.9e-01 | 0.99348293 |
| velocity | 1.1e-07 | 1.00000000 | 1.6e-03 | 0.99999999 | 1.3e-02 | 0.99999864 |

**At f32 the port is exact** — 5.3e-07 on the onset head is float32 rounding
over a 26 M-parameter graph, not a difference in the arithmetic.

The number a transcriber actually lives on is not the RMS over 82,544 mostly
near-zero cells but whether the *decisions* agree at the shipped thresholds:

| arm | onset cells agreeing | frame cells agreeing | notes decoded |
| --- | --- | --- | --- |
| f32 | **100.0000%** (257 vs 257 above 0.5) | **100.0000%** (675 vs 675) | 253 vs 253 |
| q8_0 | **100.0000%** (257 vs 257) | 99.9976% (675 vs 675) | 253 |
| q4_0 | 99.9903% (259 vs 257) | 99.8934% (647 vs 675) | 252 |

The front end is checked in the same run, against `torchaudio` on the same PCM:
max abs **3.8e-02**, mean **9.1e-05**, RMS **5.2e-04**, cos 1.00000000 in the
log domain — the max sits on near-floor bins where a tiny linear difference is
a large log one.

### Note-level F1 on MusicNet's test split

`tools/oaf_musicnet_f1.py`, all ten MusicNet test pieces (24.7 minutes, 13,589
reference notes), `mir_eval.transcription` with the default 50 ms onset and 50
cent tolerances, `offset_ratio=None` for the headline number. Every arm —
including the ONNX export under native onnxruntime — goes through **the same**
decoder and the same metric, so the rows are comparable to each other and not
merely to another project's table.

Counts are POOLED across pieces (one precision and one recall over every note),
not averaged per piece. That matters: the two conventions differ by about two
points here, and pooling is what §35.3 of the flutter_tuner report used. The
ONNX row below lands on 59.7% / 42.4% / **49.6%**, which is that report's
number to the digit, so the harness is calibrated before anything is claimed
for the port.

| arm | size | P | R | **F1** | F1 w/ offsets | solo piano | everything else |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ONNX export, onnxruntime | 101.1 MiB | 59.7% | 42.4% | **49.6%** | 13.8% | **69.0%** | 40.5% |
| ggml **f32** | 101.9 MiB | 59.7% | 42.4% | **49.6%** | 13.8% | **69.0%** | 40.5% |
| ggml **q8_0** | 30.8 MiB | 59.7% | 42.4% | **49.6%** | 13.9% | **69.0%** | 40.5% |
| ggml **q4_0** | 18.6 MiB | 60.1% | 42.0% | **49.5%** | 13.3% | **68.9%** | 40.2% |

Per piece:

| piece | 1759 | 1819 | 2106 | 2191 | 2298 | 2303 | 2382 | 2416 | 2556 | 2628 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ONNX | 59.8 | 41.4 | 27.1 | 40.5 | 51.9 | 86.5 | 20.1 | 44.0 | 70.9 | 63.1 |
| ggml f32 | 59.8 | 41.4 | 27.1 | 40.5 | 51.9 | 86.5 | 20.1 | 44.0 | 70.9 | 63.1 |
| ggml q8_0 | 59.9 | 41.1 | 27.2 | 40.5 | 52.1 | 86.5 | 20.2 | 44.0 | 70.9 | 63.1 |
| ggml q4_0 | 59.4 | 41.6 | 26.9 | 40.8 | 51.7 | 87.1 | 19.9 | 43.0 | 70.7 | 62.9 |

**At f32 the port is not approximately the model, it is the model**: identical
note lists, identical F1, on every piece in the corpus. That also closes the
loop on the decoder, which the activation diff cannot see — the script decodes
both arms in Python and separately checks the C++ runtime's own note list
against that Python decode, note for note, and reports a mismatch if there is
one. There was none, at any quantisation.

**What quantisation costs.** q8_0 is free: 49.6% / 69.0%, the same numbers to
the digit, for a third of the size. q4_0 costs **0.1 point of note F1** (49.6%
→ 49.5%) and **0.1 point on solo piano** (69.0% → 68.9%) — inside the noise of
a 13,589-note corpus, and the per-piece column shows it going both ways (2303
*gains* 0.6, 2416 loses 1.0). It is not free on the harder metric: **F1 with
offsets required drops 13.8% → 13.3%**, a 3.6% relative loss, which is what one
would expect given that the frame head is the head q4_0 perturbs most (cos
0.9935 against onset's 0.9989) and the frame head is precisely what sets a
note's duration.

So the recommendation is **q4_0 for onset-and-pitch work** — note detection,
MIDI capture, anything scored the way the table's headline column is — and
**q8_0 when note durations matter**, or simply as the default if 30.8 MiB is
affordable, because at that size the port is bit-for-bit as good as fp32 and
there is no judgement call to make. There is no case for f32 in an application:
it is 3.3× the size of q8_0 for no measurable accuracy.

**Against Basic Pitch, which is what CrispASR would otherwise reach for on
piano: 69.0% against 57.5% on the solo-piano pieces** (68.9% at q4_0, for
18.6 MiB). Neither beats MT3's 76.5% overall, and O&F is not a general
transcriber — half this corpus is strings and winds and it scores 40.5% there.
It is the piano arm.

### Cost

Intel Xeon (Skylake, IBRS), 4 vCPU, Hetzner CX-class VPS, 4 threads, and the
box was carrying a load average of 6–9 from other work throughout. Wall-clock
realtime factors on a machine that contended are not worth much, so **CPU
seconds per audio second** is the number to read; the runtime records both.

| arm | cpu-s per audio-s | wall-clock × real time (contended) |
| --- | --- | --- |
| ggml f32 | **0.73** | 0.67× |
| ggml q8_0 | 0.75 | 1.01× |
| ggml q4_0 | 0.75 | 0.77× |

Quantisation buys no speed here, and the reason is structural rather than
surprising: the quantised weights are consumed by `ggml_mul_mat`, which is the
fc and the LSTM input projection, and the profile of this model is 46%
convolution. Those run from F32 kernels either way. The wall-clock spread
between the rows is the box's load average, not the arms.

Native ONNX Runtime reaches 0.055× wall on the same box (§36.1), so the ggml
port is currently the slower runtime by roughly an order of magnitude. That is
a real finding and not a surprising one: `oaf-parity-dump` builds and frees a
fresh ggml graph and allocator per 256-frame conv chunk and per LSTM
projection, and the recurrence itself is a scalar C++ loop with no threading
at all, where ORT has a tuned fused LSTM kernel. Under real time on four
shared cores is enough for an offline transcriber; it is not the 18× headroom
ORT has, and closing that gap is a separate piece of work (reuse one allocator,
thread the two LSTM directions as `piano_transcription.cpp` does for its BiGRU).
The point of this port was the 5.5× size reduction and reaching every platform
CrispASR builds for without linking onnxruntime, and those it delivers.


## A trap that is not this model's fault

`core_mel`'s mel projection is a `cblas_sgemm`. On this VPS, CMake finds
Debian's MKL and links `libmkl_intel_thread` into a process that already
carries `libgomp` — and that combination returns a mel spectrogram whose
**upper bins are multiplied by the thread count**: exactly `ln(4)` of error in
the log domain at four threads, `ln(3)` at three, with the lower bins correct.
The same `mel.cpp.o` linked against OpenBLAS is exact to 1e-4.

It is silent. Nothing raises; the spectrum simply has the wrong shape above
about 3 kHz, and a model reading it just scores worse. It cost most of a
debugging session here, and the bisect that found it is worth writing down: the
STFT was checked against `numpy.fft` (exact), the filterbank and window were
dumped out of the loaded GGUF and checked against `torchaudio` (exact), the PCM
was dumped and checked against `soundfile` (bit-identical) — and the same
`core_mel` object file still disagreed with itself between two binaries. The
only remaining difference was which BLAS the linker had chosen.

**This is not specific to Onsets & Frames.** Every backend that uses
`core_mel` on a host with a threaded MKL is exposed to it, which is why it was
fixed separately rather than worked around here: `crispasr-core` (and `cohere`)
now look for OpenBLAS explicitly before falling back to `find_package(BLAS)`,
mirroring what `mel-band-roformer` already did for its own reason, and
`tests/test-mel-blas-parity.cpp` guards the result by running `core_mel` twice
on identical input — once through the sgemm, once through its own scalar
accumulation — and requiring 1e-3 agreement. That guard was written before the
fix and watched fail at a relative difference of 1.0 on the MKL link.

On a host where OpenBLAS is absent and MKL is all there is, the preference
cannot help and the guard is the whole defence: it fails loudly and its INFO
line names the two ways out. `tools/oaf_parity.py` and
`tools/oaf_musicnet_f1.py` still set `MKL_NUM_THREADS=1` defensively, because
they may be run against a build that predates the fix.
