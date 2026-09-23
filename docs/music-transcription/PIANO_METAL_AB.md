# The two piano arms on a GPU — wiring, and what could and could not be measured

**What this is.** The record of wiring `src/hft_transformer.cpp` and
`src/onsets_and_frames.cpp` through `core/gpu_backend_pref.h`, and of the
attempt to measure CPU against Metal on macOS CI. It contains one large
negative environmental finding, one bug the attempt exposed, and a set of
Apple Silicon **CPU** numbers that turn out to answer most of the question the
work was commissioned for.

**What it is not.** A Metal speedup number. There is not one here, and the
reason is in §2 rather than in the port.

---

## 1. What was wired

Both arms called `core_cpu_backend::init()` unconditionally, while both params
structs carried a `use_gpu` field that nothing ever read — `grep -c
"params.use_gpu"` returned 0 in both files. They were CPU-only backends with a
knob that lied to callers.

They now follow `src/crepe.cpp:346`, the cleanest instance of the
`core/gpu_backend_pref.h` pattern (issue #214) in the music family:

```cpp
const bool no_gpu = crispasr_env::get("CRISPASR_HFT_NO_GPU") != nullptr || !params.use_gpu;
ctx->backend = no_gpu ? nullptr : crispasr_init_gpu_backend();
if (!ctx->backend) ctx->backend = core_cpu_backend::init();
```

`use_gpu` is honoured rather than deleted. The env vars — `CRISPASR_HFT_NO_GPU`
and `CRISPASR_OAF_NO_GPU` — force CPU whatever the caller asked, so one binary
can run both A/B arms.

**Four things had to change with it**, none of which a grep for `init()` shows,
and all of which would have been latent crashes:

| what | why |
| --- | --- |
| `core_cpu_backend::set_n_threads()` guarded behind `is_cpu()` at all six per-graph call sites | in the ordinary build it *is* `ggml_backend_cpu_set_n_threads()`, which asserts `ggml_backend_is_cpu()`. On Metal every forward pass would have aborted. |
| `hft_to_f32()` / `oaf_to_f32()` staged through `ggml_backend_tensor_get()` | they dereferenced `tensor->data`, which is a host pointer only for a host buffer. Not theoretical for O&F: the BiLSTM runs on the host, so its `R` and `b` are read back at every load. |
| hFT's `load_weights_repack()` restricted to a CPU backend | the repack extra buffer type is a property of the CPU *device*; no GPU backend offers one. |
| the chosen backend printed at `verbosity >= 1`, asking the device rather than the return value | `crispasr_init_gpu_backend()` ends in `ggml_backend_init_best()`, which returns the **CPU** backend on a CPU-only build. The naive check would claim a GPU that is not there, and an A/B needs a line it can trust as proof. |

The parity dumps gained `CRISPASR_PARITY_USE_GPU=1` (opt-in, so nothing changes
for existing callers on a CUDA box), and `oaf-parity-dump` gained the
`peak_rss_mib` field `hft-parity-dump` always had.

---

## 2. The finding that blocked the measurement: GitHub's macOS runners have no usable GPU

`ci.yml:402` builds `macos-latest` with `-DGGML_METAL=ON`, and GitHub's macOS
arm64 images are Apple Silicon, so Metal was expected to be real there. It is
not. The runner's GPU is a **paravirtual device**:

```
ggml_metal_device_init: GPU name:   MTL0 (Apple Paravirtual device)
ggml_metal_device_init: GPU family: MTLGPUFamilyApple5  (1005)
ggml_metal_device_init: simdgroup reduction   = false
ggml_metal_device_init: simdgroup matrix mul. = false
```

`ggml_metal_device_supports_op()` consults exactly those two flags for
`GGML_OP_MUL_MAT` (`ggml/src/ggml-metal/ggml-metal-device.m:1960`), so **the
device has no matmul kernel of any kind**. hFT reached its first encoder GEMM
and the process died:

```
ggml_metal_op_encode_impl: error: unsupported op 'MUL_MAT'
ggml/src/ggml-metal/ggml-metal-ops.cpp:204: unsupported op
```

A dense-GEMM model has nothing to run on that device. **Metal throughput for
these two models cannot be measured on GitHub's hosted macOS runners**, and no
amount of workflow care changes that. Measuring it needs a self-hosted Apple
Silicon runner or a physical Mac.

This is not a Metal limitation and not an Apple Silicon one — it is GitHub's
macOS virtualisation. Real M-series hardware reports `simdgroup matrix mul =
true` and takes the `mul_mm` path.

**Asked of every hosted image, not just the one that failed first.** The
`metal-capability` job in the workflow builds one binary and runs one 3 s clip
on each, so the question costs minutes rather than hours:

| image | GPU | simdgroup reduction / matrix mul | verdict |
| --- | --- | --- | --- |
| `macos-14` | MTL0 (Apple Paravirtual device), Apple5 | false / false | **not usable** |
| `macos-15` | MTL0 (Apple Paravirtual device), Apple5 | false / false | **not usable** |
| `macos-26` | MTL0 (Apple Paravirtual device), Apple5 | false / false | **not usable** |
| `macos-latest` | MTL0 (Apple Paravirtual device), Apple5 | false / false | **not usable** |

All four, identically. There is no hosted GitHub macOS image on which a
dense-GEMM ggml model can use the GPU.

### 2a. The bug the crash exposed, and the fix

The abort is not specific to these models. Any backend that drives a **single**
backend with `ggml_gallocr` + `ggml_backend_graph_compute` — rather than a
`ggml_backend_sched` with the CPU as a fallback — has no graceful path for an op
the device declines. ggml aborts the process.

`crispasr_backend_supports_mul_mat()` (`src/core/gpu_backend_pref.h`) now asks
before committing: it builds a header-only `MUL_MAT` for **each weight dtype the
GGUF actually contains** — collected from `gguf_get_tensor_type()` while the
metadata context is open, so it is the model's real dtypes and not a guess — and
runs `ggml_backend_supports_op()`. If the device declines any of them the GPU
backend is freed and the model falls back to the CPU with a warning.

It is deliberately narrower than "this device can run this model": a device with
`MUL_MAT` but no `POOL_2D` would still abort inside O&F. The complete answer is
a `ggml_backend_sched` with a CPU fallback backend, which is a graph-lifecycle
change rather than an init-time one, and is left as future work. This catches
the case that actually occurs, because a device with no matmul kernels has
nothing to offer a transcription model anyway.

### 2b. The process bug, which nearly shipped a green lie

The first run of the A/B workflow went **green having measured 2 arms of 48**.
The driver raised on the third arm; it was piped into `tee`; the pipeline's exit
status was `tee`'s, which is 0; every later step then "succeeded" on empty
files and the artifact contained an empty table.

A perf job that reports nothing while looking healthy is the worst failure mode
available. Every such step now sets `-o pipefail`, and the A/B driver runs a
**Metal preflight** before spending 48 runs, so a runner whose GPU cannot execute
the model reports *"the Metal question is UNANSWERED here"* — not a failure, and
above all not a silent CPU fallback quietly averaged into "Metal is exactly as
fast as CPU", which is the most plausible-looking wrong answer available.

---

## 3. Correctness: the CPU path is unchanged, verified against ONNX

The wiring touches how the backend is chosen and how weight bytes are read, not
the arithmetic — but "by construction" is not a measurement, so the existing
per-stage harness was re-run against native onnxruntime on the rebuilt binary.

**Onsets & Frames, `crispasr-diff onsets-and-frames`, 26 stages, gated at cosine
0.999** — `onsets-and-frames-f32.gguf` against
`/mnt/storage/tuner-bench/onnx/onsets_and_frames.onnx`, 3 s clip:

```
onsets-and-frames diff: PASS (0 of 26 stages failing)
```

Every stage returns `cos=1.0000000`, with `max_abs` between 2.7e-07 and 9.5e-06
and `|mine|` equal to `|ref|` to six figures — the magnitude columns matter
because cosine is scale-blind. The extremes:

| stage | cosine | max abs | \|mine\| | \|ref\| |
| --- | --- | --- | --- | --- |
| `mel` | 1.0000000 | 0.000e+00 | 4.78099 | 4.78099 |
| `onset_conv0` | 1.0000000 | 9.537e-06 | 1.01310 | 1.01310 |
| `onset_bilstm` | 1.0000000 | 3.472e-06 | 0.624737 | 0.624737 |
| `onset_logits` | 1.0000000 | 9.537e-06 | 11.4813 | 11.4813 |
| `frame_logits` | 1.0000000 | 8.583e-06 | 9.50199 | 9.50199 |
| `velocity_logits` | 1.0000000 | 2.682e-07 | 0.349392 | 0.349392 |

A `mel` stage at max_abs exactly 0.0 is expected and is not a tautology: the
reference is *run on the mel the C++ runtime computed*, precisely so that a
front-end difference cannot hide behind a model one, and the stage is compared
anyway to catch a front-end change as itself.

The run also re-confirms the O&F throughput the perf doc records —
**0.545 cpu-s per audio-second at one thread**, against
`ONSETS_AND_FRAMES_PERF.md`'s 0.5301 — so the backend-selection change costs
nothing on the CPU path.

**hFT, `tools/hft_parity.py` against the pruned ONNX export**, run through a
shim that skips the `torch`/`torchaudio` resample (the clip is already canonical
16 kHz mono; `import torch` does not complete in 240 s on this box under memory
pressure). q4_0, 3 s clip, both runtimes handed the **same** mel so a model
difference cannot hide behind a front-end one:

```
log-mel      max 8.920e-02  rms 1.733e-03  cos 0.99999997  |mine| 1657  |ref| 1657
onset        max 3.073e-01  rms 8.513e-03  cos 0.99308555  |mine| 10.30  |ref| 10.64
offset       max 1.936e-01  rms 4.803e-03  cos 0.99353446  |mine| 6.267  |ref| 6.343
mpe          max 3.744e-01  rms 1.434e-02  cos 0.99475080  |mine| 20.22  |ref| 20.71
velocity     max 7.600e+01  rms 1.408e+00  cos 0.91973923  |mine| 507.2  |ref| 537.4

onset    99.9245% of 22528 decision cells agree (175 above vs 188 in the reference)
offset   99.9822%
mpe      99.7292%
velocity 99.9600% on the ignore_zero GATE, 99.7869% on the exact argmax bin
```

**Read that against `HFT_TRANSFORMER.md` §q4_0, not against 1.0**: the
documented q4_0 figures are onset 0.99350, offset 0.99168, mpe 0.99614,
velocity 0.870, and the f32 row is 1.00000000 across the board. The differences
above are q4_0 quantisation, which is what that table is for; they are not
introduced here. (A different clip, so the numbers are in the same regime rather
than identical.) **The f32 leg is the decisive check on the refactor, and it is exact:**

```
log-mel      max 8.920e-02  rms 1.733e-03  cos 0.99999997  |mine| 1657  |ref| 1657
onset        max 4.261e-06  rms 5.993e-08  cos 1.00000000  |mine| 10.64  |ref| 10.64
offset       max 2.333e-06  rms 4.606e-08  cos 1.00000000  |mine| 6.343  |ref| 6.343
mpe          max 2.550e-06  rms 1.003e-07  cos 1.00000000  |mine| 20.71  |ref| 20.71
velocity     max 0.000e+00  rms 0.000e+00  cos 1.00000000  |mine| 537.4  |ref| 537.4

onset / offset / mpe: 100.0000% of 22528 decision cells agree, in both directions
velocity: 100.0000% on the ignore_zero gate AND on the exact argmax bin
```

`velocity` at max abs **exactly 0.0** — bit-identical to onnxruntime — and the
other three within 4.3e-06 on activations of magnitude 6–21. The `log-mel` row
is the C++ front end against librosa's, which is a different computation and
always differed by ~1e-3 RMS; it is unchanged.

---

## 4. What the Apple Silicon **CPU** numbers say

Metal could not be measured (§2), but the CPU arms ran on Apple Silicon, and
they answer most of the question the work was commissioned for. These are
medians of three after a discarded cold run, every arm in its own process.

**The machine, named:** GitHub `macos-14`, `arch=ARM64`, chip reported as
**Apple M1 (Virtual)**, **3 logical cores (3 P + 0 E)**, 7 GiB, macOS 14.
`ggml_metal_device_init` reports `MTL0 (Apple Paravirtual device)`,
`MTLGPUFamilyApple5`, both simdgroup flags false — hence §2. Three threads,
clips of 3 s and 30 s, `MKL_NUM_THREADS=1`, medians of three after a discarded
cold run, every arm in its own process.

**Two independent runs**, on different runner instances, are reported side by
side. A perf number from one run of one virtual machine is a rumour; two
agreeing runs are evidence, and where they *disagree* that is reported too
rather than the nicer of the pair being quoted. Runs
[35832266132](https://github.com/CrispStrobe/CrispASR/actions/runs/35832266132)
and
[35834423457](https://github.com/CrispStrobe/CrispASR/actions/runs/35834423457).

### hFT-Transformer

| quant | clip | cpu-s / audio-s (run 1 / run 2) | × real time (run 1 / run 2) | peak RSS | notes |
| --- | --- | --- | --- | --- | --- |
| f32 | 3 s | 3.517 / 3.274 | 1.223 / 1.153 | 290 / 274 MiB | 31 |
| f32 | 30 s | 2.617 / 2.407 | **0.926 / 0.833** | 311 / 302 MiB | 274 |
| q8_0 | 3 s | 2.211 / 2.208 | 0.761 / 0.752 | 280 / 274 MiB | 31 |
| q8_0 | 30 s | 1.765 / 1.724 | **0.621 / 0.614** | 309 / 295 MiB | 274 |
| q4_0 | 3 s | 2.450 / 2.433 | 0.859 / 0.831 | 277 / 272 MiB | 27 |
| q4_0 | 30 s | 1.581 / 1.632 | **0.563 / 0.572** | 302 / 293 MiB | 261 |

Fixed vs marginal, solved from the two clip lengths (run 1):

| quant | fixed cost | marginal × real time |
| --- | --- | --- |
| f32 | 0.99 s | 0.893× |
| q8_0 | 0.47 s | 0.605× |
| q4_0 | 0.98 s | 0.530× |

**hFT runs under real time on Apple Silicon, on the CPU alone, at every
quantisation — with no GPU involved.** 0.93 / 0.83× at f32, 0.62 / 0.61× at
q8_0, 0.56 / 0.57× at q4_0 on the 30 s clip; marginally 0.89× / 0.61× / 0.53×.
Against the **2.14× real time** measured on the contended x86 VPS that is a
**2.3–3.8× difference**, and it is the difference between "cannot keep up with
live playing" and "keeps up with a third of the budget spare".

Note count is identical across runs to the note (274 / 274 / 261), so the arms
are computing the same thing and only the clock differs.

The caveat cuts the *right* way for once: this is a **virtualised 3-vCPU slice
of an M1**, the oldest Apple Silicon generation, with no E cores and a third of
a laptop's core count. A real M-series Mac, and very likely a current iPhone,
has more. **0.93× is a floor, not a ceiling.**

Quantisation buys hFT about **1.5×** (f32 → q4_0 marginal, 0.893 → 0.530) and
reproduces cleanly across both runs. It costs almost nothing in RSS here — the
weights are 22 / 7 / 5 MB, so the ~290 MiB peak is activations and the front
end, not the model.

### Onsets & Frames

| quant | clip | cpu-s / audio-s (run 1 / run 2) | × real time (run 1 / run 2) | peak RSS | notes |
| --- | --- | --- | --- | --- | --- |
| f32 | 3 s | 0.332 / 0.307 | 0.172 / 0.182 | 182 / 179 MiB | 14 |
| f32 | 30 s | 0.277 / 0.293 | **0.161 / 0.159** | 300 / 296 MiB | 150 |
| q8_0 | 3 s | 0.270 / 0.294 | 0.149 / 0.158 | 112 / 108 MiB | 13 |
| q8_0 | 30 s | 0.302 / 0.299 | **0.163 / 0.163** | 225 / 228 MiB | 149 |
| q4_0 | 3 s | 0.314 / 0.307 | 0.162 / 0.165 | 100 / 96 MiB | 17 |
| q4_0 | 30 s | 0.330 / 0.296 | **0.174 / 0.160** | 220 / 216 MiB | 141 |

**O&F is roughly 6× faster than real time on the same machine, and
quantisation does not change its speed in either direction.** Run 1 looked like
q4_0 costing 8% over f32 (0.174 vs 0.161); run 2 does not reproduce it (0.160
vs 0.159). Two runs disagreeing by more than the effect means the effect is
noise, and the honest statement is **"no measurable speed difference between
f32, q8_0 and q4_0 for O&F"** — not the more interesting negative the first run
suggested.

That null result is itself consistent with the shape of the model: the cost is
convolution and a host-side LSTM recurrence, not weight bandwidth, so there is
little for a smaller weight to buy. What quantisation *does* buy, and this does
reproduce, is **memory** — peak RSS 300 → ~220 MiB on the 30 s clip, and
180 → ~100 MiB on the short one. On a phone that may matter more than speed
that was never the constraint.

**hFT decodes 274 notes to O&F's 150 on the same 30 s clip**, at ~5× the CPU
cost. Neither is "right" — `samples/jfk.wav` is speech fed to a piano
transcriber, so nearly every detection is marginal — but it is a reminder that
the two are not interchangeable at equal settings.

---

## 5. What could not be measured, and what it would take

| question | status | what it needs |
| --- | --- | --- |
| Metal throughput for hFT | **unanswered** | a GPU with simdgroup matmul. Not available on any hosted GitHub macOS image (§2). A self-hosted Apple Silicon runner, or a Mac. |
| Metal throughput for O&F | **unanswered** | as above. Note the prior is weaker here anyway: ~46% conv, ~29% LSTM, and the LSTM is a host-side sequential recurrence a GPU cannot help with, so a device↔host copy per chunk is added against a partial win. |
| Metal *numerical* parity (fp32-vs-fp16 per op) | **unanswered** | the same. The harness for it is written and gated — per-head cosine at 0.999 against the CPU path, non-finite values a hard failure, plus a note-level pitch-sequence comparison — it simply has never had a GPU to run against. |
| whether q8_0-on-Metal hits the Chatterbox CFM trap | **unanswered, but the prior is "no"** | `docs/quantize.md` records that Metal's q8 **mat-vec** kernel requantises activations and corrupts the CFM. hFT flattens its position-wise GEMMs into one many-row `mul_mat` (`8f76e054`) and O&F's convs go through `im2col`+`mul_mat`, so both are matrix-**matrix**. That is a reason to expect safety, not evidence of it; the q8_0 arm is compared like every other one when a GPU exists to run it. |

**The honest summary of the Metal question: it is open.** Nothing here shows
Metal helping, and nothing here shows it failing to help. What is settled is
that the port can now *reach* a GPU, that it degrades rather than aborts on one
that cannot serve it, and that the measurement harness exists and is gated.

---

## 6. For the CrispTuner settings copy

The string in question is `transcriptionModelSpeedUnmeasured` in
`lib/l10n/app_en.arb` of `flutter_tuner`:

> "How fast this model runs on a phone, tablet or Mac has not been measured. It
> may not keep up with live playing."

That is still the correct thing to say, and this work does not license removing
it. What it can be narrowed by is in §4: the Apple Silicon **CPU** figures are
real measurements on Apple Silicon, not VPS extrapolations, and they are a lower
bound because GitHub's macOS runner is a virtualised 3-vCPU slice rather than a
whole laptop or phone SoC. Any GPU claim would need §5's missing runner first.

---

## 7. Reproducing

```bash
# the A/B, on demand (no push trigger, deliberately)
gh workflow run piano-metal-ab.yml --ref main

# one arm locally, either backend, from one binary
CRISPASR_PARITY_USE_GPU=1 build/bin/hft-parity-dump model.gguf clip.wav out 4   # GPU
CRISPASR_HFT_NO_GPU=1     build/bin/hft-parity-dump model.gguf clip.wav out 4   # CPU
CRISPASR_OAF_NO_GPU=1     build/bin/oaf-parity-dump model.gguf clip.wav out 4

# the ONNX anchor of §3
build/bin/oaf-parity-dump oaf-f32.gguf clip.wav ref 1
MKL_NUM_THREADS=1 python3 tools/reference_backends/onsets_and_frames.py \
    --onnx onsets_and_frames.onnx --mel ref.mel.f32 --output ref.gguf
build/bin/crispasr-diff onsets-and-frames oaf-f32.gguf ref.gguf clip.wav
```

`MKL_NUM_THREADS=1` is mandatory for any parity or timing run of a mel front-end
model here — CrispASR issue #453, "core_mel: threaded MKL silently multiplies
the upper mel bins by the thread count".
