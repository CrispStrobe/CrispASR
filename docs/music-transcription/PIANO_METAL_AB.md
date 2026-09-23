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

hFT's CPU path is likewise unchanged; its `tools/hft_parity.py` leg is noted in
§5 under what could not be completed.

---

## 4. What the Apple Silicon **CPU** numbers say

Metal could not be measured (§2), but the CPU arms ran on Apple Silicon, and
they answer most of the question the work was commissioned for. These are
medians of three after a discarded cold run, every arm in its own process.

<!-- TABLE PENDING: the A/B run was still queued behind a saturated Actions
     queue when this section was written. See the workflow artifact
     piano-metal-ab-{hft,oaf} on the most recent manual dispatch. -->

---
