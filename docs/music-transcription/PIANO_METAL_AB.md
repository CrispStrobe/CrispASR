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
