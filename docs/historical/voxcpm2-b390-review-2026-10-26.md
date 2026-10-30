# VoxCPM2 B390 performance review — 2026-10-26

Issue [#461](https://github.com/CrispStrobe/CrispASR/issues/461) remains open.
The description and all 19 comments were read, including the latest listening
report: eight inference steps sound worse than ten. Ten remains the default
in the context, context parameters and CLI. No new B390 execution or speedup
is claimed here.

## What the actual B390 logs establish

Both reports use the mixed `voxcpm2-q8_0-locdit-f16.gguf`, seed 2, eight steps,
18 AR steps and 3.04 seconds of audio on Intel Arc B390. These are historical
eight-step observations, not ten-step acceptance results.

| Stage | [September 30](https://github.com/CrispStrobe/CrispASR/issues/461#issuecomment-5901948284) | [October 8](https://github.com/CrispStrobe/CrispASR/issues/461#issuecomment-6051358760) | Difference |
|---|---:|---:|---:|
| TSLM prefill | 282.3 ms | 90.7 ms | −191.6 ms |
| AR loop | 2411.5 ms | 2340.4 ms | −71.1 ms |
| CFM average per AR step | 96.6 ms | 99.5 ms | +2.9 ms |
| VAE weight initialization | 118.07 ms | 313.35 ms | +195.28 ms |
| VAE permuted-buffer allocation/upload (inside initialization) | 14.73 ms | 120.52 ms | +105.79 ms |
| VAE graph allocation | 8.27 ms | 40.74 ms | +32.47 ms |
| VAE graph compute | 170.11 ms | 197.53 ms | +27.42 ms |
| Whole VAE | 298.63 ms | 554.33 ms | +255.70 ms |
| Whole synthesis | 3085.1 ms | 3096.5 ms | +11.4 ms |

The whole-synthesis RTFs are 1.0148 and 1.0186. Initialization accounts for
76.4% of the VAE difference. The nested rows must not be added to their parent
rows. These builds and their host/driver/cache conditions were not controlled
A/B experiments; the difference does not establish a regression in a particular
kernel. The earlier projected RTF 0.85–0.87 was already withdrawn in the latest
[maintainer follow-up](https://github.com/CrispStrobe/CrispASR/issues/461#issuecomment-6052442804).

## Source audit and timing correction

The October reporter used source `d754aa10a141ec742ff1d764c11f63845b82864b`
and ggml `c36dab89b662838f0f5d4826c399198c0b90bbfc`. Before this timing-only
change, the decoder weight initialization function was byte-identical to that
reporter's source. Current ggml is
`e2ae679a563014f46c546ce6c6448e8601e4dea6`; the B390 needs a current rerun.

`vae_wn_init_ggml()` already retains reconstructed and permuted weights in the
model context and returns immediately once its buffer exists. Starting a second
CLI process can warm the driver's shader cache, but recreates this context and
its weight buffers. It cannot demonstrate amortized context initialization.
The VAE graph itself is built per decode and allocated through the existing
context allocator. LocDiT batch-two and fused-CFM graphs are already cached;
batched TSLM prefill is already active in the latest B390 report.

The old `vae.wn ... upload=` interval also included CPU transposed-convolution
weight permutation. It was not a pure transfer measurement. The instrumentation
now emits separate `upload=` and `host_permute=` intervals. `upload=` measures
the backend tensor-set call on the host; it is not a GPU timestamp. The nested
`vae.wn perm_upload` still includes the separate permuted-buffer allocation and
uploads. No tensor contents, graph operations, cache policy or inference steps
change.

## Required B390 capture

Use a current Vulkan build on the B390, retain exact source/submodule IDs,
compiler/build options, driver version, device selection and model SHA-256.
Check load and available RAM before building or inference. Keep the mixed model,
seed 2, sentence and ten steps fixed:

```powershell
$env:CRISPASR_VOXCPM2_BENCH = '1'
$env:CRISPASR_VOXCPM2_INFERENCE_STEPS = '10'
Remove-Item Env:GGML_VK_PERF_LOGGER -ErrorAction SilentlyContinue
build-vulkan\bin\crispasr.exe --backend voxcpm2 -m voxcpm2-q8_0-locdit-f16.gguf --tts "Hello, this is a short test sentence." --tts-output b390-ten.wav --seed 2 2> b390-ten.log
```

Retain full logs and WAVs for an initial process and at least three subsequent
processes; label these as process restarts. Separately measure repeated synthesis
through one retained context if reporting initialization amortization. Record
wall time, synthesis time, audio duration/RTF, every VAE substage, CFM/TSLM
averages and AR step count. Compare distributions, rather than choosing the
fastest run. Do not clear driver caches without recording the procedure.

For a separate operation-profile capture, set `GGML_VK_PERF_LOGGER=1` and retain
the complete Vulkan operation log. Unset it afterward: this switch is enabled
by variable presence, so setting it to `0` still enables it. Its synchronization
and logging can affect timings; compare ordinary runs with the logger absent.
Use its actual small LocDiT matmul shapes and operation costs to choose the next
kernel experiment. Preserve ten-step listening quality and original-reference
numerical gates for any such change. Neither a T4 nor a driverless hosted build
substitutes for B390 timing or listening acceptance.
