# SCNet 60-second CPU cache measurement

This offline CPU task runs one actual pretrained small F16 SCNet separation.
It uses the v0.8.53 released Linux CPU CLI, whose `src/scnet.cpp` and
`src/scnet.h` match main `8d3b6aa39`. No build, GPU, external audio, auth token,
or internet connection is needed on the worker. It uses existing allocator
diagnostics; it makes no numerical changes.

`prepare.py` stages a real-model input dataset locally. It downloads the
unexpired qualified-model artifact 13574722861 from run 42371575575 and the
released CPU archive. It does not upload or submit anything. Existing downloaded
archives can be supplied with `--release` and `--models`.

```sh
python tools/kaggle/scnet-cache-cpu/prepare.py "$SCNET_INPUTS"
```

When ready to run, upload this private model dataset, then submit the CPU kernel:

```sh
python -m kaggle datasets create -p "$SCNET_INPUTS"
python tools/kaggle/kpush.py tools/kaggle/scnet-cache-cpu
```

If the dataset already exists, use `kaggle datasets version -p "$SCNET_INPUTS"
-m "Refresh SCNet cache measurement inputs"` instead of create. Account and
credentials use the existing local Kaggle setup; metadata uses the repository's
`${KAGGLE_ACCOUNT}` placeholder. Kernel submission starts execution immediately.
Preparation alone does not run it.

The attached dataset contains real model weights, their MIT notice, and the
released runtime with its license notices. It is not a tooling or cache dataset.
The runner is self-contained because CPU Kaggle workers are offline; it needs
neither a build-toolchain harness nor a secret/progress service.

The synthetic fixture is a deterministic nonconstant asymmetric stereo chirp,
exactly 60 seconds at 44.1 kHz. The actual model runs with the default 20-second
segments and one shift, seed 0, and two CPU threads. Geometry predicts six graph
lookups with frames `872,872,884,876,868,438`; the task records the observed
`metadata_hit` values rather than imposing a target hit rate. It checks exit
status and four finite stereo stems of exactly the input length.

Read `result.json`, `stdout.log`, and `stderr.log` from the output. The retained
allocator records distinguish graph metadata hits from allocation-pool reuse.
Profiling is enabled, so elapsed time is diagnostic and does not establish a
throughput result. The chirp checks execution and geometry, not separation quality.

Kernel version 4 completed on 2026-11-13: six lookups, one metadata hit (16.67%),
with actual frames `872,872,884,876,868,438`. All four stems are finite stereo
with 2,646,000 frames at 44.1 kHz. The single scheduler's allocation pool grows
once from 475,103,872 to 481,567,552 bytes, then retains the same buffer address
and capacity through the remaining metadata misses. This establishes low
metadata reuse while also demonstrating allocation-pool reuse; it does not
establish a benefit from changing shapes or padding.

For local execution in a suitable CPU environment:

```sh
python tools/kaggle/scnet-cache-cpu/run.py --inputs "$SCNET_INPUTS" --output "$SCNET_OUTPUT"
```

Model-free parser and archive/mount-layout tests use only the standard library:

```sh
python tools/kaggle/scnet-cache-cpu/test_run.py -v
```
