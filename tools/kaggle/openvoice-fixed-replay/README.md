# Frozen OpenVoice original-reference replay

This private CUDA kernel reuses the exact rejected model/reference data, with
SHA-256 verification before compilation. It builds only the two model/driver
C++ deltas against the already audited runtime; no SDK/CUDA/ggml rebuild or
Torch installation. The data attachment contains real model/reference files,
complete MIT notice and synthetic-audio attribution, without tooling caches.

The source/header base remains `6be9fda36`, ggml `0a8620765`. Only
`src/openvoice2.cpp` is overlaid from `1b874a104` and verified by SHA-256. This
keeps the prebuilt runtime ABI exact even though current main uses newer ggml.
Both base and delta source/hash are retained in the receipt. It is a bounded
model-level repair check, not whole-current-ggml or Session/CLI acceptance.

It runs the unchanged 27-stage teacher gates on CUDA and CPU. A GPU result
requires actual scheduler CUDA work and a completed vocoder output. Missing
stages, nonzero exit and missing device work fail. `replay_acceptance` is separate
from `model_acceptance`, which stays false: three-case conditioning/shared-noise,
ASR/speaker quality, current-runtime and Session/CLI tests remain to do.

Use the maintainer Kaggle usage guide and current account/session cap. Preserve
terminal source/logs/outputs before any replacement. The bootstrap records its
own hash and pins the exact shared worker. No GPU draw retries or new reference
bakes are part of this job. See [the method](../../../docs/openvoice2-baking.md).

Replay version 1 verified the posterior/flow corrections (26/27 on CPU/CUDA)
but rejected raw decoder amplitude. Version 2 adds the original final LeakyReLU
slope0.01, with all weights, reference bytes and gates unchanged. Preserve both
terminal attempts; normalized output alone hid the raw magnitude defect.
