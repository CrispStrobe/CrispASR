Prepare-only candidate; never submitted. Runtime source is frozen at
`529e073f8c9c99b688be5823e1be78443d50763e`. The uploaded script is independent
of that checkout and records its own SHA-256.

The decisive input is the **entire last five-second window** of the original
10 s JFK/noise construction: PCG64 seed 579, noise scaled on all 160000 samples
to 5 dB, then samples `[80000:160000]`. Its raw F32 SHA-256 must be
`4143c5cff09bc5b07437d027fcadc6c3b7cf1366f6856a86b4be99b990308469`.
The troubled 7–8 s segment is cropped from the resulting 5 s outputs, preserving
model context. A fresh isolated 1 s inference would answer a different question.

The script runs original fused Mamba 2.2.4 and the audited research adapter in
separate fresh GPU processes, captures all 72 named stages, and requires exactly
120 original selective-scan CUDA and 120 original causal-convolution CUDA calls
in the fused arm. It retains each compressed capture before starting the next
arm. Original source, model/configuration, wheel/checkpoint/source manifests,
input and output hashes are recorded. Hook semantics match the accepted fused
capture job. Scale-sensitive norms, relative RMS, cosine and max absolute error
are reported for every stage.

The CPU comparison reads the original retained `window-2-reference.gguf`, whose
SHA-256 must be `25da6651db2ea3ceb7910daec5ff7968db83aa29b56732f1f0f7114b2763cb02`.
Metadata attaches the existing ten-second kernel output through `kernel_sources`;
the worker must actually mount that exact 1054097472-byte archive. Missing,
duplicate or mismatched mounts fail before inference. Do not rerun/overwrite
the ten-second producer to prepare this candidate. Its original native log has
71 unconditioned and 71 conditioned PASS rows; no expensive scalar native replay
or native build is requested here. Both GPU arms use the same CUDA frontend;
CPU/GPU phase branch-cut differences remain observations, never an equality gate.
A completed diagnostic report is evidence acquisition, not a numerical acceptance
claim. No tolerance/default change or claim of product support follows.

Execution requires the already accepted fused wheels at immutable HF dataset
revision `5d53593a148d2ae5e99667a3f04a04d4fddc2107` and their recorded SHA-256
and source manifests. They require CPython 3.13, Torch 2.11.0+cu130, CUDA 13,
CXX11 ABI and compute 7.5 (T4). Other draws exit early with `conclusive:false`;
there is no rebuild, Torch reinstall or automatic redraw. Three import-compatibility
wheels are URL/SHA locked and installed with the extensions into an isolated
`--target`, `--no-index`, `--no-deps` directory. Existing small numerical packages
are checked against the producer versions. No arbitrary source is installed.
No compiler is invoked; harness heartbeats cover both captures.

Useful interpretation: if fused and adapter agree on the same GPU and both
remain poor on this exact context, the adapter is not the explanation. A first
same-GPU Mamba divergence points to fused/adapter behavior; CPU/GPU frontend
phase divergence alone cannot make that attribution. The original run's
window-2 quality and exact native/reference stitching reconstruction must be
read alongside this capture. SNR/SI-SDR on a speech proxy plus synthetic noise
are bounded diagnostics, not listening-test evidence.

Local checks: `python tests/test_reuse_quality_followup.py -v`.
No GitHub workflow changes, push, CI dispatch or Kaggle submission are part of
preparing this directory. Execution prerequisites are available quota, a compatible GPU draw,
the exact archive attachment and the pinned ABI. Execution is already authorized;
this preparation task does not launch it.

Read-only operator preflight on 2026-11-12 found no local `nvidia-smi` and
Kaggle GPU quota exhausted: used 67474.119 s, reserved 0 s, total allowed
21600 s; refresh `2026-11-14T00:00:00Z`. The live quota response, rather than
the historical nominal weekly allowance, controls submission readiness.
Do not try another account or repeated accelerator draws.

The v2 candidate checks live RSS of the parent plus all descendants every five
seconds, limits it to 10 GiB, limits output to 8 GiB and preserves 2 GiB free on
both scratch/output filesystems. It writes resource receipts before either
capture and continuously during inference. A breached budget records a bounded
failure receipt and terminates live descendants; existing references, complete
captures and partial captures are retained. Captures become final atomically.
Auth/download helper chatter and subprocess output are suppressed; exception
receipts contain only exception classes and fixed reason codes, avoiding signed
URLs or account-bearing mount paths in progress/error artifacts.
