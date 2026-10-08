# MiMo graph-phase CUDA profile

Immutable source and uploaded script SHA are recorded independently. This
kernel uses the original Q4 LM/codec and production arithmetic. It verifies
actual CUDA allocation/dispatch, CLI/session output and repeated EN/ZH output
across an ABBA profiling-toggle sequence (two warmups plus six measured calls
per clip/process). The toggle measures instrumentation overhead, not an
optimization. Detailed phases report host wall time, with synchronous graph
compute and logging excluded between phases. Scheduler `pipeline_copies` is
pipeline buffer multiplicity, not a transfer count.

The graph helper is disabled by default. GPU routing, embeddings, attention,
model files and runtime defaults are unchanged. This does not resolve PR #492's
Q4 non-flash numerical discrepancy or certify CANN. Archive terminal logs/output
before any new push and refresh the exported actual-Kaggle build cache after
success, following the maintainer Kaggle guide.
