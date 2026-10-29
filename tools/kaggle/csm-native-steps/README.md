# Native CSM graph-step diagnostic

Prepared, unsubmitted private Kaggle job. It builds the pinned CSM profiler
source for CUDA on the worker, and uses the same Q4_K CSM model, public prompt,
seed, temperature and ASR oracle as the accepted [paired pilot](../../../docs/csm-original-t4-benchmark-2026-10-29.json).
Source, binary and model hashes are recorded. It does not require original CSM
weights; the token dataset enables the established model access path.

One warmup and two measured process pairs alternate instrumentation off and on.
The worker checks host resources before build and inference; requires T4,
process-specific GPU memory and positive GPU activity; checks WAV format and
ASR WER; and requires identical generated WAV hashes across every call. The
profile parser rejects incomplete frames, duplicate rows, missing timings and
incorrect counts. Audio and subprocess logs remain private.

The native counters time synchronized scheduler graph executions separately
for backbone prefill, backbone step, depth prefill and depth step. They omit
input preparation, sampling and other frame work. Synchronization adds overhead;
the on/off control measures its aggregate process effect. Attempted frames
include an EOS attempt while accepted frames exclude it. These metrics are not
comparable to the original model's whole-frame calls without aligned boundaries,
precision and sampling. Graph-only means are available from each stage's total
and actual call count; no matching performance claim follows from this run.

The CSM model's Apache-2.0 terms and separate upstream tokenizer/codec terms are
recorded in the paired pilot and its README. Do not redistribute gated model
weights, private synthesized audio, or access credentials. The code under test is
on a validation branch and must pass hosted build and output identity before any
integration. A failed allocation or run must be retained, not silently retried.
