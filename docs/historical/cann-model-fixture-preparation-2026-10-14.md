# Small CANN model fixture preparation — 2026-10-14

The next model-level experiment starts with Moonshine Tiny. This is a
CPU fixture producer, not NPU acceptance or a runnable hardware procedure.

Diagnostic source `a149749ee492c4ea9fa80b682d0248090f9bb6af` adds
`tools/ci-heavy/cann_model_fixture.py` and four weight-free guard tests.
[Heavy CPU run38555394740](https://github.com/CrispStrobe/CrispASR/actions/runs/38555394740)
is submitted against that exact source, pinning the integrated ggml rotary
repair. The hosted result is pending; local syntax and all four guards pass.
All four secret tests pass before the diagnostic push.

The producer pins original UsefulSensors Moonshine Tiny revision
`390624ed33d594443aa4aa221f5b9f283b545b5a`, torch2.6.0 and transformers4.57.6.
Its original model card uses AutoProcessor with an attention mask and a
maximum generation length of6.5tokens per audio second. Those exact choices
are retained; input PCM/mask equality is checked before generation. Original
weights/config files, actual modeling source and generation token IDs/config
are recorded for independent audit.

The existing F32 regression model is pinned at
`cstr/moonshine-tiny-GGUF@28281de882bac021e9241e640f44a9a289607e03`.
Fresh F32 conversion must reproduce every coefficient payload. The producer
then runs the repository’s Python blueprint dumper and native diff, independently
compares complete native encoder arrays to original reference arrays with
cosine, magnitude and relative-L2 gates, and checks decoded text through CLI
and repeated auto/explicit Python Session dispatch. Double-scale, nonfinite
and shape controls reject real bad arrays. Local guard tests additionally
reject zero rows and reversed vectors. No threshold is chosen from results.

The worker retains complete configure/build/command logs, compile commands,
model/tokenizer, original reference, owned arrays, transcripts and file receipts.
Model/build work runs on GH. One collector under
`${CRISPASR_STORAGE}/crispasr/cann-model-fixture-20261014-v2` watches the existing run,
retains failed results, verifies the official full artifact digest, safely
extracts all payloads and records their hashes. Collector completion only
means raw evidence is retained; independent numerical/decoded-output audit
is still required.

Current Moonshine crispasr-diff stage dispatch is CPU-only. Passing it cannot
prove CANN execution. After this CPU bundle works, prepare explicit CANN stage
selection and real device/scheduler evidence, execute hosted negative controls,
then publish a separately tested hardware procedure. No new device request
or NPU/Q4/performance/release acceptance is made by this submission.

## Corrected regression precision before model execution

The immutable published GGUF header contains 160 tensors, all F32, despite
its previous F16 manifest description. [Header audit](moonshine-tiny-regression-dtype-2026-10-14.json)
records the exact model revision, linked whole-model size/hash and inspected
65,536-byte range hash. Whole-model hashing and numerical proof remain for
the hosted worker; header inspection alone is not inference acceptance.
The manifest description is corrected without changing weights, expected text
or thresholds. The first producer run38554714421 was cancelled while queued,
before any steps or model work; its cancellation/no-artifact metadata remains
retained. Corrected run38555394740 uses fresh F32 coefficient identity.
No failed model measurement was discarded and no tolerance was adjusted.
