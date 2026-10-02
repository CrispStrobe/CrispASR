# Index-Echo scheduler experiment

Build the runtime with `index-echo-cuda-validation.yml` on the feature branch.
Preserve the successful build artifact and upload the archive to the private
`cstr/crispasr-index-echo-cuda-validation` dataset. Fill the SDK/build commit,
bundle revision and SHA-256 pins before pushing with `tools/kaggle/kpush.py`.
The script refuses unset pins and exits inconclusive before downloading models
when the worker lacks two SM75 GPUs. Respect the single account and two-session
limit in the sibling Kaggle guide; do not re-push to seek different hardware.

The candidate runs the canonical 9B F16 independent stage/cache/magnitude,
CLI/session, five-file and three-roundtrip checks with pipeline scheduling
disabled. It then times control/candidate in both orders, each in a fresh
process, with one initial and three warm calls per clip. Both arms use the
same pinned runtime and public weights; every timed output must match the
independent source. Runtime logs must prove that the control enabled pipeline
scheduling and the candidate actually reused graphs. The receipt records
results and does not change any default.

The canonical validator accepts `INDEX_ECHO_VALIDATION_CONFIG` pointing to a
JSON file with immutable `source_commit`, `build_commit`, `build_run`,
`bundle_revision`, `bundle_sha256`, and optional `keep_models` for subsequent
GPU measurements. Without this file its published validation pins and cleanup
behavior are unchanged. Candidate speed remains unproven until the full run
completes; a successful CUDA build proves compilation only.
