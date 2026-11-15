# Issue #589 bounded first-frame diagnostic

The diagnostic patch is based on `a8151a2e8d7914571139dff1fd0a03620eaca051`.
Build this worktree using the established backend configuration. Do not reuse an
uninstrumented binary: the capture command requires a completion marker and
records the binary hash, source revision, runtime patch hash and every payload
hash. No model download is performed.

The exact runtime patch SHA-256 against that base is
`b27d8dcc6530b93abc986c7492c0c8c87b62e03adf44f4e01c4cef8ade5c0d77`.
Before building the diagnostic binary, check the runtime source pin:

```sh
test "$(git diff a8151a2e8d7914571139dff1fd0a03620eaca051 -- src/vibevoice.cpp | sha256sum | cut -d' ' -f1)" = \
  b27d8dcc6530b93abc986c7492c0c8c87b62e03adf44f4e01c4cef8ade5c0d77
```

On an idle GPU worker, with the exact neutral disclosure in
`neutral.txt` and the original 7B Q4_K pack:

```sh
python tools/vibevoice_first_frame_capture.py \
  --binary "$CRISPASR_BINARY" --gguf "$CRISPASR_7B_Q4_K" \
  --text-file neutral.txt --output "$CRISPASR_CAPTURE_ROOT/neutral-seed42"
```

Expected pack SHA-256:
`ff68a234c5a1069e379e4d3fdbf29bb07ec4981d0ad8841bb19bc442991413f3`.
The wrapper explicitly selects `--backend vibevoice-1.5b`, the base-family
dispatch used by the 7B pack. Seed 42, semantic feedback, refreshed negative history and semantic F16 recovery
are enabled; token constraint is explicitly disabled. CFG 1.3 and 20 solver steps
retain the prior configuration. The wrapper clears unrelated dump/capture gates.
The native diagnostic returns null immediately after the first completed latent,
before feedback, vocoder or audio output. CLI failure status is expected and is
recorded; only a complete, size-checked payload is success for this diagnostic.
This is never a speech quality test.

Capture uses a new directory exclusively and cannot overwrite a previous run.
Limits: 4096 text bytes, 2048 prompt tokens, 64 AR token steps before the first
speech frame, at most 100 diffusion steps, 64 MiB binary payload, 64 MiB log,
900-second default timeout (maximum 1800). Each first-frame step records both
prediction branches, CFG velocity, x0 and updated latent. Prompt IDs/embeddings,
prefill dimensions and hidden/logits, actual positive/negative conditions, previous negative
input, token choices, positions, timestep schedule, initial noise and final
latent are included. Existing unbounded dump gates are not needed.

Replay only the head and scheduler on CPU using the *same dequantized GGUF
weight values*, actual upstream module definitions, and captured native LM
conditions/noise:

```sh
python tools/vibevoice_first_frame_replay.py \
  --gguf "$CRISPASR_7B_Q4_K" \
  --expected-sha256 ff68a234c5a1069e379e4d3fdbf29bb07ec4981d0ad8841bb19bc442991413f3 \
  --capture "$CRISPASR_CAPTURE_ROOT/neutral-seed42/native" \
  --source "$CRISPASR_VIBEVOICE_SRC" \
  --output "$CRISPASR_CAPTURE_ROOT/neutral-seed42/upstream-head"
```

The replay script passed a tiny actual-upstream F32-GGUF roundtrip on 2026-11-15:
all seven prediction/CFG/solver boundaries match exactly; a deliberately perturbed
first prediction boundary reports maximum absolute error 0.25 only there.
See the [control receipt](first-frame-replay-tiny-control-2026-11-15.json).
This validates reconstruction/comparison plumbing, not native inference parity.
The 7B replay remains a prototype: head size/feasibility is unestablished. The
script refuses captures over 64 MiB and heads over 2 GiB of dequantized weights;
it loads only prediction-head tensors and computes with one CPU thread.
No streaming-weight implementation is claimed.
It checks upstream/native timestep equality and reports cosine, both norms and
maximum absolute error at each branch/CFG/solver boundary. The receipt hashes
the actual upstream head and scheduler files. A completed comparison report
records measurements; it does not impose or assert a quality acceptance gate.

This replay injects native LM conditions. Upstream 7B LM/prefill layer replay is
**not implemented**, so no prefill parity, neutral prefix fix, Metal equivalence,
Vulkan acceptance or speech quality is established. No new 7B job is launched by
preparing these tools.

Small control (actual upstream classes/scheduler, tiny generated GGUF, no model
weights or GPU):

```sh
CRISPASR_VIBEVOICE_SRC="$CRISPASR_VIBEVOICE_SRC" \
  PYTHONNOUSERSITE=1 PYTHONPATH="$CRISPASR_LOCAL_REPLAY_DEPS" \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  "$CRISPASR_LOCAL_PYTHON" tests/test_vibevoice_first_frame_replay.py
```

Validation performed locally: production scheduler configuration controls 4/4;
final native C++ syntax-only compile; Python byte-compilation; whitespace check.
The initial actual-upstream replay and negative-history controls were blocked
by shared-storage imports on 2026-11-14. The replay prerequisite was repaired
once on 2026-11-15: disable user site packages and install only
`diffusers==0.35.2` with `--no-deps` into an isolated local target. Local Torch
2.9.1, NumPy 2.2.6 and gguf 0.18.0 were reused without global package changes.
The tiny matching/perturbation control passed in 5.735 seconds. Disk reserve
remained 17 GiB, above the required 12 GiB. Native negative-history testing was
not repeated as part of this prerequisite repair.
No 7B or GPU inference was run. The fake-producer capture test validates runner
orchestration only and does not validate native first-frame tensors.

The capture runner additionally requires exact sizes for prompt IDs, prompt
embeddings, prefill logits and the selected-token trace. Its dry producer controls
cover missing and truncated versions of all four essential prefill payloads,
plus diagnostic nonzero exit, exclusive output, backend selection and latent
truncation. These controls passed without model loading.
