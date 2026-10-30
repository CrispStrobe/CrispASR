# All-step serial F32 head graph experiment

This tests the [measured CPU reduction cause](../../../docs/historical/dia-reduction-cause-2026-10-16.md)
through full native generation. It changes no sampled token, reference input,
seed, noise, filter or acceptance threshold. The same generic graph callback
runs every head at every step using only loaded native F32 weights and current
hidden state. No numerical branch depends on a token ID or generation position.

`CRISPASR_DIA_SERIAL_HEAD=serial` adds a CPU ggml custom node implementing
separately rounded F32 products and sequential F32 addition. `identity` adds the
same node/dependencies but copies native logits exactly; `off` leaves the graph
unchanged. The ordinary head matmul remains a dependency in this diagnostic,
so serial mode deliberately computes both paths. No performance win is claimed.
Unsupported precision/shapes reject. No production defaults are changed.

The patch applies only to a separate immutable8d0ad886 runtime checkout with
C36 ggml, physically verified F32 weights/cross caches, portable RNG, source
RoPE off and the original seed42/temperature1.2/top_p.95/top_k45/CFG3/capacity3072
recipe. Current main does not contain all those diagnostic paths; passing this
experiment does not establish current-main/default or GPU/Q4 support.

Four separate C-ABI processes run: pristine(before patch/build), OFF, identity,
and serial. Complete pristine/OFF/identity inputs/logits/stage captures/PCM must
be byte-exact. OFF/identity codes and final MT state also remain exact. Baseline
must reproduce the previously accepted full free-generation arrays and PCM,
plus the prior stage/prefix proof. Serial trajectory differences are reported
without assuming where the first changed input occurs. Complete code grids
and final RNG are saved without modification. Each instrumented code grid is
decoded with actual pinned original DAC; all native/original DAC outputs are
recognized by the same final recognizer/library. Both original positives must
score WER0. The worker retains all outcomes and fails its final quality gate
unless serial scores WER0 through both decoders. Complete evidence still needs
an independent audit before interpretation/integration.

Local `selftest.py` compiles the actual graph callback against verified C36
headers/libraries, checks all18,504 captured head outputs on both hidden states,
one/four threads, exact identity/native controls, a tiny fusion witness and
four rejecting controls. Whole patched runtime syntax is checked with pinned
headers. Paths are explicit arguments; no full neural model runs locally.
The hosted preflight freshly compiles/runs this same callback against its actual
native build and verifies original/identity/serial arrays on both saved hidden
states before paying for full neural inference.

```sh
gh workflow run heavy-cpu.yml --ref <candidate> \
  -f script=tools/ci-heavy/dia_serial_head.py -f timeout_minutes=300 \
  -f pip="numpy==1.26.4 torch==2.6.0 torchaudio==2.6.0 torchvision==0.21.0 pydantic>=2.11.3 huggingface_hub==0.36.0 safetensors==0.5.3 descript-audio-codec==1.0.0 gguf"
```

Full speech acceptance, current-main porting, additional prompts/seeds,
CPU/GPU/Q4 validation and profiling remain separate. The CPU custom operation
and duplicate head work are diagnostic costs, not a promoted optimization.

## Independent artifact audit

`audit.py` requires official run/lint/artifact metadata and complete raw ZIP,
explicit expected worker SHA/run ID, original source/canonical GGUF, accepted
terminal/full-native proof bundles, C36 ggml Git objects and the published small
head fixture ZIP. It validates all source/payload/physical/ELF identities,
regenerates the patch, verifies complete controls and prior baseline, checks
raw input/delayed/code alignment and independently scores every ASR/PCM row.
It freshly compiles the callback against retained actual libraries, replays all
returned preflight cases, and compares the selected actual serial hidden/logits
when that generation reaches the capture step. Fresh std::mt19937 seed42 and
one discard per decoder call must reproduce each complete final RNG state.

```sh
python tools/ci-heavy/dia-serial-head/audit_selftest.py --output FRESH_CONTROLS
python tools/ci-heavy/dia-serial-head/audit.py \
  --worker-source FULL_SHA --run-id RUN_ID \
  --bundle COMPLETE_HOSTED_BUNDLE --repo CRISPASR_REPO \
  --original-repo ORIGINAL_DIA --canonical CANONICAL_REF_GGUF \
  --terminal-bundle ACCEPTED_TERMINAL_BUNDLE \
  --native-bundle ACCEPTED_RECIPROCAL_NATIVE_BUNDLE \
  --head-fixture COMPLETE_REDUCTION_CAUSE_ZIP --ggml-repo C36_GGML_REPO \
  --output FRESH_AUDIT_JSON
```

Calibration passes26 rejecting grid/RNG/path/nonfinite/ASR/scale controls with
an actual freshly compiled MT engine. This calibrates the auditor only; the
complete future artifact audit remains pending. An expected final speech-gate
failure may still have fully verified diagnostic evidence; the auditor reports
that outcome explicitly and never turns it into speech/production acceptance.
