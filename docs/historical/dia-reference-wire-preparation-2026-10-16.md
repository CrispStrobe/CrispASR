# Dia reference/diff wiring — validation in progress

The working candidate registers the original Dia observer in the generic
Python dispatcher and adds a text-only `crispasr-diff dia` route. It replaces
the old reconstructed reference shim with hooks into the pinned original
`generate()` implementation. Runtime changes add diagnostic output marking
and readback; the current F16 cross cache, sampler, graph math, top-k and
stopping/crop behavior are preserved.

Default captures include text IDs/embedding, all12 encoder layers, normalized
encoder output, first cross K/V cache, all18 decoder layers, first cross
attention, final hidden, every decoder input/logit, original post-reversal
codes, all8 DAC stages and decoded audio. A selected decoder step supports
optional self-attention Q/K/V/attention and16 block boundaries at any requested
layer0–17. Requested extras reach the archive even when the generic dispatcher
supplies an explicit default stage set; diff reads their metadata and compares
them. Input IDs are checked exactly before any numerical interpretation.

Local checks use tiny actual original Dia classes, not reconstructed neural
math: all17 observer tests pass. Three public-dispatch/archive tests pass.
Compiled production comparison/input guards pass20 acceptance/rejection
controls against the actual current GGUF reader and ggml libraries. Doubled
scale with cosine1 is rejected, as are nonfinite, missing/truncated tensors and
invalid DAC indices. Three parser/scratch tests and one device/environment
restoration test pass. The existing original output-protocol test passes.
All237 ordered ggml computation calls match pristine9aa8df20; a graph or F16
precision mutation makes that source guard fail. Whole-runtime syntax passes.

These checks establish local contracts, not full-model neutrality or parity.
The [hosted worker](https://github.com/CrispStrobe/CrispASR/blob/c7f9da872e87ba238c579083841b3547faee772f/tools/ci-heavy/dia-reference-wire/README.md) builds
actual pristine/candidate runtimes, checks complete OFF/ON/OFF generation,
and runs the public dumper and strict diff on fresh original intermediates.
It retains failures and complete raw evidence for independent audit. No full
checklist, current-main speech, GPU, Q4 or speed claim follows yet.

Local logs and compiled controls are retained under
${CRISPASR_STORAGE}/crispasr/dia-reference-wire-20261016/local-v1. Early failures are
retained: a scratch-test extraction boundary included the new parser, the
public-list test assumed a fixed number of spaces, and the reader test assumed
GGUFWriter stored empty string metadata. The corrected tests isolate the
intended C++ body, match the public list by fields, accept omitted empty audio
metadata and close the actual mmap before cleanup on CIFS.

Candidate [c7f9da87](https://github.com/CrispStrobe/CrispASR/commit/c7f9da872e87ba238c579083841b3547faee772f)
is pushed. Hosted [full-model validation38883779736](https://github.com/CrispStrobe/CrispASR/actions/runs/38883779736),
[lint38883782438](https://github.com/CrispStrobe/CrispASR/actions/runs/38883782438)
and [general CI38883785994](https://github.com/CrispStrobe/CrispASR/actions/runs/38883785994)
all pass at that frozen source. Independent v3 verifies904 payloads and88
full stage comparisons; fresh returned-library C ABI replay passes398 synthetic
comparisons with neutral-filename detection. Earlier auditor path assumptions
and their failures are retained. The complete163537930-byte proof archive is
[public and anonymously SHA-verified](https://huggingface.co/datasets/cstr/crispasr-regression-fixtures/resolve/e6cddd222f7cade565ced28023c4d80a07a34095/dia-20261016-reference-wire/complete-hosted-v1/complete-hosted-reference-wire-v1.zip).
All metrics, pins and hashes are in [the hosted receipt](dia-reference-wire-hosted-2026-10-16.json).
Combined source fb4e5924 now runs heavy38891963272, CI38891966094 and lint38891969068;
its local actual runtime passes1194 comparisons including identity/serial heads.
Integrationa40ceaff now preserves the tested fb4e5924 executable sources exactly.
Heavy, all13 CI and10 lint pass; independent audit verifies1735 payloads/176
comparisons. [Composed proof](dia-reference-wire-composed-2026-10-16.json).
Current-main unforced speech is separately rejected; see HISTORY.md.

## Actual C++ runtime with synthetic original-class weights

A further local test now exercises the unmodified candidate C++ runtime,
not only Python hooks or isolated C++ helpers. It uses all12 encoder and18
decoder layers with small projection widths: encoder hidden1024 (the runtime's
fixed invariant), decoder hidden8, head dimension4 and MLP width16. All381 F32
GGUF payloads match independently recreated original synthetic parameters.
Original Python neural classes drive all masks, caches, attention and MLPs.
Only the unused return from original generate uses a stub codec; native has no
codec and must reject PCM only after completing neural inference. No real-model
or speech acceptance follows.

398 complete neural boundaries pass the unchanged cos_min>=.9995 and per-row
norm-error<.02 gates. Encoder/decoder IDs match exactly; complete decoder
inputs/logits are byte-identical with captures OFF/ON/OFF. All18 decoder layers
expose72 attention captures and288 block boundaries at step5. Independent
array auditing recomputes all398 metrics and rejects doubled magnitude at
every boundary.

| Captures | Count | Minimum cosine | Maximum row norm error | Maximum relative L2 |
|---|---:|---:|---:|---:|
| encoder | 15 | 1.0000000000 | 4.332987e-07 | 9.953484e-07 |
| decoder layers | 18 | 0.9999999892 | 4.385596e-05 | 0.0001197079 |
| self attention | 72 | 0.9999997781 | 0.0002090496 | 0.000165032 |
| decoder block boundaries | 288 | 0.9999997504 | 0.0003283888 | 0.000285692 |

[Compact source/audit receipt](dia-reference-wire-tiny-runtime-2026-10-16.json).
The complete1024-payload proof contains actual parameters/GGUF, sources,
compiled probe/static libraries, all reference/native arrays and independent
audit. Its27,366,062-byte ZIP has SHA256
`a5057a28d8e867a853913b4c6e0c382914e90f27578420febda4b10eaaad8e45` and
is [publicly downloadable at an immutable revision](https://huggingface.co/datasets/cstr/crispasr-regression-fixtures/resolve/60deae78ec95424d76d15f028afa948c468778a9/dia-20261016-reference-wire/tiny-runtime-v1/tiny-runtime-captures-v1.zip).
A fresh anonymous full download verifies size and SHA. Local method and raw
proof remain at ${CRISPASR_STORAGE}/crispasr/dia-reference-wire-20261016/tiny-runtime-v1.
The publication helper's duplicate metadata key failed after successful upload;
the failed log is retained and a separate read-only verifier recovered the
immutable revision and verified a second full anonymous download.

Full1.6B reference/diff, real DAC, native speech, GPU and Q4 remain separate
pending gates. These small synthetic weights prove capture plumbing and
scaled-down numerical consistency, not full-model quality.

## Nontrivial GQA head-order control

The first case's single KV head cannot distinguish head tiling from
interleaving. A second case therefore uses eight query heads and two KV heads:
correct mapping `[0,0,0,0,1,1,1,1]`, compared with deliberately wrong
`[0,1,0,1,0,1,0,1]`. Its four-way grouping matches full Dia. The unchanged
compiled C++ probe executes all12 encoder/18 decoder layers, with decoder
hidden32 and the same small head/MLP widths. All381 physical F32 tensors and
398 complete neural comparisons pass independent audit; original inputs and
OFF/ON/OFF logits remain exact. Minimum cosine0.9999996972; maximum row norm
error0.000422272. No capture or head-layout runtime repair was needed.

Feeding the exact captured normalized inputs and K/V into the actual original
Attention modules gives18 passing correct-cache controls. Replacing their
cache head order with tiling makes all18 fail the unchanged cosine/magnitude
gates. Complete positive/negative outputs are retained. The1063-payload proof
is [public at an immutable revision](https://huggingface.co/datasets/cstr/crispasr-regression-fixtures/resolve/3cf5dafa0973b884d3bace90680ebef0ec271b0d/dia-20261016-reference-wire/tiny-runtime-gqa-v2/tiny-runtime-gqa-captures-v2.zip), with a fresh anonymous full
38,113,866-byte download and SHA256
`7d96fe1a23bf8fd79a52ae3884756619fe047a5213d1ce23f58a1d83e4ce87ca`.
[Compact audit and source receipt](dia-reference-wire-gqa-runtime-2026-10-16.json).
This is synthetic neural/capture validation; the full-model and real-DAC
workers remain the acceptance gates.
