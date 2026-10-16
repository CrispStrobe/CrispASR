# Index-Echo FFN capture preparation — 2026-10-16

Four plain and three calibrated Q4 recipes fail complete acceptance. The
accepted 2B Q8/9B F16 models and all model pins, thresholds and defaults stay
unchanged. Attention Q8 is already part of the rejected sensitive recipe;
attention/SSM F16 with FFN gate/up Q4 also fails. This supports investigating
FFN error, without establishing which projection dominates it.

The calibration covers all96 FFN statistics. `first_quantized_layer` in the
old receipts is not the first failed metric: middle-recipe block4 has cosine
0.999908 and passes its0.999 gate. Find the first actually failing stage and
compare gate/up/down on identical FFN inputs before proposing another recipe.
See [plain rejection](index-echo-q4-rejection-2026-10-09.json) and
[calibrated rejection](index-echo-q4-imatrix-validation-2026-10-09.json).

The candidate adds matching diagnostic boundaries for the final prefill token:

| Suffix | Captured value |
|---|---|
| `mlp_in` | FFN input after the existing model normalization |
| `gate_out` | Gate linear projection |
| `up_out` | Up linear projection |
| `down_in` | Existing SiLU(gate) × up product |
| `down_out` | Down projection |
| `mlp_out` | FFN output, equal to down output for this blueprint |

Native stage names are `llm_ffn_LAYER_SUFFIX`, available through the existing
`index_echo_stage` C API until the next prefill. Set
`CRISPASR_INDEX_ECHO_DUMP_FFN=1` to observe existing scheduler nodes. Only literal
`1` enables it. No graph operation or weight changes. Row padding is honored,
unsupported layouts fail, and captured arrays own their storage. Existing block
capture and imatrix routing remain in place. Captures stop after prefill, so
cached generation cannot replace the retained prompt intermediates.

For the public original dumper, set `INDEX_ECHO_REF_CAPTURE_FFN=1` and supply
all three expected provenance values:

- `INDEX_ECHO_REF_TRANSFORMERS_VERSION`
- `INDEX_ECHO_REF_MODELING_SHA256`
- `INDEX_ECHO_REF_MLP_SHA256`

Pin the original checkpoint and transformers distribution first; compute these
hashes from that pinned distribution's full modeling file and full
`Qwen3_5MLP` ClassDef source. The helper verifies them and the actual model's
MLP class before installing hooks. It captures exactly forward call0, copies
all six arrays and removes hooks before subsequent generation. The reference
stores provenance in `ffn_source_provenance`. Full-generation reference mode
currently rejects this optional combination explicitly.

`crispasr-diff` recognizes the optional reference stages, enables native capture
and restores the caller's environment afterward. It uses the existing stage
cosine/magnitude gates. A missing requested stage fails rather than skipping.
Default references and production inference do not enable FFN capture.

The Python tiny controls mechanically execute the actual installed original
class source with real PyTorch/ACT2FN; they do not load model weights. Native
tiny controls execute actual ggml tensors through its scheduler, checking six
boundaries, exact OFF/ON/OFF, ownership, padded rows and rejection cases. A
first-row mutation must fail. All18 local Python controls pass. Last-token
slicing precedes device transfer: on the actual37-token tiny original MLP,
old/new captures and outputs are byte-identical, while observed copy sizes fall
from10656 to288 bytes (37× fewer elements). This measures copied data, not GPU
latency, full-model performance or quantization quality. Initial harness attempts omitted input reset
between scheduler computations; failed logs/executables are preserved. The
corrected harness reapplies identical inputs, without changing tolerance.

The full-model F16 observer control is now independently audited. Frozen
sourcea1d84150/run38938983824 and artifact12066499638 pass1946 payload checks,
including17 exact repository snapshots. On both JFK and Mandarin audio, original
69-array and native68-array OFF/ON/OFF controls are byte-exact; enabling capture
adds exactly144 FFN vectors. All690 numerical comparisons pass unchanged cosine
and magnitude gates, including16 cached greedy decisions, generation-prefill
identity, raw original/native text, CLI SRT and metadata-detected Session output.

| Case | Worst cosine | Relative L2 at that stage | Native / original norm |
|---|---:|---:|---:|
| JFK | 0.9999866486 | 0.0053496438 | 4.566902 / 4.560633 |
| Mandarin | 0.9999853969 | 0.0054465674 | 3.644588 / 3.642302 |

Both worst rows are `llm_ffn_3_down_in`. This validates the 2B F16 observer,
without attributing the rejected 9B Q4 recipes. Separate native/original inputs
can already differ because of earlier quantization. Next replay projections on
the same original FFN inputs and distinguish component matmul behavior from
full-model prefill behavior. Complete EN/ZH/cache/output/roundtrip and actual
dual-T4 validation remain required before any new Q4 recipe or speed claim.

The Hermes worker's final local source is5020e60d; rejectedf642293f and its generic
inspector remain excluded. Coordinator integration adds native observation,
public dumper/diff wiring and evidence-preserving output-directory controls.
