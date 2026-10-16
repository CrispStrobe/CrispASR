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
first-row mutation must fail. Initial harness attempts omitted input reset
between scheduler computations; failed logs/executables are preserved. The
corrected harness reapplies identical inputs, without changing tolerance.

This is capture preparation. Separate native/original full-model inputs can
already differ because of earlier quantization. The next attribution must
replay projections on the same original FFN inputs and distinguish component
matmul behavior from full-model prefill behavior. Hosted F16 identity, complete
EN/ZH stage/output checks and actual dual-T4 validation remain required before
any Q4 recipe, size, speech-quality or speed claim.

The Hermes worker's final local source is5020e60d; rejectedf642293f and its generic
inspector remain excluded. Coordinator integration adds native observation,
public dumper/diff wiring and evidence-preserving output-directory controls.
