# Same-input CPU attention operator diagnostic

This model-free probe compares actual ggml CPU score, softmax, eager attention
and flash attention. It changes no runtime math or defaults and loads no model.
Build against the exact ggml headers and libraries whose source and physical
binary hashes you record. Initialize the ggml submodule first.

```sh
c++ -std=c++17 -O2 -Wall -Wextra -Werror -I ggml/include \
  tools/ci-heavy/attention-owned-probe.cpp \
  -L "$GGML_LIB_DIR" -Wl,-rpath,"$GGML_LIB_DIR" \
  -lggml-cpu -lggml-base -o "$OUT/attention-probe"
"$OUT/attention-probe" "$OUT/fresh-native"
python3 tools/ci-heavy/attention-owned-audit.py \
  "$OUT/fresh-native" "$OUT/fresh-audit.json"
```

The probe uses 32 channels, two heads and 71 keys. All query rows within a
head are identical, with an identical key mask, so changing query count does
not change the mathematical problem. The 64 cells cover two input patterns,
1/63/64/71 query rows, one/four threads, F16/F32 K/V storage and default/scalar
reference execution. F32 storage contains exactly the same F16-representable
K/V values; only storage and kernel execution differ. Raw output layout is
`[query, head, channel]`; score/probability layout is `[head, query, key]`.

The first pattern has zero Q/K and constant V along the key dimension. Its
attention output must equal V: this is an algebraic invariant, independent of
a tolerance on a particular neural model. The auditor also predicts the scalar
F16 result by rounding the sum to F16 after each addition, then applying the
actual F32 reciprocal. A second, nonuniform pattern checks F64 score, softmax
and value-product oracles on the captured native operands. Scale, channel-order,
shape and changed-input negatives remain separate controls. Fresh directories
and audit paths are mandatory; failed results must not be overwritten.

Pinned ggml `0a862076567afd8468822ccc512c97ac6a69b340` uses an F16 value
accumulator in the scalar flash kernel and an F32 accumulator in the tiled
kernel, selected at 64 query rows for these shapes. Its patched F16 CPU dot
trait consumes F32 activations; do not assume an additional Q-to-F16 conversion.
The diagnostic deliberately retains and explains the scalar invariant failure.
A passing attribution audit does not mean that every kernel meets a numerical
quality gate.

This does not replay real MiMo Q/K/V. The older retained first-divergence capture
contains normalized inputs and projected outputs, but no raw Q/K/V tensors.
The 71-row MiMo prefill takes the tiled path, so scalar F16 accumulation alone
cannot explain its earlier flash/eager gap. Capturing identical Q/K/V, mask,
scores, probabilities and pre-output-projection values in that actual graph
remains necessary before a model-specific repair. Existing failed whole-model
cache-read, Q4 stage and speech gates remain in force. No speed, GPU or release
acceptance follows from these tiny tensors.
