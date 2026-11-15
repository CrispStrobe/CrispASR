# Issue #589 39-token base-LM prefill replay

`tools/vibevoice_prefill_replay.py` prepares an upstream Qwen2 prefill from the
exact dequantized GGUF values. It uses the original `decoder_config`, verifies
GGUF dimensions and RoPE theta, and requires exactly 39 effective prompt rows
and IDs. This path covers the neutral prompt without voice embedding splices.
Same-GGUF token embedding lookup must match the captured effective embeddings
exactly before any block executes.

It constructs one actual Transformers Qwen2DecoderLayer at a time on meta,
loads only that layer, executes the upstream causal mask/RoPE/full-attention
forward, writes its raw 39-row result, and releases the layer. The final norm
uses the original configuration epsilon; the LM head is read in 256-row chunks.
No full Qwen2Model, full embedding table or full LM head is instantiated.

Bounds are 39 tokens, hidden size at most 8192, at most 128 layers, at most
1536 MiB of dequantized weights for one layer, and at most 64 MiB of raw block
outputs. A temporary dequantization copy of the current tensor, Torch runtime
and intermediate activations are additional memory. These are checked logical
bounds; a real 7B replay has not been executed or given an RSS qualification.

The real replay currently uses upstream **F32 K/V**. It does not yet implement
the tiny control's qualified F16-K/V rounding mode. The CLI prints that limitation
and the result receipt records it, so raw native F16-cache differences must not
be called a port defect or matched-precision comparison. The tiny matched-control
results below do not qualify this optional mode in the real replay tool.

Prepared command, after obtaining a matching native capture and original model
configuration (this preparation does not launch it):

```sh
PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  "$CRISPASR_LOCAL_PYTHON" tools/vibevoice_prefill_replay.py \
  --gguf "$CRISPASR_7B_Q4_K" \
  --expected-sha256 ff68a234c5a1069e379e4d3fdbf29bb07ec4981d0ad8841bb19bc442991413f3 \
  --config "$CRISPASR_ORIGINAL_CONFIG" \
  --capture "$CRISPASR_CAPTURE_ROOT/neutral-seed42/native" \
  --output "$CRISPASR_CAPTURE_ROOT/neutral-seed42/upstream-prefill"
```

The receipt hashes the GGUF, configuration file and actual installed upstream
Qwen2 source. It reports cosine, both norms, relative L2 and maximum absolute
error for available native blocks and final hidden/logits. Missing raw native
blocks are explicitly reported; agreement or disagreement only at the final
hidden state cannot identify the first divergent native block.

## Gated native raw block taps

`run_base_tts_decoder` now captures raw blocks only when both the first-frame
directory and `CRISPASR_VIBEVOICE_TTS_CAPTURE_PREFILL_BLOCKS=1` are set. Use the
capture wrapper's explicit `--prefill-only` option for a prefill trace,
or `--prefill-blocks` for the existing first-diffusion-frame trace; setting only an inherited
environment variable would be cleared by its environment isolation.

The tap is the last `cur = ggml_add(ctx0, residual, ffn);` inside the decoder
layer loop, before final norm. Each `cur` is named and marked as an output only
under the gate. After successful graph computation, tensors are read before
scheduler reset as `tts_prefill_block_000.bin`, etc. Each has39 rows of `d_lm`
F32 values; ggml's `[d_lm,39]` matches row-major `[39,d_lm]`. Host-owned effective
prompt embeddings are captured separately, avoiding a post-compute read from an
unpreserved input buffer.

The gate requires `n_past==0`, exactly39 prefill tokens, positive dimensions
bounded by8192 hidden/128 layers, and raw output payload<=64MiB before graph
construction. One block is read back at a time. Graph scratch/allocator overhead
and weights are additional; no real7B/GPU memory bound is claimed. The wrapper
checks the marker, exact file count and every expected size before success.

The [tiny tap A/B receipt](prefill-taps-control-2026-11-15.json) records bit-exact
tap-off/tap-on/retained hidden and logits and two valid raw blocks. Output
preservation can perturb allocation on other backends, so this CPU result does
not establish Metal, Vulkan, real7B parity or speech quality. The inspected local
7B pack has a different size from the exact tested pack, blocking same-weight
capture here; see the [capture command/identity note](first-frame-capture.md).

## Tiny control

`tests/vibevoice-prefill-probe.cpp` includes the exact production runtime and
runs 39-token prefill through genuine depth-one and depth-two graphs. It refuses
anything except the generated hidden-32/two-layer fixture. Its results include
normalized last-token hidden and logits for each depth. The upstream control
compares layer-at-a-time reconstruction with a full actual Qwen2Model and then
compares those native truncated outputs; native F16 KV storage differs from the
upstream F32 computation. A separate altered-block control must detect 0.25.

Compile only against headers/libraries from the same pinned ggml submodule
`1b972bb754aa5f83e2d872ccf37a335717f6658d`, including
`src/core/gguf_loader.cpp` in the standalone probe link. Retained unrelated ggml
libraries or the protected original checkout's older headers do not satisfy
that requirement. No main-runtime build or model download is needed.

The [retained tiny receipt](prefill-tiny-control-2026-11-15.json) records the
original control's exit 1 and the separate matched control's exit 0, complete
cosine/norm/error metrics, source/payload hashes and resource context. The raw
original outputs/logs were preserved. The original F32/default-flash reference
fails at depth-two hidden relative L2 `0.0011727380403` and logits
`0.0014271802489`; the `0.001` gate was not widened.

Matching only native F16 K/V improves the default-flash depth-two hidden error
to `0.0009322081699`, while logits still fail at `0.0010402998637`. Against the
retained native **nonflash** output, the same rounded upstream reference gives
hidden/logit relative L2 `1.9252e-7`/`1.8579e-7` at depth one and
`3.3010e-7`/`3.4100e-7` at depth two; all four pass unchanged gates. The separate
first-block perturbation reports maximum error `0.25` and relative L2
`0.053240811895`, proving the comparator detects a changed boundary.

This pinned ggml's F16 key dot type is F32, so this result must not be explained
as query rounding. Its flash path uses F16 attention accumulation for F16 V;
upstream eager attention does not. The tiny flash/nonflash comparison isolates
that residual arithmetic difference without establishing any cause of #589's
real 7B output failure. Production defaults and native capture instrumentation
are unchanged.

The final upstream-only matched control held the nonblocking local heavy lock
through its child. It recorded free disk `17183596544` bytes strictly above
12 GiB, available memory `4673888` KiB above 2 GiB, and load1 `6.91845703125`
against an explicit maximum of 8. The earlier build guard did not record a load
or peak-memory check; no such claim is made for it. All work was single-process
or single-threaded CPU work on the tiny fixture, with no real model or GPU run.
