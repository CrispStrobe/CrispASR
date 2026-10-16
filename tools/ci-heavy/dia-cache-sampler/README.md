# Dia cache and sampling factorial

Private diagnostic controls, both disabled unless their value is exactly `1`:

- `CRISPASR_DIA_CROSS_CACHE_F32=1` retains projected cross-attention K/V in
  F32, reads/uploads the matching byte count and creates F32 graph inputs.
  Default remains F16; no rotary, attention or self-cache arithmetic changes.
- `CRISPASR_DIA_PORTABLE_RNG=1` uses the pinned PyTorch 2.6 MKL CPU
  exponential-race recipe. One contiguous heads × vocabulary batch consumes
  one MT19937 word per non-greedy decoder call, including masked entries.
  Default remains `std::discrete_distribution`. Current selected-index top-k,
  nucleus, CFG, stopping and crop behavior stay unchanged.

These controls reproduce only the pinned CPU recipe, not arbitrary PyTorch
versions/devices. Scalar libm and MKL vector logarithms can differ in final
bits. No performance/default/GPU/Q4 or broad speech acceptance is claimed.

`selftest.py --blueprint PINNED_ORIGINAL --output NEW_DIRECTORY` compiles the
actual production sampler/helper and executes the actual original Python
`_sample_next_token` AST. It verifies five tiny cases, 26 head decisions,
masked entries, dominant-token filtering, greedy bypass, full noise arrays
and original single-word batch consumption. Original code/hash, probabilities,
noise, tokens, native state and compile logs are retained.

`tools/ci-heavy/dia_cache_sampler.py` builds pristine composed source
fb4e5924 and the candidate. It first requires pristine/OFF/identity exact
inputs, logits, stage files and PCM. Holding the generic serial head fixed,
it runs all four F16/F32 × standard/portable configurations at the original
prompt, seed42, temperature1.2, top_p0.95, top_k45 and CFG3. Every arm retains
full native inputs/logits, actual code grids and native/original DAC outputs.
All factorial arms also run the real CLI with exact Session input/logit
checks. ASR is executed for every output and the original positive. It records
original trajectory prefix/exactness and six original step-zero stage metrics.

Complete payload receipts are checkpointed before quality assertions. The
F32/portable arm must pass unchanged stage and both-decoder/CLI WER-zero gates.
A completed failed matrix remains rejected evidence, never a successful port.
No forced inputs/output tokens or changed seed/quality threshold are allowed.
Full-array independent artifact audit and broader prompts/seeds must follow.

Local evidence under /mnt/storage/crispasr/dia-cache-sampler-20261016 is distinct
from heavy hosted validation. The default sampler body and default ggml calls
remain frozen against the pristine contract; the test explicitly admits only
the two F32 upload calls and selectable cross-input type for the optional path.
