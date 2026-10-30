# OmniVoice Metal encoder default promotion — 2026-10-23

The maintainer listened to all seven WAVs in the local listening folder and
approved the original and corrected outputs, accepting minor audible differences.
The corrected precise path is now the default for eligible Apple Silicon macOS
Metal contexts. It preserves CPU codes/audio in the measured suite and retains
independent switches for every old path.
[Raw timings and validation](omnivoice-metal-defaults-2026-10-23.json) supplement
the [precision investigation](omnivoice-semantic-precision-2026-10-23.md).

## Policy and rollback

Eligibility requires GPU use, an actual `MTL*` backend on Apple ARM macOS, a loaded
Higgs profile with768-dimensional/12-layer semantics,1024-dimensional hidden
state, eight1024-entry/64-dimensional quantizers, F16 FC/feature-projection
weights, and exclusively F16/F32 tokenizer tensors. Other devices, iOS,
quantized tokenizer tensors and unmatched profiles retain their prior policy.
Explicit full-GPU codec placement is respected.

For this profile, the unset configuration enables HYBRID, SEMANTIC_F32, GRAPH,
FUSED and pointwise FASTCONV; fixed-data caching stays on. Acoustic encoding
remains CPU, semantic encoding and quantization use Metal. These variables all
start with `CRISPASR_OMNIVOICE_ENCODER_`.

- `HYBRID=0` restores the prior CPU encoder, including its cache/pointwise defaults.
- `GRAPH=0` retains scalar FC/RVQ; `FUSED=0` retains staged HuBERT.
- `SEMANTIC_F32=0` retains the faster reduced-precision numerical control.
- `FASTCONV=0` disables the pointwise rewrite; `2` selects full baking.
- `CACHE=0` disables immutable host-data caching.
- `--no-gpu` prevents GPU encoding. Explicit switches still override defaults.

GPU allocation/compute failures reported by the semantic path retry the retained
CPU encoder, release the semantic GPU copy, and stay on CPU until tokenizer
replacement. A failed GPU attempt does not publish a reference-cache entry.
Successful reference-cache hits skip the GPU copy entirely, including across
fresh contexts. Decoder placement, licensing, consent and marking policies are
unchanged.

## Performance and memory

Same pinned F16 tokenizer/Q8 main and M1 host as the precision report. Eight-call
reversed/interleaved trials exclude the first two warmups; each median has three
samples per arm. Disk reference caching is off.

| Reference | Prior shipping CPU → unset defaults | Precise hybrid → graph/fused combined |
|---|---:|---:|
| window |1.649→0.997s|1.048→0.910s|
| station |1.445→0.876s|1.208→1.025s|
| fox |1.651→0.876s|1.097→0.924s|

All48 reference-code arrays match the independently frozen CPU encoder. The
new defaults reduce fresh encoding wall time39–47%; graph/fusion adds13–16%
versus precise hybrid alone. This is reference-encoding performance, not a
whole-generation speedup.

Three fresh-process station runs per arm also include lazy GPU setup. Median
first encoding is1.898→1.621s; median total native process wall2.16→1.98s.
The native probe includes tokenizer loading and backend initialization, but no
LLM or speech generation. OS file/shader caches were warm: these are fresh
processes, not a cold-disk claim. All individual timings are retained.

Median native-process peak RSS is0.844→1.067GB, about223MB extra, with substantial
GPU-accounting variation retained in the receipt. This excludes the full LLM
workload and is not a complete unified-memory budget. The filtered GPU weights
alone occupy223.8MB; pointwise/projection/positional caches and graph scratch
are additional. `HYBRID=0` avoids that semantic GPU copy.

## Validation and reproduction

The default-versus-explicit-CPU Session run renders16 outputs across the same
seven synthetic references and eight target/reference cases. Every candidate
has byte-identical generated codes and PCM to its CPU control, and all controls
match the previous scored outputs by SHA256. Consequently their frozen RNNT/CTC
judgments carry over: seven cases WER0; Anna's baseline English target omits
“home” (1/12) on every path. Existing broader fox/Linux failures remain open.
A final linked Session check and ordinarily marked CLI verify the final profile
guards; the CLI target codes match the baseline while retaining disclosure/C2PA.

Five focused CTest cases and ASan/UBSan pass. Policy tests cover actual backend
selection, explicit disabling, GPU permission, failed contexts, unmatched shapes
and quantized tokenizer tensors. The native lifecycle probe injects failures at
fused HuBERT, semantic bridge and graph quantization, requiring exact CPU recovery
and sticky fallback; it also verifies fresh-context cache hits and no-GPU use.
The original wrong-tie negative control still fails as expected.

Compile the probes using the encoder report's library flags. The benchmark adds
`promoted` (unset candidate versus prior CPU), `combined` (precise hybrid versus
combined), and one-call `first-cpu`/`first-default` modes. The latter pair can be
wrapped in `/usr/bin/time -l` to include process wall and peak RSS. Use three
fresh processes per arm and alternate order; retain initialization separately
from encoder time.

`tools/ci-heavy/omnivoice_encoder_lifecycle.cpp` takes tokenizer, reference WAV
and a scratch-cache parent directory. Add `-Iggml/src` when compiling it. It
creates its own cache subdirectory and verifies the real runtime; no weight or
output mock substitutes for encoding. All public receipts use portable aliases;
local listening files and synthetic system-voice fixtures remain local.
