# CSM original versus native CUDA pilot

This private Kaggle adapter pairs the official Sesame CSM model/generator
(`SesameAILabs/csm` at `836f8865`) with the v0.8.47 native CUDA binary and
qualified Q4_K GGUF. It runs one warmup, then three alternating calls per
engine on the same prompt, speaker 0, no reference context, seed 1234,
temperature 0.9 and top-k 50. The native CLI has no 10-second cap, so the run
rejects either output if it reaches 10 seconds; both must terminate by EOS
before that difference can affect the comparison. The same seed value does not
make PyTorch's multinomial RNG and the native sampler produce identical codes;
speech quality is checked separately for each output.

The official loader chooses BF16. This pilot loads its unchanged Model and
Generator at F32 because T4/P100 lack BF16 arithmetic. The native GGUF is Q4_K.
The official generator includes Mimi decoding and SilentCipher watermarking;
this pilot explicitly replaces its watermark call with an identity function
and loads no SilentCipher checkpoint, because that checkpoint's use rights
are not established. This is a **Sesame Model/Generator comparison with its
watermark disabled**, and must always be labeled that way. Process timings
include model load and are comparable only as
whole calls. Native existing `CRISPASR_CSM_BENCH` counters and original
generation time are reported separately; no cross-engine per-step speedup is
inferred from them.

Sources and terms reviewed for this pilot: Sesame source and CSM model declare
Apache-2.0; the pinned Llama-3.2 tokenizer is gated under its model license;
Mimi declares CC-BY-4.0; SilentCipher source declares MIT. No SilentCipher
weights are downloaded. The actual account must already have accepted the
gated CSM and Llama terms; the worker fails if access is absent. All permitted
weights stay in private scratch and are never redistributed. A separate terms
review is required before publishing anything beyond aggregate results.

Each output is judged with the same pinned Parakeet GGUF and a WER ≤ 0.15
against the public phrase. Generated WAVs and ASR logs are copied into the
**private** Kaggle output for evidence recovery. Only `receipt.json` may be
published after review; never publish `private-evidence/` or the complete
Kaggle output. No run has been submitted from this directory yet.

Before submission, refresh Kaggle GPU slots and resource admission. If the
original source dependency set cannot import on the available Kaggle image,
record the failure and revise the pinned environment; do not replace the
original generator with an unlabelled Transformers pipeline.
