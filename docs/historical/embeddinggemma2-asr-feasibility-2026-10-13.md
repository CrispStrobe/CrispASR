# EmbeddingGemma 2 speech recognition feasibility

Checked 2026-10-13. Technically feasible; no CrispASR backend is implemented.

Google's [model card](https://ai.google.dev/gemma/docs/embeddinggemma/model_card_2)
provides Apache-2.0 weights, modular encoders and a shared 768-dimensional space.
Upstream [llama.cpp support](https://github.com/ggml-org/llama.cpp/pull/30054) and
[official GGUFs](https://huggingface.co/ggml-org/embeddinggemma-2-GGUF) already exist.
Our local CrispEmbed tree has a validated EG2-derived typed-decision text path;
that does not establish multimodal audio or speech-reader support.

The [author's app](https://eg2.haku.sh/) describes a frozen audio encoder, trained
reader, English word-bank retrieval and punctuation. Its reported 288x speed
uses Apple's Neural Engine. Public checks located a free Mac app download, but
no reader checkpoint, word-bank artifact or complete inference blueprint.
Reader architecture/weights and their distribution license still need verification.
The 87M reader figure is from the user-supplied author post.

A port should preserve the exact temporal features and decoding recipe, execute
learned inference and batched word scoring through ggml, and integrate the C ABI,
CLI, registry, converter, bindings and reference backend per contributing.md.
Build canonical intermediate references, check cosine **and magnitude**, then
actual transcripts/WER. Validate FP32/Q8 before Q4_K and preserve sensitive tensors.
Google [documents](https://ai.google.dev/gemma/docs/embeddinggemma/multimodal-embeddinggemma-with-sentence-transformers)
incompatible FP16 activations: retain BF16/F32 computation where required.
Benchmark actual ggml CPU/Metal/CUDA only after correctness; the app's Neural
Engine figures do not predict ggml performance. If the reader stays unavailable,
training a new reader is a separate project with its own data and accuracy gates.
