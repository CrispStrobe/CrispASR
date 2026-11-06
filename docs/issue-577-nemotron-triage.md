# Issue #577: Haas and Nemotron 3.5 Transcribe

Status as of 2026-11-06, checked against the code at `f6b91a04e` and the
[issue and maintainer follow-up](https://github.com/CrispStrobe/CrispASR/issues/577).

Haas-ja-v1.0 already dispatches through the `haas` alias to the Parakeet
backend. The model registry provides F16, Q8 and Q4 auto-download entries;
`tests/test_haas_dispatch.py` and `tests/test-registry.cpp` cover dispatch and
registration. It shipped in v0.8.50. The README model table now lists it.

The requested NVIDIA Nemotron-3.5-Transcribe-30B-A4B is a separate backend
project, not a new quantization of the existing 0.6B streaming Nemotron
implementation. The [NVIDIA model card](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B)
describes a speech encoder, speaker encoder, connector, timestamp head and
hybrid Mamba2/Transformer mixture-of-experts language model. The repository has
an imported Nemotron-H MoE language model implementation, but it is not wired
up to this speech model's complete inference path.

An [independent GGUF conversion](https://huggingface.co/AtomicChat/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-GGUF)
targets NeMo-Speech.cpp and provides a separate encoder and block-32 quantized
decoders. Its authors report dimensions that do not fit the normal 256-value
K-quant blocks. Its advertised files are therefore not evidence that the
requested Q3 and Q4 variants exist or work in CrispASR. Other quantization
schemes, or a mixed representation, need conversion and quality validation.

A viable implementation would first establish reference transcripts and
intermediate tensors from the upstream model, then validate conversion and
each encoder, connector and decoder stage against those references, with
bounded CPU and GPU tests, language/timestamp coverage and end-to-end speech
quality. Quantized variants need independent accuracy and memory measurements.
This work and the large model downloads require suitable remote resources; no
claim of 30B support or of Q3/Q4 quality follows from the Haas support.
