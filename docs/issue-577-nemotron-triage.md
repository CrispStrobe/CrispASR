# Issue #577: Haas and Nemotron 3.5 Transcribe

Status as of 2026-11-06, checked against the code at `f6b91a04e` and the
[issue and maintainer follow-up](https://github.com/CrispStrobe/CrispASR/issues/577).

Haas-ja-v1.0 already dispatches through the `haas` alias to the Parakeet
backend. The model registry provides F16, Q8 and Q4 auto-download entries;
`tests/test_haas_dispatch.py` and `tests/test-registry.cpp` cover dispatch and
registration. It shipped in v0.8.50. See the [Haas usage guide](haas-ja.md)
for commands, model variants and the scope of the dispatch tests.

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

## Resource and quantization feasibility (2026-11-07)

The [NVIDIA checkpoint inventory](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B/tree/c17667641642c4265ff2b7037dd101b7d4567634)
lists a 67,496,013,734-byte safetensors file. This is a storage requirement,
not an inference-memory measurement. NVIDIA's published BF16 example targets
an 80 GB H100; actual memory depends on the runtime and workload.

The separate [AtomicChat GGUF inventory](https://huggingface.co/AtomicChat/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-GGUF/tree/dbd06c5cad7a2f33c8e06be2fb3ef814dbd9aa59)
lists a 1,670,410,080-byte F16 encoder plus distinct decoder files, including
IQ4_NL (19,678,656,672 bytes), Q8_0 (35,004,643,232 bytes) and Q2_0_M
(13,572,208,800 bytes). Encoder plus Q2_0_M is about 15.24 GB on disk before
runtime buffers, caches and operating-system overhead. IQ4_NL is a four-bit
alternative, not a Q4_K measurement; the inventory does not list Q3. The
publisher's accuracy and hardware statements are their results for their own
runtime and datasets, not CrispASR measurements.

AtomicChat reports a 2,200-utterance comparison against its BF16 decoder: for
its published decoders of 15.55 GB and larger, English, German, French and
Spanish word error differed by at most 0.18 percentage points on its tested
sets. Its smaller Q2_0_M variant showed a 1.53-point Chinese character-error
increase, with a reported paired-bootstrap 95% interval of 0.85–2.22 points.
The publisher measured Q2_0_M peak memory at 15.6 GiB on an A100; it explicitly
did not measure this on a 16 GB card. These are third-party NeMo-Speech.cpp
measurements, not Q3 results or CrispASR qualification.

The [pinned NVIDIA configuration](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B/blob/c17667641642c4265ff2b7037dd101b7d4567634/config.json)
specifies a 2,688-wide language-model hidden state and 1,856-wide expert
intermediate state; neither is divisible by 256. That rules out naively applying
256-element block quantization along those dimensions. A converter would need
to inspect each actual tensor layout and choose a compatible representation;
the config alone does not establish that a complete Q3 or Q4 conversion works.

The current maintainer host cannot load the full model locally. A useful next
experiment needs a suitably sized remote machine: pin the upstream model and a
fixed multilingual ASR/translation corpus, capture reference transcripts and
stage outputs, then compare a runnable quantized baseline with matched decode
settings. Record peak memory, latency, WER/CER and translation quality, and
compare Q3 only if a valid Q3 artifact can actually be constructed. Do not
claim support from file availability or extrapolate quality from bit width.
