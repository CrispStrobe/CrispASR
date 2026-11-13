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

## Actual GGUF layout and CPU quantizer follow-up (2026-11-13)

The issue description, both comments, and the reporter's latest request to try
Q3/Q4 were reread. Haas is already documented and registered; the remaining
request concerns the distinct 30B speech model.

Bounded HTTP ranges inspected the actual Q8_0 decoder, AD-IQ4_NL decoder, and
F16 encoder at the AtomicChat revision above. Only 25 MiB of prefixes were
downloaded in this investigation; no full checkpoint or model inference ran.
The [layout results](historical/issue577-gguf-layout-20261113.json) and
`tools/inspect_nemotron_35_layout.py` make this a tensor-layout observation,
not an inference from the model configuration.

| Published file | Actual architecture | Actual tensor types |
|---|---|---|
| Q8_0 decoder | `nemotron_h_moe` | 173 Q8_0, 243 F32, 1 BF16 |
| AD-IQ4_NL decoder | `nemotron_h_moe` | 48 IQ4_NL, 125 Q8_0, 243 F32, 1 BF16 |
| F16 encoder | `pee` | 259 F16, 627 F32 |

All 173 quantized decoder matrices have rows divisible by 32. Only 31 have
rows divisible by 256, accounting for 344,752,128 of 32,904,044,544 matrix
elements (1.04775%). In particular, 24 up-expert banks have GGML dimensions
`[2688,1856,128]`, and 24 down-expert banks have `[1856,2688,128]`.
Those banks contain 30,651,973,632 elements (93.15564% of the quantized matrix
elements). Neither row width fits a 256-element K block. The IQ4_NL header
confirms that those 48 banks use the compatible 32-element format while the
other 125 quantized matrices retain Q8_0.

The pre-fix released CrispASR quantizer excludes these 3D expert banks
from its normal 2D weight selection. A 516,320-byte synthetic F32 GGUF with
one `[2688,16,2]` expert bank and one `[2688,16]` shared-expert matrix was run
through the released CPU quantizer with both `q3_k` and `q4_k`. Both outputs
copy the 3D bank as F32 and convert the 2D matrix to **Q4_0**, the existing
row-width fallback. The output tensor descriptors were read back independently;
these are execution controls, not speech-quality measurements. The quantizer
source matches main `bf4b88ce1`.

A block-32 four-bit representation is therefore structurally feasible and
already present in the published IQ4_NL file. In that pre-fix release, a plain `q3_k`/`q4_k` invocation
does not establish the requested conversion: it skips the dominant expert
banks, and a `q3_k` fallback may actually be four-bit. No new quantization
recipe or backend change is justified by header inspection alone.

The concrete remaining implementation work is the `pee` encoder/speaker/CTC
pipeline and its coupling to the hybrid-MoE decoder, audio/text prompting and
decoding. The imported text-model implementation is not that speech pipeline,
and the existing 0.6B streaming backend cannot substitute for it. Once a
runnable complete baseline exists, remote CPU recognition/translation tests
can measure four-bit quality; Q3 additionally needs an implemented compatible
representation. No new GPU or CI job was dispatched for this follow-up.

The diagnostic reuses the existing GGUF header parser. Its parser self-tests
and two model-free layout tests pass; the layout tests distinguish row-width
compatibility from total-element divisibility and count all expert dimensions.

## Bounded quantizer implementation follow-up (2026-11-13)

The quantizer now recognizes `nemotron_h_moe` 3D `*_exps.weight` banks as
row-wise matrices. It preserves the expert dimension and processes all rows
through the existing quantization writer. This selection change is limited to
that architecture and tensor naming convention; convolution tensors retain
their existing rules. The writer buffers an entire tensor, including F32
conversion and quantized output buffers, so a large expert bank can require
substantial temporary memory.

Rows still use the existing compatible-type fallback: 2688- and 1856-wide
banks requested as Q3_K or Q4_K become **Q4_0**, not three-bit or K-quant
storage. Aggregate warnings report the actual fallback type and affected
tensor/element counts separately for the global target and explicit tensor-type
overrides. Unsupported expert dimensions or row widths produce a
separate copy warning, while explicit overrides and architecture precision
floors retain their authority. The final summary reports changed tensor types
and quantized output storage, including idempotent and floating-point casts.

Model-free controls compare every output byte of synthetic 3D banks with
identical flattened 2D matrices for both real row widths and a 256-wide control,
including Q8-source requantization. This establishes storage behavior only.
The full 30B speech pipeline, large-bank memory use and quantized recognition
or translation quality remain unvalidated.
