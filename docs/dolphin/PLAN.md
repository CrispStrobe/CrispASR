# §436 — Dolphin (DataoceanAI) ASR family

## NOW — active work

Branch `feat/436-dolphin`. Target: `DataoceanAI1/dolphin-cn-dialect-small-streaming`
(asked for in #436), with the runtime written for the whole Dolphin family
(base / small / cn / cn.streaming / cn.prompt share one architecture).
X-ASR (the other half of #436, an icefall Zipformer2 transducer) is a separate
encoder family and follows Dolphin.

## Blueprint (read from DataoceanAI/Dolphin @ 78ea615, not the card)

`dolphin/transcribe.py::transcribe` → `ASRModel.decode(methods=["attention_rescoring"],
beam_size=10)` with the defaults of `decode()`:

| knob | value | where |
| --- | --- | --- |
| features | Kaldi fbank, 80 bins, 25/10 ms, dither only in training | `train.yaml` fbank_conf |
| normalisation | global CMVN from `global_cmvn` (JSON) | cmvn_conf |
| encoder | E-Branchformer, 12 blocks, d=768, 12 heads, cgMLP 3072 (kernel 31), merge conv kernel 31, rel-pos self-attention, conv2d subsampling, **causal**, dynamic chunk trained | encoder_conf |
| inference chunking | `decoding_chunk_size=-1` → full context even for the streaming checkpoint | `decode()` default |
| CTC | vocab 18173, blank 0 | ctc_conf |
| decoder | Transformer, 12 blocks, 12 heads, 3072 FFN | decoder_conf |
| search | CTC prefix beam (10) → attention rescoring, `ctc_weight=0.0`, `reverse_weight=0.0` | `decode()` defaults |
| prompt | `<sos>` + language token + region token (two-level, e.g. `<zh>` `<CN>`) | tokenizer / transcribe |
| hotwords | optional deep biasing (`context_module: cppn`, 2 layers) — out of scope for v1 | hotword.py |

Checkpoint SHA-256 is pinned in `dolphin/model_registry.py`
(small.cn.streaming: `bba8688e…`) — verify before converting (see LEARNINGS:
a CIFS download silently corrupted a checkpoint on 2026-09-23).

## Steps

1. Read `model.py` E-Branchformer / rel-pos attention / cgMLP line by line.
2. Converter (`models/convert-dolphin-to-gguf.py`), CMVN + units baked in.
3. Reference dump backend (`tools/reference_backends/dolphin.py`) — Kaggle CPU.
4. Runtime + diff harness arm; parity per stage; then decoded-output check.
5. Registry, quantizer rules, CLI/C-ABI wiring (12-point checklist).
