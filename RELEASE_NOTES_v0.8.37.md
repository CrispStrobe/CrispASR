# CrispASR v0.8.37

Three new model arms — Raon-Speech-9B speech-to-text, moondream's
parakeet-ultra / parakeet-redux, and NVIDIA's Nemotron-3-Diarization with
streaming presets — native qwen3-tts on Vulkan, Whisper phrase scoring for
known-vocabulary recognition, and a text-to-speech fix that made letters
followed by digits ("d4", "B52") lose their number.

---

## New backends

### Raon-Speech-9B speech-to-text (#455)

The Qwen3-Omni audio tower + a 2-layer embedding adaptor + a Qwen3 36L/4096
LLM, on the qwen3-asr runtime as variant `raon-speech` (3ee6be51). The front
end reproduces `RaonPipeline.stt` (16k -> 24k, 8 s chunks, per-chunk mel,
13 Hz -> 12.5 Hz truncation); `core/torchaudio_resample.h` is a port of
torchaudio's sinc resampler, equal to it to 1 ULP. Every diff stage passes on
jfk (two chunks) and a Korean clip; F16/Q8_0 transcripts equal the remote-code
greedy reference (f65e70e2). CC-BY-NC-4.0: NC-gated registry entry (8a59d373).
GGUFs: `cstr/raon-speech-9b-GGUF`.

### parakeet-ultra and parakeet-redux (#454)

moondream's HF-transformers `ParakeetForTDT` checkpoints convert with
`convert-parakeet-to-gguf.py --hf` (931146ea); redux's base-3 packed ternary
encoder weights are expanded exactly. Transcripts equal transformers and
moondream Photon at every quant. Registry entries and README rows (83d3f3af,
23d3184a).

### Nemotron-3-Diarization / streaming Sortformer v3 (#466)

`--diarize-method sortformer` (alias `nemotron3-diar`), C-ABI method 5,
auto-downloading NVIDIA's q8_0 (OpenMDW-1.1) (a6a5bd79). F32/F16 probabilities
cos 1.000000 and every p>0.5 decision identical to transformers; AMI frame DER
29.54% for both (ed851dff). Streaming presets `--sortformer-mode
offline|low_latency|very_low_latency|ultra_low_latency` and a live session API
(`nemotron3_diar_stream_*`), with catch-up chunking for a caller that falls
behind (cccdfae1, ccc611c5). Flash attention is the CPU default, measured
exact in every mode (5c65abcb).

## Whisper: phrase scoring and strict grammars

For recognising one of a known set of phrases — the legal moves of a chess
position, spoken (7bf34f2b, 93d10a59):

- **`whisper_score_texts` / `crispasr_session_score_texts`** (Python
  `score_texts`, Dart `scoreTexts`): teacher-forced log P(text | audio) per
  candidate, with token counts for per-token comparison. The audio is encoded
  once and all candidates are decoded in ONE batch as a prefix tree (tree
  attention through sequence ids); `CRISPASR_WHISPER_SCORE_SEQUENTIAL=1` keeps
  the one-token-per-call path. Optional priming prompt.
- **`grammar_strict`** (`--grammar-strict`, `crispasr_session_set_grammar_strict`):
  no end-of-text until the grammar can be complete — the check upstream
  whisper.cpp ships commented out, without which a constrained decode stops
  mid-phrase ("knight to f" for "knight to f3"). Off by default.

Measured on 52 Piper TTS utterances of chess moves (EN + DE, whisper base):
per-token scoring picks the right move 81%, strict grammar decoding 62%, a
free transcript matched afterwards 23%. Use 1-2 threads: each candidate token
is a tiny decoder step, and on a busy 4-core CPU 4 threads took 345 ms per
step against 18 ms for 2.

## Text to speech

- **Letters followed by digits kept their digits** (35c1ee53). Every language's
  number speller skipped digits after a letter, and nothing else read them:
  Kokoro said "knight f3" as "knight f" and "d4" as silence. Such tokens are
  now split before number expansion in all five G2Ps (en, de, es, fr, ru);
  English matches misaki exactly ("d4" `dˈi fˈɔɹ`, "B52" `bˈi fˈɪfti tˈu`).
  German reads a lone letter by its name, as espeak-ng does ("d" `dˈeː`, was
  `t`).
- **qwen3-tts runs natively on Vulkan by default** (#337, 3048063c): both root
  causes fixed — the Vulkan flash-attention mask stride in the ggml fork
  (0f4269f6) and the F16 code-predictor down-projection overflow on every
  backend whose F16 GEMM narrows activations, CUDA included (ce6a436c).
  Verified on RX 7900 XT and gated in CI on lavapipe.
  `CRISPASR_QWEN3_TTS_VULKAN_CPU=1` restores the CPU pin.
- **voxcpm2**: CFG cond + uncond LocDiT forwards as one batch-2 graph — T4 cfm
  -38%, CPU -11% with identical WAV (6d7c1180, 31f18a6b).

## Fixes

- **Weights a GPU backend cannot bind stay on the CPU** (248fce7b, d3aa769f):
  a tensor past Vulkan's `maxStorageBufferRange` used to load and then abort
  in the scheduler; `load_weights` now routes it to a CPU buffer.
  `CRISPASR_GGUF_NO_FIT=1` turns it off. qwen3-asr keeps one KV tensor per
  layer so 128 MiB-range devices can bind it (f0484e4c).
- **crisp_audio** reflect-pads the STFT like torch.stft (911ca65a) and loads
  only the audio tower's tensors, not the host LLM a second time (14888f65).
- **Standalone punctuation API** loads every punctuation family and fails at
  init on a wrong GGUF instead of segfaulting later (#460, 30c45e25, bffba39c).
- **qwen3-tts** partial-frame codec mask fully initialised (25297a72).
- Reference dumpers: every capture owns its storage, and aliased captures are
  reported (64286eff).
