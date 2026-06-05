---
license: apache-2.0
language:
- zh
- en
- ja
- ko
- yue
- fr
- de
- es
- pt
- it
- ru
base_model:
- FunAudioLLM/Fun-CosyVoice3-0.5B-2512
tags:
- tts
- text-to-speech
- gguf
- crispasr
- cosyvoice3
- multilingual
- voice-cloning
- zero-shot
- 24khz
---

# Fun-CosyVoice3-0.5B-2512 — GGUF

GGUF conversion of [FunAudioLLM/Fun-CosyVoice3-0.5B-2512](https://huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B-2512)
for use with [CrispASR](https://github.com/CrispStrobe/CrispASR)
(`--backend cosyvoice3-tts`).

CosyVoice3 is a streaming, multilingual, zero-shot voice-cloning TTS
system from Alibaba's FunAudioLLM team. The 0.5B-2512 release is
Apache-2.0 licensed and supports **9 languages plus 18 Chinese
dialects**. Output is 24 kHz mono.

The model is a three-stage pipeline:

```
text (Qwen2 BPE) → CosyVoice3LM (Qwen2-0.5B + speech-token AR head)
                 → speech tokens ∈ [0, 6561)
                 → Flow (DiT + CausalConditionalCFM, 10-step Euler ODE)
                 → mel @ 24 kHz / 480-hop
                 → CausalHiFTGenerator (HiFi-GAN + NSF + iSTFT)
                 → 24 kHz PCM
```

## Files

| File | Quantisation | Size |
|---|---|---|
| `cosyvoice3-llm-f16.gguf` | F16 | 1.29 GB |
| `cosyvoice3-llm-q4_k.gguf` | Q4_K (Q4_0 fallback on 896-wide rows; head + embeddings stay F16) | 384 MB |
| `cosyvoice3-flow-f16.gguf` | F16 | 665 MB |
| `cosyvoice3-flow-q8_0.gguf` | Q8_0 (input_embd + spk_affine stay F16) | 361 MB |
| `cosyvoice3-hift-f16.gguf` | F16 — too small to benefit from quant | 42 MB |
| `cosyvoice3-voices.gguf` | F32 voice-clone bank: 8 baked voices (zero_shot + en/de/zh/ja/fr/es/ko) | 665 KB |
| `cosyvoice3-s3tok-f16.gguf` | F16 speech_tokenizer_v3 — **byte-exact vs ONNX** | 462 MB |
| `cosyvoice3-s3tok-q4_k.gguf` | Q4_K s3tok (FSQ proj stays F16); ~0.6% token drift — optional smaller variant | 139 MB |
| `cosyvoice3-campplus-f16.gguf` | F16 CAMPPlus 192-D speaker encoder | 13 MB |

Pick **one LLM + one flow + HiFT + voices**. The smallest viable
combo is `llm-q4_k + flow-q8_0 + hift-f16 + voices` at **745 MB
total**; the F16 reference is 1.96 GB. The s3tok + campplus companions
are only needed for **arbitrary-WAV runtime cloning** (below) — not for
synthesis with a baked voice.

## Quant validation (ASR roundtrip on smoke prompt)

Synthesis used the default zero-shot voice (upstream
`asset/zero_shot_prompt.wav`) at `--temperature 0.8 --seed 42`. The
generated WAV was transcribed with `parakeet-tdt-0.6b-v3-q4_k` and
compared against the prompt text.

| Combo | Synthesis size | ASR transcript of TTS output | WER |
|---|---|---|---|
| `llm-f16  + flow-f16 ` | 1.96 GB | "Hello, this is a test." | 0% |
| `llm-f16  + flow-q8_0` | 1.66 GB | "Hello, this is a test." | 0% |
| `llm-q4_k + flow-f16 ` | 1.05 GB | "Hello? This is a test." | 0% (punct only) |
| `llm-q4_k + flow-q8_0` | 745 MB  | "Hello? This is a test." | 0% (punct only) |
| `llm-q4_k + flow-q8_0` (German) | — | "Hallo? Das ist ein Test." | 0% (punct only) |

Q4_K LLM introduces a small punctuation drift (commas occasionally
read as question-intonation) but content is fully preserved across
languages. Q8_0 flow is perceptually indistinguishable from F16.

## Usage

### CrispASR (recommended)

```bash
# Auto-discovers flow + hift + voices as siblings of the LLM.
crispasr -m cosyvoice3-llm-q4_k.gguf \
         --backend cosyvoice3-tts \
         --tts "Hello, this is a test." \
         --voice zero_shot \
         --tts-output out.wav
```

The CLI auto-discovers companion GGUFs in this order:

* **Flow** — `cosyvoice3-flow-*.gguf` next to the LLM, or `--codec-model PATH`.
* **CAMPPlus** — `cosyvoice3-campplus-f16.gguf` next to the LLM, or `COSYVOICE3_CAMPPLUS_PATH` for the native WAV-clone path.
* **S3Tokenizer** — `cosyvoice3-s3tok-f16.gguf` next to the LLM, or `COSYVOICE3_S3TOK_PATH` for the native WAV-clone path.
* **HiFT** — `cosyvoice3-hift-*.gguf` next to the LLM, or `COSYVOICE3_HIFT_PATH` env var.
* **Voices** — `cosyvoice3-voices.gguf` next to the LLM, or `COSYVOICE3_VOICES_PATH` env var.

Greedy decode is disabled by default (CV3 falls into a documented
"silent_tokens" loop within ~5 steps). The backend overrides
`--temperature 0` to 0.8 so the RAS sampler engages — pass a different
positive value to override.

### Voices

`cosyvoice3-voices.gguf` ships a small multilingual voice bank — pass
the name to `--voice`:

| `--voice` | Language | Prompt source |
|---|---|---|
| `zero_shot` | Mandarin | upstream `asset/zero_shot_prompt.wav` (~3.5 s) |
| `fleurs-en` | English | FLEURS en (CC BY 4.0) |
| `fleurs-de` | German | FLEURS de (CC BY 4.0) |
| `fleurs-zh` | Mandarin | FLEURS zh (CC BY 4.0) |
| `fleurs-ja` | Japanese | FLEURS ja (CC BY 4.0) |
| `fleurs-fr` | French | FLEURS fr (CC BY 4.0) |
| `fleurs-es` | Spanish | FLEURS es (CC BY 4.0) |
| `fleurs-ko` | Korean | FLEURS ko (CC BY 4.0) |

The `fleurs-*` prompts are ~4–6 s clips from Google's
[FLEURS](https://huggingface.co/datasets/google/fleurs) corpus
(CC BY 4.0), loudness-normalised before baking. CV3 clones the prompt's
timbre *and* level, so quiet prompts yield quiet output — normalise your
own prompt clips for a consistent level. More voices can be baked with
the converter in the CrispASR tree:

```bash
python models/convert-cosyvoice3-voices-to-gguf.py \
    --manifest my-voices.json \
    --upstream-base /path/to/CosyVoice-clone \
    --output my-voices.gguf
```

Each manifest entry is `{name, wav, prompt_text}`. The script needs
`campplus.onnx` (CV2/CV3 speaker encoder) and
`speech_tokenizer_v3.onnx` (CV3 token extractor); both auto-download
from HF on first run.

### Arbitrary-WAV cloning (native, no Python pre-bake)

With the `cosyvoice3-s3tok-f16.gguf` + `cosyvoice3-campplus-f16.gguf`
companions present (siblings of the LLM, or pulled by `-m auto`), you
can clone from any 16 kHz WAV at runtime:

```bash
crispasr -m cosyvoice3-llm-q4_k.gguf \
         --backend cosyvoice3-tts \
         --voice my_reference.wav \
         --ref-text "exact transcription of my_reference.wav" \
         --tts "The text to speak in the cloned voice." \
         --tts-output out.wav
```

The runtime ports all three front-end extractors to ggml: the
**speech_tokenizer_v3** token extractor (12 FSMN/attention blocks +
FSQ head — byte-exact vs the ONNX reference, validated stage-by-stage
with `crispasr-diff`), the **CAMPPlus** 192-D speaker encoder, and the
**matcha 24 kHz** reference mel. The legacy Python pre-bake bridge
(`convert-cosyvoice3-voices-to-gguf.py`) remains as an automatic
fallback when the companions are absent.

## Tensor naming

Conventional naming for all three GGUFs:

* **LLM** — llama.cpp-standard `token_embd`, `blk.K.{attn,ffn}_*`,
  `output_norm`, `output`, plus CV3-specific
  `cosyvoice3.speech_embd.weight` (input embedding, vocab 6761) and
  `cosyvoice3.speech_lm_head.weight` (output head).
* **Flow** — `cosyvoice3.flow.{input_embd,pre_la,spk_affine,dit.*}`
  matching the upstream `CausalMaskedDiffWithDiT` module tree.
* **HiFT** — `cosyvoice3.hift.{conv_pre,ups.K,resblocks.K.*,source_*,
  m_source,f0.*,conv_post}` with weight-norm pre-resolved on the
  Python converter side (g · v / ‖v‖).

## License

The model weights are **Apache-2.0** (inherited from the upstream
model). Free for commercial use. The `zero_shot` voice prompt is the
`asset/zero_shot_prompt.wav` clip from the Apache-2.0 CosyVoice repo.

The `fleurs-{en,de,zh,ja,fr,es,ko}` voice prompts are derived (trimmed +
loudness-normalised) from Google's **FLEURS** corpus, licensed
**[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)** —
commercial use permitted, attribution required:

> FLEURS (Few-shot Learning Evaluation of Universal Representations of
> Speech), Conneau et al., 2022 — <https://huggingface.co/datasets/google/fleurs>,
> licensed CC BY 4.0. The prompt clips here are trimmed excerpts,
> loudness-normalised; no other modification.

All eight baked voices are therefore clean for commercial use under
permissive licenses (Apache-2.0 / CC BY 4.0).

## Related links

* Upstream: [FunAudioLLM/Fun-CosyVoice3-0.5B-2512](https://huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B-2512)
* Project page: [funaudiollm.github.io/cosyvoice3](https://funaudiollm.github.io/cosyvoice3/)
* Code: [github.com/FunAudioLLM/CosyVoice](https://github.com/FunAudioLLM/CosyVoice)
* CrispASR: [github.com/CrispStrobe/CrispASR](https://github.com/CrispStrobe/CrispASR)
