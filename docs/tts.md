# Text-to-Speech (TTS)

CrispASR ships **62 registered open-weights TTS backends** — roughly three
dozen distinct engine families once variants are folded; the authoritative
auto-generated list is [`docs/feature-matrix.md`](feature-matrix.md) — behind
the same `crispasr` binary, each with a distinct voice / quality / footprint
trade-off:

## Contents

- [dots.tts](#dotstts--voice-cloning-and-performance) — voice cloning, mixed quant, flow-match knobs
- [TADA](#tada--multilingual-and-voice-cloning) — built-in per-language voices, baking a custom voice
  - [Switching voice at query time (server, #201)](#switching-voice-at-query-time-server-201)
  - [Timing quality (`CRISPASR_TADA_NUM_CANDIDATES`)](#timing-quality-crispasr_tada_num_candidates)
- [CosyVoice3](#cosyvoice3--voice-cloning-from-a-wav) — WAV cloning, `--ref-text`, base vs RL talker
- [Output language and cross-lingual cloning](#output-language-and-cross-lingual-cloning--tl---sl) — `-tl` / `-sl`, what each backend does
- [G2P Phonemization](#g2p-phonemization---g2p-dict) — `--g2p-dict`, number expansion, phoneme dialects
  - [Driving the phonemes directly (`--tts-phonemes`)](#driving-the-phonemes-directly---tts-phonemes)
- [Kokoro](#kokoro--multilingual-smallest) — multilingual, smallest
- [Qwen3-TTS](#qwen3-tts--voice-cloning-highest-fidelity) — voice cloning, highest fidelity
  - [qwen3-tts environment switches](#qwen3-tts-environment-switches)
  - [pocket-tts languages, voices, and environment switches](#pocket-tts-languages-voices-and-environment-switches)
- [VibeVoice](#vibevoice--realtime-streaming-tts) — realtime streaming TTS
- [VibeVoice 1.5B](#vibevoice-15b--base-tts-with-wav-cloning) — base TTS with WAV cloning
- [Orpheus](#orpheus--llama-32-3b--snac-codec) — Llama-3.2-3B + SNAC codec
- [Chatterbox](#chatterbox--flow-matching-tts-voice-cloning--multilingual) — flow-matching TTS, multilingual
  - [Multilingual language selection](#multilingual-language-selection)
  - [Voice cloning](#voice-cloning)
- [Parler TTS](#parler-tts--prompt-conditioned-voice-description) — prompt-conditioned voice description
- [TTS GGUF downloads](#tts-gguf-downloads)
- [F5-TTS](#f5-tts--dit-flow-matching-voice-cloning) — DiT flow-matching voice cloning
  - [Performance (issue #294)](#performance-issue-294)
- [IndexTTS](#indextts--chineseenglish-voice-cloning) — Chinese/English voice cloning
  - [Chinese text normalization](#indextts-chinese-text-normalization)
- [Irodori-TTS](#irodori-tts--japanese-voice-cloning--emoji-emotion-control) — Japanese, emoji emotion control
- [Reference-conditioning cache](#reference-conditioning-cache)
- [Local speaker output (`--tts-play`)](#local-speaker-output---tts-play)
- [AI-generated audio provenance & watermarking](#ai-generated-audio-provenance--watermarking)
  - [Disabling the watermark](#disabling-the-watermark-operator-opt-out)
  - [Voice cloning consent gate](#voice-cloning-consent-gate)
  - [Spoken disclaimer (voice clones only)](#spoken-disclaimer-voice-clones-only)

| Backend | Why pick it | Voice cloning | First-run download |
|---|---|---|---|
| **`melotts`** | Multilingual VITS2 (MeloTTS). 4 English speakers (US/BR/India/AU). 44.1 kHz output, ~102 MB GGUF. Neural G2P + CMU dict. BERT companion (Q4_K 52 MB) auto-downloads with `-m auto`; also via `--codec-model` or `CRISPASR_MELOTTS_BERT` env. | No (per-speaker ID) | ~154 MB via `-m auto` |
| [`piper`](#piper--community-voices) | Tiniest footprint (30 MB). rhasspy/piper VITS; 250+ community voices across 30+ languages. Built-in G2P (CMUdict + LTS rules) for English — no espeak-ng needed. Optional espeak-ng for other langs (loaded via dlopen). 22 kHz output. Use `--g2p-dict` to select dictionary source. | No (per-voice GGUF) | ~30 MB default via `-m auto`; other voices are manual downloads |
| [`kokoro`](#kokoro--multilingual-smallest) | Smallest + fastest. 82 M-param StyleTTS2-derived model. Multilingual via built-in G2P or espeak-ng (dlopen/popen fallback). | No (preset voice packs) | Manual `wget` (no `-m auto`) |
| [`qwen3-tts`](#qwen3-tts--voice-cloning-highest-fidelity) | Highest fidelity / strongest cloning. Speech-LLM (talker + code predictor + 12 Hz codec). Default voice auto-downloaded with `-m auto`; or supply your own WAV + ref-text. | Optional (auto default voice; or WAV + ref-text or baked voice GGUF) | ~1.3 GB via `-m auto` |
| **`miotts`** | MioTTS-0.6B (Qwen3 LLM + MioCodec-v2). EN/JA. Single GGUF, 24 kHz output. Codec-aware mixed quantization (LLM Q4_K + codec F16). | Yes — `--voice preset.emb.gguf` (preset speaker embeddings) | 502 MB Q4_K via `-m auto` |
| **`moss-tts`** | MOSS-TTS-v1.5 (MossTTSDelay): Qwen3-8B backbone emitting 32 RVQ audio codebooks under a delay pattern, decoded by a 1.6B transformer codec. Needs the codec companion (`--codec-model`, or auto-downloaded/sibling). | Yes — `--voice ref.wav` (the codec encoder clones the reference speaker) | ~5 GB Q4_K backbone + ~3.5 GB F16 codec via `-m auto` |
| **`moss-tts-local`** | MOSS-TTS-Local-Transformer-v1.5 (MossTTSLocal, 4B): Qwen3-4B backbone + a 1-layer local/depth transformer that autoregressively emits 12 RVQ codebooks per frame (RQ-Transformer; no delay pattern), decoded to 48 kHz stereo by MOSS-Audio-Tokenizer-v2 (downmixed to mono). Needs the codec companion (`--codec-model`, or auto-downloaded/sibling). | `--voice ref.wav --i-have-rights` (needs the encoder-carrying codec, `--codec-model moss-tts-local-v1.5-codec-enc.gguf`; the plain `-codec.gguf` is decode-only and falls back to the default voice) | ~9.1 GB F16 backbone + ~2.1 GB codec via `-m auto` (F16 is the reliable target; Q4_K long-form runs away) |
| **`omnivoice`** | 600+ languages, selected with `-l` / `-tl` (or `"language"` on `/v1/audio/speech`) — see the language note below. Qwen3-0.6B backbone with masked iterative 8-codebook TTS (SoundStorm-style). Zero-shot voice cloning from reference audio. Supports finetunes (omnivoice-singing). | Yes (`--voice <wav> --ref-text "..."`) | ~1.2 GB F16 + ~400 MB tokenizer |
| [`vibevoice-tts`](#vibevoice--realtime-streaming-tts) | Lowest-latency streaming TTS, designed for realtime. | Preset voice packs | ~636 MB via `-m auto` |
| [`vibevoice-1.5b`](#vibevoice-15b--base-tts-with-wav-cloning) | Base VibeVoice TTS model with WAV cloning. | Yes (`CRISPASR_VIBEVOICE_VOICE_AUDIO=<wav>` or `--voice <wav>`) | ~1.6 GB via `-m auto` |
| [`orpheus`](#orpheus--llama-32-3b--snac-codec) | Llama-3.2-3B talker + SNAC 24 kHz codec. 8 baked English speakers; expressive output. Greedy loops — pass `--temperature 0.6`. | Preset names via `--voice tara/leah/...` | ~3.5 GB via `-m auto` (talker Q8 + 26 MB SNAC) |
| [`chatterbox`](#chatterbox--flow-matching-tts-voice-cloning--multilingual) | T3 AR + S3Gen flow-matching + HiFTGenerator. Built-in voice baked into the T3 GGUF; clones via a baked voice GGUF (see workflow below). EN/AR/DE variants share runtime. | Yes (`--voice <voice.gguf>`, baked from a WAV with `models/bake-chatterbox-voice-from-wav.py`) | ~880 MB via `-m auto` (T3 Q8 + S3Gen Q8) |
| **`outetts`** | OuteTTS-0.3-1B: OLMo-1B LLM + WavTokenizer single-codebook VQ-GAN. CC-BY-NC-SA-4.0 (non-commercial, ShareAlike). 24 kHz output. | Yes (`--voice <speaker.json>`, created with `tools/reference_backends/outetts_create_speaker.py`) | ~2.5 GB via `-m auto` (talker F16 + WavTokenizer decoder) |
| [`f5-tts`](#f5-tts--dit-flow-matching-voice-cloning) | F5-TTS v1 Base: 22-layer DiT flow-matching TTS + Vocos iSTFT vocoder. MIT license. High-quality zero-shot voice cloning from 3-15s reference audio. 24 kHz output. English + Chinese (built-in pinyin g2p, #294). | Yes (`--voice <ref.wav> --ref-text "transcript"`) | ~953 MB via `-m auto` (single F16 GGUF, DiT + Vocos) |
| [`raon`](#f5-tts--dit-flow-matching-voice-cloning) | Raon-OpenTTS 0.3B (KRAFTON): F5-TTS DiT on the same runtime, paired with a 16 kHz HiFi-GAN vocoder (sbhifigan16k, slaney mel). English zero-shot voice cloning. **CC-BY-NC-4.0** (non-commercial; auto-download prints the restriction). TTS→ASR roundtrip validated (0.90). Note: CPU vocoder ~40s/utterance. | Yes (`--voice <ref.wav> --ref-text "transcript"`) | ~959 MB via `-m auto` (single GGUF: DiT + HiFi-GAN) |
| [`irodori-tts`](#irodori-tts--japanese-voice-cloning--emoji-emotion-control) | Irodori-TTS: RF-DiT flow-matching TTS with LowRankAdaLN + JointAttention + half-RoPE + SwiGLU. 48 kHz via Semantic-DACVAE-Japanese-32dim codec. MIT license. Japanese-focused (llm-jp-3 tokenizer). Zero-shot voice cloning from any reference WAV (DAC-VAE encoder + speaker CFG); emoji emotion control; duration predictor for output length. **VoiceDesign** (600M-v3): adds caption encoder for style/emotion control via text descriptions (`--instruct "calm adult male, deep voice"`); independent text/speaker/caption CFG. | Yes (`--voice <ref.wav> --i-have-rights`) | ~526 MB Q4_K (VoiceDesign) / ~852 MB Q4_K (base) + DAC-VAE codec |
| [`indextts`](#indextts--chineseenglish-voice-cloning) | IndexTTS-1.5: GPT-2 AR (24L/1280d) mel-code generator + BigVGAN vocoder. Designed for Chinese+English. Zero-shot voice cloning from any reference WAV. | Yes (`--voice <ref.wav>`) | ~2.4 GB via `-m auto` (GPT F16 + BigVGAN F16) |
| [`cosyvoice3-tts`](#cosyvoice3--voice-cloning-from-a-wav) | Fun-CosyVoice3-0.5B-2512: Qwen2-0.5B AR speech-token LM + DiT-CFM (10-step Euler) + HiFT (NSF + iSTFT) @ 24 kHz. 9 languages + 18 Chinese dialects. Ships an 8-voice baked bank (`zero_shot` + `fleurs-{en,de,zh,ja,fr,es,ko}`). | Yes — baked-bank name via `--voice <name>`, **or** native arbitrary-WAV cloning via `--voice <ref.wav> --ref-text "..."` (ports speech_tokenizer_v3 + CAMPPlus + matcha mel to ggml; speech tokens byte-exact vs ONNX). | ~1.2 GB via `-m auto` (Q4_K LLM + Q8_0 flow + HiFT + s3tok + campplus + voices) |

## Piper — community voices

The default US Lessac voice needs no manual model download:

```powershell
.\crispasr.exe --backend piper -m auto --tts "Hello from Piper." --tts-output piper.wav
```

For another voice, download its GGUF from
[`cstr/piper-voices-GGUF`](https://huggingface.co/cstr/piper-voices-GGUF).
Either pass the full path, or put it in `CRISPASR_MODELS_DIR` and pass its bare
filename. CrispASR resolves that exact file; it does not replace an unknown
community voice with the registered US default (#397).

```powershell
$env:CRISPASR_MODELS_DIR = 'D:\ai\crispasr'
.\crispasr.exe --backend piper -m piper-en_GB-cori-medium-f16.gguf `
  --tts "Hello from the Cori voice." --tts-output cori.wav
```

For the HTTP server and PowerShell, keep `-t 8` separate from `-l en` and call
the real curl executable (PowerShell aliases `curl` in some versions):

```powershell
.\crispasr.exe --server --backend piper `
  -m piper-en_GB-cori-medium-f16.gguf -l en -t 8 --port 8089 `
  --no-spoken-disclaimer --accept-marking-responsibility

curl.exe -sS http://localhost:8089/v1/audio/speech `
  -H 'Content-Type: application/json' `
  --data-raw '{"model":"piper","input":"Hello, how are you today?","spoken_disclaimer":false,"response_format":"wav"}' `
  --output piper-server.wav
```

Writing a WAV with `--output` avoids sending binary audio through PowerShell's
object pipeline. To test raw streaming in `cmd.exe`, request
`"stream":true,"response_format":"pcm"` and pipe `curl.exe` to
`ffplay.exe -f s16le -ar 22050 -ac 1 -nodisp -`.

## More TTS backends

| Backend | Why pick it | Voice cloning | First-run download |
|---|---|---|---|
| **`csm`** | Sesame CSM-1B: Llama-3.2 1B backbone (first-codebook AR) + 100M depth decoder (codebooks 1–31) + Kyutai Mimi codec (32-codebook RVQ → SEANet) @ 24 kHz. Single GGUF. Apache-2.0. | No (single built-in voice) | ~1.4 GB via `-m auto` (single Q4_K GGUF) |
| **`dia`** | Nari Labs Dia 1.6B: byte-level text encoder (12L) + AR audio decoder (18L GQA) + 9-codebook DAC codec @ 44.1 kHz. CFG-guided, dialogue-style with `[S1]`/`[S2]` speaker tags. Apache-2.0. | No (dialogue via speaker tags) | ~1.6 GB via `-m auto` |
| **`zonos-tts`** | Zyphra Zonos-v0.1-transformer: 26-layer GQA AR transformer → 9-codebook DAC @ 44.1 kHz. Rich conditioning: speaker embedding + text + emotion + FWHM pitch/tempo. CFG guided. Voice cloning from any reference WAV (pass via `CRISPASR_ZONOS_SPEAKER_EMB_PATH` or `--voice <ref.wav>`). Apache-2.0. | Yes (`--voice <ref.wav>`) | ~1.6 GB Q8_0 (default) or ~931 MB selective-Q4_K (heads/embeddings kept F16, auto-retry guard) or ~3.0 GB F16, via `-m auto` + 104 MB DAC codec. |
| **`bark`** | Suno Bark: 3-stage GPT-2 (text→semantic→coarse→fine) + EnCodec 24 kHz decoder. All sub-models packed into one GGUF. Supports speaker conditioning via `.npz` prompts. MIT license. | Yes (`--voice <speaker.npz>`) | ~423 MB via `-m auto` (selective Q4_K) |
| **`speecht5`** | Microsoft SpeechT5 80M: char-level encoder (12L) + AR mel decoder (6L) + 5-conv postnet + HiFi-GAN @ 16 kHz. MIT. Speaker via 512-d x-vector. | Yes (`--voice <xvector.bin>`, raw float32) | ~300 MB via `-m auto` (F16 GGUF) |
| **`fastpitch`** | NVIDIA FastPitch 60M: non-autoregressive parallel TTS — 6L FFTransformer encoder + duration/pitch predictors + length regulator + 6L FFTransformer decoder + HiFi-GAN @ 22 kHz. Deterministic (no sampling). CC-BY-4.0. | No (single speaker) | ~230 MB via `-m auto` (Q8_0 GGUF) |
| **`bananamind-tts`** | BananaMind-TTS-V2.1 13M: Tacotron-lite (char tokenizer + Conv1d+BN+ReLU encoder + BiLSTM + AR GRU decoder with location-sensitive attention + postnet) + HiFi-GAN @ 22 kHz. English (LJ Speech) and German (ThorstenVoice). Apache-2.0. Runtime is designed as a template for standard Tacotron2 ports — add `decoder_rnn_type=lstm` to the GGUF to switch to the LSTM decoder path ([architecture notes](architecture.md#bananamind-tts)). | No (fixed voice per locale) | ~40 MB Q8_0 / ~50 MB F32 per locale |
| [`parler-tts`](#parler-tts--prompt-conditioned-voice-description) | Parler TTS Mini v1.1 (~900M): T5 encoder + MusicGen decoder + DAC 44.1 kHz. Apache-2.0. Prompt-conditioned: describe the voice in natural language via `--instruct`. | No (prompt-conditioned) | ~900 MB via `-m auto` (Q8_0 GGUF) |
| **`voxcpm2-tts`** | VoxCPM2: 2B Qwen2 backbone + flow matching + VAE decoder @ 48 kHz. Zero-shot voice cloning via `--voice <ref.wav>`. | Yes | ~2.4 GB via `-m auto` |
| [`pocket-tts`](#pocket-tts-languages-voices-and-environment-switches) | Kyutai Pocket TTS 100M: continuous-latent AR @ 12.5 Hz + one-step LSD flow head + Mimi VAE decoder → 24 kHz. English, German, Spanish, Italian, Portuguese, plus the upstream French 24L preview. CC-BY-4.0 plus gated-use conditions. Voice cloning via `--voice ref.wav`. | Yes (`--voice`) | ~124 MB Q8_0 per non-English 6L model; English F16 ~220 MB; French 24L Q8_0 ~365 MB |
| **`kugelaudio`** | KugelAudio-0-Open: 7B Qwen2.5 backbone + 4-layer DiT diffusion head (20-step SDE-DPMSolver++) + acoustic VAE decoder → 24 kHz. 23 languages. MIT. | Pre-encoded voices (`--voice voice.gguf`) | ~17.3 GB F16 via `-m auto` — needs >16 GB VRAM, else `--no-gpu`. The ~5.7 GB Q4_K is **not** a usable substitute: it stutters and loops (WER 0.72 vs 0.056 for F16) |
| [`tada-1b`](#tada--multilingual-and-voice-cloning) | HumeAI TADA 1B: Llama-3.2-1B backbone + per-token flow-matching diffusion head + TADA codec → 24 kHz. **English-only.** `-m auto` downloads model + default `tada-ref.gguf`. | Yes (`--voice <tada-ref.gguf>`, English voice refs only) | ~1.7 GB Q4_K + ~1 GB codec |
| [`tada` / `tada-3b-ml`](#tada--multilingual-and-voice-cloning) | HumeAI TADA 3B Multilingual: same architecture, 3B params. Supports **ar, ch, de, es, fr, it, ja, pl, pt** in addition to English. `-l <lang>` auto-downloads `tada-ref-<lang>.gguf`. | Yes (`--voice <tada-ref.gguf>`) | ~4 GB Q4_K + ~1 GB codec |
| **`lfm2-audio`** | LiquidAI LFM2.5-Audio 1.5B: FastConformer encoder + LFM2 hybrid conv+attention backbone + 6L depthformer (8-codebook Mimi) + ISTFT detokenizer → 24 kHz. Interleaved text+audio generation. Also does ASR and speech-to-speech. LFM Open License v1.0 ($10M revenue cap). | No | ~1.5 GB Q4_K (JP) / ~1.6 GB Q5_K (EN) + ~157 MB detokenizer companion |
| [`dots-tts`](#dotstts--voice-cloning-and-performance) | rednote-hilab dots.tts-soar: Qwen2.5-1.5B LLM + 24L VAESemanticEncoder + 18L DiT flow-matching head (16-step Euler CFG) + BigVGAN vocoder → 48 kHz. Continuous-latent AR (patch-by-patch). Apache-2.0. **The CFG flow-match needs an F16 DiT — a full-q8 core derails; use the mixed quant (`crispasr-quantize` keeps the DiT at F16, quantizes the LLM+PatchEncoder to Q8_0 or Q4_K).** CAM++ reference-WAV voice cloning is supported when the speaker companion is present. | Yes (`--voice ref.wav --i-have-rights`) | ~3.1 GB mixed-Q8 / ~2.2 GB mixed-Q4_K core + 330–345 MB vocoder companion |
| **`confucius4-tts`** | NetEase Youdao Confucius4-TTS: GPT-2 T2S (24L/1280d, beam-sample num_beams=3, LlamaTokenizer vocab baked) + flow-matching DiT+WaveNet S2A (25-step Euler CFG 0.7, native w2v-BERT + CAMPPlus style/reference-mel conditioning) + BigVGAN vocoder → 22.05 kHz. Zero-shot voice cloning; 14 languages via Chinese `LANGUAGE_TOKEN_MAP` prompts (`-l <lang>`). Apache-2.0. **Zero-shot only: without `--voice` conditioning the output is unintelligible by design.** | Yes (`--voice ref.wav --i-have-rights`) | ~376 MB Q4_K T2S + ~135 MB S2A + ~214 MB BigVGAN + w2v companion via `-m auto` |
| **`mini-omni2`** | gpt-omni/mini-omni2: Whisper-small encoder + Qwen2-0.5B LLM with 8-stream architecture + SNAC 24 kHz decoder → 24 kHz. Also does ASR and speech-to-speech. MIT license. Requires `--codec-model snac-24khz.gguf` companion. | No | ~1.0 GB Q4_K + ~80 MB SNAC companion |
| **`voxtral-tts`** | Mistral Voxtral-4B-TTS-2603: Ministral-3B AR backbone (26L GQA, NORMAL/adjacent-pair RoPE) + 3L bidirectional flow-matching acoustic transformer (8-step Euler ODE + CFG α=1.2, no positional encoding) + Voxtral codec decoder (ALiBi sliding-window attention + reflect-causal conv + ConvTranspose upsampling) → 24 kHz. 20 preset voices across 9 languages (en/fr/de/es/it/pt/nl/ar/hi); strong on French technical text. CC-BY-NC-4.0. | No (20 preset voices via `--voice <name>`, e.g. `fr_female`) | ~2.4 GB Q4_K / ~4.3 GB Q8_0 / ~8.2 GB F16 via `-m auto` |

All backends write mono WAV via `--tts-output` (22 kHz for piper/fastpitch/bananamind-tts/confucius4-tts, 16 kHz for speecht5, 44.1 kHz for melotts/dia/parler-tts/zonos-tts, 48 kHz for voxcpm2-tts/dots-tts/irodori-tts/moss-tts-local/sidon, 24 kHz for most others). Programmatic callers don't need this table: `crispasr_session_output_sample_rate()` returns the active backend's output rate, with `crispasr_session_input_channels()` / `output_channels()` alongside (everything is mono today) — #332.

## dots.tts — voice cloning and performance

`dots-tts` supports zero-shot voice cloning from a reference WAV. Place the
CAM++ speaker companion (`dots-tts-soar-spk-f16.gguf`) beside the core model,
then pass the reference recording with the consent attestation:

```bash
crispasr --backend dots-tts \
  -m dots-tts-soar-q4_k.gguf \
  --codec-model dots-tts-soar-vocoder-q8_0.gguf \
  --voice reference.wav --i-have-rights \
  --tts "Hello from CrispASR." --tts-output cloned.wav
```

The speaker companion is loaded only when `--voice` is supplied. Without it,
the same backend produces text-conditioned speech using its default model
conditioning. If the speaker companion is missing, the CLI warns and ignores
the voice prompt; it does not silently claim that cloning succeeded.

The practical low-memory choice is the mixed Q4_K core: the LLM and
PatchEncoder are quantized, while the DiT, projections, embeddings, and
conditioning statistics remain at the precision required by the flow-matching
loop. The Q8 core is the safer quality/size default; do not quantize the DiT.

The dominant cost is the DiT flow-match, not tokenization or the recurrent
LLM step. A local Apple M1 measurement using the Q4_K core, Q8 vocoder, Metal,
and an eight-patch cap took 24.2 s at 16 Euler steps: about 13.7 s in
flow-match, 10.4 s in the BigVGAN vocoder, and under 0.5 s in the LLM and
PatchEncoder. With the same seed and cap, `CRISPASR_DOTS_ODE_STEPS=8` took
16.2 s (1.49x faster); this is a speed/quality trade-off, not a parity-preserving
optimization. Full utterances add patch-by-patch AR cost, so benchmark with a
known patch/audio cap and inspect the generated WAV.

For further experiments, `CRISPASR_DOTS_FUSED_STEP=0` restores the legacy DiT
path for A/B testing; the persistent fused graph is the default on CPU/Metal.
`CRISPASR_DOTS_CFG_INTERVAL=2` skips some unconditional CFG evaluations but is
approximate and should remain opt-in until the decoded output is checked.
`CRISPASR_DOTS_TTS_BENCH=1` prints the stage timings used above.

## TADA — multilingual and voice cloning

TADA ships as two variants; only the 3B multilingual model can synthesise
non-English text:

| Backend | Model | Languages |
|---|---|---|
| `tada-1b` | HumeAI/tada-1b | English only |
| `tada` / `tada-3b-ml` | HumeAI/tada-3b-ml | en + ar, ch, de, es, fr, it, ja, pl, pt |

### Use case (a) — built-in default voice for a language

TADA uses Llama 3.2 materials and is distributed under the Llama 3.2
Community License. Managed model and auxiliary downloads therefore require
`--accept-license llama3.2`; redistribution must retain the Llama notice and
follow its acceptable-use terms. The language reference packs are derived
from CC-BY-4.0 FLEURS clips and retain that attribution requirement.

Pass `-l <lang>` with the `tada-3b-ml` backend. CrispASR auto-downloads
`tada-ref-<lang>.gguf` from `cstr/tada-tts-3b-ml-GGUF` on first use (<200 KB
per language):

```bash
crispasr --backend tada-3b-ml -m auto -l fr \
    --tts "La justice sans force est impuissante." \
    --tts-output justice.wav
```

The `tada-ref-fr.gguf` encodes the acoustic fingerprint (speaker identity,
prosody) extracted offline from a 10 s FLEURS CC-BY-4.0 French clip. Once
cached it is reused for every subsequent French synthesis call. Available
language codes: `ar`, `ch`, `de`, `es`, `fr`, `it`, `ja`, `pl`, `pt`.

#### Switching voice at query time (server, #201)

On the HTTP server the voice is **per request** — point `voice` at a different
`tada-ref-*.gguf` and the backend reloads it on the next call, no container
restart needed. Omitting `voice` (or `"default"`/`"auto"`) keeps the
currently-loaded reference, so requests that don't care pay nothing.

```bash
# request 1: French built-in voice
curl -s :8080/v1/audio/speech -d '{"input":"Bonjour.","voice":"tada-ref-fr.gguf"}' -o fr.wav
# request 2: German voice — switched live, same running server
curl -s :8080/v1/audio/speech -d '{"input":"Guten Tag.","voice":"tada-ref-de.gguf"}' -o de.wav
```

The name resolves like any other model path (absolute path, or a cache/registry
name that auto-downloads). Embedders going through the session C ABI get the
same capability via `crispasr_session_set_voice(s, "tada-ref-de.gguf", NULL)`.
Generating a brand-new reference from raw audio+transcript at query time is not
yet wired into the server (it needs the encoder+aligner GGUFs loaded) — bake the
ref offline with the `--make-ref` pipeline below, then switch to it live.

### Use case (b) — custom voice cloning

To speak in a specific person's voice, bake a ref GGUF from ~10 s of their
speech. Two options — pure C++ (no Python) or the Python converter:

#### Option 1: C++ `--make-ref` (no Python needed)

Place `tada-encoder-f16.gguf` and `tada-aligner-en.gguf` (they ship in the
same repo as the model — [cstr/tada-tts-3b-ml-GGUF](https://huggingface.co/cstr/tada-tts-3b-ml-GGUF)
for `tada` / `tada-3b-ml`, [cstr/tada-tts-1b-GGUF](https://huggingface.co/cstr/tada-tts-1b-GGUF)
for `tada-1b`) next to your TADA model GGUF, then:

```bash
# Step 1 — bake the ref GGUF (one-time per speaker):
# --i-have-rights is required: baking IS the cloning step. The pack is
# stamped as clone-derived so step 2 still gets the consent gate and the
# spoken AI disclosure.
crispasr --backend tada-3b-ml -m auto \
    --make-ref \
    --voice speaker_10s.wav \
    --ref-text "Exact words spoken in the audio." \
    --make-ref-output tada-ref-custom.gguf \
    --i-have-rights

# Step 2 — synthesise with that voice:
crispasr --backend tada-3b-ml -m auto \
    --voice tada-ref-custom.gguf \
    --tts "Bonjour, comment allez-vous ?" \
    --tts-output result.wav \
    --i-have-rights
```

The `--make-ref` pipeline runs entirely in C++: wav2vec2 aligner → BPE
tokenization → DP alignment → WavEncoder → LocalAttentionEncoder → GGUF
output. Auto-discovers the encoder + aligner GGUFs next to the model file;
override with `--make-ref-encoder` / `--make-ref-aligner` if needed.

#### Option 2: Python converter

```bash
pip install hume-tada
python models/convert-tada-ref-to-gguf.py \
    --audio speaker_10s.wav \
    --language fr \
    --output tada-ref-custom.gguf
```

A 5–15 s clip of clean speech (no music/noise) produces the best fingerprint.
`--language` (Python) selects the language-specific TADA aligner; it must match
the language of the text you will synthesise.

### Encoder / aligner GGUFs

The encoder pipeline (WAV+transcript → voice reference) is ported to C++/ggml.
Pre-converted GGUFs sit alongside the model in
[cstr/tada-tts-3b-ml-GGUF](https://huggingface.co/cstr/tada-tts-3b-ml-GGUF)
(and [cstr/tada-tts-1b-GGUF](https://huggingface.co/cstr/tada-tts-1b-GGUF) for
`tada-1b`); `--auto-download` fetches them into the cache on demand:

| File | Size | Description |
|------|------|-------------|
| `tada-encoder-f16.gguf` | 178 MB | Shared WavEncoder + 6-layer LocalAttentionEncoder + hidden linear |
| `tada-aligner-en.gguf` | 1.1 GB | English aligner (wav2vec2-large + 128K-class Llama CTC head) |

The encoder GGUF is loaded by `src/tada_encoder.{h,cpp}`. The aligner GGUF
is loaded by the existing `wav2vec2_load()` runtime (same architecture).

To convert language-specific aligners:
```bash
python models/convert-tada-aligner-to-gguf.py \
    --codec-repo HumeAI/tada-codec --language fr \
    --output tada-aligner-fr.gguf
```

If `--voice` is omitted, the runtime uses `tada-ref-<lang>.gguf` when `-l
<lang>` is set, then falls back to `tada-ref.gguf` (the built-in English
voice).

### Timing quality (`CRISPASR_TADA_NUM_CANDIDATES`)

TADA predicts each token's duration with a per-token flow-matching head that
is **noise-sensitive**: an unlucky noise draw can collapse durations into a
rushed, unintelligible utterance (a known property of the model — the PyTorch
reference behaves identically with the same noise). CrispASR can generate
several flow-matching candidates per token and keep the best one by
reconstruction likelihood (the same `num_acoustic_candidates` ranking the
reference implements).

The default is **1 candidate**, matching upstream `InferenceOptions` — best-of-N
is opt-in because the reconstruction scorer looks at acoustic dims only and can
prefer a duration outlier ("…four hours" → "…and forth", #192). Override with
`CRISPASR_TADA_NUM_CANDIDATES`:

```bash
# Default: single noise draw (may occasionally rush/garble timing):
CRISPASR_TADA_NUM_CANDIDATES=1 crispasr --backend tada-3b-ml -m auto -l fr \
    --tts "Bonjour, comment allez-vous ?" --tts-output out.wav

# Higher quality / more robust timing (slower):
CRISPASR_TADA_NUM_CANDIDATES=8 crispasr --backend tada-3b-ml -m auto -l de \
    --tts "Guten Tag, wie geht es Ihnen?" --tts-output out.wav
```

All candidates for a step are solved in a single batched flow-matching forward,
so raising the count adds little wall-clock on top of the (model-load-dominated)
baseline. `1` reproduces a single draw and is the fastest.

The same **default of 1** applies through the session C ABI, so the bindings
and HTTP server track the CLI. Bindings can override it at
runtime with `set_tts_num_candidates(n)` (Python/Go/Rust/Ruby),
`SetTtsNumCandidates` (C#/Java), or `setTtsNumCandidates` (Dart/JS) — and the
`CRISPASR_TADA_NUM_CANDIDATES` env var is honoured by every consumer, not just the CLI.

### Talker text sampling (`CRISPASR_TADA_TEMPERATURE`, `CRISPASR_TADA_TOP_P`, …)

`CRISPASR_TADA_NUM_CANDIDATES` tunes only the **duration** flow-matching head, not the
**content** (which words are spoken). The talker text decoder is a separate
knob. It samples by default, matching upstream `InferenceOptions`
(do_sample=True, temperature=0.6, top_k=0, top_p=0.9, repetition_penalty=1.1);
pure greedy decoding has no repetition control and loops, cuts words off, or
adds trailing noise — worst on harder or non-English text.

| Env var | Default | Notes |
|---|---|---|
| `CRISPASR_TADA_DO_SAMPLE` | `1` | `0` = greedy argmax (the old behaviour); also `set_do_sample` |
| `CRISPASR_TADA_TEMPERATURE` | `0.6` | also set by `--temperature` / `set_temperature` |
| `CRISPASR_TADA_TOP_P` | `0.9` | nucleus; also `set_top_p` |
| `CRISPASR_TADA_TOP_K` | `0` | `0` = disabled; also `set_top_k` |
| `CRISPASR_TADA_REPETITION_PENALTY` | `1.1` | `1.0` = none; also `set_repetition_penalty` |

Honoured by the CLI, C ABI, bindings and server. Raising
`CRISPASR_TADA_REPETITION_PENALTY` measurably reduces repeats; with sampling on, `--seed`
changes the wording.

On the HTTP server these are **per-request** JSON fields on `POST
/v1/audio/speech` — `temperature`, `top_p`, `top_k`, `repetition_penalty`,
`do_sample`, `num_candidates` — so a long-running container can be retuned at
query time without a restart. A field that is omitted falls back to the value
the server was started with (env / flags), so requests don't leak settings into
each other.

### Acoustic fidelity — quick vs accurate (`CRISPASR_TADA_NUM_FM_STEPS`, …)

The **acoustic** flow-matching head (which renders the predicted features into
codec frames) has its own knobs, separate from the talker sampler and the
duration-candidate ranking. These are the upstream `InferenceOptions` fields
(`tada.py`) the reporter in #197 flagged as the "quick and dirty" vs "slow and
accurate" axis. `num_flow_matching_steps` is the primary lever — more ODE steps
cost proportionally more wall-clock but improve fidelity (e.g. ~4 steps is fast
and intelligible; ~25 is noticeably slower and crisper).

| Env var | Default | Notes |
|---|---|---|
| `CRISPASR_TADA_NUM_FM_STEPS` | `10` | Flow-matching ODE steps (Python `num_flow_matching_steps`); higher = slower/more accurate. Also `set_tts_steps` |
| `CRISPASR_TADA_ACOUSTIC_CFG` | `1.6` | Acoustic classifier-free-guidance scale (Python `acoustic_cfg`). Also `set_cfg_weight` |
| `CRISPASR_TADA_NOISE_TEMP` | `0.9` | FM noise temperature (Python `noise_temp`). Also `set_tts_noise_temp` |

On the HTTP server these are also **per-request** JSON fields on `POST
/v1/audio/speech`: `num_steps` (→ FM steps), `cfg_scale` (→ acoustic CFG), and
`noise_temp`. Same omit-falls-back-to-default, no-leak semantics as the sampler.

```bash
# Slow and accurate: more ODE steps for crisper acoustics.
curl -s http://localhost:8080/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"input":"Hi. My name is Bob.","num_steps":25}' -o out.wav
```

### Reproducible / diverse generation (`--seed`)

Pass `--seed N` (any non-zero integer) for **reproducible** output —
the same seed + prompt + voice produces identical audio across runs.
Pass `--seed 0` (the default) for non-deterministic sampling, where
each run can produce a different prosody or phrasing.

```bash
# Reproducible:
./build/bin/crispasr --backend qwen3-tts -m auto \
    --tts "Good morning." --seed 42 --tts-output out1.wav
./build/bin/crispasr --backend qwen3-tts -m auto \
    --tts "Good morning." --seed 42 --tts-output out2.wav
# out1.wav == out2.wav (bit-identical)

# Diverse — different seeds produce different renderings:
./build/bin/crispasr --backend qwen3-tts -m auto \
    --tts "Good morning." --seed 1 --tts-output variant1.wav
./build/bin/crispasr --backend qwen3-tts -m auto \
    --tts "Good morning." --seed 2 --tts-output variant2.wav
```

The seed is wired through the sampling-capable TTS backends:
qwen3-tts, chatterbox, vibevoice, orpheus, tada, indextts, f5-tts, voxcpm2, and parler-tts. It
also works for ASR backends with temperature sampling (parakeet,
canary, cohere, qwen3-asr, voxtral4b, granite, glm-asr, kyutai-stt,
moonshine). The server API accepts `"seed"` in the `/v1/audio/speech`
JSON body.

Local live checks on `/Volumes/backups/ai/crispasr` confirmed the visible
effect on `qwen3-tts-customvoice`, `chatterbox`, `vibevoice-tts`, and
`vibevoice-1.5b`: same seed = bit-identical WAV, different seed =
different WAV hash. `IndexTTS` accepts the seed too, but on the tested
prompt/reference pair the default beam-search path stayed identical
across seeds, so treat it as a low-visibility knob unless you expose a
stochastic decode path. `Kokoro` uses `CRISPASR_KOKORO_SEED` instead of
`--seed`.

For HTTP usage, see [`docs/server.md`](server.md) — `POST
/v1/audio/speech` is the OpenAI-compatible TTS endpoint, available on
any `crispasr --server` instance whose loaded backend declares
`CAP_TTS`. Routes register on every backend; per-request `voice`,
`speed`, and `instructions` pass through to the backend's
`whisper_params`. Long-form input is auto-chunked on sentence
boundaries.

## CosyVoice3 — voice cloning from a WAV

```bash
./build/bin/crispasr --backend cosyvoice3-tts -m auto \
    --voice ref.wav --i-have-rights \
    --tts "Text to speak." --tts-output out.wav
```

**Leave `--ref-text` off unless you have an exact transcript.** The reference
is auto-transcribed (whisper by default, `--ref-asr` to pick another backend)
and the result is cached beside the clip as `<voice>.cv3reftext`, so only the
first run pays for ASR.

The reason this matters is issue #334: a transcript that does not match the
clip is the one input that quietly ruins the output. The talker LM is
conditioned on the pair *(reference transcript, reference speech)* and infers
the speaker's rate from it. Hand it 18 s of audio labelled with one sentence
and it concludes the speaker says a sentence in 18 s, then either stops
immediately — `AR decode produced 0 tokens`, no audio — or crams the line you
asked for into far too few 40 ms frames, which is heard as rushed, pitched-up
speech. Measured on a 17.7 s reference: with a one-sentence guess the requested
line vanished from the output entirely; auto-transcribed, it came out in full.

If you do pass `--ref-text`, it must be a complete, exact transcription.
Upstream's decode only ever spends between 2 and 20 speech tokens per text
token, so the runtime checks what you passed against that same band and warns
when the pairing cannot be a transcript:

```
cosyvoice3_tts: WARNING: the reference clip holds 17.72 s of speech but
--ref-text is only 7 token(s) long (63.3 speech frames per text token; a
matching transcript lands between 2 and 20). …
```

If you cannot transcribe the whole clip, either drop `--ref-text` and let the
backend do it, or **trim the clip to the part you did transcribe** — a clean
4–10 s excerpt clones better than 20 s with an approximate transcript.

### Reference sample rate

Any rate works and none is preferred: the reference is resampled internally to
16 kHz for the speech tokenizer and the CAMPPlus speaker encoder, and to 24 kHz
for the prompt mel. 8/16/22.05/24/32/44.1/48 kHz references of the same
recording all round-trip to the same transcript with the same speaking rate.
The output is always 24 kHz mono regardless of the reference.

### Base vs RL talker

`FunAudioLLM/Fun-CosyVoice3-0.5B-2512` ships two talker checkpoints: `llm.pt`
(pre-trained) and `llm.rl.pt` (reinforcement-learning tuned by the authors for
speech quality, pronunciation accuracy and generation stability). Only the
talker differs — flow, HiFT, CAMPPlus, the speech tokenizer and the voice bank
are shared, so the RL build swaps exactly one 384 MB file:

```bash
./build/bin/crispasr --backend cosyvoice3-tts-rl -m auto \
    --voice fleurs-en --i-have-rights \
    --tts "The northern lights can be heard as well as seen." --tts-output out.wav
```

`cosyvoice3-tts-rl` is the same engine; the alias exists so `-m auto` fetches
`cosyvoice3-llm-rl-q4_k.gguf` instead of the base talker. Both live in
[`cstr/cosyvoice3-0.5b-2512-GGUF`](https://huggingface.co/cstr/cosyvoice3-0.5b-2512-GGUF)
(F16 and Q4_K), and pointing `-m` at either file directly works too — the
companions are found by name.

To rebuild either talker from the upstream checkpoint:

```bash
python models/convert-cosyvoice3-to-gguf.py \
    --input FunAudioLLM/Fun-CosyVoice3-0.5B-2512 --output-dir out \
    --llm-checkpoint llm.rl.pt --skip flow --skip hift   # → cosyvoice3-llm-rl-f16.gguf
```

## Output language and cross-lingual cloning (`-tl` / `-sl`)

Issue #329: *"there is usually an option to select the target language, but I
don't find it for these engines."* It exists — here is where it lives on each
surface, and what each backend does with it.

**The target language is the language to SPEAK.** For a TTS backend `-l`, `-tl`
and the server's `"language"` all mean the same thing; `-tl` /
`--target-lang` is the spelling to reach for when you are dubbing, and it wins
over `-l` where both are given.

```bash
# CLI — synthesize German
./build/bin/crispasr --backend cosyvoice3-tts -m auto -tl de \
    --tts "Guten Morgen, wie geht es Ihnen?" --tts-output out.wav

# CLI — clone an ENGLISH reference clip, speaking GERMAN
./build/bin/crispasr --backend cosyvoice3-tts -m auto \
    -tl de -sl en \
    --voice ref_en.wav --ref-text "This is my reference recording." \
    --i-have-rights \
    --tts "Guten Morgen, wie geht es Ihnen?" --tts-output out.wav
```

```jsonc
// POST /v1/audio/speech
{
  "input":       "Guten Morgen, wie geht es Ihnen?",
  "voice":       "ref_en.wav",
  "language":    "de",     // what to speak    (alias: "target_lang")
  "source_lang": "en",     // what the REFERENCE is spoken in (alias: "ref_lang")
  "consent_attestation": "..."
}
```

```python
# Python / Rust / Flutter session API
s.set_target_language("de")          # what to speak
s.set_tts_reference_language("en")   # what the reference clip is spoken in
```

**`-sl` / `--source-lang` / `"source_lang"` names the language of the CLONING
REFERENCE, not of the output.** It only matters for cross-lingual cloning, and
only cosyvoice3 acts on it today.

### What each backend does with it

| Backend | Effect |
|---|---|
| **`cosyvoice3-tts`** | When the target language differs from the reference voice's language, mirrors upstream `frontend_cross_lingual`: the reference **transcript** is dropped from the LM prompt (and the LM's reference speech tokens with it) while the flow keeps the reference speech + mel for **timbre**. That is what stops a German sentence coming out with the English reference's accent. Same language (or no target) = plain zero-shot, unchanged. |
| **`qwen3-tts`** | Sets the talker's explicit `codec_language_id` in the prefill instead of the model's auto ("nothink") path. Applies to preset voices, cloned voices and VoiceDesign alike. Languages come from the model's own `codec_language_names` table; one outside it prints a warning and falls back to auto. |
| **`moss-tts` / `moss-tts-local`** | Fills the `- Language:` field of the generation prompt. |
| **`kokoro`, `zonos`, `piper`** | Selects the eSpeak voice / G2P language. |
| **`voxcpm2`, `f5-tts`, `vibevoice`, …** | Language-agnostic — they read the script of the input text itself, which is why they need no knob (and why voxcpm2 "just works" in #329). |

### When cosyvoice3 says it could not determine the reference language

To decide *whether* to go cross-lingual, cosyvoice3 needs the reference clip's
language. It resolves that in this order:

1. `-sl` / `"source_lang"` / `set_tts_reference_language()` — you said so.
2. The voice-bank entry, for the baked `fleurs-<lang>` voices.
3. Detection over the reference transcript (`--ref-text`): writing system first,
   then function words for Latin-script languages
   (`src/core/tts_lang.h`, covered by `tests/test-tts-lang.cpp`).

Step 3 answers "unknown" rather than guessing when the transcript is short or
its language is not one it knows. In that case the target language cannot be
acted on and you get:

```
cosyvoice3_tts: target language 'de' requested but the reference voice's language
could not be determined (transcript too short or unsupported script) —
synthesising zero-shot, which keeps the reference's accent. Pass
--source-lang <lang> (server: "source_lang") to enable cross-lingual synthesis.
```

Passing `-sl` resolves it. Before #329, detection covered only Hangul / Kana /
Han / Cyrillic, so **every Latin-script pair** — en↔de, en↔fr, es↔it, the ones a
subtitle-dubbing workflow actually asks for — landed in that unknown case, with
no message and no way to override it.

### OmniVoice: ISO 639-3 ids, and what the tag can and cannot do

OmniVoice carries the language as a literal string in its prompt, so the value
has to be one the model was trained on: an **ISO 639-3 id** from its own
646-entry map, or the English name for one. `-l de`, `-l German` and
`{"language": "German"}` all produce byte-identical output; anything else —
`de-DE`, `en_US`, a typo — is not a language the model knows and falls back to
language-agnostic synthesis with a warning naming the nearest match:

```
crispasr[omnivoice]: language 'de-DE' is not one of the model's 646 language IDs — did you mean 'de'?
crispasr[omnivoice]: falling back to language-agnostic synthesis. Pass an ISO 639-3 id (e.g. 'en', 'de', 'arb') or an English name (e.g. 'German').
```

**If you request nothing, the language is guessed from the text.** An explicit
`-l` / `-tl` / `"language"` always wins; the guess only fills the gap that used
to be filled by "language-agnostic", and only when it is confident enough to
land on an id the model knows (short lines usually are not, and stay agnostic).
`CRISPASR_OMNIVOICE_AUTO_LANG=0` turns it off. This is why OmniVoice speaks
German subtitles as German through a client that never sends a language field.

Two things that surprise people:

- **Macrolanguage codes are absent.** The map lists individual languages, so
  Arabic is `arb` (Standard), `arz` (Egyptian), `ary` (Moroccan) and ~20 more —
  there is no `ar`. Same shape for other macrolanguages. This is the model's
  vocabulary, not a gap in ours.
- **The tag is not an accent control.** Unlike CosyVoice3, OmniVoice has no
  cross-lingual mode: a voice clone always conditions on the
  (reference transcript, reference audio) pair, and dropping the transcript
  would desynchronize it from the audio. So cloning an English speaker onto
  German text may still carry the speaker's accent, and `-tl de` is not
  guaranteed to remove it — measured over three German sentences, whisper LID
  could not tell the tagged and untagged output apart. If accent-free output in
  the target language matters more than exact timbre, use a reference clip
  already spoken in that language, or use CosyVoice3 with `-sl` (above), which
  does have a transcript-dropping cross-lingual path.

### OmniVoice: `--instruct` is a closed vocabulary, and it is validated

Voice design (`--instruct`, server `"instructions"`) is not free prose for this
backend. OmniVoice was trained on a fixed 48-item vocabulary in one exact
spelling, and the string reaches the prompt literally — `Male, British Accent`
and `male, british accent` share **no** token ids, so an unnormalised value is
a different request as far as the model is concerned.

CrispASR mirrors upstream's resolver: items are lowercased, half/full-width
commas are both accepted, and everything is unified into one language before it
reaches the prompt. Anything outside the vocabulary is **rejected** — the CLI
exits non-zero, the server returns `400 invalid_instructions` — rather than
being dropped, because a voice-design request that silently does nothing is the
failure this replaced:

```
crispasr[omnivoice]: unsupported instruct item 'britsh accent' — did you mean 'british accent'?
```

The categories, at most one item from each:

| category | items |
|---|---|
| gender | `male`, `female` |
| age | `child`, `teenager`, `young adult`, `middle-aged`, `elderly` |
| pitch | `very low pitch`, `low pitch`, `moderate pitch`, `high pitch`, `very high pitch` |
| style | `whisper` |
| accent (English only) | `american accent`, `british accent`, `australian accent`, `canadian accent`, `indian accent`, `chinese accent`, `japanese accent`, `korean accent`, `portuguese accent`, `russian accent` |
| dialect (Chinese only) | `河南话`, `陕西话`, `四川话`, `贵州话`, `云南话`, `桂林话`, `济南话`, `石家庄话`, `甘肃话`, `宁夏话`, `青岛话`, `东北话` |

Each item has a Chinese counterpart (`male` ↔ `男`), and the whole instruct is
unified to one language before synthesis: a dialect forces Chinese, an accent
forces English, otherwise it follows the language of the text being spoken. So
`--instruct "male, elderly"` on Chinese text becomes `男，老年`. Mixing a
dialect with an accent is rejected — they belong to different speech.

## G2P Phonemization (`--g2p-dict`)

TTS backends that use IPA phonemes (piper, kokoro) need a
grapheme-to-phoneme (G2P) engine to convert text to IPA. CrispASR
ships pre-generated IPA pronunciation dictionaries for 4 languages —
**no espeak-ng required**:

| Language | IPA dict (primary) | Fallback | Match rate |
|----------|--------------------|----------|------------|
| English | 126K words, 3 MB | CMUdict + ARPAbet→IPA + LTS rules | 99.5% |
| German | 667K words, 23 MB | OLaPh + LTS rules (Auslautverhärtung, compound splitting) | 100% |
| French | 257K words, 6.6 MB | OLaPh + LTS rules (nasals, silent finals, s-voicing) | — |
| Spanish | 600K words, 18 MB | OLaPh + LTS rules (seseo, lenition, yeísmo) | — |

Dictionaries are auto-downloaded from
[cstr/g2p-dicts](https://huggingface.co/datasets/cstr/g2p-dicts)
on first use and cached at `~/.cache/crispasr/`.

### Numbers are spelled out first

Digits appear in no pronunciation dictionary and no letter-to-sound rule, so
before v0.8.24 a numeric token produced **no phonemes at all** and vanished from
the audio — "a model with 82 million parameters" was spoken as "…with million
parameters" (#316). Numbers are now expanded to words ahead of phonemization,
following misaki's reading:

| input | spoken | | input | spoken |
|---|---|---|---|---|
| `82` | eighty two | | `1234` | twelve thirty four |
| `101` | one hundred one | | `1005` | one thousand five |
| `1984` | nineteen eighty four | | `1100` | eleven hundred |
| `2026` | twenty twenty six | | `1000` | one thousand |
| `3.14` | three point one four | | `1st` | first |
| `50%` | fifty percent | | `$20` | twenty dollars |

A four-digit number reads as a **year** (two halves) unless it ends in `00` or
its last two digits are below ten — `1234` is "twelve thirty four" but `1005` is
"one thousand five". A digit inside a word (`mp3`, `x64`) is left alone.

> **English only.** The expansion lives in the English G2P, so it also fixes
> piper's EN voices. German, French and Spanish have their own G2P front ends
> and still drop numeric tokens — `82` phonemizes to nothing there. Each needs
> its own number grammar (German compounds "zweiundachtzig" rather than reading
> the digits in order), so this is not a shared routine.

### Phoneme dialects — one G2P, several models

The G2P emits **espeak-style IPA** (`tʃ`, `oʊ`, `ɜː`, length marks), which is
what piper expects. Kokoro was trained on [misaki](https://github.com/hexgrad/misaki),
its own G2P, whose alphabet differs: `ʧ`, `O`, `ɜɹ`, and **no length marks at
all**. Both spellings exist in Kokoro's vocabulary, so feeding it the espeak
forms neither errors nor drops anything — the model just receives tokens it was
never trained on and drifts, which is the "sounds British / old English" of #316
(`ː` is the RP length mark).

CrispASR converts per backend (`src/core/phoneme_dialect.h`):

| Dialect | Spelling | Used by |
|---|---|---|
| `EspeakIpa` | `tʃ`, `oʊ`, `ɜː`, `ɾ`, `ɚ`, length marks | piper — and the G2P's own output |
| `Misaki` | `ʧ`, `O`, `ɜɹ`, `T`, `əɹ`, no length marks | kokoro |

`CRISPASR_KOKORO_MISAKI_IPA=0` restores the raw espeak spelling for A/B.

### Phoneme parity with misaki

Alphabet alone is not enough: CrispASR's CMUdict-based G2P and misaki disagree
about stress and unstressed vowels, so kokoro was fed pronunciations it was not
trained on even after the spelling was fixed. Five further pieces close that gap,
each measured against misaki 0.9.4:

| stage | 10k frequency list | 400-sentence prose corpus |
|---|---|---|
| CMUdict + alphabet conversion | — | 81.6% |
| **+ misaki's own lexicon** (Tier 0) | — | 88.5%\* |
| **+ morphological fallback** (`-s`/`-ed`/`-ing` from the stem) | — | 98.1%\* |
| **+ contextual function words** (`the`/`to`/`a`) | — | 90.8% |
| **+ Unicode punctuation** in the tokenizer | — | 93.0% |
| **+ capitalisation stress + phrase-final variants** | **99.30%** | **99.12%** |

\* measured on a harder obscure-word corpus at that stage.

Three of those deserve a note because they are not obvious:

- **Capitalisation, not grammar, drives stress.** misaki derives it from the
  token's case (`lowercase` → the lexicon form, `Titlecase` → insert secondary
  stress, `ALLCAPS` → primary): `and` / `And` / `AND` → `ænd` / `ˌænd` / `ˈænd`.
  No part-of-speech tagger is involved.
- **`the`/`to`/`a` depend on the FOLLOWING word.** `the apple` is `ði`, `the box`
  is `ðə`; `to open` is `tʊ`, `to walk` is `tə`. Reading the citation form
  everywhere is the "old English" diction of #316.
- **misaki's `'None'` lexicon key is not a POS tag** — it is the phrase-final
  reading, chosen when nothing follows (`…is she?` → `ʃˌi`). 32 entries.

What is left needs a POS tagger and is deliberately not guessed: `that` wants
`ðˈæt` as a determiner but `ðæt` as a conjunction, and `DEFAULT` is measurably
the better single choice (68% vs 31%). Same for `used`, `read`, `desert`.

### Driving the phonemes directly (`--tts-phonemes`)

```bash
crispasr --tts "unused" --tts-phonemes "həlˈO wˈɜɹld" --backend kokoro \
         -m kokoro-82m-q8_0.gguf --voice kokoro-voice-af_heart.gguf --tts-output out.wav
```

Synthesizes the phoneme string verbatim, skipping the G2P. This is the seam
between text processing and the acoustic model: feeding another
implementation's phonemes through our model separates "our G2P is wrong" from
"our model is wrong" in a single run — which is exactly how #316 was diagnosed.
kokoro only; other backends refuse rather than silently synthesizing the text,
so an A/B cannot be misread.

The `--g2p-dict` flag selects the dictionary source:

```bash
# Default: pre-generated IPA dicts (piper-compatible, auto-download)
crispasr --backend piper -m auto --tts "Hello world"

# Use CMUdict + ARPAbet→IPA conversion instead (76% piper match)
crispasr --backend piper -m auto --g2p-dict cmudict --tts "Hello world"

# Use OLaPh MIT dicts (British IPA conventions)
crispasr --backend piper -m auto --g2p-dict olaph --tts "Hello world"

# Use your own dictionary file
crispasr --backend piper -m auto --g2p-dict /path/to/my/dict.txt --tts "Hello world"
```

The phonemization cascade tries in order:
1. Pre-generated IPA dict (99.5% piper-compatible) — auto-download
2. CMUdict + ARPAbet→IPA conversion (EN) / OLaPh dict (DE/FR/ES)
3. LTS letter-to-sound rules — always available, zero dependencies
4. espeak-ng via dlopen (loaded at runtime if installed)
5. espeak-ng via popen (subprocess fallback)

Override per-language dict paths with env vars:
`CRISPASR_CMUDICT_PATH`, `CRISPASR_DE_DICT_PATH`,
`CRISPASR_FR_DICT_PATH`, `CRISPASR_ES_DICT_PATH`.

### Kokoro G2P strategy (`CRISPASR_KOKORO_G2P`)

Kokoro's phonemization order can be controlled via the
`CRISPASR_KOKORO_G2P` environment variable:

| Value | Behavior |
|-------|----------|
| `builtin-first` | Built-in G2P first, espeak fallback (default) |
| `espeak-first` | espeak-ng first, built-in fallback |
| `espeak-only` | espeak-ng only, no built-in G2P |
| `builtin-only` | Built-in G2P only, no espeak-ng |

The built-in G2P includes text normalization for common technical tokens
(`C++` → "C plus plus", `C#` → "C sharp", `.NET` → "dot net", etc.).
If you encounter mispronunciations with the built-in path, try
`CRISPASR_KOKORO_G2P=espeak-first` to prefer espeak-ng when available.

Dictionary sources at [cstr/g2p-dicts](https://huggingface.co/datasets/cstr/g2p-dicts):
- **Pre-generated IPA** (primary): piper-compatible phonetic transcriptions for EN/DE/FR/ES
- **CMUdict** (BSD): [cmusphinx/cmudict](https://github.com/cmusphinx/cmudict), English ARPAbet
- **OLaPh** (MIT): [iisys-hof/olaph](https://github.com/iisys-hof/olaph), 13 languages

## Kokoro — multilingual, smallest

Kokoro is the 82 M-param StyleTTS2-derived model. It does not
currently support `-m auto`; drop the GGUFs into a directory of your
choice (`~/.cache/crispasr/` works) and pass explicit paths.

```bash
# English — uses the official Kokoro-82M with the bundled af_heart voice.
./build/bin/crispasr \
    --backend kokoro \
    -m ~/.cache/crispasr/kokoro-82m-f16.gguf \
    --voice ~/.cache/crispasr/kokoro-voice-af_heart.gguf \
    --tts "Hello, how are you today?" -l en \
    --tts-output hello.wav

# German — pass `-l de` and the CLI auto-routes:
#   1. If kokoro-de-hui-base-f16.gguf sits next to kokoro-82m-f16.gguf,
#      the German-trained backbone (dida-80b/kokoro-german-hui-
#      multispeaker-base, Apache-2.0; HUI corpus CC0) is loaded instead
#      of the official one.
#   2. If --voice is omitted, a per-language fallback voice is picked
#      from <model_dir>/kokoro-voice-<name>.gguf in the cascade
#      df_victoria → df_eva → ff_siwis. Drop any of these into the
#      model directory; the first that exists wins.
./build/bin/crispasr \
    --backend kokoro \
    -m ~/.cache/crispasr/kokoro-82m-f16.gguf \
    --tts "Guten Tag, dies ist ein Test des deutschen Phonemizers." \
    -l de --tts-output guten_tag.wav
```

| Voice (German) | Source | License | Roundtrip on the test phrase (parakeet-v3, -l de) |
|---|---|---|---|
| `dm_martin` | [`kikiri-tts/kikiri-german-martin`](https://huggingface.co/kikiri-tts/kikiri-german-martin) | Apache-2.0 | "...Phonemizers." (perfect) |
| `df_victoria` | [`kikiri-tts/kikiri-german-victoria`](https://huggingface.co/kikiri-tts/kikiri-german-victoria) | Apache-2.0 | "...Tester des Deutschen Phonemizers." (1 word boundary err) |
| `dm_bernd` | Tundragoon (recovered from `r1di/kokoro-fastapi-german`'s Git LFS) | Apache-2.0 | "...Phonemetzers." (1 word boundary err) |
| `df_eva` | Tundragoon (recovered from `r1di/kokoro-fastapi-german`'s Git LFS) | Apache-2.0 | "...Phonemetzes." (1 word boundary err) |

All four voices clear the gate (peak ≥ 8000, RMS ≥ 1000) on the
dida-80b backbone — see `PLAN.md` §56 Option 2b for the full
methodology. The `crispasr_kokoro_resolve_*_abi` C ABI in
`src/kokoro.h` exposes the same routing logic to wrappers; from
Python it surfaces as
`crispasr.kokoro_resolve_for_lang(model_path, lang)` returning a
`KokoroResolved(model_path, voice_path, voice_name, backbone_swapped)`
record.

Kokoro supports runtime speaking-rate control via the session API:

| Setter | Default | Purpose |
|---|---|---|
| `set_length_scale(s)` | 1.0 | Per-phoneme duration multiplier. `> 1.0` = slower; `< 1.0` = faster. Clamped to `[0.25, 4.0]`. |

### Kokoro environment switches

| Variable | Default | Effect when set |
|---|---|---|
| `CRISPASR_KOKORO_GEN_GPU` | unset | Route the iSTFTNet generator (the vocoder) onto the main GPU backend instead of the Metal-hang-workaround CPU pin. Use on CUDA / Vulkan where the stride-10 ConvTranspose1d M1 hang doesn't apply and CPU vocoder is the bottleneck. Mirrors `CRISPASR_QWEN3_TTS_CODEC_GPU`. |
| `CRISPASR_KOKORO_GEN_FORCE_METAL` | unset | Same effect as `CRISPASR_KOKORO_GEN_GPU`, but the name carries the original "reproduce the M1 hang" debug intent. Kept for back-compat; new deployments should prefer `CRISPASR_KOKORO_GEN_GPU`. |

## Qwen3-TTS — voice cloning, highest fidelity

Speech-LLM (talker + code predictor + 12 Hz codec). Needs both a
talker GGUF and a codec / tokenizer GGUF. With `-m auto` both are
pulled into `~/.cache/crispasr/` on first run (Q8_0 talker + F16
codec by default).

```bash
# Zero-setup: auto-downloads talker + codec + default voice pack (~1.3 GB):
./build/bin/crispasr \
    --backend qwen3-tts -m auto \
    --tts "Hello there" \
    --tts-output hello.wav

# Runtime WAV clone — supply your own reference:
./build/bin/crispasr \
    --backend qwen3-tts -m auto \
    --voice samples/qwen3_tts/clone.wav \
    --ref-text "Okay. Yeah. I resent you. I love you. I respect you. But you know what? You blew it! And thanks to you." \
    --tts "Hello there" \
    --tts-output hello.wav

# F16 reference baseline (1.83 GB talker; strict-fidelity):
./build/bin/crispasr \
    --backend qwen3-tts \
    -m ~/.cache/crispasr/qwen3-tts-12hz-0.6b-base.gguf \
    --voice samples/qwen3_tts/clone.wav \
    --ref-text "Okay, yeah. I resent you, I love you, I respect you. But you know what - You blew it, and thanks to you." \
    --tts "Hello there" \
    --tts-output hello.wav

# Baked voice-pack GGUF (skips the WAV+ref-text step):
./build/bin/crispasr \
    --backend qwen3-tts -m auto \
    --voice my-voice.gguf \
    --tts "Hello there" \
    --tts-output hello.wav

# Larger 1.7B talker (~2.07 GB Q8_0 / ~3.86 GB F16; same ICL contract):
./build/bin/crispasr \
    --backend qwen3-tts-1.7b-base -m auto \
    --voice samples/qwen3_tts/clone.wav \
    --ref-text "Okay, yeah. I resent you, I love you, I respect you. But you know what - You blew it, and thanks to you." \
    --tts "Hello there" \
    --tts-output hello.wav

# VoiceDesign — describe the voice in natural language. No reference WAV,
# no preset speaker. 1.7B-only (~1.9 GB Q8_0). Pass --instruct instead of
# --voice; the codec bridge omits the speaker frame and the description
# is prepended to the prefill as a `<|im_start|>user\n…<|im_end|>\n`
# block.
./build/bin/crispasr \
    --backend qwen3-tts-1.7b-voicedesign -m auto \
    --instruct "A young female voice with a slight British accent, energetic, slightly fast paced" \
    --tts "Hello, I'm an excited engineer." \
    --tts-output hello.wav
```

Notes:
- **No `--voice` needed**: `-m auto` downloads a baked default voice pack
  (`qwen3-tts-voice-default.gguf`) alongside the talker and codec so the
  Base model works out of the box. The default voice is auto-selected when
  no `--voice` flag is given and the GGUF sits next to the talker.
- When `--voice` points to a `.wav`, `--ref-text` is required. When it
  points to a `.gguf`, it is treated as a baked voice pack and
  `--ref-text` is ignored.
- With an explicit `-m`, the CLI auto-discovers the codec when
  `qwen3-tts-tokenizer-12hz.gguf` sits next to the talker; otherwise
  pass `--codec-model`.
- Quantization is **not** quality-equivalent across variants. The
  reference baseline is `f16` talker + `f16` codec. The recommended
  deployment quant is `q8_0` talker + `f16` codec — used by `-m auto`,
  ~986 MB, audibly indistinguishable from F16 on the test prompts in
  LEARNINGS.md. Lower-bit talker quants (`q6_k`, `q5_k`, `q4_k`)
  drift noticeably in strict tensor diffs. Quantizing the codec
  hurts earlier than quantizing the talker — keep
  `qwen3-tts-tokenizer-12hz.gguf` at `f16`.

### qwen3-tts environment switches

Diagnostic / experimental knobs. Leave them unset for normal use — the
defaults reproduce the validated, end-to-end-tested code path.

| Variable | Default | Effect when set |
|---|---|---|
| `CRISPASR_QWEN3_TTS_SEED` | `42` | Override the AR sampling seed (superseded by `--seed N` on the CLI). |
| `CRISPASR_QWEN3_TTS_MAX_FRAMES` | auto | Hard cap on AR decode steps, overriding the computed one. The default is the talker's KV ceiling (`1500` when unknown), then tightened per input to `max(240, codepoints × 12)` (#337) so a prompt that fails to sample `codec_eos` cannot run to a 4096-frame / 340 s "successful" synthesis. |
| `CRISPASR_QWEN3_TTS_O15` | unset | Pin code-predictor `Lk = cp_kv_max_ctx` and reuse one cached T=1 graph across AR steps 2..14 (saves ~14-19 ms/frame on Mac/Metal — alloc+build collapse from ~20 ms/frame to ~1.6 ms/frame). Default flipped back to OFF after [#56](https://github.com/CrispStrobe/CrispASR/issues/56): the cached-graph reuse asserts on the CUDA backend (`GGML_ASSERT` in `ggml_backend_tensor_set` on first `code_pred_generate_15` call, Jetson Orin AGX sm_87). M1 Metal users who want the speedup should set `CRISPASR_QWEN3_TTS_O15=1`. Default goes back to ON once the CUDA path is verified. Largely superseded by `CRISPASR_QWEN3_TTS_CP_DIRECT`, which gets the same graph-reuse win without touching the scheduler. |
| `CRISPASR_QWEN3_TTS_CP_DIRECT` | auto (GPU on, CPU off) | §232/#245: dispatch the code predictor through two persistent sched-free graphs (one T=2 step-0 graph, one T=1 step graph with the O15 topology), gallocr-allocated once on the code-pred backend. Each of the 15 per-frame steps is then just input `tensor_set` + one `ggml_backend_graph_compute` — no scheduler reset/alloc, so the sched-reuse breakage behind #56 cannot occur. Validated md5-identical WAV vs. the sched path on M1 Metal and CUDA P100 (0.6B Q8_0, seed 42; ASR roundtrip verbatim). Default ON when the code predictor runs on a GPU backend: Metal is ~equal on an idle box but ~3x faster under load (the sched path degrades badly under contention), CUDA P100 is ~11% faster. Default OFF on CPU, where there is no dispatch cost to save and the per-step lm_head slot blit (~2.2 MB memcpy ×14/frame) makes it ~2x slower. Set `=1`/`=0` to override either way; falls back to the sched path automatically when an op is unsupported on the backend or the code predictor is CPU-pinned. |
| `CRISPASR_QWEN3_TTS_LK_BUCKET` | unset | Bucketed fixed-`Lk` talker AR steps (256/512/1024/2048/4096), one persistent graph per bucket. Since §232 the bucket graphs are gallocr-allocated once and dispatched sched-free (the previous sched-plan reuse segfaulted on current ggml — nil-buffer inputs, same root cause as #56). Stays opt-in: on Metal the fixed-`Lk` attention costs ~10% at short outputs; on CUDA P100 it was the fastest config (~5% under CP_DIRECT alone) — CUDA users can enable both. |
| `CRISPASR_QWEN3_TTS_FUSED_QKV` | unset | Fuse Q+K+V weights into one matmul per talker layer at load time (F16/F32 talker only; auto-skipped for Q8_0/Q4_K). Bit-identical to the unfused path on M1 Metal; speed effect is machine-dependent. |
| `CRISPASR_QWEN3_TTS_BENCH` | unset | Print per-call build/alloc/compute/read timings for `talker_kv` and `code_pred_kv`. |
| `CRISPASR_QWEN3_TTS_PROF` | unset | Per-op profiler (more granular than `BENCH`). |
| `CRISPASR_QWEN3_TTS_CP_BACKEND` | unset | Pin the code predictor to a chosen backend. `cpu`, `cpu-f16`, `cpu-f32` keep its weights on the CPU backend — useful when isolating bugs to the talker vs. code-predictor or when comparing CPU and Metal end-to-end. |
| `CRISPASR_QWEN3_TTS_DUMP_DIR` | unset | Write per-frame intermediate tensors into the named directory. Bulky; intended for diff-harness work (`tools/dump_reference.py --backend qwen3-tts`). |
| `CRISPASR_QWEN3_TTS_CODEC_GPU` | auto | Force codec weights and decode through the GPU scheduler. GPU is now the default on all GPU backends including Metal — the `CONV_TRANSPOSE_1D` hang was fixed in `f8fc8b8e` and the op replaced by `mul_mat+col2im_1d` in `5f600f25`. Distinct from `CRISPASR_QWEN3_TTS_CODEC_FORCE_METAL`, which also enables a per-op trace callback for debugging. |
| `CRISPASR_QWEN3_TTS_CODEC_CPU` | unset | Force codec weights and decode through the CPU-only `codec_sched`. Useful for A/B timing and regression bisection. |
| `CRISPASR_QWEN3_TTS_CODEC_FASTCONV` | **on** (set `=0` to opt out) | §232: three codec conv rewrites — K=1 convs run as channel matmuls (the im2col of a 1×1 conv is a pure ~300 MB copy at 24 kHz T), K>1 causal convs pad inside im2col and crop (removes the CPU-placed asymmetric PAD nodes Metal can't run), and K>1 F16 conv kernels are baked to F32 once at load (removes a ~70 ms per-decode kernel cast). Validated 2026-07-11 on 0.6B Q8_0, seed 42: WAV **md5-identical** on M1 Metal; on CPU within 1 int16 LSB (PCM cos 1.00000000). Codec decode ~3× faster on Metal (3.9 s → 1.3 s for 4.6 s audio), ~2.1× on CPU. The `=0` path is the legacy graph, kept for regression bisection. |
| `CRISPASR_QWEN3_TTS_CODEC_CHUNK` | `150` (`64` on CUDA) | Maximum generated codec frames per decode chunk. CUDA clamps values above `64` and treats `0` as `64` unless `CRISPASR_QWEN3_TTS_CODEC_ALLOW_FULL=1` is also set, avoiding oversized `mul_mat+col2im_1d` allocations on 10 GB cards. |
| `CRISPASR_QWEN3_TTS_CODEC_CTX` | `128` (`96` on CUDA) | Left-context codec frames prepended to each chunk. Values below the codec sliding window are raised; CUDA clamps larger values unless `CRISPASR_QWEN3_TTS_CODEC_ALLOW_FULL=1` is set. |
| `CRISPASR_QWEN3_TTS_SKIP_REF_DECODE` | **on** (set `=0` to opt out) | Skip the codec decode of the reference audio in `qwen3_tts_synthesize`. The default-on path emits `codec_decode_codes(gen)` directly; the opt-out path concatenates `ref_codes + gen_codes`, decodes both, then trims the ref portion. With a 26 s reference (~334 codec frames at 12 Hz), the ref half adds ~16 s of constant codec compute regardless of how much new audio is generated (Jetson Orin AGX, issue #64). End-to-end RTF on Orin drops from ~7-9 → ~1.5; the win compounds N× under `/v1/audio/speech` long-form chunking. Bit-identity verified 2026-05-05 on Apple Silicon Metal, qwen3-tts-customvoice 0.6B Q8_0: max\|diff\| = 0, cosine similarity = 1.0 — equivalence holds because the codec is a straight-line forward pass with no rolling state. Set `CRISPASR_QWEN3_TTS_SKIP_REF_DECODE=0` only for A/B verification or if a future codec graph variant grows rolling state. |

### pocket-tts languages, voices, and environment switches

Pocket-TTS uses one checkpoint per language. With `-m auto`, the base backend
routes `-l de`, `es`, `it`, `pt`, or `fr` to the corresponding model; omit
`-l` (or use `-l en`) for English. The explicit backend names are
`pocket-tts-de`, `pocket-tts-es`, `pocket-tts-it`, `pocket-tts-pt`, and
`pocket-tts-fr`. French is Kyutai's 24-layer preview checkpoint; the other new
languages are the distilled 6-layer releases.

```bash
./build/bin/crispasr --backend pocket-tts -m auto -l es \
  --accept-license pocket-tts-terms \
  --tts "Hola, este modelo ya habla español." \
  --voice samples/jfk.wav --i-have-rights --tts-output pocket-es.wav
```

`--voice` accepts either reference audio or Kyutai's prepared voice states:

- **Absolute/relative path** — `--voice /path/to/ref.wav` (requires `--i-have-rights`).
- **Official embedding** — `--voice alba.safetensors`. Files from Kyutai's
  `embeddings_v3` directory are prefilled transformer K/V states, so they work
  even with a GGUF that omits the Mimi voice-cloning encoder. Match the
  embedding to the model language/layer count.
- **Bare name + `--voice-dir`** — `--voice alice --voice-dir voices/` resolves to
  `voices/alice.safetensors` when present, then `voices/alice.wav`. This is what
  `--server` / `{"voice": "<name>"}` requests use, so a single server can serve
  multiple voices from one directory (issues #255 and #411).
- **Unset** — auto-loads `samples/jfk.wav` as a default; without any voice the
  output is near-silent.

Prepared embeddings are preset identities, not a claim that the voice is safe
to impersonate. CrispASR retains its disclosure/marking warning; follow the
embedding publisher's license and personality-rights terms.

| Variable | Default | Effect when set |
|---|---|---|
| `CRISPASR_POCKET_MANUAL_MIMI` | unset | Force the CPU Mimi decoder path (bypass the ggml/GPU decode). |
| `CRISPASR_POCKET_MANUAL_BACKBONE` | unset | Force the CPU FlowLM backbone path. |
| `CRISPASR_POCKET_VULKAN_MIMI_MAX_FRAMES` | `120` | Vulkan-only guard (issue #256): fall back to the CPU Mimi decoder when a generation exceeds this many frames, avoiding the `maxComputeWorkGroupCount` abort on constrained iGPUs (e.g. AMD 780M / gfx1103). Set `<=0` to disable the guard and always attempt the GPU decode. No effect on non-Vulkan backends. |

## VibeVoice — realtime streaming TTS

Lowest-latency TTS engine. Uses `--voice` for its voice prompt or
preset; the realtime `0.5B` flow is typically driven by a voice GGUF.

```bash
# First run downloads ~636 MB to ~/.cache/crispasr/ (Q4_K talker + emma
# voice from cstr/vibevoice-realtime-0.5b-GGUF), then runs from cache.
./build/bin/crispasr \
    --backend vibevoice-tts -m auto \
    --tts "Hello, how are you today?" \
    --tts-output hello.wav
```

The realtime backend preserves the beginning of the sigma-VAE decoder output.
Older builds trimmed a fixed 100 ms warmup window, which could skip the clean
first decoded chunk and create a click by starting on a later waveform peak.
For parity debugging, `CRISPASR_VIBEVOICE_TTS_LATENTS=/path/to/latents.bin` can replay a
raw float32 latent stack, `CRISPASR_VIBEVOICE_TTS_DUMP=/dir` writes `tts_scaled_latent`
and `tts_raw_audio`, and `CRISPASR_VIBEVOICE_TTS_DUMP_DECODER=1` adds per-stage decoder
dumps.

## VibeVoice 1.5B — base TTS with WAV cloning

The 1.5B base model supports both a generic no-clone voice and WAV
reference cloning through `CRISPASR_VIBEVOICE_VOICE_AUDIO`.

```bash
# Generic output, no voice reference.
./build/bin/crispasr \
    --backend vibevoice-1.5b -m auto \
    --tts "Hello, how are you today?" \
    --tts-output hello.wav

# Clone from a 24 kHz mono WAV reference.
CRISPASR_VIBEVOICE_VOICE_AUDIO=samples/qwen3_tts/clone.wav \
./build/bin/crispasr \
    --backend vibevoice-1.5b -m auto \
    --tts "Hello, how are you today?" \
    --tts-output hello-clone.wav
```

## Orpheus — Llama-3.2-3B + SNAC codec

Llama-3.2-3B-Instruct talker emitting `<custom_token_N>` LM tokens
that SNAC decodes to 24 kHz PCM. 8 baked English speakers (`tara`,
`leah`, `jess`, `leo`, `dan`, `mia`, `zac`, `zoe`). The talker GGUF
and the SNAC codec live in two separate HF repos and download
together via `-m auto`.

The talker is derived from Llama 3.2. Use `--accept-license llama3.2` before
the first managed download and preserve the Llama 3.2 Community License and
its attribution/acceptable-use terms when redistributing it. The SNAC codec
is MIT-licensed.

```bash
# First run pulls ~3.5 GB (Q8_0 talker) + 26 MB (SNAC codec) into
# ~/.cache/crispasr/.  --temperature 0.6 is the upstream
# engine_class.py default — DO NOT skip it. Greedy (--temperature 0)
# enters a 7-slot loop after a few super-frames and produces unusable
# audio.
./build/bin/crispasr \
    --backend orpheus -m auto \
    --voice tara --temperature 0.6 \
    --tts "Hello, my name is Tara." \
    --tts-output hello.wav
```

Drop-in DE checkpoint variants are shipped: pass
`--backend kartoffel-orpheus-de-natural` for a 19-speaker German
fine-tune trained on natural speech recordings,
`--backend kartoffel-orpheus-de-synthetic` for a 4-speaker variant
with explicit emotion + outburst control (`Martin - Sad: Oh, ich
bin so traurig.`), or `--backend lex-au-orpheus-de` for lex-au's
German Q8_0 mirror. All three reuse the same orpheus runtime + SNAC
codec.

## Chatterbox — flow-matching TTS, voice cloning + multilingual

ResembleAI's chatterbox is a two-GGUF pipeline: **T3** (AR text →
speech tokens) and **S3Gen** (flow-matching tokens → 24 kHz audio
via Conformer encoder + UNet1D CFM denoiser + HiFTGenerator vocoder).
The default voice is baked into the T3 GGUF (`conds.*` tensors); a
reference WAV switches into voice-cloning mode through the VoiceEncoder
LSTM + CAMPPlus x-vector.

```bash
# English base — auto-download pulls T3 + S3Gen (~880 MB) on first run.
./build/bin/crispasr \
    --backend chatterbox -m auto \
    --tts "Hello there, this is chatterbox speaking." \
    --tts-output out.wav
```

Four variants share the same runtime — the architecture flag in the
T3 GGUF metadata switches between the Llama-T3 path (base/lahgtna)
and the GPT-2-T3 path (turbo/kartoffelbox-turbo):

```bash
# Distilled English (350 M, 2-step meanflow S3Gen — faster than base):
./build/bin/crispasr --backend chatterbox-turbo -m auto --tts "..." --tts-output out.wav

# Nano (#382): GPT2-small T3 (12L/768, ~345 MB Q8) on the SAME Turbo S3Gen —
# the registry auto-downloads the Turbo S3Gen as companion:
./build/bin/crispasr --backend chatterbox-nano -m auto --tts "..." --tts-output out.wav

# Finnish Nano v0.1.3 (#382 reporter fine-tune; same Turbo S3Gen companion):
./build/bin/crispasr --backend chatterbox-finnish-nano -m auto -l fi \
    --tts "Hyvää huomenta. Tämä on Chatterbox Finnish Nano." --tts-output out-fi.wav

# German fine-tune of Turbo (SebastianBodza/Kartoffelbox_Turbo):
./build/bin/crispasr --backend kartoffelbox-turbo -m auto -l de \
    --tts "Hallo, das ist Kartoffelbox-Turbo." --tts-output out-de.wav

# Arabic Llama-T3 fine-tune (oddadmix/lahgtna-chatterbox-v1):
./build/bin/crispasr --backend lahgtna-chatterbox -m auto -l ar \
    --tts "مرحباً" --tts-output out-ar.wav
```

### Multilingual language selection

The base `chatterbox` backend uses the upstream multilingual V3 T3 checkpoint
`t3_mtl23ls_v3.safetensors` at pinned source revision
`5bb1f6ee58e50c3b8d408bc82a6d3740c2db6e18`. It is paired with upstream's
production `s3gen.pt` weights (converted from their tensor-equivalent
`s3gen.safetensors`), not the retired `s3gen_v3` experiment. The explicit
`chatterbox-v3-*` filenames prevent an older generic artifact from being
mistaken for this pair. Pass `-l <code>` / `--language <code>` to
select the language token for multilingual synthesis:

```bash
./build/bin/crispasr --backend chatterbox -m auto -l fr \
    --tts "bonjour tout le monde" \
    --tts-output out-fr.wav
```

The flag is wired into the T3 prompt, not concatenated into the spoken text.
With the rebuilt 2026-06-18 GGUFs, `-l fr` inserts the `[fr]` token after
`[START]` (token id 634 in the multilingual tokenizer) and changes the
generated speech-token stream. A local Q4_K smoke check with seed 123 showed
that no-language `bonjour tout le monde` roundtripped through Parakeet as
`Bonjour tout monde.`, while `-l fr` roundtripped as
`Bonjour tout le monde.`.

Quality is still model-dependent. The rebuilt artifacts fix the previous
tokenizer/model mismatch and make `-l` active, but some French Q4_K samples
remain heavily accented. Treat language-token checks as a wiring smoke test,
not a guarantee of native pronunciation.

For voice cloning, pass `--source-lang <code>` when the reference recording's
language differs from the requested output language (`-l` or `--target-lang`).
Chatterbox then follows the upstream V3 cross-lingual recommendation and uses
CFG weight 0 unless an explicit `--tts-cfg-scale` override is supplied. This keeps
the reference speaker's timbre while reducing transfer of the reference
language's accent. In server/session use, set the output language with the
target/source language setter and the recording language with
`set_tts_reference_language()`.

On the multilingual path the text is **NFKD-normalized** (then ASCII-lowercased)
before tokenization, matching upstream `MTLTokenizer.preprocess_text`. This
matters for scripts with precomposed diacritics: e.g. Arabic `أ`
(ALEF-WITH-HAMZA) decomposes to base alef + combining hamza, the form the model
was trained on. Without it, partial-diacritic Arabic produced spurious onset
letters (#170). Script-specific normalizers (zh cangjie / ja kakasi / he dicta /
ko jamo / ru stress) are not yet implemented.

> **Note on legacy GGUFs.** Some older multilingual T3 artifacts pair a
> 2352-token tokenizer with a 2454-vocab T3; the loader rejects that mismatch.
> Repair locally with `models/patch-chatterbox-gguf-add-merges.py` and the
> matching `grapheme_mtl_merged_expanded_v1.json` (2454 tokens).

### Performance

The compute-bound T3 AR decode is the slow stage. It runs on CPU by default on
Metal (GPU has higher per-step kernel-launch overhead for the many T=1 steps).
The CPU thread count defaults to `min(8, hardware_concurrency)`; override with
`CRISPASR_CHATTERBOX_THREADS=<n>` (e.g. dial down on a heavily shared host).
Output is bit-identical regardless of thread count.

On **Vulkan** the T3 GPT-2 (turbo/nano) attention uses the explicit
softmax(QK^T)V path by default instead of `ggml_flash_attn_ext`: the Vulkan
flash-attention pipeline crashes on RADV (Radeon 780M, issue #402) while the
explicit path completes full syntheses there with both F16 and Q4_K weights.
Opt back into flash on Vulkan with `CRISPASR_CHATTERBOX_FLASH_ATTN=1`;
`CRISPASR_CHATTERBOX_NAIVE_ATTN=1` still forces the explicit path on every
backend (debug gate).

### Voice cloning

Two paths are supported. **The recommended path is the python baker
+ baked GGUF** — it's the workflow the upstream chatterbox project
ships, our parity is exact, and the C++ runtime treats the resulting
GGUF the same way it treats the built-in default voice. The native
24 kHz WAV path described below the baker is functional but
experimental — it ships its own caveats (see "Known issues" later).

**Step 1 — bake the voice GGUF (one-time per reference speaker):**

```bash
# Requires the upstream chatterbox-tts python package (pip install
# chatterbox-tts) or RESEMBLE_CHATTERBOX_SRC=/path/to/clone/src for a
# local source checkout. The model loads on CPU by default; pass
# --device mps / cuda for faster baking. Reference WAV can be any
# sample rate / channel count — the baker resamples to 16 kHz for
# the VoiceEncoder + S3Tokenizer paths and 24 kHz for the prompt mel.
python models/bake-chatterbox-voice-from-wav.py \
    --input samples/jfk.wav \
    --output my_voice.gguf \
    --i-have-rights
```

`--i-have-rights` is required — baking is the cloning step, and everything
downstream just replays the pack. The baker stamps
`crispasr.voice.cloned_from_recording` into the output so the runtime's
consent gate and Art. 50(4) spoken disclosure recognise the `.gguf` as a
clone at synthesis time. Chatterbox has **no `.wav` cloning path**, so
before that stamp existed a baked voice could never trip either gate; see
[`eu-ai-act.md`](eu-ai-act.md#62-art-504--deepfake-disclosure). A pack baked
by an older CrispASR carries no stamp and reads as a preset — re-bake it.

The baker runs upstream `ChatterboxTTS.prepare_conditionals(wav)` and
writes five tensors plus two scalar metadata keys, using the same
names the runtime already accepts for the built-in default voice
(`conds.t3.{speaker_emb, speech_prompt_tokens}`,
`conds.gen.{prompt_token, prompt_feat, embedding}`,
`chatterbox.conds.{emotion_adv, gen_prompt_token_len}`). Output
size is ~150-200 KB regardless of reference WAV length.

**Step 2 — synthesise with the baked voice:**

```bash
./build/bin/crispasr --backend chatterbox -m auto \
    --voice my_voice.gguf \
    --tts "Cloned voice synthesising arbitrary text." \
    --tts-output cloned.wav \
    --i-have-rights
```

`--voice` is per-call cached, so server callers (`--server` mode) can
switch voices between requests without reloading on every synthesise.

**Direct `--voice <path>.wav` — native cloning, no python required**
(experimental). The C++ runtime now runs the full VoiceEncoder +
S3Tokenizer V2 + CAMPPlus + 24 kHz Matcha mel pipeline in-process and
forks on the input sample rate:

- **24 kHz mono PCM16/F32 WAV** — atomic clone. Resamples 24 → 16 kHz
  via a Kaiser-windowed sinc polyphase resampler, then computes all
  five conds (`speaker_emb`, `speech_prompt_tokens`, `gen.prompt_token`,
  `gen.prompt_feat`, `gen.embedding`) from the same source audio and
  installs them together. ASR roundtrip on `samples/jfk.wav`
  (resampled to 24 kHz) with prompt "Ask not what your country can
  do for you." returns the prompt verbatim — the cloned voice path
  works end-to-end.
- **16 kHz mono PCM16/F32 WAV** — **NOT a real clone**. Only the
  T3-side conds (`speaker_emb`, `speech_prompt_tokens`) are
  installed; S3Gen renders with the **default voice's** `gen.*`
  bundle. The output sounds like the default voice, not the
  reference speaker. The path exists as a stepping stone in the
  module ladder; for actual voice cloning, use the 24 kHz WAV
  branch above OR the python baker (recommended). Re-encode the
  reference at 24 kHz mono (`ffmpeg -i in.* -ar 24000 -ac 1 ref.wav`)
  to get a real clone.

**Known issues for the native path**:
- **Default backend split is hardware-dependent.** On Metal (Apple
  Silicon) the default is full CPU because (a) T3's 30-layer × 86-step
  AR loop is dominated by Metal kernel-launch overhead — measured 50 s
  on CPU vs 75 s on T3-GPU + S3Gen-CPU on M1 — and (b) S3Gen UNet1D
  on Metal has compound per-op precision drift across mul_mat / FA /
  norm / add / gelu / tanh / softplus that the 10-step CFM Euler
  solver amplifies ~1000× into the documented `s3gen_mel cos≈0.92`
  collapse. On non-Metal GPU builds (CUDA / Vulkan), the default keeps
  T3 on GPU and S3Gen on CPU — the S3Gen-on-GPU drift has only been
  bisected on Metal but the same compound-precision class likely
  applies to CUDA wmma / Vulkan cooperative-matrix paths with F16
  intermediate state, so the safer S3Gen-CPU default ships on every
  GPU backend.
- **Env knobs to override the default:**
  - `CRISPASR_CHATTERBOX_FULL_CPU=1` — force everything to CPU.
  - `CRISPASR_CHATTERBOX_T3_GPU=1` — opt T3 back into GPU on Metal
    (useful for benchmarking on M3+ where tensor cores may flip the
    balance, or to verify correctness on CUDA-without-S3Gen).
  - `CRISPASR_CHATTERBOX_FORCE_GPU=1` — put T3 *and* S3Gen on the GPU
    backend. Output is garbled on Metal (vocoder amplifies drifted /
    NaN mel into saturated audio). Kept as a diagnostic.
  - `CRISPASR_S3GEN_UNET_CPU=1` / `CRISPASR_S3GEN_ENCODER_CPU=1` /
    `CRISPASR_S3GEN_VOCODER_CPU=1` — pin individual S3Gen sub-graphs
    to CPU when the parent context uses GPU. Diagnostic: the
    documented Metal scheduler "upgrade to higher-priority backend"
    pass undoes user-set CPU assignments for host-mapped buffers, so
    these don't fully isolate sub-graphs on unified-memory devices.
  - `CRISPASR_S3GEN_UNET_PIN_CPU_OP=<op>` /
    `CRISPASR_S3GEN_UNET_KEEP_GPU_OP=<op>` — op-type bisect (handover
    round 7). Pins the named op to CPU (or keeps only that op on GPU)
    inside `build_graph_unet1d`. Useful for localising the next
    suspect op in further drift work; doesn't fix end-to-end TTS
    audio on its own because the vocoder still amplifies whatever
    mel the UNet produces.
- **F16 vs Q-mat weights:** F16 mul_mat is bit-identical CPU↔GPU on
  every quant. Quantised weights (Q4_K/Q5_K/Q6_K/Q8_0) need the
  CrispASR ggml-metal patches (`kernel_mul_mv_q4_K_q8_K`,
  `kernel_quantize_q8_K_f32`, `kernel_mul_mm_*_hp`) plus
  `GGML_PREC_F32` op tagging to reach the F32-precise path on Metal.
  T3 carries those tags (see `chatterbox.cpp:build_graph_t3_kv`);
  S3Gen tagging was tested and helped mul_mat alone but didn't break
  the compound-drift chain, so it isn't currently applied. F16 T3 +
  CPU S3Gen is the safest config if you need T3 on GPU and care
  about exact-match output on Metal.
- T3 sampling can produce unrelated text on long technical prompts
  (sampler drift). Short, common phrases work reliably; if a prompt
  produces gibberish, try a different seed via `--seed <n>` (or the
  legacy env `CRISPASR_CHATTERBOX_T3_SEED=<n>`).

The parity-quality compute kernels are bit- or fp32-rounding-tight
against PyTorch — verified via `crispasr-diff chatterbox` on the
`ve_*`, `s3tok_*`, `campplus_fbank`, `campplus_xvector`, and
`prompt_feat_24k` stages. End-to-end output may drift from the
python baker due to the resampler differing slightly from librosa's
`kaiser_fast`; for perfect baker-equivalent cloning the
`models/bake-chatterbox-voice-from-wav.py` workflow remains
recommended.

If the WAV is not 16 kHz mono PCM16/F32, the runtime falls back to
the same hint-then-error path as before, pointing at the baker or
suggesting `ffmpeg -i in.* -ar 16000 -ac 1 ref.wav`.

The same `my_voice.gguf` works across all four chatterbox variants
(`chatterbox`, `chatterbox-turbo`, `kartoffelbox-turbo`,
`lahgtna-chatterbox`) since the cond tensor contract is shared.

**Optional: `--exaggeration`** is baked into the voice at conversion
time via `--exaggeration <float>` (default `0.5`); pass a different
value to the baker to produce a more / less expressive variant of
the same speaker. The C++ runtime reads
`chatterbox.conds.emotion_adv` from the loaded voice GGUF, so the
flag is honored without further wiring.

Companion sharing — the registry deliberately points multiple variants
at the same S3Gen file to avoid redundant downloads. Kartoffelbox-turbo
and chatterbox-turbo share the meanflow S3Gen verbatim; lahgtna and
chatterbox-base share the original S3Gen. Pulling any variant first
warms the cache for the rest.

| Variant | T3 default | S3Gen companion | Total |
|---|---|---|---:|
| `chatterbox`         | T3 Q8_0 (610 MB)  | base S3Gen Q8_0  (348 MB) | ~960 MB |
| `chatterbox-turbo`   | T3 F16  (963 MB)  | turbo S3Gen F16  (627 MB) | ~1.6 GB |
| `kartoffelbox-turbo` | T3 Q8_0 (623 MB)  | turbo S3Gen F16  (shared)  | ~1.25 GB |
| `lahgtna-chatterbox` | T3 F16  (1059 MB) | base S3Gen Q8_0  (shared)  | ~1.4 GB |

Sampling controls:

| Flag | Default | Purpose |
|---|---|---|
| `--temperature` | runtime default 0.8 | AR sampling temperature (0 = greedy; runtime falls back to 0.8 when global default 0.0) |
| `--seed N` | 0 (non-deterministic) | RNG seed — same seed + same text = bit-identical audio |
| `--tts-steps N` | 10 (base/lahgtna) / 2 (turbo/kartoffelbox-turbo meanflow) | CFM Euler steps for the S3Gen mel-decoder |
| `--codec-model FNAME` | sibling autodetect | Explicit S3Gen GGUF path (overrides `-m auto` companion) |

Session-API knobs (runtime-settable via the session API, not CLI flags):

| Setter | Default | Purpose |
|---|---|---|
| `set_top_p(p)` | 1.0 | Top-p nucleus-sampling threshold for the AR T3 token loop |
| `set_min_p(p)` | 0.05 | Min-p sampling threshold |
| `set_repetition_penalty(r)` | 1.2 | Repetition penalty (1.0 = no penalty; > 1 discourages repeated tokens) |
| `set_cfg_weight(w)` | 0.5 | Classifier-free-guidance weight. 0 = unconditional; 0.5 = upstream default |
| `set_exaggeration(e)` | 0.5 | Emotion-exaggeration scalar. Raise for dramatic delivery, lower for monotone |
| `set_max_speech_tokens(n)` | 1000 | Upper bound on AR speech tokens per call (≈ 20 s at 50 Hz codes) |
| `set_tts_steps(n)` | (see CLI table) | Same as `--tts-steps`; settable without reloading the session |

**Quantized variants** (Q8_0, Q4_K) are supported — the
`crispasr-quantize` tool skips vocoder, F0-predictor, and embedding
tensors automatically (see [docs/quantize.md](quantize.md)). Turbo
size table for the alternate quants:

| Variant | T3 | S3Gen | Total |
|---|---:|---:|---:|
| Turbo F16  | 964 MB | 628 MB | 1,592 MB |
| Turbo Q8_0 | 629 MB | 350 MB |   979 MB |
| Turbo Q4_K | 457 MB | 245 MB |   702 MB |

The Conformer rel-pos parity gap that previously affected the C++
encoder closed in §80 (5 fixes: PE ordering, pos_bias_u/v transpose,
missing up_layer.conv, missing xscale-after-up_embed, attention
output head layout). encoder_out is now bit-exact to the Python
reference.

## Parler TTS — prompt-conditioned voice description

Parler TTS Mini v1.1 is a prompt-conditioned TTS model (~900M params):
T5 encoder processes a natural-language voice description, MusicGen-style
decoder generates audio codes, DAC codec decodes to 44.1 kHz PCM.
Apache-2.0 license. No reference audio or voice packs needed --- describe
the voice you want in text via `--instruct`.

```bash
# Auto-download (~900 MB Q8_0 GGUF on first run):
./build/bin/crispasr --backend parler-tts -m auto \
    --instruct "A female speaker with a warm voice in a quiet room." \
    --tts "Hello, this is a test of Parler TTS." \
    --tts-output output.wav --seed 42

# Explicit model path:
./build/bin/crispasr --backend parler-tts \
    -m parler-tts-mini-v1.1-q8_0.gguf \
    --instruct "A young male speaker with an energetic tone." \
    --tts "Welcome to CrispASR text-to-speech." \
    --tts-output welcome.wav
```

The `--instruct` flag sets the voice description. If omitted, a default
description ("A female speaker with a warm, clear voice in a quiet room.")
is used. Output is 44.1 kHz mono PCM. Temperature (default 1.0) and seed
are supported for reproducible / diverse generation.

**Model file:**
[`cstr/parler-tts-mini-v1.1-GGUF`](https://huggingface.co/cstr/parler-tts-mini-v1.1-GGUF)

## TTS GGUF downloads

[`cstr/vibevoice-realtime-0.5b-GGUF`](https://huggingface.co/cstr/vibevoice-realtime-0.5b-GGUF) ·
[`cstr/vibevoice-1.5b-GGUF`](https://huggingface.co/cstr/vibevoice-1.5b-GGUF) ·
[`cstr/qwen3-tts-0.6b-base-GGUF`](https://huggingface.co/cstr/qwen3-tts-0.6b-base-GGUF) ·
[`cstr/qwen3-tts-1.7b-base-GGUF`](https://huggingface.co/cstr/qwen3-tts-1.7b-base-GGUF) ·
[`cstr/qwen3-tts-1.7b-voicedesign-GGUF`](https://huggingface.co/cstr/qwen3-tts-1.7b-voicedesign-GGUF) ·
[`cstr/qwen3-tts-tokenizer-12hz-GGUF`](https://huggingface.co/cstr/qwen3-tts-tokenizer-12hz-GGUF) ·
[`cstr/orpheus-3b-0.1-ft-GGUF`](https://huggingface.co/cstr/orpheus-3b-0.1-ft-GGUF) ·
[`cstr/kartoffel-orpheus-3b-german-natural-GGUF`](https://huggingface.co/cstr/kartoffel-orpheus-3b-german-natural-GGUF) ·
[`cstr/kartoffel-orpheus-3b-german-synthetic-GGUF`](https://huggingface.co/cstr/kartoffel-orpheus-3b-german-synthetic-GGUF) ·
[`cstr/snac-24khz-GGUF`](https://huggingface.co/cstr/snac-24khz-GGUF) ·
[`cstr/chatterbox-GGUF`](https://huggingface.co/cstr/chatterbox-GGUF) ·
[`cstr/chatterbox-turbo-GGUF`](https://huggingface.co/cstr/chatterbox-turbo-GGUF) ·
[`cstr/kartoffelbox-turbo-GGUF`](https://huggingface.co/cstr/kartoffelbox-turbo-GGUF) ·
[`cstr/lahgtna-chatterbox-v1-GGUF`](https://huggingface.co/cstr/lahgtna-chatterbox-v1-GGUF) ·
[`cstr/indextts-1.5-GGUF`](https://huggingface.co/cstr/indextts-1.5-GGUF) ·
[`cstr/parler-tts-mini-v1.1-GGUF`](https://huggingface.co/cstr/parler-tts-mini-v1.1-GGUF)

## F5-TTS — DiT flow-matching voice cloning

F5-TTS v1 Base is a DiT-based flow-matching TTS model with zero-shot
voice cloning from 3-15s of reference audio. MIT license. Architecture:
ConvNeXtV2 text encoder → 22-layer Diffusion Transformer with AdaLN-Zero
→ 32-step Euler ODE solver with CFG → Vocos iSTFT vocoder. Single GGUF
(~1.3 GB) containing both DiT and Vocos weights.

```bash
# Basic synthesis with voice cloning
./build/bin/crispasr --backend f5-tts -m auto \
    --voice samples/jfk.wav \
    --ref-text "Ask not what your country can do for you" \
    --tts "Hello, how are you today?" \
    --tts-output hello.wav --seed 42

# Without voice cloning (requires ref audio for now)
# F5-TTS always needs a reference audio + transcript pair.
```

The `--ref-text` flag provides the transcript of the reference audio.
This is required for F5-TTS (unlike indextts which conditions on audio
only). Output is 24 kHz mono PCM.

**Tokenization — English and Chinese (#294).** The model uses a 2545-entry
character/pinyin vocab. English (and other Latin text) tokenizes per character.
Chinese text is converted to pinyin the same way the upstream model expects
(`convert_char_to_pinyin`): a built-in g2p (`src/core/pinyin_g2p.*`) does
phrase-based segmentation → `pypinyin`-style TONE3 syllables (e.g. `zhong1`) with
tone-sandhi (不/一/third-tone), embedded from `pypinyin` via
`tools/gen_pinyin_data.py`. Token parity vs the reference is ~99.7% on the
`tests/test-pinyin-g2p` corpus; the residual is segmentation-boundary polyphones
that need full jieba (not embedded). Before #294 Chinese produced no audio (every
Han char hit the unknown token). Mixed English+Chinese in one sentence works.

**Output length (#294).** F5 estimates the number of mel frames from the
reference's speech rate (`ref_frames / ref_text_len`) × the generated text
length. A slow or expressive reference must not be truncated: the rate guard is
asymmetric (a generous upper bound, matching upstream which has none — an
over-estimate is just trailing silence that gets trimmed), so long sentences are
no longer cut off. `CRISPASR_F5_DURATION_CLAMP=0` restores the exact upstream
formula. Give an accurate `--ref-text`; a wrong/short reference transcript
distorts the rate.

**Model file:**
[`cstr/f5-tts-GGUF`](https://huggingface.co/cstr/f5-tts-GGUF)

### Performance (issue #294)

The whole cost is the ODE denoising loop — 32 steps × 2 (CFG) = **64 full 22-layer
DiT passes** over the ref+gen sequence. The vocoder and text encoder are <1%.
**Quantization is not an option** — flow matching accumulates per-op error 1408×
per synthesis, so anything below F16 is unintelligible (F16 is the only viable
format; see the model README). All speed comes from *fewer / smaller passes*:

| Lever | Speedup | How | Quality |
|-------|---------|-----|---------|
| **`--tts-steps 7`** | **~4.7×** | EPSS non-uniform step schedule (built in for n=5/6/7/10/12/16) | verified intact; drop to 10/12 (~3×) if a voice sounds rough |
| `--tts-steps 16` | ~2.0× | fewer uniform-ish steps | intact |
| `CRISPASR_F5_CFG_INTERVAL=2` | ~1.3× | reuse the unconditional CFG velocity between steps | intact at ≥16 steps |
| shorter reference clip (3–5 s) | scales with T | the DiT denoises ref+gen jointly, so a long ref inflates *every* pass (attention is O(T²)) | intact |
| `CRISPASR_F5_BATCH_CFG=1` | GPU-dependent | one 2×-batch CFG forward (matches upstream) instead of two; try on CUDA | intact |
| `CRISPASR_F5_EMBED_GPU=1` | ~1.3× (more on slow-CPU) | run the InputEmbedding (input_proj + conv-pos) on the GPU instead of the CPU; removes the per-step CPU stall | intact (byte-identical gate-off; roundtrip-identical gate-on). GPU builds only |
| `CRISPASR_F5_DIT_SKIP=2` | ~1.9× | DiTReducio temporal skip — reuse the cached step velocity every other step | intact at 32 steps; ≈ using `--tts-steps 16`, so use one or the other |

Stacking: `~4 s ref` + `--tts-steps 7` (+ `CRISPASR_F5_EMBED_GPU=1` on a GPU) compound.
**Do not** combine a low step count with `CFG_INTERVAL` or `DIT_SKIP` — too few
non-uniform steps + a stale/reused velocity degrades to noise (the runtime warns
below 16 steps). `EMBED_GPU` is orthogonal and stacks with any step setting.

`CRISPASR_F5_BENCH=1` prints the per-stage split (host_embed / dit_graph / vocos).
`-nfa` has no effect on F5 — the DiT always uses flash attention.

## IndexTTS — Chinese/English voice cloning

IndexTTS-1.5 is a zero-shot voice cloning TTS model. Given a short
reference WAV (~3-10 s), it reproduces the speaker's voice for arbitrary
text. Architecture: Conformer+Perceiver conditioning encoder → GPT-2
autoregressive mel-code generator (24 layers, 1280d, beam search) →
BigVGAN vocoder (24 kHz).

```bash
# Auto-download (~2.4 GB: GPT F16 + BigVGAN F16)
./build/bin/crispasr --backend indextts -m auto \
    --tts "Hello world, this is a test." \
    --voice reference_speaker.wav \
    --tts-output cloned.wav

# Explicit paths
./build/bin/crispasr --backend indextts \
    --model indextts-gpt.gguf \
    --codec-model indextts-bigvgan.gguf \
    --tts "Hello world." \
    --voice reference_speaker.wav \
    --tts-output hello.wav
```

The `--voice` flag points to any mono WAV file (16 kHz or 24 kHz) of the
target speaker. Longer clips (5-10 s) give better cloning fidelity.

The BigVGAN vocoder runs anti-aliased SnakeBeta on the CPU by default
(BigVGAN v2's upsample→activate→downsample sandwich; the raw activation
emits harmonics above Nyquist that fold back as audible click/buzz). On
M1 this adds ~5 % to the vocoder stage versus the aliased path. Two env
knobs for power users:

- `CRISPASR_INDEXTTS_VOCODER_RAW=1` — opt out of AA; fully GPU-graphable but
  produces ~2 k impossible inter-sample jumps on speech. Use only for
  reproducing the legacy / aliased benchmark.
- `--no-gpu` — keeps the whole vocoder graph on CPU. Recommended for
  IndexTTS specifically: the GPT codes generate quickly either way, and
  the AA custom op forces a GPU↔CPU sync per AMP block when mixed with
  Metal, leaving GPU + AA the slowest of the four combinations.

Set `CRISPASR_INDEXTTS_DEBUG=1` for per-stage intermediate dumps (mel, conformer
blocks, perceiver output) useful for diff-testing against Python.

### IndexTTS Chinese text normalization

**Default in-process pipeline** handles ~95 % of real prompts:

- A port of upstream `tokenize_by_CJK_char` (every CJK codepoint is
  whitespace-surrounded so SentencePiece emits one `▁<char>` piece per
  hanzi, matching the model's training distribution).
- The relevant subset of `TextNormalizer.char_rep_map` — full-width
  CJK punctuation `，。：；！？、…“”‘’（）《》【】「」—～·` mapped to ASCII
  before the CJK splitter runs. Notably `。` (U+3002) sits **inside**
  the CJK Unicode range used by `tokenize_by_CJK_char`, so it must be
  punct-mapped first or it gets split as a CJK character and the
  model hallucinates an extra trailing syllable.
- ASCII upper-case for non-CJK runs (matches upstream's
  `do_upper_case=True`).

Token IDs are bit-identical to Python `sentencepiece.SentencePieceProcessor.Encode`
on the preprocessed string. Mixed CJK+English
(`他用Python写了一个程序。`) works without special flags. Pass `-v` to
print the preprocessed string and token count (`indextts: text "..." ->
N tokens`) if output sounds wrong; the full BPE id dump
(`indextts: text_ids[...]`) needs verbosity ≥ 2, which only the library /
session API reaches — the CLI adapter pins verbosity to 1.

**ASR-validate Chinese output with a real Chinese ASR.** `whisper-base`
over-counts CER ~5× on Mandarin (we measured 21 % whisper-base vs
3.8 % `qwen3-asr-0.6b` on the same audio); use Qwen3-ASR, the upstream
Cohere transcribe API, or `whisper-large-v3` for CER numbers you can
trust.

**What the default does NOT do:** number → hanzi (`2025年` stays as
literal digits, which the model can't pronounce cleanly), pinyin tone
digits (`xuan4` is not restored to its real pronunciation),
English contractions inside Chinese text, dates / times / currency /
phone-number expansion. These need a full WFST-based normalizer
(upstream IndexTTS uses [wetext](https://github.com/pengzhendong/wetext)
on macOS, [WeTextProcessing](https://github.com/wenet-e2e/WeTextProcessing)
elsewhere). We deliberately don't vendor that engine — OpenFST + the
~1 MB of compiled `.fst` rule data is a heavy dependency for a feature
most TTS prompts don't need.

**Optional hook: `CRISPASR_INDEXTTS_TEXT_NORMALIZER`** delegates normalization
to any user-provided shell command that reads UTF-8 text on stdin and
writes the normalized text to stdout. The output is then run through
the default pipeline (CJK split + punct map + upper) before
SentencePiece.

Recommended setup with upstream wetext:

```bash
pip install wetext
# Then on every IndexTTS invocation:
CRISPASR_INDEXTTS_TEXT_NORMALIZER="python $REPO/tools/wetext-normalize.py" \
    ./build/bin/crispasr --backend indextts -m auto \
        --tts "我有 3 个苹果，2025 年买的。" \
        --voice ref.wav --tts-output out.wav
```

The wrapper at `tools/wetext-normalize.py` is a thin stdin → stdout
shim around `wetext.Normalizer`; it accepts wetext's normal flags
(`--lang zh`, `--traditional-to-simple`, `--remove-erhua`, etc.).
Without `wetext` installed, the wrapper passes input through
unchanged so the hook still degrades gracefully.

**Failure modes — all fall back silently to the raw text:**

- Hook command exits non-zero → warning to stderr, raw text used.
- Hook stdout is empty → raw text used.
- Hook isn't installed / `mkstemp` fails → warning, raw text used.
- Hook hangs → currently no timeout; the synthesis blocks until the
  subprocess exits (no protection against a buggy normalizer). Use
  with shell commands you trust.

**Security:** the env var IS passed to `system()` — the user is the one
setting it, so don't expose this hook to text inputs from untrusted
sources unless you also vet the env var.

When the hook fires you'll see no extra log line by default; pass
`-v` to confirm the post-hook tokenization (`indextts: text "..." ->
N tokens`).

---

## Irodori-TTS — Japanese voice cloning + emoji emotion control

Irodori-TTS v3 (500M) is an RF-DiT flow-matching model at 48 kHz via the
Semantic-DACVAE-Japanese-32dim codec. Needs the companion codec GGUF next to the
model (auto-discovered) or via `--codec-model`.

```bash
# Plain synthesis:
crispasr --backend irodori-tts -m irodori-tts-500m-v3-q8_0.gguf \
    --codec-model dacvae-ja-32dim-f16.gguf --tts "こんにちは、世界。" --tts-output out.wav

# Zero-shot voice cloning from any reference WAV:
crispasr --backend irodori-tts -m irodori-tts-500m-v3-q8_0.gguf \
    --codec-model dacvae-ja-32dim-f16.gguf \
    --voice reference.wav --i-have-rights --tts "テスト。" --tts-output cloned.wav

# VoiceDesign (600M-v3): style/emotion control via --instruct:
crispasr --backend irodori-tts -m irodori-tts-600m-v3-voicedesign-q4_k.gguf \
    --codec-model dacvae-ja-32dim-f16.gguf \
    --instruct "落ち着いた大人の男性。深く響く声で丁寧に話している。" \
    --tts "こんにちは、世界。" --tts-output voicedesign.wav
```

**Voice cloning** encodes the reference through the DAC-VAE encoder (resample →
−16 LUFS → latent) and conditions the DiT via speaker CFG. `--i-have-rights` is
the consent attestation; a spoken AI-disclosure is prepended unless
`--no-spoken-disclaimer`. A short, clean 5–15 s reference clones as well as a
long one and is faster.

**Emoji emotion control** — Irodori's emoji drive prosody (👂 whisper, 😮‍💨
breath, 😭 crying, …); include them in the text. See the model's
`EMOJI_ANNOTATIONS.md` for the supported set. Emoji outside the trained set are
harmlessly ignored.

**Output length** is set by the model's duration predictor (kanji unpack to a
variable number of mora, so a fixed chars/sec heuristic truncates). Pin the exact
frame count with `CRISPASR_IRODORI_T_LATENT=N`; the runtime's `duration_scale`
multiplier has no CLI flag today (the CLI adapter leaves it at 1.0), so the env
var is the only lever from the command line.

**Knobs (env):**

| Variable | Effect |
| --- | --- |
| `CRISPASR_IRODORI_CFG_SPEAKER` | Speaker-CFG strength for cloning (default 5.0). |
| `CRISPASR_IRODORI_T_LATENT` | Force the exact output latent-frame count. |
| `CRISPASR_IRODORI_DECODE_CHUNK` / `_CTX` | Overlap-save codec-decode window / context (auto for long outputs; `CHUNK=0` disables). Bounds peak decode memory; exact (byte-identical) output. |
| `CRISPASR_IRODORI_CODEC_GPU=1` | Run the DAC-VAE codec on the GPU under Vulkan (CPU by default there — validated clean on MoltenVK; confirm on your driver). `CRISPASR_IRODORI_CODEC_CPU=1` forces CPU. |

See also the shared [reference-conditioning cache](#reference-conditioning-cache)
and [streaming TTS output](streaming.md#streaming-synthesized-audio-out).

## Reference-conditioning cache

Voice cloning encodes the reference into a small conditioning blob — a DAC-VAE
latent (irodori) or Conformer/Perceiver + ECAPA conditioning (indextts) — which
is slow for long references. CrispASR caches it **content-addressed on the
reference audio**, in the runtime, so **every entry point** (CLI, server, C ABI,
language wrappers) skips the encode on a repeat reference automatically — output
is byte-identical to a fresh encode.

- Cache directory: `CRISPASR_TTS_REF_CACHE_DIR` (default: a `crispasr-tts-refcache`
  folder under the system temp dir).
- Disable entirely with `CRISPASR_TTS_REF_CACHE=0`.

No flag is needed to enable it — it's on by default. (f5-tts caches its
auto-transcribed reference transcript separately, next to the voice file.)

---

## Local speaker output (`--tts-play`)

Pass `--tts-play` to play TTS output through the local speaker immediately
after synthesis, in addition to (or instead of) writing a file. The
spread-spectrum watermark is embedded before playback (unless disabled via
`--no-watermark` / `CRISPASR_NO_WATERMARK`), so the audio leaving the speaker
carries the provenance marker.

```bash
# Synthesize and play through default speaker
crispasr --tts "Hello world." --tts-play -m model.gguf

# Write file AND play
crispasr --tts "Hello world." --tts-play -m model.gguf --tts-output output.wav

# Select a non-default output device (miniaudio playback device index)
crispasr --tts "Hello world." --tts-play --tts-play-device 2 -m model.gguf
```

Playback is synchronous — the CLI blocks until audio drains, then exits.
Device -1 (the default) selects the system default output device.

**Implementation note:** the device is opened at the hardware-native sample
rate (`sampleRate=0 / channels=0`). The model's mono float32 PCM is
pre-resampled via linear interpolation before the device starts. This avoids
miniaudio's 4× upsampler artefacts on devices that run natively at 96 kHz
(MacBook Air Speakers and many Core Audio devices).

## AI-generated audio provenance & watermarking

All TTS output is marked as AI-generated by default through multiple
complementary layers. The waveform watermark is on by default and can be turned
off by an operator who takes on the marking responsibility themselves (see
[Disabling the watermark](#disabling-the-watermark-operator-opt-out) below); the
file-metadata tags are always written.

### Spread-spectrum watermark (built-in, on by default)

A frequency-domain watermark embedded in the PCM signal after synthesis.
Survives re-encoding, volume normalization, and moderate compression.
The embedder writes only a ramped watermark delta back into the signal and
leaves under-covered FFT boundary samples untouched, so quiet starts/ends do
not become click impulses.

```bash
# Detect watermark in any audio file (C API)
crispasr_watermark_detect(pcm, n_samples)  # returns confidence 0..1
```

### AudioSeal neural watermark (optional upgrade)

Meta's AudioSeal (MIT) provides stronger robustness via a learned
SEANet encoder-decoder. 100% cosine parity with the PyTorch reference.

```bash
# Convert model (requires pip install audioseal gguf)
python3 models/convert-audioseal-to-gguf.py -o audioseal.gguf

# Use with TTS
crispasr --tts "hello" -m kokoro.gguf --watermark-model audioseal.gguf

# Or let the registry fetch it: "auto" / "default" resolves the AudioSeal
# GGUF (auto-downloading when --auto-download is on). An unloadable model
# falls back to the built-in spread-spectrum watermark rather than failing.
crispasr --tts "hello" -m kokoro.gguf --watermark-model auto --auto-download

# Debug: CRISPASR_AUDIOSEAL_DEBUG=1 for shape traces, CRISPASR_AUDIOSEAL_DUMP_STAGES=1 for binary dumps
```

### Disabling the watermark (operator opt-out)

The waveform watermark is on by default. It can be turned off two ways:

- **CLI flag**: `--no-watermark` — **requires `--accept-marking-responsibility`**,
  else the run is refused (exit 12) with a `[MARKING]` audit line when it is
  honoured. The same gate covers `--no-spoken-disclaimer` and `--no-c2pa`, on
  the CLI and at server startup.
- **Env var**: `CRISPASR_NO_WATERMARK=1` — takes effect without the attestation
  flag, so it is the escape hatch for embedders, not the documented CLI path.

Either one disables watermark embedding for the whole process and logs, once:

```
crispasr: warning: watermarking disabled. AI usage marking responsibility rests with the operator.
```

**The opt-out has a floor.** When the chosen output container cannot carry a
C2PA manifest (raw `.aac` / `.opus` with `CRISPASR_NO_C2PA_REMUX=1`, or a build
with C2PA compiled out) the watermark is the only machine-readable mark left, so
`--no-watermark` is overridden and the mark is kept, with a note naming the file.

**Rationale.** Disabling the mark does not remove any AI-disclosure obligation
that may apply to the generated audio — it transfers responsibility for meeting
it to whoever runs the binary. The warning text is deliberately
jurisdiction-neutral and names no statute at runtime, because the marking
obligation is jurisdiction-specific (e.g. EU AI Act Art. 50, and analogous US
state and other rules) and CrispASR is a tool/component, not the "provider" that
those rules bind. A downstream operator that ships synthetic media is that
provider. Keeping the mark on is the zero-config default so the compliant path
is the path of least resistance; the opt-out exists for legitimate cases
(content not represented as authentic, research/testing, or an operator that
substitutes its own compliant marking scheme). See
[`docs/issue-260/PLAN.md`](issue-260/PLAN.md) for the full background.

> **Scope.** The flag/env opt-out applies to the `crispasr` CLI and the server
> (process-wide — there is no per-request toggle). It does **not** affect the
> C-API/language bindings: `synthesize()` always watermarks. Binding consumers
> that need unwatermarked PCM call `synthesize_raw()` and skip
> `watermark_embed()`, thereby assuming the marking responsibility themselves
> (see [`bindings.md`](bindings.md)).

### File metadata (always active)

- **WAV**: `LIST`/`INFO` chunk with `ISFT="CrispASR (AI-generated audio)"` and `ICMT` notice
- **MP3**: ID3v2 `TXXX` frames: `AI_GENERATED=true`, `GENERATOR=CrispASR`

### C2PA Content Credentials

Cryptographically-signed provenance manifests with
`digitalSourceType=trainedAlgorithmicMedia` (a C2PA `c2pa.created` action),
embedded directly in the output file. This is the "signed metadata" layer that
complements the waveform watermark.

**Build.** WAV signing needs **no build flags and no external library** — it is
handled by the built-in native signer (`src/core/crispasr_c2pa_native.h`),
a pure-C++ implementation of C2PA (hand-built CBOR / JUMBF / COSE_Sign1, ES256
via the vendored BSD-2 micro-ecc + a header-only SHA-256). It compiles into
`crispasr-lib` on every platform with zero dependencies. Its output validates
in the c2pa-rs reference reader. To additionally sign **MP3/M4A/FLAC**, fetch the
optional c2pa-rs library:

```bash
./scripts/fetch-c2pa.sh          # downloads the prebuilt c2pa-rs lib → third_party/c2pa
# then cmake reconfigure; look for "C2PA signing enabled" at configure time
```

(Set `-DCRISPASR_NO_C2PA_NATIVE=ON` to drop the built-in signer, e.g. a build
that wants only the c2pa-rs path.)

**On by default (self-signed), on every platform.** Output is signed
automatically — no flags needed. Signing uses a fixed self-signed certificate
**baked into the binary** (`crispasr_c2pa_default_cert.h`), so it works
identically on desktop, mobile, and in the **WASM browser sandbox** (no
filesystem or openssl needed at runtime). Self-signed manifests are valid and
machine-readable (EU AI Act Art. 50); C2PA verifiers show "unverified signer".
The bundled private key is intentionally public — it only marks content as
AI-generated, it is not a trust anchor. (Regenerate with
`scripts/gen-default-cert-header.sh`.)

```bash
crispasr --tts "hello" -m kokoro.gguf --tts-output out.wav   # signed automatically
crispasr --detect-watermark out.wav                          # (or verify at contentcredentials.org)
```

**Bring your own cert** for a *trusted* signer identity (a CA-issued
code-signing cert; verifiers then show the named issuer):

```bash
./scripts/generate-c2pa-cert.sh   # or use your CA-issued cert + key
crispasr --tts "hello" --c2pa-cert crispasr-c2pa.crt --c2pa-key crispasr-c2pa.key
```

**Format support.** **WAV** (RIFF), **MP3** (ID3v2 GEOB), **M4A/MP4** (ISO BMFF,
`c2pa.hash.bmff.v3`), and **FLAC** (ID3v2 GEOB prepend) are **all** signed by the
built-in native signer (the `c2pa-audio` submodule) — **c2pa-rs is no longer
needed for any audio container**; `fetch-c2pa.sh` is optional/legacy.

**AAC and Opus** have no C2PA embedding path in their raw streaming containers
(ADTS / Ogg) — and neither does c2pa-rs. So when C2PA is active, CrispASR
**muxes AAC/Opus into an MP4 container** (`.aac` → `.m4a`, `.opus` → `.mp4`, via
the in-tree glint encoder + `crispasr_mp4_writer.h`) and signs that natively.
Explicit `.m4a`/`.mp4` output is always AAC-in-MP4. Set
`CRISPASR_NO_C2PA_REMUX=1` to keep the raw `.aac`/`.opus` container (watermark +
file-metadata tag only). The muxed MP4 output validates in the c2pa-rs reference
reader and plays in standard players.

**Bindings / mobile.** C2PA lives in the core C API (`crispasr_c2pa_sign` /
`crispasr_c2pa_free`, and `c2paSign()` in the wasm/JS binding), so any consumer
of `crispasr-lib` can sign — pass a WAV/MP3 container and get signed bytes back.
Build the library for the target with `-DCRISPASR_C2PA_FETCH=ON`:
- **Android**: links `libc2pa_c.so`; bundle it in the APK `jniLibs/<abi>/` next to
  `libwhisper.so` (verified: the arm64 prebuilt links cleanly with the NDK).
- **iOS**: links `libc2pa_c.a` statically; `crispasr_enable_c2pa` auto-links the
  required Apple frameworks (Security / CoreFoundation / SystemConfiguration)
  (verified: the arm64 prebuilt links cleanly with the iOS SDK).
- **WASM** — WAV signing works out of the box (no `--c2pa`, no ~10 MB c2pa-rs).
  Three ways, in order of preference:
  1. **Native-JS signer (module-free).** `bindings/javascript/c2pa.mjs` +
     `c2pa-verify.mjs` — **pure WebCrypto** C2PA sign + verify (ECDSA P-256/ES256
     + SHA-256, hand-built canonical CBOR / JUMBF / COSE_Sign1) for WAV/MP3/M4A/
     FLAC — **no c2pa-rs, no wasm module at all**. These are **vendored from the
     `c2pa-audio` submodule** (`third_party/c2pa-audio/js/`, the single source of
     truth) — CMake copies them in at configure time; don't edit the copies. Runs
     in any browser, Node ≥16, Deno, or a Worker. Also on npm/pub.dev as
     `c2pa-audio` / `c2pa_audio`. Usage:
     ```js
     import { c2paSignWav } from 'crispasr/c2pa';
     const wav    = Module.pcmToWav(float32Pcm, 24000);          // interop WAV + AI tag
     const signed = await c2paSignWav(wav, certPem, keyPem);      // full C2PA manifest
     ```
     Covered by `npm test` in `bindings/javascript/` — 12 hermetic unit tests
     plus a live parity test through the c2pa-python reference reader.
  2. **Built-in native C++ signer via `c2paSign()`.** The native signer
     (`crispasr_c2pa_native`) compiles into every wasm build, so
     `Module.c2paSign(wavBytes, "audio/wav")` works **without `--c2pa`** and adds
     no c2pa-rs weight — use this if you already hold a wasm `Module`. Both 1 and
     2 emit the same manifest and validate in the c2pa-rs reference reader (only
     status `signingCredential.untrusted`, expected for the self-signed cert).
  3. **c2pa-rs in wasm (opt-in, MP3/M4A only)** via `./build-wasm.sh --c2pa` —
     adds ~10 MB (the full Rust stack), off by default. When enabled, the module
     is built with **`-fwasm-exceptions`** + **`-sSUPPORT_LONGJMP=wasm`** because the
     prebuilt c2pa-rs emscripten lib uses **native wasm exceptions** (it imports the
     `__cpp_exception` tag — provided only by native wasm EH, not the default
     no-exceptions runtime nor JS-based `-fexceptions`); needs a wasm-EH browser
     (all modern browsers, 2023+). Only needed for `c2paSign(bytes, "audio/mpeg")`
     and other non-WAV containers in the browser.

**VLC Playback Bug (Silence Padding).** Because the C2PA JUMBF metadata chunk is extremely large (often 2-3 KB), some audio players (notably **VLC**) stall their playback thread for ~1.5 seconds while parsing it from the end of the file before playing the stream. For short TTS clips, this causes the first 1-2 seconds of the generated speech to be silently dropped during playback.
To work around this while keeping the C2PA metadata intact, you can explicitly pad silence frames at the beginning of the audio. By the time the player unblocks, the silence has played out and the speech begins unharmed.
- **CLI:** `--tts-pad-silence-ms 1500`
- **C ABI:** `crispasr_session_set_tts_pad_silence_ms(session, 1500)`
- **HTTP Server:** Add `"pad_silence_ms": 1500` to the JSON request body.

### Voice cloning consent gate

Voice cloning (`.wav` reference files) requires explicit consent:

```bash
# CLI: --i-have-rights flag required
crispasr --tts "hello" --voice speaker.wav --i-have-rights

# Server: consent_attestation field required in JSON body
curl -X POST http://localhost:8080/v1/audio/speech \
  -d '{"input":"hello","voice":"speaker.wav","consent_attestation":"I have consent"}'
```

All consent attestations are logged with ISO 8601 timestamps.

### Post-embed watermark verification (automatic)

After writing a watermarked WAV in TTS mode, CrispASR automatically
reads back the in-memory PCM and runs watermark detection on it. If the
score lands in the `No watermark detected` band (see the verdict table
below), a warning is emitted to stderr. The self-check only runs for the
built-in spread-spectrum detector (not AudioSeal) and only on clips long
enough to score — **≥ 2.5 s** for the default per-frame statistic, **≥ 5 s**
for the legacy sign test — because the detector averages across frames and a
short clip reads near chance even when the embed worked. A bare `< 0.6` bar
used to warn on most sub-two-second clips. No extra flags needed; the
self-check is a diagnostic and never the marking gate (embedding is
unconditional).

### `--detect-watermark PATH` — standalone watermark detection

Reads a WAV file, runs watermark detection (spread-spectrum by default,
or AudioSeal if `--watermark-model` is given), prints the file, the detector
used, the analysed duration, the confidence score and a human-readable
verdict, then exits.

The default spread-spectrum statistic is the **per-frame** one (t-statistic +
decoy specificity); `CRISPASR_WATERMARK_DETECT=sign` selects the legacy
averaged-spectrum bin-sign test, which additionally prints the exact
probability of the score arising without a watermark.

| Detector | Score | Verdict |
|---|---|---|
| spread-spectrum, per-frame (default) | > 0.65 | `AI-GENERATED WATERMARK DETECTED` |
| spread-spectrum, per-frame (default) | > 0.5 – 0.65 | `INCONCLUSIVE - consistent with a watermark, but not statistically significant` |
| spread-spectrum, per-frame (default) | ≤ 0.5 | `No watermark detected (this does NOT mean the audio is human-made)` |
| spread-spectrum, sign (legacy) | p < 0.01 | `AI-GENERATED WATERMARK DETECTED` |
| spread-spectrum, sign (legacy) | p < 0.20 | `INCONCLUSIVE …` |
| spread-spectrum, sign (legacy) | otherwise (or score ≤ 0.5) | `No watermark detected …` |
| AudioSeal (neural) | > 0.5 | `AI-GENERATED WATERMARK DETECTED` |

AudioSeal returns a probability rather than a bin-agreement fraction, so no
p-value is quoted for it. Note that an unwatermarked file scores ~0.5 on the
spread-spectrum detector, not 0.

```bash
# Detect watermark using the built-in spread-spectrum detector
crispasr --detect-watermark output.wav

# Detect with AudioSeal neural watermark model
crispasr --detect-watermark output.wav --watermark-model audioseal.gguf
```

### Spoken disclaimer (voice clones only)

Voice-cloned output is automatically prefixed with "This audio was
generated by artificial intelligence" using a neutral default voice
(not the cloned voice), with a 300ms silence gap before the cloned audio.

The spoken disclaimer can be disabled per-request while keeping all
machine-readable provenance (watermark + C2PA) intact:

- **CLI**: `--no-spoken-disclaimer` — **requires `--accept-marking-responsibility`**,
  else the run is refused (exit 12).
- **Server**: `"spoken_disclaimer": false` in the request body — **requires a
  `"marking_attestation": "<text>"` field alongside it**, or a server launched
  with `--accept-marking-responsibility` (the operator accepts the duty for every
  response the process serves, which covers each request).

When the spoken disclaimer is suppressed, the caller assumes
responsibility for providing appropriate AI-disclosure to end users
(e.g. a visual label in the UI) — that is exactly what the attestation
records. The spread-spectrum watermark and C2PA metadata are always
embedded regardless of this setting.

**An unattested opt-out is denied, not refused** (#312, from v0.8.24). A server
request that sets `"spoken_disclaimer": false` without any attestation still gets
its audio — with the spoken disclaimer applied, i.e. the documented default. The
denial is announced, never silent:

```
X-Crispasr-Spoken-Disclaimer: applied
X-Crispasr-Marking-Warning:   'spoken_disclaimer': false ignored - it requires a
                              'marking_attestation' field (or a server launched with
                              --accept-marking-responsibility); served with the
                              spoken AI-disclaimer
```

plus a `[MARKING] … no_spoken_disclaimer=DENIED` line in the server log. Serving
the *stronger* default can't emit weaker-than-default output, and it doesn't hard-
break clients written before the field existed — v0.8.22/v0.8.23 returned
`400 marking_attestation_required` for that request instead, which took out voice
cloning for every Subtitle Edit build up to v5.1.0-rc16. The CLI keeps its hard
refusal: there the operator can just add the flag.

**Latency (by design).** The disclaimer is a *full synthesis pass* of the
neutral sentence on the loaded backend — not a pre-recorded clip — so it adds
latency proportional to that one fixed sentence. It is cached per process and
reused: a long-running **server** pays this cost once (on the first voice-clone
request) and not again, whereas the **CLI** re-synthesizes it on every
invocation because each run is a fresh process with a cold cache. For
continuous-latent AR backends such as dots.tts the disclaimer is a complete
autoregressive generation, so on a short clone it can roughly double the
wall-clock time of a single CLI call (a constant overhead that becomes
negligible for longer text and for repeated server requests). Pass
`--no-spoken-disclaimer` / `"spoken_disclaimer": false` to skip it when you
provide AI-disclosure another way — the watermark and C2PA provenance are
still embedded.
