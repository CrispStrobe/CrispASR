# Text-to-Speech (TTS)

CrispASR ships **fourteen open-weights TTS engines** behind the same
`crispasr` binary, each with a distinct voice / quality / footprint
trade-off:

| Backend | Why pick it | Voice cloning | First-run download |
|---|---|---|---|
| **`melotts`** | Multilingual VITS2 (MeloTTS). 4 English speakers (US/BR/India/AU). 44.1 kHz output, ~102 MB GGUF. Neural G2P + CMU dict. BERT companion (Q4_K 52 MB) auto-downloads with `-m auto`; also via `--codec-model` or `MELOTTS_BERT` env. | No (per-speaker ID) | ~154 MB via `-m auto` |
| **`piper`** | Tiniest footprint (30 MB). rhasspy/piper VITS; 250+ community voices across 30+ languages. Built-in G2P (CMUdict + LTS rules) for English — no espeak-ng needed. Optional espeak-ng for other langs (loaded via dlopen). 22 kHz output. Use `--g2p-dict` to select dictionary source. | No (per-voice GGUF) | Manual `wget` |
| **`kokoro`** | Smallest + fastest. 82 M-param StyleTTS2-derived model. Multilingual via built-in G2P or espeak-ng (dlopen/popen fallback). | No (preset voice packs) | Manual `wget` (no `-m auto`) |
| **`qwen3-tts`** | Highest fidelity / strongest cloning. Speech-LLM (talker + code predictor + 12 Hz codec). Default voice auto-downloaded with `-m auto`; or supply your own WAV + ref-text. | Optional (auto default voice; or WAV + ref-text or baked voice GGUF) | ~1.3 GB via `-m auto` |
| **`vibevoice-tts`** | Lowest-latency streaming TTS, designed for realtime. | Preset voice packs | ~636 MB via `-m auto` |
| **`vibevoice-1.5b`** | Base VibeVoice TTS model with WAV cloning. | Yes (`VIBEVOICE_VOICE_AUDIO=<wav>` or `--voice <wav>`) | ~1.6 GB via `-m auto` |
| **`orpheus`** | Llama-3.2-3B talker + SNAC 24 kHz codec. 8 baked English speakers; expressive output. Greedy loops — pass `--temperature 0.6`. | Preset names via `--voice tara/leah/...` | ~3.5 GB via `-m auto` (talker Q8 + 26 MB SNAC) |
| **`chatterbox`** | T3 AR + S3Gen flow-matching + HiFTGenerator. Built-in voice baked into the T3 GGUF; clones via a baked voice GGUF (see workflow below). EN/AR/DE variants share runtime. | Yes (`--voice <voice.gguf>`, baked from a WAV with `models/bake-chatterbox-voice-from-wav.py`) | ~880 MB via `-m auto` (T3 Q8 + S3Gen Q8) |
| **`outetts`** | OuteTTS-0.3-1B: OLMo-1B LLM + WavTokenizer single-codebook VQ-GAN. Lightweight (1B params), CC BY 4.0 license. 24 kHz output. | Yes (`--voice <speaker.json>`, created with `tools/reference_backends/outetts_create_speaker.py`) | ~2.5 GB via `-m auto` (talker F16 + WavTokenizer decoder) |
| **`f5-tts`** | F5-TTS v1 Base: 22-layer DiT flow-matching TTS + Vocos iSTFT vocoder. MIT license. High-quality zero-shot voice cloning from 3-15s reference audio. 24 kHz output, character-level tokenization. | Yes (`--voice <ref.wav> --ref-text "transcript"`) | ~953 MB via `-m auto` (single F16 GGUF, DiT + Vocos) |
| **`indextts`** | IndexTTS-1.5: GPT-2 AR (24L/1280d) mel-code generator + BigVGAN vocoder. Designed for Chinese+English. Zero-shot voice cloning from any reference WAV. | Yes (`--voice <ref.wav>`) | ~2.4 GB via `-m auto` (GPT F16 + BigVGAN F16) |
| **`cosyvoice3-tts`** | Fun-CosyVoice3-0.5B-2512: Qwen2-0.5B AR speech-token LM + DiT-CFM (10-step Euler) + HiFT (NSF + iSTFT) @ 24 kHz. 9 languages + 18 Chinese dialects. Ships an 8-voice baked bank (`zero_shot` + `fleurs-{en,de,zh,ja,fr,es,ko}`). | Yes — baked-bank name via `--voice <name>`, **or** native arbitrary-WAV cloning via `--voice <ref.wav> --ref-text "..."` (ports speech_tokenizer_v3 + CAMPPlus + matcha mel to ggml; speech tokens byte-exact vs ONNX). | ~1.2 GB via `-m auto` (Q4_K LLM + Q8_0 flow + HiFT + s3tok + campplus + voices) |
| **`csm`** | Sesame CSM-1B: Llama-3.2 1B backbone (first-codebook AR) + 100M depth decoder (codebooks 1–31) + Kyutai Mimi codec (32-codebook RVQ → SEANet) @ 24 kHz. Single GGUF. Apache-2.0. | No (single built-in voice) | ~1.4 GB via `-m auto` (single Q4_K GGUF) |
| **`dia`** | Nari Labs Dia 1.6B: byte-level text encoder (12L) + AR audio decoder (18L GQA) + 9-codebook DAC codec @ 44.1 kHz. CFG-guided, dialogue-style with `[S1]`/`[S2]` speaker tags. Apache-2.0. | No (dialogue via speaker tags) | ~1.6 GB via `-m auto` |
| **`zonos-tts`** | Zyphra Zonos-v0.1-transformer: 26-layer GQA AR transformer → 9-codebook DAC @ 44.1 kHz. Rich conditioning: speaker embedding + text + emotion + FWHM pitch/tempo. CFG guided. Voice cloning from any reference WAV (pass via `ZONOS_SPEAKER_EMB_PATH` or `--voice <ref.wav>`). Apache-2.0. | Yes (`--voice <ref.wav>`) | ~1.6 GB Q8_0 (default) or ~931 MB selective-Q4_K (heads/embeddings kept F16, auto-retry guard) or ~3.0 GB F16, via `-m auto` + 104 MB DAC codec. |
| **`bark`** | Suno Bark: 3-stage GPT-2 (text→semantic→coarse→fine) + EnCodec 24 kHz decoder. All sub-models packed into one GGUF. Supports speaker conditioning via `.npz` prompts. MIT license. | Yes (`--voice <speaker.npz>`) | ~423 MB via `-m auto` (selective Q4_K) |
| **`speecht5`** | Microsoft SpeechT5 80M: char-level encoder (12L) + AR mel decoder (6L) + 5-conv postnet + HiFi-GAN @ 16 kHz. MIT. Speaker via 512-d x-vector. | Yes (`--voice <xvector.bin>`, raw float32) | ~300 MB via `-m auto` (F16 GGUF) |
| **`fastpitch`** | NVIDIA FastPitch 60M: non-autoregressive parallel TTS — 6L FFTransformer encoder + duration/pitch predictors + length regulator + 6L FFTransformer decoder + HiFi-GAN @ 22 kHz. Deterministic (no sampling). CC-BY-4.0. | No (single speaker) | ~230 MB via `-m auto` (Q8_0 GGUF) |
| **`parler-tts`** | Parler TTS Mini v1.1 (~900M): T5 encoder + MusicGen decoder + DAC 44.1 kHz. Apache-2.0. Prompt-conditioned: describe the voice in natural language via `--instruct`. | No (prompt-conditioned) | ~900 MB via `-m auto` (Q8_0 GGUF) |
| **`voxcpm2-tts`** | VoxCPM2: 2B Qwen2 backbone + flow matching + BigVGAN @ 48 kHz (decimated to 24 kHz). Zero-shot voice cloning via `--voice <ref.wav>`. | Yes | ~2.4 GB via `-m auto` |
| **`pocket-tts`** | Kyutai Pocket TTS 100M: continuous-latent AR @ 12.5 Hz + one-step LSD flow head + Mimi VAE decoder → 24 kHz. MIT / CC-BY-4.0. Voice cloning via `--voice ref.wav`. | Yes (`--voice`) | ~220 MB via `-m auto` (F16 GGUF) |
| **`kugelaudio`** | KugelAudio-0-Open: 7B Qwen2.5 backbone + 4-layer DiT diffusion head (20-step SDE-DPMSolver++) + acoustic VAE decoder → 24 kHz. 23 languages. MIT. | Pre-encoded voices (`--voice voice.gguf`) | ~5.3 GB Q4_K / ~16 GB F16 via `-m auto` |
| **`tada-1b`** | HumeAI TADA 1B: Llama-3.2-1B backbone + per-token flow-matching diffusion head + TADA codec → 24 kHz. **English-only.** `-m auto` downloads model + default `tada-ref.gguf`. | Yes (`--voice <tada-ref.gguf>`, English voice refs only) | ~1.7 GB Q4_K + ~1 GB codec |
| **`tada` / `tada-3b-ml`** | HumeAI TADA 3B Multilingual: same architecture, 3B params. Supports **ar, ch, de, es, fr, it, ja, pl, pt** in addition to English. `-l <lang>` auto-downloads `tada-ref-<lang>.gguf`. | Yes (`--voice <tada-ref.gguf>`) | ~4 GB Q4_K + ~1 GB codec |
| **`lfm2-audio`** | LiquidAI LFM2.5-Audio 1.5B: FastConformer encoder + LFM2 hybrid conv+attention backbone + 6L depthformer (8-codebook Mimi) + ISTFT detokenizer → 24 kHz. Interleaved text+audio generation. Also does ASR and speech-to-speech. LFM Open License v1.0 ($10M revenue cap). | No | ~1.5 GB Q4_K (JP) / ~1.6 GB Q5_K (EN) + ~157 MB detokenizer companion |
| **`mini-omni2`** | gpt-omni/mini-omni2: Whisper-small encoder + Qwen2-0.5B LLM with 8-stream architecture + SNAC 24 kHz decoder → 24 kHz. Also does ASR and speech-to-speech. MIT license. Requires `--codec-model snac-24khz.gguf` companion. | No | ~1.0 GB Q4_K + ~80 MB SNAC companion |

All backends write mono WAV via `--tts-output` (22 kHz for piper/fastpitch, 16 kHz for speecht5, 24 kHz for most others, 44.1 kHz for melotts/dia/parler-tts/zonos-tts, 48 kHz for voxcpm2-tts).

## TADA — multilingual and voice cloning

TADA ships as two variants; only the 3B multilingual model can synthesise
non-English text:

| Backend | Model | Languages |
|---|---|---|
| `tada-1b` | HumeAI/tada-1b | English only |
| `tada` / `tada-3b-ml` | HumeAI/tada-3b-ml | en + ar, ch, de, es, fr, it, ja, pl, pt |

### Use case (a) — built-in default voice for a language

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

### Use case (b) — custom voice cloning

To speak in a specific person's voice, bake a ref GGUF from ~10 s of their
speech. Two options — pure C++ (no Python) or the Python converter:

#### Option 1: C++ `--make-ref` (no Python needed)

Place `tada-encoder-f16.gguf` and `tada-aligner-en.gguf` (from
[cstr/tada-encoder-GGUF](https://huggingface.co/cstr/tada-encoder-GGUF))
next to your TADA model GGUF, then:

```bash
# Step 1 — bake the ref GGUF (one-time per speaker):
crispasr --backend tada-3b-ml -m auto \
    --make-ref \
    --voice speaker_10s.wav \
    --ref-text "Exact words spoken in the audio." \
    --make-ref-output tada-ref-custom.gguf

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
Pre-converted GGUFs are on HuggingFace at
[cstr/tada-encoder-GGUF](https://huggingface.co/cstr/tada-encoder-GGUF):

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

### Timing quality (`TADA_NUM_CANDIDATES`)

TADA predicts each token's duration with a per-token flow-matching head that
is **noise-sensitive**: an unlucky noise draw can collapse durations into a
rushed, unintelligible utterance (a known property of the model — the PyTorch
reference behaves identically with the same noise). To make output robust,
CrispASR generates several flow-matching candidates per token and keeps the
best one by reconstruction likelihood (the same `num_acoustic_candidates`
ranking the reference implements).

The CLI defaults to **4 candidates**. Override with `TADA_NUM_CANDIDATES`:

```bash
# Fastest, single noise draw (may occasionally rush/garble timing):
TADA_NUM_CANDIDATES=1 crispasr --backend tada-3b-ml -m auto -l fr \
    --tts "Bonjour, comment allez-vous ?" --tts-output out.wav

# Higher quality / more robust timing (slower):
TADA_NUM_CANDIDATES=8 crispasr --backend tada-3b-ml -m auto -l de \
    --tts "Guten Tag, wie geht es Ihnen?" --tts-output out.wav
```

All candidates for a step are solved in a single batched flow-matching forward,
so raising the count adds little wall-clock on top of the (model-load-dominated)
baseline. `1` reproduces a single draw and is the fastest.

The same **default of 4** applies through the session C ABI, so the bindings
and HTTP server get robust timing out of the box. Bindings can override it at
runtime with `set_tts_num_candidates(n)` (Python/Go/Rust/Ruby),
`SetTtsNumCandidates` (C#/Java), or `setTtsNumCandidates` (Dart/JS) — and the
`TADA_NUM_CANDIDATES` env var is honoured by every consumer, not just the CLI.

### Talker text sampling (`TADA_TEMPERATURE`, `TADA_TOP_P`, …)

`TADA_NUM_CANDIDATES` tunes only the **duration** flow-matching head, not the
**content** (which words are spoken). The talker text decoder is a separate
knob. It samples by default, matching upstream `InferenceOptions`
(do_sample=True, temperature=0.6, top_k=0, top_p=0.9, repetition_penalty=1.1);
pure greedy decoding has no repetition control and loops, cuts words off, or
adds trailing noise — worst on harder or non-English text.

| Env var | Default | Notes |
|---|---|---|
| `TADA_DO_SAMPLE` | `1` | `0` = greedy argmax (the old behaviour) |
| `TADA_TEMPERATURE` | `0.6` | also set by `--temperature` / `set_temperature` |
| `TADA_TOP_P` | `0.9` | nucleus; also `set_top_p` |
| `TADA_TOP_K` | `0` | `0` = disabled |
| `TADA_REPETITION_PENALTY` | `1.1` | `1.0` = none; also `set_repetition_penalty` |

Honoured by the CLI, C ABI, bindings and server. Raising
`TADA_REPETITION_PENALTY` measurably reduces repeats; with sampling on, `--seed`
changes the wording.

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
stochastic decode path. `Kokoro` uses `KOKORO_SEED` instead of
`--seed`.

For HTTP usage, see [`docs/server.md`](server.md) — `POST
/v1/audio/speech` is the OpenAI-compatible TTS endpoint, available on
any `crispasr --server` instance whose loaded backend declares
`CAP_TTS`. Routes register on every backend; per-request `voice`,
`speed`, and `instructions` pass through to the backend's
`whisper_params`. Long-form input is auto-chunked on sentence
boundaries.

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
| `KOKORO_GEN_GPU` | unset | Route the iSTFTNet generator (the vocoder) onto the main GPU backend instead of the Metal-hang-workaround CPU pin. Use on CUDA / Vulkan where the stride-10 ConvTranspose1d M1 hang doesn't apply and CPU vocoder is the bottleneck. Mirrors `QWEN3_TTS_CODEC_GPU`. |
| `KOKORO_GEN_FORCE_METAL` | unset | Same effect as `KOKORO_GEN_GPU`, but the name carries the original "reproduce the M1 hang" debug intent. Kept for back-compat; new deployments should prefer `KOKORO_GEN_GPU`. |

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
| `QWEN3_TTS_SEED` | `42` | Override the AR sampling seed (superseded by `--seed N` on the CLI). |
| `QWEN3_TTS_MAX_FRAMES` | `1500` | Hard cap on AR decode steps. Short prompts that fail to sample `codec_eos` would otherwise run to the 1500-frame ceiling. |
| `QWEN3_TTS_O15` | unset | Pin code-predictor `Lk = cp_kv_max_ctx` and reuse one cached T=1 graph across AR steps 2..14 (saves ~14-19 ms/frame on Mac/Metal — alloc+build collapse from ~20 ms/frame to ~1.6 ms/frame). Default flipped back to OFF after [#56](https://github.com/CrispStrobe/CrispASR/issues/56): the cached-graph reuse asserts on the CUDA backend (`GGML_ASSERT` in `ggml_backend_tensor_set` on first `code_pred_generate_15` call, Jetson Orin AGX sm_87). M1 Metal users who want the speedup should set `QWEN3_TTS_O15=1`. Default goes back to ON once the CUDA path is verified. |
| `QWEN3_TTS_FUSED_QKV` | unset | Fuse Q+K+V weights into one matmul per talker layer at load time (F16/F32 talker only; auto-skipped for Q8_0/Q4_K). Bit-identical to the unfused path on M1 Metal; speed effect is machine-dependent. |
| `QWEN3_TTS_BENCH` | unset | Print per-call build/alloc/compute/read timings for `talker_kv` and `code_pred_kv`. |
| `QWEN3_TTS_PROF` | unset | Per-op profiler (more granular than `BENCH`). |
| `QWEN3_TTS_CP_BACKEND` | unset | Pin the code predictor to a chosen backend. `cpu`, `cpu-f16`, `cpu-f32` keep its weights on the CPU backend — useful when isolating bugs to the talker vs. code-predictor or when comparing CPU and Metal end-to-end. |
| `QWEN3_TTS_DUMP_DIR` | unset | Write per-frame intermediate tensors into the named directory. Bulky; intended for diff-harness work (`tools/dump_reference.py --backend qwen3-tts`). |
| `QWEN3_TTS_CODEC_GPU` | auto | Force codec weights and decode through the GPU scheduler. GPU is now the default on all GPU backends including Metal — the `CONV_TRANSPOSE_1D` hang was fixed in `f8fc8b8e` and the op replaced by `mul_mat+col2im_1d` in `5f600f25`. Distinct from `QWEN3_TTS_CODEC_FORCE_METAL`, which also enables a per-op trace callback for debugging. |
| `QWEN3_TTS_CODEC_CPU` | unset | Force codec weights and decode through the CPU-only `codec_sched`. Useful for A/B timing and regression bisection. |
| `QWEN3_TTS_CODEC_CHUNK` | `150` (`64` on CUDA) | Maximum generated codec frames per decode chunk. CUDA clamps values above `64` and treats `0` as `64` unless `QWEN3_TTS_CODEC_ALLOW_FULL=1` is also set, avoiding oversized `mul_mat+col2im_1d` allocations on 10 GB cards. |
| `QWEN3_TTS_CODEC_CTX` | `128` (`96` on CUDA) | Left-context codec frames prepended to each chunk. Values below the codec sliding window are raised; CUDA clamps larger values unless `QWEN3_TTS_CODEC_ALLOW_FULL=1` is set. |
| `QWEN3_TTS_SKIP_REF_DECODE` | **on** (set `=0` to opt out) | Skip the codec decode of the reference audio in `qwen3_tts_synthesize`. The default-on path emits `codec_decode_codes(gen)` directly; the opt-out path concatenates `ref_codes + gen_codes`, decodes both, then trims the ref portion. With a 26 s reference (~334 codec frames at 12 Hz), the ref half adds ~16 s of constant codec compute regardless of how much new audio is generated (Jetson Orin AGX, issue #64). End-to-end RTF on Orin drops from ~7-9 → ~1.5; the win compounds N× under `/v1/audio/speech` long-form chunking. Bit-identity verified 2026-05-05 on Apple Silicon Metal, qwen3-tts-customvoice 0.6B Q8_0: max\|diff\| = 0, cosine similarity = 1.0 — equivalence holds because the codec is a straight-line forward pass with no rolling state. Set `QWEN3_TTS_SKIP_REF_DECODE=0` only for A/B verification or if a future codec graph variant grows rolling state. |

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
For parity debugging, `VIBEVOICE_TTS_LATENTS=/path/to/latents.bin` can replay a
raw float32 latent stack, `VIBEVOICE_TTS_DUMP=/dir` writes `tts_scaled_latent`
and `tts_raw_audio`, and `VIBEVOICE_TTS_DUMP_DECODER=1` adds per-stage decoder
dumps.

## VibeVoice 1.5B — base TTS with WAV cloning

The 1.5B base model supports both a generic no-clone voice and WAV
reference cloning through `VIBEVOICE_VOICE_AUDIO`.

```bash
# Generic output, no voice reference.
./build/bin/crispasr \
    --backend vibevoice-1.5b -m auto \
    --tts "Hello, how are you today?" \
    --tts-output hello.wav

# Clone from a 24 kHz mono WAV reference.
VIBEVOICE_VOICE_AUDIO=samples/qwen3_tts/clone.wav \
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
./build/bin/crispasr --backend chatterbox-turbo -m auto --tts "..." -ow out.wav

# German fine-tune of Turbo (SebastianBodza/Kartoffelbox_Turbo):
./build/bin/crispasr --backend kartoffelbox-turbo -m auto -l de \
    --tts "Hallo, das ist Kartoffelbox-Turbo." -ow out-de.wav

# Arabic Llama-T3 fine-tune (oddadmix/lahgtna-chatterbox-v1):
./build/bin/crispasr --backend lahgtna-chatterbox -m auto -l ar \
    --tts "مرحباً" -ow out-ar.wav
```

### Multilingual language selection

The base `chatterbox` backend uses the upstream multilingual v3 T3 weights
from `cstr/chatterbox-GGUF`. Pass `-l <code>` / `--language <code>` to
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

On the multilingual path the text is **NFKD-normalized** (then ASCII-lowercased)
before tokenization, matching upstream `MTLTokenizer.preprocess_text`. This
matters for scripts with precomposed diacritics: e.g. Arabic `أ`
(ALEF-WITH-HAMZA) decomposes to base alef + combining hamza, the form the model
was trained on. Without it, partial-diacritic Arabic produced spurious onset
letters (#170). Script-specific normalizers (zh cangjie / ja kakasi / he dicta /
ko jamo / ru stress) are not yet implemented.

> **Note on published GGUFs.** Some published multilingual T3 artifacts pair a
> 2352-token tokenizer with a 2454-vocab T3; the loader rejects that mismatch.
> Repair locally with `models/patch-chatterbox-gguf-add-merges.py` and the
> matching `grapheme_mtl_merged_expanded_v1.json` (2454 tokens).

### Performance

The compute-bound T3 AR decode is the slow stage. It runs on CPU by default on
Metal (GPU has higher per-step kernel-launch overhead for the many T=1 steps).
The CPU thread count defaults to `min(8, hardware_concurrency)`; override with
`CRISPASR_CHATTERBOX_THREADS=<n>` (e.g. dial down on a heavily shared host).
Output is bit-identical regardless of thread count.

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
    --output my_voice.gguf
```

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
    --tts-output cloned.wav
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
| `set_min_p(p)` | 0.0 | Min-p sampling threshold |
| `set_repetition_penalty(r)` | 1.0 | Repetition penalty (1.0 = no penalty; > 1 discourages repeated tokens) |
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
    -m parler-mini-v1.1-q8_0.gguf \
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
only). The model uses character-level tokenization (2545 vocab, pinyin
for Chinese). Output is 24 kHz mono PCM.

**Model file:**
[`cstr/f5-tts-GGUF`](https://huggingface.co/cstr/f5-tts-GGUF)

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

- `INDEXTTS_VOCODER_RAW=1` — opt out of AA; fully GPU-graphable but
  produces ~2 k impossible inter-sample jumps on speech. Use only for
  reproducing the legacy / aliased benchmark.
- `--no-gpu` — keeps the whole vocoder graph on CPU. Recommended for
  IndexTTS specifically: the GPT codes generate quickly either way, and
  the AA custom op forces a GPU↔CPU sync per AMP block when mixed with
  Metal, leaving GPU + AA the slowest of the four combinations.

Set `INDEXTTS_DEBUG=1` for per-stage intermediate dumps (mel, conformer
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
(`他用Python写了一个程序。`) works without special flags. Pass `-vv` to
dump the BPE token IDs (`indextts: text_ids[...]`) for diffing against
the Python reference if output sounds wrong.

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

**Optional hook: `INDEXTTS_TEXT_NORMALIZER`** delegates normalization
to any user-provided shell command that reads UTF-8 text on stdin and
writes the normalized text to stdout. The output is then run through
the default pipeline (CJK split + punct map + upper) before
SentencePiece.

Recommended setup with upstream wetext:

```bash
pip install wetext
# Then on every IndexTTS invocation:
INDEXTTS_TEXT_NORMALIZER="python $REPO/tools/wetext-normalize.py" \
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

## Local speaker output (`--tts-play`)

Pass `--tts-play` to play TTS output through the local speaker immediately
after synthesis, in addition to (or instead of) writing a file. The
spread-spectrum watermark is always embedded before playback, so the audio
leaving the speaker carries the provenance marker.

```bash
# Synthesize and play through default speaker
crispasr --tts --tts-play -m model.gguf "Hello world."

# Write file AND play
crispasr --tts --tts-play -m model.gguf -o output.wav "Hello world."

# Select a non-default output device (index from --list-audio-devices)
crispasr --tts --tts-play --tts-play-device 2 -m model.gguf "Hello world."
```

Playback is synchronous — the CLI blocks until audio drains, then exits.
Device -1 (the default) selects the system default output device.

**Implementation note:** the device is opened at the hardware-native sample
rate (`sampleRate=0 / channels=0`). The model's mono float32 PCM is
pre-resampled via linear interpolation before the device starts. This avoids
miniaudio's 4× upsampler artefacts on devices that run natively at 96 kHz
(MacBook Air Speakers and many Core Audio devices).

## AI-generated audio provenance & watermarking

All TTS output is automatically marked as AI-generated through multiple
complementary layers. This is non-optional and cannot be bypassed.

### Spread-spectrum watermark (built-in, always active)

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

# Debug: AUDIOSEAL_DEBUG=1 for shape traces, AUDIOSEAL_DUMP_STAGES=1 for binary dumps
```

### File metadata (always active)

- **WAV**: `LIST`/`INFO` chunk with `ISFT="CrispASR (AI-generated audio)"` and `ICMT` notice
- **MP3**: ID3v2 `TXXX` frames: `AI_GENERATED=true`, `GENERATOR=CrispASR`

### C2PA Content Credentials (optional, compile-time)

Signed provenance manifests with `digitalSourceType=trainedAlgorithmicMedia`.
Requires `c2pa-c` library and a self-signed certificate:

```bash
# Generate certificate
./scripts/generate-c2pa-cert.sh

# Use with TTS
crispasr --tts "hello" --c2pa-cert crispasr-c2pa.crt --c2pa-key crispasr-c2pa.key
```

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
reads back the in-memory PCM and runs watermark detection on it. If
the detected confidence is below 0.6, a warning is emitted to stderr.
This catches cases where the watermark was degraded during synthesis
or encoding — no extra flags needed.

### `--detect-watermark PATH` — standalone watermark detection

Reads a WAV file, runs watermark detection (spread-spectrum by default,
or AudioSeal if `--watermark-model` is given), prints the confidence
score and a human-readable verdict, then exits.

| Confidence | Verdict |
|---|---|
| > 0.65 | `AI-GENERATED WATERMARK DETECTED` |
| 0.4 – 0.65 | `UNCERTAIN` |
| < 0.4 | `No watermark detected` |

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

- **CLI**: `--no-spoken-disclaimer`
- **Server**: `"spoken_disclaimer": false` in the request body

When the spoken disclaimer is suppressed, the caller assumes
responsibility for providing appropriate AI-disclosure to end users
(e.g. a visual label in the UI). The spread-spectrum watermark and
C2PA metadata are always embedded regardless of this setting.
