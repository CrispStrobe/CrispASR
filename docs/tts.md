# Text-to-Speech (TTS)

CrispASR ships **seven open-weights TTS engines** behind the same
`crispasr` binary, each with a distinct voice / quality / footprint
trade-off:

| Backend | Why pick it | Voice cloning | First-run download |
|---|---|---|---|
| **`kokoro`** | Smallest + fastest. 82 M-param StyleTTS2-derived model. Multilingual via espeak-ng + native German backbone. | No (preset voice packs) | Manual `wget` (no `-m auto`) |
| **`qwen3-tts`** | Highest fidelity / strongest cloning. Speech-LLM (talker + code predictor + 12 Hz codec). | Yes (WAV + ref-text or baked voice GGUF) | ~1.3 GB via `-m auto` |
| **`vibevoice-tts`** | Lowest-latency streaming TTS, designed for realtime. | Preset voice packs | ~636 MB via `-m auto` |
| **`vibevoice-1.5b`** | Base VibeVoice TTS model with WAV cloning. | Yes (`VIBEVOICE_VOICE_AUDIO=<wav>` or `--voice <wav>`) | ~1.6 GB via `-m auto` |
| **`orpheus`** | Llama-3.2-3B talker + SNAC 24 kHz codec. 8 baked English speakers; expressive output. Greedy loops — pass `--temperature 0.6`. | Preset names via `--voice tara/leah/...` | ~3.5 GB via `-m auto` (talker Q8 + 26 MB SNAC) |
| **`chatterbox`** | T3 AR + S3Gen flow-matching + HiFTGenerator. Built-in voice baked into the T3 GGUF; clones via a baked voice GGUF (see workflow below). EN/AR/DE variants share runtime. | Yes (`--voice <voice.gguf>`, baked from a WAV with `models/bake-chatterbox-voice-from-wav.py`) | ~880 MB via `-m auto` (T3 Q8 + S3Gen Q8) |
| **`indextts`** | IndexTTS-1.5: GPT-2 AR (24L/1280d) mel-code generator + BigVGAN vocoder. Designed for Chinese+English. Zero-shot voice cloning from any reference WAV. | Yes (`--voice <ref.wav>`) | ~2.4 GB via `-m auto` (GPT F16 + BigVGAN F16) |
| **`cosyvoice3-tts`** | Fun-CosyVoice3-0.5B-2512: Qwen2-0.5B AR speech-token LM + DiT-CFM (10-step Euler) + HiFT (NSF + iSTFT) @ 24 kHz. 9 languages + 18 Chinese dialects. Ships an 8-voice baked bank (`zero_shot` + `fleurs-{en,de,zh,ja,fr,es,ko}`). | Yes — baked-bank name via `--voice <name>`, **or** native arbitrary-WAV cloning via `--voice <ref.wav> --ref-text "..."` (ports speech_tokenizer_v3 + CAMPPlus + matcha mel to ggml; speech tokens byte-exact vs ONNX). | ~1.2 GB via `-m auto` (Q4_K LLM + Q8_0 flow + HiFT + s3tok + campplus + voices) |

All seven write 24 kHz mono WAV via `--tts-output`.

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
qwen3-tts, chatterbox, vibevoice, orpheus, indextts, and voxcpm2. It
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
# Auto-download, runtime WAV clone (~1.3 GB on first run):
./build/bin/crispasr \
    --backend qwen3-tts -m auto \
    --voice samples/qwen3_tts/clone.wav \
    --ref-text "Okay, yeah. I resent you, I love you, I respect you. But you know what - You blew it, and thanks to you." \
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
    --voice /tmp/qwen3-tts-voice-pack.gguf \
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
| `QWEN3_TTS_CODEC_GPU` | unset | Route the codec decode through the main GPU scheduler instead of the CPU-only `codec_sched`. The codec is pinned to CPU by default to dodge the M1 Metal hang; on CUDA / Vulkan / etc. the hang does not apply and CPU codec is dramatically slower. On Jetson Orin AGX, codec on CPU is ~50× slower than CUDA. Distinct from `QWEN3_TTS_CODEC_FORCE_METAL`, which also moves the codec to the main GPU sched but additionally enables a per-op `ggml_backend_synchronize` trace callback for reproducing the Metal hang — useful for debugging, not for production. |
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
| `chatterbox`         | T3 Q8_0 (541 MB)  | base S3Gen Q8_0  (342 MB) | ~880 MB |
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

## TTS GGUF downloads

[`cstr/vibevoice-realtime-0.5b-GGUF`](https://huggingface.co/cstr/vibevoice-realtime-0.5b-GGUF) ·
[`cstr/vibevoice-1.5b-GGUF`](https://huggingface.co/cstr/vibevoice-1.5b-GGUF) ·
[`cstr/qwen3-tts-0.6b-base-GGUF`](https://huggingface.co/cstr/qwen3-tts-0.6b-base-GGUF) ·
[`cstr/qwen3-tts-1.7b-base-GGUF`](https://huggingface.co/cstr/qwen3-tts-1.7b-base-GGUF) ·
[`cstr/qwen3-tts-1.7b-voicedesign-GGUF`](https://huggingface.co/cstr/qwen3-tts-1.7b-voicedesign-GGUF) ·
[`cstr/qwen3-tts-tokenizer-12hz-GGUF`](https://huggingface.co/cstr/qwen3-tts-tokenizer-12hz-GGUF) ·
[`cstr/orpheus-3b-base-GGUF`](https://huggingface.co/cstr/orpheus-3b-base-GGUF) ·
[`cstr/kartoffel-orpheus-3b-german-natural-GGUF`](https://huggingface.co/cstr/kartoffel-orpheus-3b-german-natural-GGUF) ·
[`cstr/kartoffel-orpheus-3b-german-synthetic-GGUF`](https://huggingface.co/cstr/kartoffel-orpheus-3b-german-synthetic-GGUF) ·
[`cstr/snac-24khz-GGUF`](https://huggingface.co/cstr/snac-24khz-GGUF) ·
[`cstr/chatterbox-GGUF`](https://huggingface.co/cstr/chatterbox-GGUF) ·
[`cstr/chatterbox-turbo-GGUF`](https://huggingface.co/cstr/chatterbox-turbo-GGUF) ·
[`cstr/kartoffelbox-turbo-GGUF`](https://huggingface.co/cstr/kartoffelbox-turbo-GGUF) ·
[`cstr/lahgtna-chatterbox-v1-GGUF`](https://huggingface.co/cstr/lahgtna-chatterbox-v1-GGUF) ·
[`cstr/indextts-1.5-GGUF`](https://huggingface.co/cstr/indextts-1.5-GGUF)

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
