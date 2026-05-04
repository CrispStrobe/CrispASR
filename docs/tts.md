# Text-to-Speech (TTS)

CrispASR ships **five open-weights TTS engines** behind the same
`crispasr` binary, each with a distinct voice / quality / footprint
trade-off:

| Backend | Why pick it | Voice cloning | First-run download |
|---|---|---|---|
| **`kokoro`** | Smallest + fastest. 82 M-param StyleTTS2-derived model. Multilingual via espeak-ng + native German backbone. | No (preset voice packs) | Manual `wget` (no `-m auto`) |
| **`qwen3-tts`** | Highest fidelity / strongest cloning. Speech-LLM (talker + code predictor + 12 Hz codec). | Yes (WAV + ref-text or baked voice GGUF) | ~1.3 GB via `-m auto` |
| **`vibevoice-tts`** | Lowest-latency streaming TTS, designed for realtime. | Preset voice packs (and a 1.5 B-base WAV cloning path) | ~636 MB via `-m auto` |
| **`orpheus`** | Llama-3.2-3B talker + SNAC 24 kHz codec. 8 baked English speakers; expressive output. Greedy loops — pass `--temperature 0.6`. | Preset names via `--voice tara/leah/...` | ~3.5 GB via `-m auto` (talker Q8 + 26 MB SNAC) |
| **`chatterbox`** | T3 AR + S3Gen flow-matching + HiFTGenerator. Built-in voice baked into the T3 GGUF; clones from a reference WAV. EN/AR/DE variants share runtime. | Yes (`--voice <wav>` via VE/CAMPPlus) | ~880 MB via `-m auto` (T3 Q8 + S3Gen Q8) |

All five write 24 kHz mono WAV via `--tts-output`.

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
| `QWEN3_TTS_MAX_FRAMES` | `1500` | Hard cap on AR decode steps. Short prompts that fail to sample `codec_eos` would otherwise run to the 1500-frame ceiling. |
| `QWEN3_TTS_O15` | unset | Pin code-predictor `Lk = cp_kv_max_ctx` and reuse one cached T=1 graph across AR steps 2..14 (saves ~14-19 ms/frame on Mac/Metal — alloc+build collapse from ~20 ms/frame to ~1.6 ms/frame). Default flipped back to OFF after [#56](https://github.com/CrispStrobe/CrispASR/issues/56): the cached-graph reuse asserts on the CUDA backend (`GGML_ASSERT` in `ggml_backend_tensor_set` on first `code_pred_generate_15` call, Jetson Orin AGX sm_87). M1 Metal users who want the speedup should set `QWEN3_TTS_O15=1`. Default goes back to ON once the CUDA path is verified. |
| `QWEN3_TTS_FUSED_QKV` | unset | Fuse Q+K+V weights into one matmul per talker layer at load time (F16/F32 talker only; auto-skipped for Q8_0/Q4_K). Bit-identical to the unfused path on M1 Metal; speed effect is machine-dependent. |
| `QWEN3_TTS_BENCH` | unset | Print per-call build/alloc/compute/read timings for `talker_kv` and `code_pred_kv`. |
| `QWEN3_TTS_PROF` | unset | Per-op profiler (more granular than `BENCH`). |
| `QWEN3_TTS_CP_BACKEND` | unset | Pin the code predictor to a chosen backend. `cpu`, `cpu-f16`, `cpu-f32` keep its weights on the CPU backend — useful when isolating bugs to the talker vs. code-predictor or when comparing CPU and Metal end-to-end. |
| `QWEN3_TTS_DUMP_DIR` | unset | Write per-frame intermediate tensors into the named directory. Bulky; intended for diff-harness work (`tools/dump_reference.py --backend qwen3-tts`). |
| `QWEN3_TTS_CODEC_GPU` | unset | Route the codec decode through the main GPU scheduler instead of the CPU-only `codec_sched`. The codec is pinned to CPU by default to dodge the M1 Metal hang; on CUDA / Vulkan / etc. the hang does not apply and CPU codec is dramatically slower. On Jetson Orin AGX, codec on CPU is ~50× slower than CUDA. Distinct from `QWEN3_TTS_CODEC_FORCE_METAL`, which also moves the codec to the main GPU sched but additionally enables a per-op `ggml_backend_synchronize` trace callback for reproducing the Metal hang — useful for debugging, not for production. |
| `QWEN3_TTS_SKIP_REF_DECODE` | unset | Skip the codec decode of the reference audio in `qwen3_tts_synthesize`. Default behavior concatenates `ref_codes + gen_codes` and decodes both together, then trims the ref portion off the output. With a 26 s reference (~334 codec frames at 12 Hz), the ref half adds ~16 s of constant codec compute regardless of how much new audio is generated (Jetson Orin AGX measurement, issue #58). The codec is a straight-line forward pass without rolling state, so decoding gen-only is mathematically equivalent to "decode ref+gen, trim ref" — same audio out, less wall-clock. Use this together with `QWEN3_TTS_CODEC_GPU=1` for the strongest perf combo on CUDA / Vulkan hosts. Will flip to default-on once bit-identity is validated across all qwen3-tts variants and platforms. |

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

Voice cloning from a reference WAV (24 kHz mono, ~5–10 s):

```bash
./build/bin/crispasr --backend chatterbox -m auto \
    --voice samples/jfk.wav \
    --tts "Cloned voice synthesizing arbitrary text." \
    --tts-output cloned.wav
```

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
| `--tts-steps` | 10 (base/lahgtna) / 2 (turbo/kartoffelbox-turbo meanflow) | CFM Euler steps |
| `--codec-model` | sibling autodetect | explicit S3Gen GGUF path (overrides `-m auto` companion) |

**Status — known parity gap:** the Conformer encoder still has an
open ~30 % rel-pos attention magnitude gap vs the Python reference
(matrix_bd rms 7.08 vs 10.06 — see commits `b4f4afe`, `8dc8bc6`).
The vocoder side is bit-tight (per-stage cosine 1.000, waveform
cosine 0.93). Audio synthesizes end-to-end and is intelligible, but
voice fidelity / prosody may differ from the Python reference until
the rel-pos gap closes.

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
[`cstr/lahgtna-chatterbox-v1-GGUF`](https://huggingface.co/cstr/lahgtna-chatterbox-v1-GGUF)
