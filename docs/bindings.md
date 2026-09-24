# Language bindings

All wrappers are thin shells over the same C-ABI surface in
`src/crispasr_c_api.cpp`. Anything the CLI can do — transcribe, VAD,
diarize, LID, align, download — is one function call in every
language.

## Session audio-format getters (#332)

Four read-only getters describe the PCM a session consumes and produces, so
callers don't hard-code per-backend rates:

| C-ABI getter | Returns |
|---|---|
| `crispasr_session_input_sample_rate(s)` | Rate (Hz) the backend expects for input PCM — 16000 for Whisper-family, the model's native rate otherwise. Pair with `crispasr_audio_load_at_rate` to avoid a double resample. |
| `crispasr_session_output_sample_rate(s)` | Rate (Hz) of the PCM `synthesize` / `synthesize_streaming` / `get_disclaimer_pcm` / `speech_to_speech` return — the "backend-native rate" those calls document. `0` = the backend produces no audio output (ASR-only). |
| `crispasr_session_input_channels(s)` | `1` (mono) for every current backend. Source separation is the stereo exception and has its own surface (`separate*`). |
| `crispasr_session_output_channels(s)` | `1` (mono), or `0` when the backend produces no audio output. |

All return `0` on a NULL/invalid session. Exposed as `output_sample_rate` /
`input_channels` / `output_channels` (Rust, Ruby), `outputSampleRate()` /
`inputChannels()` / `outputChannels()` (Java, C# `OutputSampleRate()` etc.),
and `sessionOutputSampleRate()` etc. in the WASM/JS binding.

## Session setter reference

All generation-control setters are available in every binding. Each
call is a thin proxy over the C-ABI function of the same name.
Setters that return an error code `-2` are soft no-ops — the active
backend doesn't expose that knob, but the call is safe to make.

| C-ABI setter | Bindings name (Python/Rust/Go/Java) | Notes |
|---|---|---|
| `set_temperature(temp, seed)` | `set_temperature` / `set_temperature` / `SetTemperature` / `setTemperature` | ASR + TTS backends that sample; rc=-2 = no backend supports it |
| `set_tts_seed(seed)` | `set_tts_seed` / `set_tts_seed` / `SetTTSSeed` / `setTtsSeed` | Chatterbox, vibevoice, qwen3-tts, orpheus; rc=-2 for others |
| `set_max_new_tokens(n)` | `set_max_new_tokens` / `set_max_new_tokens` / `SetMaxNewTokens` / `setMaxNewTokens` | AR backends; ≤ 0 clears override |
| `set_frequency_penalty(f)` | `set_frequency_penalty` / `set_frequency_penalty` / `SetFrequencyPenalty` / `setFrequencyPenalty` | AR backends; ≤ 0 disables |
| `set_tts_steps(n)` | `set_tts_steps` / `set_tts_steps` / `SetTTSSteps` / `setTtsSteps` | Chatterbox S3Gen CFM steps; vibevoice DPM-Solver++ steps; kugelaudio; tada FM steps; irodori flow-matching ODE steps |
| `set_tts_num_candidates(n)` | `set_tts_num_candidates` / `set_tts_num_candidates` / `SetTTSNumCandidates` / `setTtsNumCandidates` | TADA flow-matching timing candidates ranked per token (default 4); rc=-2 for others |
| `set_top_p(p)` | `set_top_p` / `set_top_p` / `SetTopP` / `setTopP` | Chatterbox AR T3 loop |
| `set_top_k(k)` | `set_top_k` / `set_top_k` / `SetTopK` / `setTopK` | TADA talker sampler (0 = disabled); rc=-2 for others |
| `set_do_sample(enable)` | `set_do_sample` / `set_do_sample` / `SetDoSample` / `setDoSample` | TADA talker: false = greedy; rc=-2 for others |
| `set_min_p(p)` | `set_min_p` / `set_min_p` / `SetMinP` / `setMinP` | Chatterbox AR T3 loop |
| `set_repetition_penalty(r)` | `set_repetition_penalty` / `set_repetition_penalty` / `SetRepetitionPenalty` / `setRepetitionPenalty` | Chatterbox (1.0 = no penalty) |
| `set_cfg_weight(w)` | `set_cfg_weight` / `set_cfg_weight` / `SetCFGWeight` / `setCfgWeight` | Chatterbox (0.5 = upstream default; 0 = unconditional); TADA acoustic_cfg |
| `set_tts_noise_temp(t)` | `set_tts_noise_temp` / `set_tts_noise_temp` / `SetTtsNoiseTemp` / `setTtsNoiseTemp` | TADA flow-matching noise temperature (0.9 = upstream default) |
| `set_exaggeration(e)` | `set_exaggeration` / `set_exaggeration` / `SetExaggeration` / `setExaggeration` | Chatterbox emotion scalar (0.5 = upstream default) |
| `set_max_speech_tokens(n)` | `set_max_speech_tokens` / `set_max_speech_tokens` / `SetMaxSpeechTokens` / `setMaxSpeechTokens` | Chatterbox AR loop token budget (default 1000 ≈ 20 s) |
| `set_length_scale(s)` | `set_length_scale` / `set_length_scale` / `SetLengthScale` / `setLengthScale` | Kokoro phoneme duration multiplier (1.0 = normal) |
| `set_best_of(n)` | `set_best_of` / `set_best_of` / `SetBestOf` / `setBestOf` | Best-of-N sampling for temperature > 0 |
| `set_beam_size(n)` | `set_beam_size` / `set_beam_size` / `SetBeamSize` / `setBeamSize` | Beam search width |
| `set_return_logits(enable)` | `set_return_logits` / `set_return_logits` / `SetReturnLogits` / `setReturnLogits` | Opt-in dense CTC grid capture for backends that expose frame-level CTC scores |
| `set_grammar_text(gbnf, root, penalty)` | `set_grammar_text` / `set_grammar_text` / `SetGrammarText` / `setGrammarText` | GBNF constrained decoding (whisper); empty string clears |
| `set_fallback_thresholds(...)` | `set_fallback_thresholds` / `set_fallback_thresholds` / `SetFallbackThresholds` / `setFallbackThresholds` | Whisper entropy/logprob/no-speech thresholds + temp-inc |
| `set_sensitivity(preset)` | `set_sensitivity` / `set_sensitivity` / `SetSensitivity` / `setSensitivity` | The four thresholds above as one named bundle: `conservative` / `balanced` / `aggressive` (aliases `strict` / `default` / `loose`). `balanced` is the shipped defaults, so it is always a no-op. **rc=-2 means an unknown preset and every wrapper raises** — a typo must never decode silently at the defaults. A later `set_fallback_thresholds` overrides it. HTTP: the `sensitivity` form field, applied before the individual threshold fields so those still win. |
| `set_alt_n(n)` | `set_alt_n` / `set_alt_n` / `SetAltN` / `setAltN` | Per-token alternative candidates (whisper greedy) |
| `set_whisper_decode_extras(...)` | `set_whisper_decode_extras` / `set_whisper_decode_extras` / `SetWhisperDecodeExtras` / `setWhisperDecodeExtras` | suppress_nst, suppress_regex, carry_initial_prompt |
| `set_ask(prompt)` | `set_ask` / `set_ask` / `SetAsk` / `setAsk` | Free-form prompt for instruct-tuned audio-LLM backends (granite, voxtral, qwen3-asr, raon-speech, glm-asr, gemma4-e2b, mimo-asr, higgs-stt, ark-asr, moss-audio, moss-diarize, mini-omni2, lfm2-audio). Empty string clears. |
| `set_punc_model(alias\|path)` | `set_punc_model` / `set_punc_model` / `SetPuncModel` / `setPuncModel` | Load FireRedPunc/PCS punctuation restoration on the session (`auto`/`firered`/`fullstop`/`punctuate-all`/`pcs`/path; auto-downloads). Restores punctuation on backends that emit none (parakeet RNNT/CTC, …). `"none"`/`""` unloads. (Also Java/Ruby.) |
| `set_hotwords(words, boost)` | `set_hotwords` / `set_hotwords` / `SetHotwords` / `setHotwords` | Comma-separated contextual-biasing hotwords, boosted per token match (parakeet CTC/TDT trie; LLM-backend prompt injection). Empty string clears. (All six wrappers.) |
| `set_tts_phonemes(ipa)` | `set_tts_phonemes` / `set_tts_phonemes` / `SetTTSPhonemes` / `setTtsPhonemes` | #316: synthesize the given phonemes verbatim, skipping the G2P — the seam between text processing and the acoustic model. Use it to reproduce another implementation's pronunciation, or to tell a G2P bug from a model bug. Empty clears; rc=-2 on a backend with no phonemes-in call (kokoro and piper have one). Server: `"phonemes"` on `/v1/audio/speech`. CLI: `--tts-phonemes`. (All wrappers.) |
| `set_g2p_dict(source)` | `set_g2p_dict` / `set_g2p_dict` / `SetG2PDict` / `setG2pDict` | Select the G2P pronunciation dictionary for TTS phonemization (`olaph`/`open-dict`/path). (All six wrappers.) |

## Result field reference

Everything a transcription result carries is read back through per-index
accessors on the opaque `crispasr_session_result`, so adding a field is a new
symbol rather than a layout change — old callers keep working. Every wrapper
exposes them as struct/class members on its segment type.

| C-ABI accessor | Member (Python/Rust/Go/Dart/Java/C#/Ruby) | Notes |
|---|---|---|
| `result_segment_text(r, i)` | `text` / `text` / `Text` / `text` / `text` / `Text` / `:text` | The segment transcript. |
| `result_segment_t0/t1(r, i)` | `start`,`end` / `start`,`end` / `T0`,`T1` / `start`,`end` / `t0`,`t1` / `T0`,`T1` / `:t0`,`:t1` | Centiseconds on the C ABI. **Python/Rust/Dart/C# divide to seconds; Go, Java and Ruby hand the raw centiseconds through** (each documents the unit on the field — note the `t0`/`t1` naming marks the centisecond side). C# joined the seconds side after issue #291 — it had been reporting centiseconds through `Segment`/`Word`/`AlignedWord` while every other time value in that binding (`VadSpan`, the music types) was seconds (issue #291). A backend with no timing for a unit reports `-1` there, and the seconds bindings pass that sentinel through unscaled rather than reporting `-0.01`. |
| `result_segment_no_speech_prob(r, i)` | `no_speech_prob` / … / `NoSpeechProb` / `noSpeechProb` / … / `:no_speech_prob` | Whisper only; `-1.0` sentinel = no data. |
| `result_segment_speaker(r, i)` | `speaker` / — / `Speaker` / `speaker` / `speaker` / `Speaker` / `:speaker` | **New in v0.8.24.** Not surfaced by the Rust wrapper yet (`SessionSegment` has no `speaker` field). Native per-segment speaker label in the `"(Speaker N) "` form, `""` when the backend does not diarize natively. |
| `result_n_words` + `result_word_*` | `words` (list of word objects) | Per-word text/timings/confidence, plus `alts` where the backend emits them. |

> **`speaker` ordinals are CHUNK-LOCAL.** They come from the backend's own
> per-call diarization, so `Speaker 1` in one `transcribe()` is not guaranteed to
> be the same voice as `Speaker 1` in the next — nothing clusters across calls
> here. Use `diarize_segments` / `DiarizeSegments` (§ session-scoped clustering
> in [`diarization-speakers.md`](diarization-speakers.md)) when you need labels
> stable across a whole recording. Populated today by `vibevoice`, whose model
> answers with a Start/End/Speaker/Content array; `moss-diarize` and `granite`
> in `--diarize` mode populate it on the CLI/server surfaces.

> **Older libraries.** The Python and Dart wrappers probe for
> `result_segment_speaker` (`hasattr` / `providesSymbol`) before calling it and
> fall back to `""`, so a wrapper built after v0.8.24 still runs against a
> pre-v0.8.24 `libcrispasr`. Go, Java, Ruby and C# bind the symbol directly and
> need a v0.8.24+ library. The reverse (old wrapper, new library) is always
> fine — it simply ignores the symbol.

> **Tip — chunk-boundary dedup for bindings.** When a binding drives a
> CAP_UNBOUNDED_INPUT backend (parakeet, canary, …) chunk-by-chunk and
> needs to stitch the output, call `crispasr_lcs_dedup_prefix_count`
> between adjacent chunks. It returns the number of leading tokens of
> `chunk[i]` that duplicate the tail of `chunk[i-1]` (NeMo-style
> sub-word LCS over emitted token ids). The binding then drops that
> many tokens from `chunk[i]` and rebuilds its own segment / word /
> text representation. The C declaration lives in `include/crispasr.h`;
> see also the `--lcs-dedup` / `--lcs-min-length` CLI flags.

> **CTC logits and vocab.** `transcribe_with_logits` / `TranscribeWithLogits`
> enables `set_return_logits(true)` for a single call, copies the result-owned
> dense CTC grid into language-owned memory, and then disables capture again.
> The grid is frame-major (`data[t * n_vocab + v]`). Omni CTC and wav2vec2
> return raw pre-softmax logits; canary/FastConformer CTC returns log-probs.
> `ctc_vocab` / `CtcVocab` returns raw token pieces where the backend exposes a
> CTC vocabulary.

| Language | Status | Surface |
|---|---|---|
| C / C++ | ✓ | Full (the C-ABI is the source of truth) |
| Python | ✓ | Full — transcribe, VAD, diarize, LID, align, registry |
| Rust | ✓ | Full — same surface as Python |
| Dart / Flutter | ✓ | Full — used by [CrisperWeaver](https://github.com/CrispStrobe/CrisperWeaver) |
| Go | ✓ | Full (all 11 capabilities) |
| Java | ✓ | Transcribe + align + LID; full session-setter parity (JNA) |
| Ruby | ✓ | Transcribe; full session-setter parity (C ext) |
| C# / .NET | ✓ | Transcribe + align + LID + VAD + the music task surface; full session-setter parity (P/Invoke). CI-tested on ubuntu + windows (`bindings-csharp.yml`) |
| JavaScript / WASM | ✓ | `asrOpen`/`asrTranscribe` + session setters (backend-agnostic); plus the whisper-only `init`/`full_default` and the TTS surface. Built with emcc. |

> **Setter parity.** Python, Rust (`crispasr-sys` + `crispasr` at the repo root),
> Go, Dart, Java, Ruby, and C# all expose the complete `crispasr_session_set_*`
> surface declared in `include/crispasr_session.h` and `include/crispasr.h`
> (a handful — `set_sensitivity`, `set_pcm_sample_rate` — live in the latter).
> The native Node addon
> (`examples/addon.node`) reaches it via `transcribeSession`; the WASM/JS binding
> (`bindings/javascript/emscripten.cpp`) via the `asr*` functions
> (`asrOpen`/`asrTranscribe`/`asrSet…`).
>
> **Long audio auto-chunks.** `transcribe` slices audio longer than
> ~30 s at energy minima and transcribes each piece (like the CLI/server),
> collapsing any decode-loop repetition — so short-segment models (e.g.
> moonshine) don't degrade or hang on a single long pass. Disable with
> `CRISPASR_SESSION_AUTOCHUNK=0`; window via `CRISPASR_SESSION_CHUNK_SECONDS`.
> `transcribe_chunked` remains the explicit, tunable long-form control (and
> parakeet has its own internal long-audio handling either way).
>
> **Chunked long-form + progress (issue #208).** `transcribe_chunked` forces
> the Parakeet backend through its bounded long-form path (inert on other
> backends) and is exposed in **every** binding:
>
> _Issue #257:_ when `chunk_seconds > 0` is passed explicitly, the non-JA
> Parakeet path now runs one coherent internal-streamed decode at the model's
> quality encoder window (complete text, bounded VRAM — small encoder windows
> degrade this full-attention FastConformer) and returns **~`chunk_seconds`-second
> segments** (per-segment `start`/`end`/`words`), instead of one giant segment.
> `chunk_seconds <= 0` keeps the single-merged-segment #208 contract.
>
> Binding names:
> `Session.transcribe_chunked` (Python), `CrispasrSession.TranscribeChunked`
> (Go), `.transcribeChunked` (Java/Dart), `Session.transcribe_chunked` (Ruby),
> `asrTranscribeChunked` (WASM), `transcribeSession({chunk_seconds,…})` (Node
> addon), and Rust `Session::transcribe_chunked[_with_language]`. Two ways to
> surface per-window progress:
> 1. **Poll (universal, no callback).** `crispasr_get_progress()` returns
>    `0..100` (-1 idle) and now tracks the chunked-merge windows in lockstep (it
>    was previously only fed by whisper). Exposed as `Session.get_progress`
>    (Python/Ruby), `GetProgress()` (Go), `.getProgress()` (Java),
>    `getTranscriptionProgress()` (Dart), `asrGetProgress()` (WASM). This is the
>    Dart-friendly path (Dart FFI can't take C function-pointer callbacks).
> 2. **Native callback.** `crispasr_session_set_progress_callback(s, cb,
>    user_data)` — `cb(processed_samples, total_samples, user_data)` fires once
>    per finished window on the transcribe thread. Exposed where native
>    callbacks are idiomatic and safe: C/C++, Rust
>    (`Session::transcribe_chunked_with_progress`), and Python
>    (`transcribe_chunked(..., progress=fn)`). The other bindings use the poll.

## Python

```python
from crispasr import (
    Session, diarize_segments, diarize_segments_with_turns,
    detect_language_pcm,
    align_words, cache_ensure_file, registry_default_bundle,
    # Diarize pipeline primitives (#107):
    SpeakerEmbedder, PyannoteCache, agglomerative_cluster,
)

# Transcribe (any of the 24 ASR backends via one session object)
sess = Session("parakeet-tdt-0.6b-v3-q4_k.gguf")
sess.set_max_new_tokens(256)       # AR backends; <= 0 clears
sess.set_frequency_penalty(0.4)    # AR backends; <= 0 disables
segs = sess.transcribe_vad(pcm, "silero-v6.2.0.bin")  # stitched VAD pass

# Run each shared post-step standalone
lang = detect_language_pcm(pcm, model_path="ggml-tiny.bin")
diarize_segments(my_segs, pcm, method=DiarizeMethod.VAD_TURNS)
ok, turns = diarize_segments_with_turns(
    my_segs, pcm, method=DiarizeMethod.FOXNOSE,
    foxnose_embedder_path="wespeaker-resnet34-lm.gguf",
)
words = align_words("canary-ctc-aligner.gguf", "hello world", pcm)

# Inspect the canonical bundle used by `-m auto` (no quant suffix).
# NOTE: this does not apply a preferred quant, so it does NOT reproduce
# `-m auto:q8_0` — that rewrites both filename and URL. Use registry_lookup()
# with a preferred quant for those.
bundle = registry_default_bundle("omnivoice")
assert not bundle.requires_acceptance  # prompt/attest before restricted downloads
paths = [cache_ensure_file(a.filename, a.url) for a in bundle.artifacts]

# Custom diarize pipeline: pluggable embedder + cosine clustering.
# Same building blocks as `--diarize-embedder` in the CLI.
emb = SpeakerEmbedder("auto", n_threads=4)             # 'titanet'/'indextts'/.gguf
embeddings = [emb.embed(pcm[int(s.start*16000):int(s.end*16000)]) for s in segs]
labels = agglomerative_cluster(embeddings, merge_threshold=0.5, max_speakers=8)
emb.close()
```

Install: `pip install crispasr` (or build locally from `python/`).

## Rust

```rust
use crispasr::{
    Session, DiarizeMethod, DiarizeOptions, DiarizeSegment, DiarizeTurn,
    LidMethod, detect_language_pcm, align_words,
    cache_ensure_file, registry_default_bundle,
    // Diarize pipeline primitives (#107):
    SpeakerEmbedder, PyannoteCache, agglomerative_cluster,
};

let sess = Session::open("cohere-transcribe-q4_k.gguf")?;   // or open_with_backend(path, "cohere", 4)
sess.set_max_new_tokens(256)?;
sess.set_frequency_penalty(0.4)?;
let segs = sess.transcribe_vad(&pcm, "silero-v6.2.0.bin", None)?;

let bundle = registry_default_bundle("canary")?.unwrap();
assert!(!bundle.requires_acceptance); // obtain explicit acceptance when true
for artifact in bundle.artifacts {
    cache_ensure_file(&artifact.filename, &artifact.url, false, None)?;
}

// Custom diarize pipeline: pluggable embedder + cosine clustering.
let emb = SpeakerEmbedder::new("auto", 4, None)?;     // "titanet"/"indextts"/.gguf
let mut flat: Vec<f32> = Vec::new();
for s in &segs {
    if let Some(v) = emb.embed(&pcm[(s.start * 16000.0) as usize .. (s.end * 16000.0) as usize]) {
        flat.extend(v);
    }
}
let labels = agglomerative_cluster(&flat, (flat.len() / emb.dim() as usize) as i32,
                                   emb.dim(), 0.5, 8)?;

// #395: labels alone can only ever be as fine as the grid you send in. Ask for
// the turns FoxNose derived from the AUDIO to split a segment that spans two
// speakers. Segments are labelled exactly as diarize_segments() labels them;
// the other methods return an empty Vec, which is not an error.
let mut grid: Vec<DiarizeSegment> = /* your coarse, well-clustering segments */ vec![];
let turns: Vec<DiarizeTurn> =
    crispasr::diarize_segments_with_turns(&mut grid, &pcm, None, false, &opts)?;
for t in &turns {
    println!("{:.2}–{:.2} speaker {}", t.t0, t.t1, t.speaker); // caller's timeline
}
```

Crates: `crispasr-sys/` (raw FFI) + `crispasr/` (high-level) at the repo
root, both published on crates.io. The `-sys` crate's `build.rs` builds
`libcrispasr` with cmake from the CrispASR sources, or links a pre-built copy
when `CRISPASR_LIB_DIR` is set. The crates.io package does not vendor the C/C++
sources, so consume via a **git dependency** to build from source
(`crispasr = { git = "https://github.com/CrispStrobe/CrispASR" }`), or use
`crispasr = "0.8"` from crates.io together with a pre-built lib + `CRISPASR_LIB_DIR`.

## Dart / Flutter

```dart
import 'package:crispasr/crispasr.dart' as crispasr;

final sess = crispasr.CrispasrSession.open(modelPath, backend: 'parakeet');
final segs = sess.transcribeVad(pcm, vadModelPath);

final lang = crispasr.detectLanguagePcm(
  pcm: pcm, method: crispasr.LidMethod.whisper, modelPath: tinyPath);
final words = crispasr.alignWords(
  alignerModel: ctcPath, transcript: text, pcm: pcm);
```

Package: `flutter/crispasr/`.

**Reference application:**
[CrisperWeaver](https://github.com/CrispStrobe/CrisperWeaver) — a
cross-platform Flutter desktop/mobile transcription app built on
`package:crispasr`. Ships with a model browser + downloader (all 10
backends + quants), drag-and-drop files, mic capture, SRT/VTT/TXT
export, per-run performance metrics, and full en/de i18n. The v0.5.4
release uses `transcribeVad` so every non-whisper backend benefits
from stitched Silero VAD with zero CrisperWeaver-side work.

## Go

```go
import whisper "github.com/CrispStrobe/CrispASR/bindings/go"

sess, _ := whisper.SessionOpen("parakeet.gguf", 4)   // or SessionOpenExplicit(path, "parakeet", 4)
defer sess.Close()
_ = sess.SetMaxNewTokens(256)
_ = sess.SetFrequencyPenalty(0.4)
res, _ := sess.TranscribeVAD(pcm, 16000, "silero-v6.2.0.bin")   // or sess.Transcribe(pcm)
```

Module: `bindings/go` (package name `whisper`); the whisper-only high-level
wrapper lives in `bindings/go/pkg/whisper`.

## Java

```java
import io.github.ggerganov.whispercpp.CrispasrSession;

try (var sess = CrispasrSession.open("granite-speech.gguf", 4)) {
    sess.setMaxNewTokens(256);
    sess.setFrequencyPenalty(0.4f);
    var segs = sess.transcribe(pcm);
}
```

JAR: `bindings/java/`.

## Ruby

```ruby
require "whisper"

# Every Session entry point is a MODULE method taking the opaque handle first.
handle = Whisper::CrispASR::Session.open("parakeet.gguf", 4)
segs   = Whisper::CrispASR::Session.transcribe(handle, pcm)
Whisper::CrispASR::Session.close(handle)
```

Gem: `bindings/ruby/` (gem name `whispercpp`, required as `whisper`).

## Node.js addon

`examples/addon.node` is a native N-API addon (built via cmake-js). Besides the
legacy whisper-only `whisper()` entry point, it exposes `transcribeSession()`
over the `crispasr_session` C-ABI — reaching every ASR backend plus the session
post-processors (punctuation, `punc_model`, beam, translate, src/tgt lang):

```js
const { transcribeSession } = require('./build/Release/addon.node');
const { promisify } = require('util');
const run = promisify(transcribeSession);

const r = await run({
  model: 'parakeet.gguf', backend: 'parakeet', language: 'en',
  punctuation: true, punc_model: 'fullstop',   // restore punctuation
  fname_inp: 'audio.wav',
});
// { language, transcription: [[t0, t1, text], ...] }
```

For browser / pure-WASM use, see `bindings/javascript` (emscripten).

## Mobile

```bash
./build-ios.sh                    # iOS xcframework with Metal
./build-android.sh --vulkan       # Android NDK with Vulkan GPU
```

`build-ios.sh` emits `build-ios/CrispASR.xcframework`, which drops into a
Swift/Objective-C app as an embedded binary framework; the Android NDK
build produces an `.so` that Flutter or native Android consumes through
`package:crispasr`'s FFI layer.

## Text-to-speech

Every binding above (Python, Rust, Dart/Flutter, Go, Java, JavaScript,
Ruby, C#) reaches all TTS backends through the same two unified-C-API calls,
so there is nothing TTS-specific per wrapper:

- `synthesize(text) -> float32 PCM (mono, backend-native rate — 24 kHz
  for most, 48 kHz for irodori/voxcpm2)` (`crispasr_session_synthesize`)
- `synthesize_streaming(text, cb, user)` — same, but fires `cb(pcm,
  n_samples, is_final, user)` once per sentence chunk as it's produced, for
  progressive playback (`crispasr_session_synthesize_streaming`). The PCM is
  owned by the call; copy it in the callback if you need to keep it.
- `set_voice(path, ref_text?)` — `path` is a preset/baked-voice name
  **or** a `*.wav` clone reference (`ref_text` required for a WAV);
  `set_instruct(...)` for qwen3-tts VoiceDesign.

For cloning backends whose reference encode is expensive (irodori, indextts),
the encoded conditioning is cached automatically (content-addressed on the
reference audio) so a repeated reference skips the encode — this happens in the
runtime, so wrappers get it for free. Control with `CRISPASR_TTS_REF_CACHE=0`
(disable) / `CRISPASR_TTS_REF_CACHE_DIR` (location).

Open the TTS model GGUF like any other; the backend auto-detects from
the GGUF architecture. There are ~33 `CAP_TTS` backends; the commonly
used ones are `kokoro`, `qwen3-tts`
(+ customvoice), `vibevoice-tts` / `vibevoice-1.5b`, `orpheus`,
`chatterbox`, `indextts`, `voxcpm2-tts`, `cosyvoice3-tts`,
`lfm2-audio`, and `mini-omni2`. See
[`tts.md`](tts.md) for per-backend cloning + voice details.

**Provenance:** `synthesize()` automatically embeds the AI-generated
watermark (spread-spectrum or AudioSeal) into the returned PCM. No
manual step needed — all binding consumers get watermarked audio by
default. For advanced use cases that need DSP (speed change, mixing,
concatenation) before watermarking, use `synthesize_raw()` +
`watermark_embed()` instead. The spoken disclaimer is not applied at
the C API level (see
[`tts.md`](tts.md#spoken-disclaimer-voice-clones-only)).

### Whose voice is a preset voice? (`set_speaker_identity`)

`synthesize()` marks every clip (Art. 50(2)), and the voice-clone gate handles
cloning. Neither covers the third case: a **preset** voice shipped inside a
model can be an identifiable individual — a named donor, or a corpus speaker
such as VCTK's `p225` — and Art. 3(60) attaches to the audio resembling that
person, not to which pipeline produced it. CrispASR resolves this automatically
where it has researched the model, and warns once per model where it has not.

When you know the answer and CrispASR does not, say so:

| Binding | Call |
|---|---|
| Python | `session.set_speaker_identity("real_person")` |
| Rust | `session.set_speaker_identity("real_person")?` |
| Go | `session.SetSpeakerIdentity("real_person")` |
| Java | `session.setSpeakerIdentity("real_person")` |
| C# | `session.SetSpeakerIdentity("real_person")` |
| Dart | `session.setSpeakerIdentity('real_person')` |
| Ruby | `CrispASR::Session.set_speaker_identity(handle, "real_person")` |
| WASM | `Module.ttsSetSpeakerIdentity("real_person")` |

Values are `real_person`, `synthetic` and `unknown`. An unrecognised value
**raises** rather than silently becoming `unknown` — a typo must not quietly
remove a duty you meant to declare.

`real_person` makes the Art. 50(4) reminder fire for a non-cloned voice. It
does **not** require a consent attestation: whether the donor agreed to the
model being trained is a licensing matter settled upstream, which you cannot
attest to. Do not reach for `synthetic` to quiet a warning you have not
checked — of every model whose provenance this project has resolved, the large
majority turned out to be real people.

> **The CLI `--no-watermark` flag and the `CRISPASR_NO_WATERMARK` env var do
> NOT affect the bindings.** They are wired into the `crispasr` CLI and server
> only; `synthesize()` and `crispasr_watermark_embed()` watermark
> unconditionally. A binding consumer that legitimately needs unwatermarked
> output uses `synthesize_raw()` and simply does not call `watermark_embed()` —
> and thereby assumes the AI-content marking responsibility itself (see
> [`tts.md`](tts.md#disabling-the-watermark-operator-opt-out) for what that
> means).

```python
# Python (identical shape in every binding)
s = crispasr.Session("cosyvoice3-llm-f16.gguf")   # backend auto-detected
s.set_voice("fleurs-de")                          # baked-bank voice name
pcm = s.synthesize("Hallo, das ist ein Test.")    # float32 @ 24 kHz
# Voice cloning from a WAV:
s.set_voice("ref.wav", ref_text="exact transcription of ref.wav")
pcm = s.synthesize("Clone my voice.")
```

## Voice conversion (SVC / RVC)

**The session C ABI is the only surface — there is no CLI verb.** RVC's input is
ContentVec features, which CrispASR does not produce (the caller owns the
content encoder), so a command line has nothing to feed it.

```c
// content: n_frames * content_dim, frame-major. f0_hz: n_frames, 0.0 = unvoiced.
// The coarse mel-quantised pitch is derived internally — those constants are
// model-side and replicating them in the caller guarantees drift.
int crispasr_session_convert(crispasr_session* s, const float* content, int n_frames,
                             const float* f0_hz, int speaker_id,
                             const float* noise_zp, const float* noise_sine);
const float* crispasr_session_convert_audio(crispasr_session* s, int* out_n_samples);
int crispasr_session_convert_content_dim(crispasr_session* s);   // 256 (v1) or 768 (v2)
int crispasr_session_convert_n_speakers(crispasr_session* s);
int crispasr_session_convert_sample_rate(crispasr_session* s);   // 32k/40k/48k
```

### Check `convert_content_dim()` before you call

256 means v1 (ContentVec layer 9 + `final_proj`); 768 means v2 (final layer).
Feeding a v2 encoder's features to a v1 checkpoint is **silent** — it produces
audio that merely sounds poor. This accessor exists so the mismatch can be
refused loudly, which the caller cannot detect on its own.

### Conversion is STOCHASTIC — that is not a bug

Two independent RNG sites (the latent sample and the sine source's additive
noise) mean output varies run to run by design. Pass `NULL` for both noise
buffers in production.

Passing explicit buffers replays a specific draw and makes the call
**bit-identical**. That is the only way to compare against another
implementation: correlating waveforms against a reference run is invalid here,
because the reference disagrees with itself.

| buffer | size |
|---|---|
| `noise_zp` | `inter_channels * n_frames` (192 × N for every shipped config) |
| `noise_sine` | `n_frames * (sample_rate / 100)` |

Feature and F0 rate is **100 Hz** — derived as `sample_rate / prod(upsample_rates)`,
not configured. See `docs/music-transcription/SVC_RECORD_SHAPES.md` for the full
wire contract and `RVC_BLUEPRINT.md` for the ~15 implementation details the port
reproduces.

### Licence

RVC's code is MIT, but **checkpoints are not uniformly so** — community voice
models have unclear provenance and some forks add non-commercial terms. Each
GGUF carries its own tag and the registry gate matches on it, exactly as for the
BTC chord weights.

## Chord recognition

The `btc-chords` backend is a standalone task (CLI `--chords`) — audio in, a
chord timeline out. It is exposed on the session C-ABI
(`include/crispasr_session.h`) and, on top of that, in the WASM/JS binding:

- `crispasr_session_chords(s, pcm, n_samples, sample_rate)` — returns the span
  count, `-1` on error or on a backend with no chord arm. Input is mono
  float32 at any rate; it is resampled internally to the model's 22050 Hz.
- `crispasr_session_chords_n_spans(s)`
- `crispasr_session_chords_spans(s, &n)` — flat, session-owned float view,
  4 floats per span: `{start_ms, end_ms, label, confidence}`.
- `crispasr_session_chords_span_name(s, idx)` — resolves `label` to a chord
  name (`"C"`, `"Am"`, `"G:7"`, `"N"` for no-chord).
- `crispasr_session_chords_vocab_size(s)` — `25` or `170`, `0` if the session
  has no chord arm; usable as a capability probe.

`CRISPASR_BTC_MAJ_MIN=1` collapses the 170-class output to the 25-class
maj/min vocabulary (default off — full 170-class).

```js
// JavaScript / WASM (bindings/javascript/emscripten.cpp)
const vocab = Module.sessionChordsVocabSize();   // 25 | 170 | 0 (no chord arm)
const spans = Module.sessionChords(audio, sampleRate);
// [{ startMs, endMs, chord, confidence }, ...]
```

C# binds the whole chord surface in `bindings/csharp/CrispASR/SessionMusic.cs`
(`Session.Chords(pcm, sampleRate)` + `Session.ChordsVocabSize`). The Go binding
links `-lbtc-chords` (cgo LDFLAGS resynced) but adds no hand-written wrapper
function; Python, Rust, Dart, Java and Ruby have no dedicated wrapper yet — the
C ABI above is the surface for all of them.

> **Weights are non-commercial.** The upstream BTC code is MIT and CrispASR
> itself is MIT, but the shipped weights (`cstr/btc-chords-GGUF`) are
> CC-BY-NC-SA — trained on Isophonics / Robbie Williams / UsPop2002 chord
> annotations. The registry refuses to download them without
> `--accept-license cc-by-nc-sa-4.0` (or the `CRISPASR_ACCEPT_LICENSE` env
> var). A commercial product must supply its own weights.

## Speech-to-speech

Backends with S2S capability (`lfm2-audio`, `mini-omni2`, `sidon`,
`voxcpm2-vae`) support
end-to-end audio-in → audio-out transformation through a single model
pass. Bound in every wrapper (Python, Rust, Go, Dart/Flutter, Java, Ruby, C#,
WASM/JS) and on the HTTP server (`POST /v1/audio/speech-to-speech`).

- `speech_to_speech(pcm) -> (float32 PCM, transcript)`
  (`crispasr_session_speech_to_speech`)

Input defaults to 16 kHz mono float32 PCM. Python callers with another input
rate call `set_pcm_sample_rate(rate)` before `speech_to_speech()`; Sidon and
VoxCPM2 AudioVAE then resample internally to 16 kHz. Conversational S2S
backends return 24 kHz; Sidon and VoxCPM2 AudioVAE return 48 kHz audio and an
empty transcript.

```python
# Python
import numpy as np, soundfile as sf
s = crispasr.Session("lfm2-audio-1.5b-q5_k.gguf")
audio, sr = sf.read("input.wav", dtype="float32")  # must be 16 kHz mono
out_pcm, transcript = s.speech_to_speech(audio)
print(f"Transcript: {transcript}")
sf.write("output.wav", out_pcm, 24000)
```

```python
# Sidon restoration from a 24 kHz source
s = crispasr.Session("sidon-v0.1-f16.gguf")
audio, sr = sf.read("input.wav", dtype="float32")
s.set_pcm_sample_rate(sr)
restored, _ = s.speech_to_speech(audio)
sf.write("restored.wav", restored, 48000)
```

```python
# VoxCPM2 AudioVAE upscaling
s = crispasr.Session("voxcpm2-vae-f32.gguf")
audio, sr = sf.read("input.wav", dtype="float32")
s.set_pcm_sample_rate(sr)
upscaled, _ = s.speech_to_speech(audio)
sf.write("upscaled.wav", upscaled, 48000)
```

```go
// Go
s, _ := whisper.SessionOpen("lfm2-audio-1.5b-q5_k.gguf", 4)
defer s.Close()
result, _ := s.SpeechToSpeech(inputPCM)
fmt.Println("Transcript:", result.Transcript)
// result.PCM is []float32 at 24 kHz
```
