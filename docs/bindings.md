# Language bindings

All wrappers are thin shells over the same C-ABI surface in
`src/crispasr_c_api.cpp`. Anything the CLI can do — transcribe, VAD,
diarize, LID, align, download — is one function call in every
language.

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
| `set_tts_steps(n)` | `set_tts_steps` / `set_tts_steps` / `SetTTSSteps` / `setTtsSteps` | Chatterbox S3Gen CFM steps; vibevoice DPM-Solver++ steps |
| `set_top_p(p)` | `set_top_p` / `set_top_p` / `SetTopP` / `setTopP` | Chatterbox AR T3 loop |
| `set_min_p(p)` | `set_min_p` / `set_min_p` / `SetMinP` / `setMinP` | Chatterbox AR T3 loop |
| `set_repetition_penalty(r)` | `set_repetition_penalty` / `set_repetition_penalty` / `SetRepetitionPenalty` / `setRepetitionPenalty` | Chatterbox (1.0 = no penalty) |
| `set_cfg_weight(w)` | `set_cfg_weight` / `set_cfg_weight` / `SetCFGWeight` / `setCfgWeight` | Chatterbox (0.5 = upstream default; 0 = unconditional) |
| `set_exaggeration(e)` | `set_exaggeration` / `set_exaggeration` / `SetExaggeration` / `setExaggeration` | Chatterbox emotion scalar (0.5 = upstream default) |
| `set_max_speech_tokens(n)` | `set_max_speech_tokens` / `set_max_speech_tokens` / `SetMaxSpeechTokens` / `setMaxSpeechTokens` | Chatterbox AR loop token budget (default 1000 ≈ 20 s) |
| `set_length_scale(s)` | `set_length_scale` / `set_length_scale` / `SetLengthScale` / `setLengthScale` | Kokoro phoneme duration multiplier (1.0 = normal) |
| `set_best_of(n)` | `set_best_of` / `set_best_of` / `SetBestOf` / `setBestOf` | Best-of-N sampling for temperature > 0 |
| `set_beam_size(n)` | `set_beam_size` / `set_beam_size` / `SetBeamSize` / `setBeamSize` | Beam search width |
| `set_grammar_text(gbnf, root, penalty)` | `set_grammar_text` / `set_grammar_text` / `SetGrammarText` / `setGrammarText` | GBNF constrained decoding (whisper); empty string clears |
| `set_fallback_thresholds(...)` | `set_fallback_thresholds` / `set_fallback_thresholds` / `SetFallbackThresholds` / `setFallbackThresholds` | Whisper entropy/logprob/no-speech thresholds + temp-inc |
| `set_alt_n(n)` | `set_alt_n` / `set_alt_n` / `SetAltN` / `setAltN` | Per-token alternative candidates (whisper greedy) |
| `set_whisper_decode_extras(...)` | `set_whisper_decode_extras` / `set_whisper_decode_extras` / `SetWhisperDecodeExtras` / `setWhisperDecodeExtras` | suppress_nst, suppress_regex, carry_initial_prompt |
| `set_ask(prompt)` | `set_ask` / `set_ask` / `SetAsk` / `setAsk` | Free-form prompt for instruct-tuned audio-LLM backends (granite, voxtral, qwen3-asr, glm-asr, gemma4-e2b, mimo-asr). Empty string clears. |

> **Tip — chunk-boundary dedup for bindings.** When a binding drives a
> CAP_UNBOUNDED_INPUT backend (parakeet, canary, …) chunk-by-chunk and
> needs to stitch the output, call `crispasr_lcs_dedup_prefix_count`
> between adjacent chunks. It returns the number of leading tokens of
> `chunk[i]` that duplicate the tail of `chunk[i-1]` (NeMo-style
> sub-word LCS over emitted token ids). The binding then drops that
> many tokens from `chunk[i]` and rebuilds its own segment / word /
> text representation. The C declaration lives in `include/crispasr.h`;
> see also the `--lcs-dedup` / `--lcs-min-length` CLI flags.

| Language | Status | Surface |
|---|---|---|
| C / C++ | ✓ | Full (the C-ABI is the source of truth) |
| Python | ✓ | Full — transcribe, VAD, diarize, LID, align, registry |
| Rust | ✓ | Full — same surface as Python |
| Dart / Flutter | ✓ | Full — used by [CrisperWeaver](https://github.com/CrispStrobe/CrisperWeaver) |
| Go | ✓ | Full (all 11 capabilities) |
| Java | ✓ | Transcribe + align + LID |
| Ruby | ✓ | Transcribe |
| JavaScript | partial | WebAssembly approach; see PLAN.md #59 |

## Python

```python
from crispasr import (
    Session, diarize_segments, detect_language_pcm,
    align_words, cache_ensure_file, registry_lookup,
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
words = align_words("canary-ctc-aligner.gguf", "hello world", pcm)

# Auto-download a canonical model
entry = registry_lookup("parakeet")
path  = cache_ensure_file(entry.filename, entry.url)

# Custom diarize pipeline: pluggable embedder + cosine clustering.
# Same building blocks as `--diarize-embedder` in the CLI.
emb = SpeakerEmbedder("auto", n_threads=4)             # 'titanet'/'indextts'/.gguf
embeddings = [emb.embed(pcm[s.t0*16000:s.t1*16000]) for s in segs]
labels = agglomerative_cluster(embeddings, merge_threshold=0.5, max_speakers=8)
emb.close()
```

Install: `pip install crispasr` (or build locally from `python/`).

## Rust

```rust
use crispasr::{
    Session, DiarizeMethod, DiarizeOptions, DiarizeSegment,
    LidMethod, detect_language_pcm, align_words,
    cache_ensure_file, registry_lookup,
    // Diarize pipeline primitives (#107):
    SpeakerEmbedder, PyannoteCache, agglomerative_cluster,
};

let sess = Session::open("cohere-transcribe-q4_k.gguf", 4)?;
sess.set_max_new_tokens(256)?;
sess.set_frequency_penalty(0.4)?;
let segs = sess.transcribe_vad(&pcm, "silero-v6.2.0.bin", None)?;

let entry = registry_lookup("canary")?.unwrap();
let path  = cache_ensure_file(&entry.filename, &entry.url, false, None)?;

// Custom diarize pipeline: pluggable embedder + cosine clustering.
let emb = SpeakerEmbedder::new("auto", 4, None)?;     // "titanet"/"indextts"/.gguf
let mut flat: Vec<f32> = Vec::new();
for s in &segs {
    if let Some(v) = emb.embed(&pcm[(s.t0 * 16000.0) as usize .. (s.t1 * 16000.0) as usize]) {
        flat.extend(v);
    }
}
let labels = agglomerative_cluster(&flat, (flat.len() / emb.dim() as usize) as i32,
                                   emb.dim(), 0.5, 8)?;
```

Crates: `crispasr-sys/` (raw FFI) + `crispasr/` (high-level) at the repo
root; published as `crispasr-sys` / `crispasr` on crates.io.

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
import "github.com/CrispStrobe/CrispASR/bindings/go/crispasr"

sess, _ := crispasr.OpenSession("parakeet.gguf", crispasr.SessionOpts{Threads: 4})
defer sess.Close()
_ = sess.SetMaxNewTokens(256)
_ = sess.SetFrequencyPenalty(0.4)
segs, _ := sess.Transcribe(pcm, crispasr.TranscribeOpts{Vad: true})
```

Module: `bindings/go/crispasr/`.

## Java

```java
import org.crispasr.CrispASR;

try (var sess = CrispASR.openSession("granite-speech.gguf")) {
    sess.setMaxNewTokens(256);
    sess.setFrequencyPenalty(0.4f);
    var segs = sess.transcribe(pcm);
}
```

JAR: `bindings/java/`.

## Ruby

```ruby
require "crispasr"

sess = CrispASR::Session.open("parakeet.gguf")
segs = sess.transcribe(pcm)
```

Gem: `bindings/ruby/`.

## Mobile

```bash
./build-ios.sh                    # iOS xcframework with Metal
./build-android.sh --vulkan       # Android NDK with Vulkan GPU
```

The xcframework drops into a Swift/Objective-C app via `package add
crispasr.xcframework`; the Android NDK build produces an `.so` that
Flutter or native Android consumes through `package:crispasr`'s FFI
layer.

## Text-to-speech

Every binding above (Python, Rust, Dart/Flutter, Go, Java, JavaScript,
Ruby) reaches all TTS backends through the same two unified-C-API calls,
so there is nothing TTS-specific per wrapper:

- `synthesize(text) -> float32 PCM @ 24 kHz mono`
  (`crispasr_session_synthesize`)
- `set_voice(path, ref_text?)` — `path` is a preset/baked-voice name
  **or** a `*.wav` clone reference (`ref_text` required for a WAV);
  `set_instruct(...)` for qwen3-tts VoiceDesign.

Open the TTS model GGUF like any other; the backend auto-detects from
the GGUF architecture. Supported TTS backends: `kokoro`, `qwen3-tts`
(+ customvoice), `vibevoice-tts` / `vibevoice-1.5b`, `orpheus`,
`chatterbox`, `indextts`, `voxcpm2-tts`, and `cosyvoice3-tts`. See
[`tts.md`](tts.md) for per-backend cloning + voice details.

**Provenance responsibility:** `synthesize()` returns raw PCM with no
watermark or spoken disclaimer — those are applied at the CLI/server
layer. When using the C API or any binding directly, callers **must**
embed the watermark themselves via `watermark_embed()` (available in
every binding). The spoken disclaimer is optional (see
[`tts.md`](tts.md#spoken-disclaimer-voice-clones-only)), but
machine-readable provenance (watermark) should always be applied to
voice-cloned output for EU AI Act compliance.

```python
# Python (identical shape in every binding)
s = crispasr.Session("cosyvoice3-llm-f16.gguf")   # backend auto-detected
s.set_voice("fleurs-de")                          # baked-bank voice name
pcm = s.synthesize("Hallo, das ist ein Test.")    # float32 @ 24 kHz
# Voice cloning from a WAV:
s.set_voice("ref.wav", ref_text="exact transcription of ref.wav")
pcm = s.synthesize("Clone my voice.")
```
