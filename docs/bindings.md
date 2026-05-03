# Language bindings

All wrappers are thin shells over the same C-ABI surface in
`src/crispasr_c_api.cpp`. Anything the CLI can do — transcribe, VAD,
diarize, LID, align, download — is one function call in every
language.

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
)

# Transcribe (any of the 24 ASR backends via one session object)
sess = Session("parakeet-tdt-0.6b-v3-q4_k.gguf")
segs = sess.transcribe_vad(pcm, "silero-v5.1.2.bin")  # stitched VAD pass

# Run each shared post-step standalone
lang = detect_language_pcm(pcm, model_path="ggml-tiny.bin")
diarize_segments(my_segs, pcm, method=DiarizeMethod.VAD_TURNS)
words = align_words("canary-ctc-aligner.gguf", "hello world", pcm)

# Auto-download a canonical model
entry = registry_lookup("parakeet")
path  = cache_ensure_file(entry.filename, entry.url)
```

Install: `pip install crispasr` (or build locally from `python/`).

## Rust

```rust
use crispasr::{
    Session, DiarizeMethod, DiarizeOptions, DiarizeSegment,
    LidMethod, detect_language_pcm, align_words,
    cache_ensure_file, registry_lookup,
};

let sess = Session::open("cohere-transcribe-q4_k.gguf", 4)?;
let segs = sess.transcribe_vad(&pcm, "silero-v5.1.2.bin", None)?;

let entry = registry_lookup("canary")?.unwrap();
let path  = cache_ensure_file(&entry.filename, &entry.url, false, None)?;
```

Crate: `bindings/rust/`.

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
segs, _ := sess.Transcribe(pcm, crispasr.TranscribeOpts{Vad: true})
```

Module: `bindings/go/crispasr/`.

## Java

```java
import org.crispasr.CrispASR;

try (var sess = CrispASR.openSession("granite-speech.gguf")) {
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
