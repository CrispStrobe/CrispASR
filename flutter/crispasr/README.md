# crispasr

Dart / Flutter FFI bindings for [CrispASR](https://github.com/CrispStrobe/CrispASR) — lightweight on-device speech recognition and speech tooling via ggml.

The native project currently ships 143 backends, 69 TTS engines, language detection, punctuation, diarization, translation and chat/text helpers. This package exposes the Dart FFI API and expects the native `libcrispasr` library to be installed or bundled by the application.

## Install

```yaml
dependencies:
  crispasr: ^0.8.10
```

This package is **pure Dart FFI** and does **not** bundle the native library. Install `libcrispasr` separately or ship it with your app:

```bash
git clone https://github.com/CrispStrobe/CrispASR
cd CrispASR
cmake -B build && cmake --build build -j
sudo cmake --install build   # installs libcrispasr.{so,dylib,dll}
```

If `libcrispasr` is in a non-standard location, pass `libPath:` to the constructors.

## Quick start

```dart
import 'package:crispasr/crispasr.dart';

final model = CrispASR.open('ggml-base.en.bin');
final segments = model.transcribe('audio.wav');
for (final seg in segments) {
  print('[${seg.start}s - ${seg.end}s] ${seg.text}');
}
model.close();
```

See the [main repo](https://github.com/CrispStrobe/CrispASR) for full API docs, the model registry, and the CLI.

## License

MIT — see [LICENSE](LICENSE).
