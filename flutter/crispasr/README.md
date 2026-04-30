# crispasr

Dart / Flutter FFI bindings for [CrispASR](https://github.com/CrispStrobe/CrispASR) — lightweight on-device speech recognition via ggml.

Supports 17 ASR backends including Whisper, Qwen3-ASR, FastConformer, Canary, Parakeet, Cohere, Granite-Speech, Voxtral, wav2vec2, GLM-ASR, Kyutai-STT, Moonshine, FireRed, OmniASR, and VibeVoice-ASR.

## Install

```yaml
dependencies:
  crispasr: ^0.4.9
```

This package is **pure Dart FFI** and does **not** bundle the native library — install `libcrispasr` separately, the same way `crispasr`'s bindings work:

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
