# crispasr

Safe Rust wrapper for [CrispASR](https://github.com/CrispStrobe/CrispASR) — lightweight on-device speech recognition via ggml.

Supports 17 ASR backends including Whisper, Qwen3-ASR, FastConformer, Canary, Parakeet, Cohere, Granite-Speech, Voxtral, wav2vec2, GLM-ASR, Kyutai-STT, Moonshine, FireRed, OmniASR, and VibeVoice-ASR.

## Install

```toml
[dependencies]
crispasr = "0.1"
```

You also need `libcrispasr` installed on the system — this crate is a thin wrapper around `libcrispasr` and does **not** build it. Same pattern as CrispASR's other language bindings.

```bash
git clone https://github.com/CrispStrobe/CrispASR
cd CrispASR
cmake -B build && cmake --build build -j
sudo cmake --install build
```

If `libcrispasr` is in a non-standard location, set `CRISPASR_LIB_DIR`.

## Quick start

```rust
use crispasr::CrispAsr;

let model = CrispAsr::open("ggml-base.en.bin")?;
for seg in model.transcribe_file("audio.wav")? {
    println!("[{:.1}s - {:.1}s] {}", seg.start, seg.end, seg.text);
}
```

See the [main repo](https://github.com/CrispStrobe/CrispASR) for full documentation, the model registry, and the CLI.

## License

MIT — see [LICENSE](LICENSE).
