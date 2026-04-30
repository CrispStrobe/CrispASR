# crispasr-sys

Raw FFI bindings for [CrispASR](https://github.com/CrispStrobe/CrispASR) — lightweight on-device speech recognition via ggml.

This crate is a thin `extern "C"` shim and **does not** build the native library. Users install `libcrispasr.{so,dylib,dll}` separately — same pattern as CrispASR's other language bindings.

## Install

```toml
[dependencies]
crispasr-sys = "0.1"
```

You also need the native library:

```bash
git clone https://github.com/CrispStrobe/CrispASR
cd CrispASR
cmake -B build && cmake --build build -j
sudo cmake --install build
```

If `libcrispasr` is in a non-standard location, point the linker at it:

```bash
export CRISPASR_LIB_DIR=/path/to/lib
```

The legacy `libwhisper` alias also works:

```bash
export CRISPASR_LIB_NAME=whisper
```

For the safe high-level wrapper see the [`crispasr`](https://crates.io/crates/crispasr) crate.

## License

MIT — see [LICENSE](LICENSE).
