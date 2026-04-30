// Link to a pre-installed `libcrispasr` (or its legacy `libwhisper` alias).
//
// This crate is a thin FFI shim — it does NOT build the native library.
// The user is expected to have `libcrispasr.{so,dylib,dll}` installed
// (Homebrew, apt, or built from source). Same pattern as crispasr's
// language bindings, the Python wrapper in this repo, and the Dart
// wrapper in `flutter/crispasr/`.
//
// Override the search path with `CRISPASR_LIB_DIR=/path/to/lib`.
// Override the library name with `CRISPASR_LIB_NAME=crispasr` (default)
// or `CRISPASR_LIB_NAME=whisper` for the legacy alias.

use std::env;
use std::path::Path;

fn main() {
    println!("cargo:rerun-if-env-changed=CRISPASR_LIB_DIR");
    println!("cargo:rerun-if-env-changed=CRISPASR_LIB_NAME");

    if let Ok(dir) = env::var("CRISPASR_LIB_DIR") {
        println!("cargo:rustc-link-search=native={dir}");
    }

    // Standard install prefixes — the linker probes these in order.
    for d in &[
        "/opt/homebrew/lib", // macOS arm64 Homebrew
        "/usr/local/lib",    // macOS x64 Homebrew, /usr/local installs
        "/usr/lib",
        "/usr/lib/x86_64-linux-gnu", // Debian/Ubuntu multiarch
        "/usr/lib/aarch64-linux-gnu",
    ] {
        if Path::new(d).is_dir() {
            println!("cargo:rustc-link-search=native={d}");
        }
    }

    let lib = env::var("CRISPASR_LIB_NAME").unwrap_or_else(|_| "crispasr".to_string());
    println!("cargo:rustc-link-lib=dylib={lib}");
}
