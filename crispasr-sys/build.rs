// Build (or locate) `libcrispasr` for the FFI shim.
//
// Decision tree (first match wins):
//
//   1. CRISPASR_LIB_DIR is set                  → legacy path: assume the
//                                                 library is already
//                                                 installed at that prefix
//                                                 (Homebrew/apt). No build.
//                                                 Standard system prefixes
//                                                 are added as a fallback.
//   2. CRISPASR_SYS_LIB_DIR is set + valid      → use that cmake build dir
//                                                 (lib expected under
//                                                 `<dir>/src/lib*`).
//   3. An in-tree `<repo>/build*` dir exists    → use it (covers users who
//                                                 already ran the cmake
//                                                 flow themselves; same
//                                                 convention as
//                                                 CrisperWeaver's
//                                                 `build-flutter-bundle`).
//   4. Fallback                                  → run cmake ourselves into
//                                                 OUT_DIR and link against
//                                                 the freshly built lib.
//                                                 Honours the cuda / metal
//                                                 / vulkan cargo features.
//
// In every case the `crispasr` library name is the default; override with
// `CRISPASR_LIB_NAME=whisper` for the legacy alias.

use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

fn link_lib_name() -> String {
    env::var("CRISPASR_LIB_NAME").unwrap_or_else(|_| "crispasr".to_string())
}

fn add_link_search(dir: &Path) {
    println!("cargo:rustc-link-search=native={}", dir.display());
}

fn add_build_dir_search(build_dir: &Path) {
    // Lay out paths the way CrispASR's CMakeLists actually places its
    // outputs: libs go under `<build>/src/` (and `src/Release/` for
    // MSVC multi-config); ggml shared libs go under `<build>/ggml/src/`.
    for sub in ["src", "src/Release", "ggml/src", "ggml/src/Release"] {
        add_link_search(&build_dir.join(sub));
    }
    // Also probe the build dir root for "flat" CMake generators.
    add_link_search(build_dir);
}

fn print_link_lib(lib_name: &str) {
    println!("cargo:rustc-link-lib=dylib={lib_name}");
    match env::var("CARGO_CFG_TARGET_OS").unwrap_or_default().as_str() {
        "linux" => println!("cargo:rustc-link-lib=dylib=stdc++"),
        "macos" => println!("cargo:rustc-link-lib=dylib=c++"),
        _ => {}
    }
}

/// Emit `cargo:rustc-link-arg=-Wl,-rpath,...` directives so the executable
/// can locate `libcrispasr` at runtime.
///
/// macOS dylibs are linked with their install name as `@rpath/...`, so the
/// loader needs at least one LC_RPATH that resolves. Without this, even
/// `cargo run` after a successful link fails with "no LC_RPATH's found".
///
/// We add two entries:
///
/// * `<build_dir>/src` — absolute path to the freshly built lib. Lets
///   `cargo run` / `cargo test` work directly out of the workspace
///   without any post-build copy step.
/// * `@executable_path/../Frameworks` (macOS) /
///   `$ORIGIN/../lib` (Linux) — relative to the executable, so an app
///   bundled by Tauri (or any downstream consumer) finds the lib once
///   it has been copied into `Contents/Frameworks/` (macOS) or
///   `<bin>/../lib/` (Linux). Windows needs no rpath — DLLs are found
///   alongside the .exe.
fn emit_runtime_rpath(build_dir: &Path) {
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let lib_subdir = build_dir.join("src");
    let lib_subdir_str = lib_subdir.display();
    let ggml_subdir = build_dir.join("ggml").join("src");
    let ggml_subdir_str = ggml_subdir.display();
    match target_os.as_str() {
        "macos" => {
            println!("cargo:rustc-link-arg=-Wl,-rpath,{lib_subdir_str}");
            println!("cargo:rustc-link-arg=-Wl,-rpath,{ggml_subdir_str}");
            println!("cargo:rustc-link-arg=-Wl,-rpath,@executable_path/../Frameworks");
            println!("cargo:rustc-link-arg=-Wl,-rpath,@loader_path/../Frameworks");
        }
        "linux" => {
            println!("cargo:rustc-link-arg=-Wl,-rpath,{lib_subdir_str}");
            println!("cargo:rustc-link-arg=-Wl,-rpath,{ggml_subdir_str}");
            // $ORIGIN must be quoted carefully so the linker (not the
            // shell) sees it; cargo:rustc-link-arg passes through verbatim.
            println!("cargo:rustc-link-arg=-Wl,-rpath,$ORIGIN/../lib");
            println!("cargo:rustc-link-arg=-Wl,-rpath,$ORIGIN");
        }
        _ => {} // Windows: DLL search path includes the exe's directory.
    }
}

fn has_built_lib(build_dir: &Path, lib_name: &str) -> bool {
    let candidates = [
        // Linux / macOS, Ninja or Unix Makefiles.
        build_dir.join("src").join(format!("lib{lib_name}.dylib")),
        build_dir.join("src").join(format!("lib{lib_name}.so")),
        // Legacy `whisper` alias produced by the same target.
        build_dir.join("src").join("libwhisper.dylib"),
        build_dir.join("src").join("libwhisper.so"),
        // MSVC multi-config: import lib lives under `src/Release/`.
        build_dir
            .join("src")
            .join("Release")
            .join(format!("{lib_name}.lib")),
        build_dir.join("src").join("Release").join("whisper.lib"),
        // Flat layouts (e.g., users who pointed CRISPASR_SYS_LIB_DIR at a
        // directory that already contains the libs without the `src/`
        // prefix).
        build_dir.join(format!("lib{lib_name}.dylib")),
        build_dir.join(format!("lib{lib_name}.so")),
        build_dir.join(format!("{lib_name}.lib")),
    ];
    candidates.iter().any(|p| p.exists())
}

fn try_existing_build(src_root: &Path, lib_name: &str) -> Option<PathBuf> {
    if let Ok(dir) = env::var("CRISPASR_SYS_LIB_DIR") {
        let path = PathBuf::from(dir);
        if has_built_lib(&path, lib_name) {
            return Some(path);
        }
    }
    let candidates = [
        src_root.join("build-cuda"),
        src_root.join("build-vulkan"),
        src_root.join("build-metal"),
        // CrisperWeaver convention so a CrispSorter dev who already built
        // for CrisperWeaver doesn't pay the cost twice.
        src_root.join("build-flutter-bundle"),
        src_root.join("build"),
    ];
    candidates
        .into_iter()
        .find(|p| has_built_lib(p, lib_name))
}

fn run(cmd: &mut Command, what: &str) {
    let status = cmd
        .status()
        .unwrap_or_else(|err| panic!("failed to start {what}: {err}"));
    if !status.success() {
        panic!("{what} failed with status {status}");
    }
}

fn configure_and_build(src_root: &Path) -> PathBuf {
    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR not set"));
    let build_dir = out_dir.join("crispasr-build");

    let mut configure = Command::new("cmake");
    configure
        .arg("-S")
        .arg(src_root)
        .arg("-B")
        .arg(&build_dir)
        .arg("-DBUILD_SHARED_LIBS=ON")
        .arg("-DCRISPASR_BUILD_TESTS=OFF")
        .arg("-DCRISPASR_BUILD_EXAMPLES=OFF")
        .arg("-DCRISPASR_BUILD_SERVER=OFF")
        .arg("-DCMAKE_BUILD_TYPE=Release");

    if cfg!(feature = "cuda") {
        configure.arg("-DGGML_CUDA=ON");
    }
    if cfg!(feature = "metal") {
        configure.arg("-DGGML_METAL=ON");
        configure.arg("-DGGML_METAL_EMBED_LIBRARY=ON");
    }
    if cfg!(feature = "vulkan") {
        configure.arg("-DGGML_VULKAN=ON");
    }

    run(&mut configure, "cmake configure");

    let mut build = Command::new("cmake");
    build
        .arg("--build")
        .arg(&build_dir)
        .arg("--config")
        .arg("Release")
        .arg("--target")
        .arg("crispasr");
    run(&mut build, "cmake build");

    build_dir
}

fn add_system_lib_search_paths() {
    if let Ok(dir) = env::var("CRISPASR_LIB_DIR") {
        add_link_search(Path::new(&dir));
    }
    for d in &[
        "/opt/homebrew/lib",
        "/usr/local/lib",
        "/usr/lib",
        "/usr/lib/x86_64-linux-gnu",
        "/usr/lib/aarch64-linux-gnu",
    ] {
        if Path::new(d).is_dir() {
            add_link_search(Path::new(d));
        }
    }
}

fn main() {
    println!("cargo:rerun-if-env-changed=CRISPASR_SYS_LIB_DIR");
    println!("cargo:rerun-if-env-changed=CRISPASR_LIB_DIR");
    println!("cargo:rerun-if-env-changed=CRISPASR_LIB_NAME");

    let lib_name = link_lib_name();

    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let src_root = manifest_dir
        .parent()
        .expect("crispasr-sys must live inside the CrispASR repo root");

    // (1) Legacy path: explicit CRISPASR_LIB_DIR with no other build hint.
    let legacy_system_install = env::var("CRISPASR_LIB_DIR").is_ok()
        && env::var("CRISPASR_SYS_LIB_DIR").is_err()
        && try_existing_build(src_root, &lib_name).is_none();
    if legacy_system_install {
        add_system_lib_search_paths();
        print_link_lib(&lib_name);
        return;
    }

    // (2) / (3) An existing cmake build tree we can link against.
    if let Some(build_dir) = try_existing_build(src_root, &lib_name) {
        add_build_dir_search(&build_dir);
        emit_runtime_rpath(&build_dir);
        // Expose the resolved location to consuming crates' build scripts
        // via Cargo's `links` metadata channel (we declare `links =
        // "crispasr"` in Cargo.toml, so consumers see DEP_CRISPASR_LIB_DIR).
        println!("cargo:LIB_DIR={}", build_dir.display());
        print_link_lib(&lib_name);
        return;
    }

    // (4) Build it ourselves.
    let build_dir = configure_and_build(src_root);
    add_build_dir_search(&build_dir);
    emit_runtime_rpath(&build_dir);
    println!("cargo:LIB_DIR={}", build_dir.display());
    print_link_lib(&lib_name);
}
