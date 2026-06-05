// crispasr_diagnostics.h — startup diagnostics + --version / --diagnostics
//
// The Docker / CUDA bring-up surface is a popular failure mode for users
// (#31 chiefly), and the only debug information they could attach to a bug
// report used to be `nvidia-smi` from outside the container. This module
// gives the binary a way to dump everything it knows about its own build
// and the runtime environment in one place, gated either by --verbose,
// --diagnostics, or --version.
//
// Three independent printers, all idempotent and side-effect-free apart
// from writing to the FILE * passed in:
//
//   crispasr_print_build_info  — version, git SHA, build date, compiler,
//                                compiled-in ggml backends, supported
//                                CUDA archs. Pure compile-time info.
//   crispasr_print_runtime_env — env vars relevant to GPU bring-up,
//                                LD_LIBRARY_PATH, /etc/ld.so.conf.d
//                                contents, /usr/local/cuda/compat presence.
//   crispasr_print_devices     — calls ggml_backend_load_all() and
//                                enumerates the registered devices,
//                                prefixed with the backend reg name and
//                                feature flags. Tolerant of CUDA
//                                init failures: prints the error code
//                                and continues instead of aborting.

#pragma once

#include <cstdio>

void crispasr_print_build_info(FILE* out);
void crispasr_print_runtime_env(FILE* out);
void crispasr_print_devices(FILE* out);

// Convenience wrapper used at startup when --verbose / --diagnostics is set.
// Calls all three printers above with a header in between.
void crispasr_print_full_diagnostics(FILE* out);

// Compact one-line build banner ("crispasr <version> (<git>) <backends>").
// Always safe to call. Used unconditionally in the docker entrypoint and
// at the top of --verbose runs so a copy-paste of the first 5 lines of any
// log is enough to identify the build.
void crispasr_print_short_banner(FILE* out);
