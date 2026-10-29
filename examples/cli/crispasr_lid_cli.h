// crispasr_lid_cli.h — CLI-side language-identification shim.
//
// The whisper-tiny and native silero algorithms live in
// `src/crispasr_lid.{h,cpp}` so every CrispASR consumer reaches them
// through the shared library. This header keeps the CLI flag
// translation + the sherpa-onnx subprocess fallback + the
// auto-download / `~/.cache/crispasr` resolution that's CLI UX policy.

#pragma once

#include <string>

struct whisper_params; // fwd

// CLI-facing LID result. Same shape as the shared-lib result, plus
// historical naming for downstream tools.
struct crispasr_lid_result {
    std::string lang_code;
    float confidence = -1.0f;
    std::string source;
};

// Run LID from CLI flags. Dispatches to either the shared library
// (whisper / silero-native) or the CLI-local sherpa-onnx subprocess
// fallback. Resolves `params.lid_model` (auto-download on empty / "auto")
// and picks a sensible default model when omitted. With
// `params.lid_full_recording`, scores 15-second windows across the input.
bool crispasr_detect_language_cli(const float* samples, int n_samples, const whisper_params& params,
                                  crispasr_lid_result& out);

class CrispasrBackend; // fwd

// Ask the ALREADY-LOADED backend to identify the language itself, before
// reaching for an external detector. Returns false when the backend has no
// such path (the default), when the user named an external `--lid-backend`
// explicitly, or when the probe came back empty — in every case the caller
// should fall through to crispasr_detect_language_cli().
//
// Worth preferring where available for a correctness reason, not only to skip
// a download: a backend probing its OWN language set cannot return a language
// it does not support, while whisper-tiny LID knows 99 and regularly does.
// Three surfaces (file CLI, session path, server) run the same pre-step, so
// the decision lives here rather than being written out three times.
bool crispasr_backend_probe_language(CrispasrBackend& backend, const float* samples, int n_samples,
                                     const whisper_params& params, crispasr_lid_result& out);
