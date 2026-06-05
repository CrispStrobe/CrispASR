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
// and picks a sensible default model when omitted.
bool crispasr_detect_language_cli(const float* samples, int n_samples, const whisper_params& params,
                                  crispasr_lid_result& out);
