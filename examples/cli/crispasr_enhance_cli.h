// crispasr_enhance_cli.h — CLI side of the selectable denoise pre-pass.
//
// `--enhance SPEC` (or `--denoise` = rnnoise) cleans the decoded input before
// VAD, chunking and transcription, on both the whisper path (cli.cpp) and the
// backend dispatcher (crispasr_run.cpp). `--enhance-output FNAME` is a
// standalone verb: enhance each input, write it, exit. The method itself lives
// in the library (src/crispasr_enhancer.h) and is shared with the Session ABI.

#pragma once

#include <string>
#include <vector>

struct whisper_params;

/// True when params ask for enhancement (a non-"none" spec).
bool crispasr_cli_enhance_requested(const whisper_params& params);

/// Enhance `mono` (and every channel in `stereo`) in place at `sample_rate`.
/// Returns false only when the requested method cannot be loaded — the caller
/// should then fail, since the user explicitly asked for it. A processing
/// failure on one buffer warns and leaves it unchanged.
bool crispasr_cli_apply_enhance(const whisper_params& params, std::vector<float>& mono,
                                std::vector<std::vector<float>>& stereo, int sample_rate);

/// Standalone `--enhance-output` verb. Returns a process exit code.
int crispasr_run_enhance(const whisper_params& params);
