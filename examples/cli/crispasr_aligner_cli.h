// crispasr_aligner_cli.h — CLI-side aligner shim.
//
// The dispatch + inference for both canary-ctc and qwen3-forced-aligner
// now lives in `src/crispasr_aligner.{h,cpp}`. This header keeps the
// thin CLI adapter that returns results as `crispasr_word` (the CLI
// type) instead of the library's `CrispasrAlignedWord`.

#pragma once

#include "crispasr_backend.h"

#include <string>
#include <vector>

std::vector<crispasr_word> crispasr_ctc_align(const std::string& aligner_model, const std::string& transcript,
                                              const float* samples, int n_samples, int64_t t_offset_cs, int n_threads);
