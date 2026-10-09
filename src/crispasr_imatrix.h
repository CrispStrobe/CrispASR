// crispasr_imatrix.h — importance-matrix (activation-statistics) collector.
//
// When the environment variable CRISPASR_IMATRIX_OUT is set to a file path,
// installing the collector on a ggml scheduler makes every ggml_mul_mat node
// whose src0 is a model weight accumulate the per-column sum-of-squares of its
// activation input (src1). At process exit the accumulated statistics are
// merged with any pre-existing file and written back, so a calibration corpus
// can be streamed across many CLI invocations that all point at the same
// output file.
//
// The resulting file is consumed by `crispasr-quantize --imatrix <file>`,
// which feeds the per-tensor importance vector to ggml_quantize_chunk so that
// k-quant / IQ-quant error is minimised on the weights the calibration data
// actually exercised (activation-weighted error instead of plain L2 — the same
// idea as llama.cpp's `llama-imatrix`).
//
// This is a no-op unless CRISPASR_IMATRIX_OUT is set; production paths are
// unaffected. Ported from CrispEmbed src/imatrix.cpp.

#pragma once

#include "ggml-backend.h"

#ifdef __cplusplus
extern "C" {
#endif

// Install the imatrix eval-callback on a scheduler. Safe no-op when
// CRISPASR_IMATRIX_OUT is unset. Call once, right after ggml_backend_sched_new.
void crispasr_imatrix_install(ggml_backend_sched_t sched);

// Obtain the collector for runtimes (e.g. llama) that own their scheduler.
// NULL when neither imatrix nor activation dumping is enabled; otherwise
// registers the same exit flush as install(). Compose with existing callbacks
// rather than replacing stage capture, and pass NULL as collector user data.
ggml_backend_sched_eval_callback crispasr_imatrix_callback(void);


// Merge the accumulated statistics with any existing output file and write it.
// Registered via atexit() on first install; may also be called explicitly.
void crispasr_imatrix_flush(void);

#ifdef __cplusplus
}
#endif
