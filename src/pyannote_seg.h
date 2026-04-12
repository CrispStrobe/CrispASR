// pyannote_seg.h — native ggml runtime for pyannote-segmentation-3.0.
//
// Takes raw 16 kHz mono audio, outputs per-frame speaker activity
// probabilities (7 classes: silence + 3 speakers × 2 states).

#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct pyannote_seg_context;

struct pyannote_seg_context * pyannote_seg_init(const char * gguf_path, int n_threads);
void pyannote_seg_free(struct pyannote_seg_context * ctx);

// Run segmentation. Returns a malloc'd (T_out, 7) F32 buffer of
// log-softmax speaker activity probabilities. Caller frees.
// T_out is written back via out_T.
float * pyannote_seg_run(struct pyannote_seg_context * ctx,
                         const float * samples, int n_samples,
                         int * out_T);

#ifdef __cplusplus
}
#endif
