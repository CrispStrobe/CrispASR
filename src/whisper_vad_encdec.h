#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct whisper_vad_encdec_context;

struct whisper_vad_encdec_segment {
    float start_sec;
    float end_sec;
};

// Initialize from GGUF. Returns nullptr on failure.
struct whisper_vad_encdec_context* whisper_vad_encdec_init(const char* model_path);

// Run VAD detection. Returns speech segments (caller frees *segments with free()).
// probs_out: if non-null, filled with malloc'd array of n_frames probabilities.
int whisper_vad_encdec_detect(struct whisper_vad_encdec_context* ctx,
                              const float* samples, int n_samples,
                              struct whisper_vad_encdec_segment** segments, int* n_segments,
                              float threshold, float min_speech_sec, float min_silence_sec,
                              float** probs_out, int* n_frames_out);

void whisper_vad_encdec_free(struct whisper_vad_encdec_context* ctx);

#ifdef __cplusplus
}
#endif
