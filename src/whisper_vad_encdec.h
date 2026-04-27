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
// encoder_out/encoder_out_size: if non-null, filled with the whisper encoder output
//   [d_model * n_frames] that can be injected into whisper's cross-attention state
//   to skip the ASR encoder pass (experimental — encoder was fine-tuned for VAD).
int whisper_vad_encdec_detect(struct whisper_vad_encdec_context* ctx,
                              const float* samples, int n_samples,
                              struct whisper_vad_encdec_segment** segments, int* n_segments,
                              float threshold, float min_speech_sec, float min_silence_sec,
                              float** probs_out, int* n_frames_out,
                              float** encoder_out, int* encoder_out_size);

void whisper_vad_encdec_free(struct whisper_vad_encdec_context* ctx);

#ifdef __cplusplus
}
#endif
