// firered_vad.h — FireRedVAD (DFSMN-based voice activity detection).
//
// Architecture: 8-block DFSMN (Feedforward Sequential Memory Network)
//   Input: 80-dim fbank → fc1(256) → fc2(128) → 8× FSMN blocks → dnn(256) → out(1)
//   588K params, 2.4 MB GGUF
//
// Usage: load model, feed 16kHz PCM audio, get speech segments.

#pragma once

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct firered_vad_context;

struct firered_vad_segment {
    float start_sec;
    float end_sec;
};

struct firered_vad_context* firered_vad_init(const char* model_path);
void firered_vad_free(struct firered_vad_context* ctx);

// Detect speech segments in 16kHz mono PCM audio.
// Returns array of segments, caller must free with free().
// n_segments is set to the number of segments found.
int firered_vad_detect(struct firered_vad_context* ctx, const float* samples, int n_samples,
                       struct firered_vad_segment** segments, int* n_segments, float threshold, float min_speech_sec,
                       float min_silence_sec);

#ifdef __cplusplus
}
#endif
