#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct marblenet_vad_context;

struct marblenet_vad_segment {
    float start_sec;
    float end_sec;
};

struct marblenet_vad_context* marblenet_vad_init(const char* model_path);

int marblenet_vad_detect(struct marblenet_vad_context* ctx, const float* samples, int n_samples,
                         struct marblenet_vad_segment** segments, int* n_segments, float threshold,
                         float min_speech_sec, float min_silence_sec);

void marblenet_vad_free(struct marblenet_vad_context* ctx);

#ifdef __cplusplus
}
#endif
