// SCNet / SCNet-large four-stem music separation (MIT).
// Blueprint: starrytong/SCNet, commit 5d95bf96b19c3eede63248d171efeca8e3abb948.
#pragma once
#include <stdint.h>

#ifndef __cplusplus
#include <stdbool.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

struct scnet_context;

struct scnet_params {
    int n_threads;
    bool use_gpu;
    int gpu_device;
    int segment_seconds; // <= 0: official inference's 20-second segments
    bool no_segment;     // diagnostic raw model.forward; no normalization/shift/overlap wrapper
    int shifts;          // official inference default: one shift per overlapping chunk
    uint64_t shift_seed; // Python random.Random(seed), reset at each separation call
};

struct scnet_result {
    int n_sources;
    int n_channels;
    int n_samples; // frames per channel, exactly the input length
    int sample_rate;
    float** sources;           // each source is interleaved [samples, channels]
    const char** source_names; // drums, bass, other, vocals
};

struct scnet_params scnet_default_params(void);
struct scnet_context* scnet_init_from_file(const char* path, struct scnet_params params);
void scnet_free(struct scnet_context* ctx);
struct scnet_result* scnet_separate(struct scnet_context* ctx, const float* pcm, int n_samples, int in_channels);
void scnet_result_free(struct scnet_result* result);
int scnet_sample_rate(const struct scnet_context* ctx);
int scnet_n_sources(const struct scnet_context* ctx);
const char* scnet_source_name(const struct scnet_context* ctx, int idx);
const char* scnet_backend_name(const struct scnet_context* ctx);
const char* scnet_compute_precision(const struct scnet_context* ctx);

// Strict complete-stage comparison against independently captured original Python.
// F16 cosine > .999, Q8_0 > .98, Q4_K > .94; shapes, finite values and amplitude
// are checked independently. Missing reference stages fail the comparison.
int scnet_diff(const char* model_gguf, const char* ref_gguf, const char* audio_wav, int verbosity);

#ifdef __cplusplus
}
#endif
