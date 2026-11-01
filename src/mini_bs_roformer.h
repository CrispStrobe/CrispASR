// Mini-BS-RoFormer V2.5 four-stem music separation.
// Blueprint: HiDolen/Mini-BS-RoFormer-V2.5-46.2M (MIT), pinned by the converter.
#pragma once

#ifdef __cplusplus
extern "C" {
#endif

struct mini_bs_roformer_context;

struct mini_bs_roformer_params {
    int n_threads; // <= 0: runtime default
    bool use_gpu;  // prefer a GPU, with an explicit CPU fallback
    int gpu_device;
    int segment_seconds; // <= 0: checkpoint's trained 8-second chunk
    bool no_segment;     // raw whole-buffer forward (diagnostic; no overlap-add)
};

struct mini_bs_roformer_result {
    int n_sources;
    int n_channels;
    int n_samples; // per channel, exactly the input length
    int sample_rate;
    float** sources;           // each is interleaved [samples, channels]
    const char** source_names; // bass, drums, other, vocals
};

mini_bs_roformer_params mini_bs_roformer_default_params(void);
mini_bs_roformer_context* mini_bs_roformer_init_from_file(const char* path, mini_bs_roformer_params params);
void mini_bs_roformer_free(mini_bs_roformer_context* ctx);
mini_bs_roformer_result* mini_bs_roformer_separate(mini_bs_roformer_context* ctx, const float* pcm, int n_samples,
                                                   int in_channels);
void mini_bs_roformer_result_free(mini_bs_roformer_result* result);
int mini_bs_roformer_sample_rate(const mini_bs_roformer_context* ctx);
int mini_bs_roformer_n_sources(const mini_bs_roformer_context* ctx);
const char* mini_bs_roformer_source_name(const mini_bs_roformer_context* ctx, int idx);
// Selected compute backend, including the optional CPU BLAS accelerator.
// The returned string is borrowed; null means the context is unavailable.
const char* mini_bs_roformer_backend_name(const mini_bs_roformer_context* ctx);
// Effective operand/accumulation policy; returned storage is borrowed.
const char* mini_bs_roformer_compute_precision(const mini_bs_roformer_context* ctx);

// Compares every required original-blueprint stage, including four complete
// waveforms. Missing stages, shape mismatches and nonfinite values are failures.
// Q8_0 requires global cosine > .98; Q4_K > .94; F16 > .999.
int mini_bs_roformer_diff(const char* model_gguf, const char* ref_gguf, const char* audio_wav, int verbosity);

#ifdef __cplusplus
}
#endif
