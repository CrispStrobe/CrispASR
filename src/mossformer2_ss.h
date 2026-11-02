// ClearerVoice-Studio MossFormer2_SS_16K two-speaker speech separation.
// Weights: alibabasglab/MossFormer2_SS_16K (Apache-2.0), converted by
// models/convert-mossformer2-ss-to-gguf.py; network code
// modelscope/ClearerVoice-Studio @6b3774dc.  Shared MossFormer2 blocks:
// mossformer2_blocks.h.
#pragma once

#ifdef __cplusplus
extern "C" {
#endif

struct mossformer2_ss_context;

struct mossformer2_ss_params {
    int n_threads; // <= 0: runtime default
    bool use_gpu;  // prefer a GPU, with an explicit CPU fallback
    int gpu_device;
    bool no_segment; // one forward over the whole (window-padded) input; diagnostic
};

struct mossformer2_ss_result {
    int n_sources;             // 2
    int n_channels;            // 1
    int n_samples;             // exactly the input length
    int sample_rate;           // 16000
    float** sources;           // mono, one buffer per speaker
    const char** source_names; // "spk1", "spk2"
};

struct mossformer2_ss_params mossformer2_ss_default_params(void);
struct mossformer2_ss_context* mossformer2_ss_init_from_file(const char* path, struct mossformer2_ss_params params);
void mossformer2_ss_free(struct mossformer2_ss_context* ctx);

// Separate `n_samples` frames of 16-kHz audio.  `in_channels` 1 (mono) or 2
// (interleaved stereo, averaged to mono).  Runs upstream
// decode_one_audio_mossformer2_ss_16k exactly: window padding, 2-s windows with
// 0.75 stride and give-up edges for inputs longer than 2 s, and each output
// renormalised to the input RMS.  Returns null on error.
struct mossformer2_ss_result* mossformer2_ss_separate(struct mossformer2_ss_context* ctx, const float* pcm,
                                                      int n_samples, int in_channels);
void mossformer2_ss_result_free(struct mossformer2_ss_result* result);

int mossformer2_ss_sample_rate(const struct mossformer2_ss_context* ctx);
int mossformer2_ss_n_sources(const struct mossformer2_ss_context* ctx);
const char* mossformer2_ss_source_name(const struct mossformer2_ss_context* ctx, int idx);
const char* mossformer2_ss_backend_name(const struct mossformer2_ss_context* ctx);

// Stage-by-stage comparison against a tools/dump_mossformer2_ss_reference.py
// GGUF archive (crispasr-diff mossformer2-ss).  Every stage present in the
// reference is compared; F32/F16 need cosine > 0.999, quantized models > 0.99.
// Returns 0 when every stage passes.
int mossformer2_ss_diff(const char* model_gguf, const char* ref_gguf, const char* audio_wav, int verbosity);

#ifdef __cplusplus
}
#endif
