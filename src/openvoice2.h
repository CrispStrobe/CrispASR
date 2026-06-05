// openvoice2.h — OpenVoice V2 Tone Color Converter (voice cloning)
//
// Converts source audio to match a reference speaker's voice timbre.
// Architecture: ReferenceEncoder → speaker embedding,
//               PosteriorEncoder (WaveNet) → latent z,
//               ResidualCouplingFlow (WaveNet) forward/reverse → voice conversion,
//               HiFi-GAN decoder → waveform.
//
// Usage:
//   auto * ctx = openvoice2_init_from_file("openvoice2-tcc-f16.gguf", params);
//   openvoice2_convert(ctx, src_pcm, n_src, ref_pcm, n_ref, &out_pcm, &n_out);
//   openvoice2_free(ctx);

#ifndef OPENVOICE2_H
#define OPENVOICE2_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct openvoice2_context;

struct openvoice2_context_params {
    int n_threads;
    int verbosity; // 0=silent, 1=info, 2=debug
    bool use_gpu;
    float tau; // posterior encoder sampling temperature (default 0.3)
};

struct openvoice2_context_params openvoice2_context_default_params(void);

// Load the Tone Color Converter GGUF model.
struct openvoice2_context* openvoice2_init_from_file(const char* path, struct openvoice2_context_params params);

// Convert source audio to match reference speaker's voice.
// src_pcm/ref_pcm: mono float PCM (any sample rate, resampled internally to 22050).
// out_pcm: allocated by callee, caller must free().
// Returns true on success.
bool openvoice2_convert(struct openvoice2_context* ctx, const float* src_pcm, int n_src, int src_sr,
                        const float* ref_pcm, int n_ref, int ref_sr, float** out_pcm, int* n_out);

// Extract speaker embedding from reference audio (256-d).
// Returns true on success.
bool openvoice2_extract_speaker_embedding(struct openvoice2_context* ctx, const float* ref_pcm, int n_ref, int ref_sr,
                                          float* out_embedding); // must point to float[256]

void openvoice2_free(struct openvoice2_context* ctx);

// Diff harness: dump intermediates to directory.
void openvoice2_set_dump_dir(struct openvoice2_context* ctx, const char* dir);

#ifdef __cplusplus
}
#endif

#endif // OPENVOICE2_H
