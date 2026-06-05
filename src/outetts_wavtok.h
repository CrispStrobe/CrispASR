// outetts_wavtok.h -- WavTokenizer decoder C ABI.
//
// Single-codebook VQ-GAN: 4096 entries x 512-d -> ConvNeXt backbone ->
// ISTFTHead -> 24 kHz PCM.  Used as the codec decoder for OuteTTS.

#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct wavtok_decoder_ctx;

struct wavtok_decoder_params {
    int n_threads;
    int verbosity;
    int use_gpu;
    const char* dump_dir; // if non-null, dump per-stage .bin files for diff testing
};

struct wavtok_decoder_params wavtok_decoder_default_params(void);

// Load the WavTokenizer decoder GGUF (arch="wavtokenizer").
struct wavtok_decoder_ctx* wavtok_decoder_init_from_file(const char* path, struct wavtok_decoder_params params);

// Decode audio codes -> 24 kHz float32 PCM.
// codes: int32 array of codebook indices (0-4095), length n_codes.
// Returns malloc'd PCM buffer; caller frees with free(). *out_n_samples
// set on success.
float* wavtok_decoder_decode(struct wavtok_decoder_ctx* ctx, const int32_t* codes, int n_codes, int* out_n_samples);

void wavtok_decoder_free(struct wavtok_decoder_ctx* ctx);

#ifdef __cplusplus
}
#endif
