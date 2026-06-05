#pragma once

// Standalone SNAC 24 kHz decoder (hubertsiuzdak/snac_24khz, MIT).
//
// The Orpheus talker emits a stream of <custom_token_N> LM tokens; every
// 7 emitted tokens form one "super-frame" that de-interleaves into 1
// codes_0 / 2 codes_1 / 4 codes_2 entries (orpheus_tts_pypi/orpheus_tts/
// decoder.py convert_to_audio). This module loads a SNAC GGUF emitted by
// `models/convert-snac-to-gguf.py` and runs the decoder graph on the
// three codebook streams to synthesise 24 kHz mono PCM.
//
// The talker AR forward (orpheus_synthesize_codes) lands in slice (c) of
// PLAN #57 Phase 2; this module is exercised independently via
// `crispasr-diff orpheus`, which compares each named stage tensor
// against tools/reference_backends/orpheus_snac.py.
//
// Stage names (must match the Python reference dump):
//   snac_quant_out   — quantizer.from_codes output, (latent_dim=768, T_q)
//   snac_dec_pre     — after decoder.model[0..1] (in convs), (1024, T_q)
//   snac_dec_blk0    — after DecoderBlock 0 (stride 8), (512, 8·T_q)
//   snac_dec_blk1    — after DecoderBlock 1 (stride 8), (256, 64·T_q)
//   snac_dec_blk2    — after DecoderBlock 2 (stride 4), (128, 256·T_q)
//   snac_dec_blk3    — after DecoderBlock 3 (stride 2), (64, 512·T_q)
//   snac_pcm         — final tanh head, (T_q · 512,) at 24 kHz

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct snac_decoder_ctx;

struct snac_decoder_params {
    int n_threads;
    int verbosity; // 0=silent, 1=normal, 2=verbose
    bool use_gpu;
};

struct snac_decoder_params snac_decoder_default_params(void);

struct snac_decoder_ctx* snac_decoder_init_from_file(const char* path, struct snac_decoder_params params);

void snac_decoder_free(struct snac_decoder_ctx* ctx);

uint32_t snac_decoder_sample_rate(const struct snac_decoder_ctx* ctx); // 24000
uint32_t snac_decoder_n_codebooks(const struct snac_decoder_ctx* ctx); // 3
uint32_t snac_decoder_hop_length(const struct snac_decoder_ctx* ctx);  // 512
// Fills out[0..n_codebooks-1] with the per-codebook strides ([4, 2, 1]).
void snac_decoder_vq_strides(const struct snac_decoder_ctx* ctx, uint32_t out[3]);

// Decode 3 SNAC codebook streams to mono float32 PCM at 24 kHz.
//
// Constraints:
//   * n0 = T_super, n1 = 2 · T_super, n2 = 4 · T_super  (vq_strides=[4,2,1])
//   * Output length = n2 · 512 = 2048 · T_super
//
// Caller frees the returned buffer with `free()`. Returns nullptr on
// shape / argument errors. *out_n_samples is set on success.
float* snac_decoder_decode(struct snac_decoder_ctx* ctx, const int32_t* c0, int n0, const int32_t* c1, int n1,
                           const int32_t* c2, int n2, int* out_n_samples);

// Run the decode graph and extract a named intermediate tensor by its
// `ggml_set_name` string. Returns a malloc'd float32 buffer of *out_n
// elements (laid out channels-innermost: ne[0]=C, ne[1]=T) on success.
// The caller frees with `free()`. Returns nullptr on stage-not-found.
float* snac_decoder_extract_stage(struct snac_decoder_ctx* ctx, const int32_t* c0, int n0, const int32_t* c1, int n1,
                                  const int32_t* c2, int n2, const char* stage_name, int* out_n);

#ifdef __cplusplus
}
#endif
