#pragma once
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

struct index_echo_context;
struct index_echo_context_params {
    int n_threads;
    int verbosity;
    bool use_gpu;
    bool flash_attn;
};
struct index_echo_context_params index_echo_context_default_params(void);
struct index_echo_context* index_echo_init_from_file(const char* path, struct index_echo_context_params params);
void index_echo_free(struct index_echo_context* ctx);

// Target language is en/ja/es. Empty glossary/context keeps the released prompt
// byte-identical. Context is retained between windows of one call, never files.
bool index_echo_set_target_lang(struct index_echo_context* ctx, const char* lang);
void index_echo_set_glossary(struct index_echo_context* ctx, const char* glossary);
void index_echo_set_ask(struct index_echo_context* ctx, const char* instruction);
void index_echo_set_temperature(struct index_echo_context* ctx, float temperature, uint32_t seed);
void index_echo_set_max_new_tokens(struct index_echo_context* ctx, int limit);
bool index_echo_set_vad_model(struct index_echo_context* ctx, const char* path);

struct index_echo_cue {
    double start_seconds, end_seconds;
    char* transcript;
    char* translation;
};
struct index_echo_result {
    int n_cues;
    struct index_echo_cue* cues;
    char* raw_text;
    int parse_warnings;
};
struct index_echo_result* index_echo_transcribe(struct index_echo_context* ctx, const float* samples, int n_samples);
void index_echo_result_free(struct index_echo_result* result);

// Stage helpers for crispasr-diff. All returned buffers are malloc-owned except
// stage() and prompt_ids(), which remain valid until the next prefill.
float* index_echo_compute_mel(struct index_echo_context*, const float*, int, int* n_mels, int* frames);
float* index_echo_run_encoder(struct index_echo_context*, const float* mel, int n_mels, int frames, int* rows,
                              int* dim);
float* index_echo_prefill(struct index_echo_context*, const float* audio_embd, int rows, int dim, int* n_vocab);
float* index_echo_decode_token(struct index_echo_context*, int32_t token, int* n_vocab);
const float* index_echo_stage(struct index_echo_context*, const char* name, int* count);
const int32_t* index_echo_prompt_ids(struct index_echo_context*, int* count);
int index_echo_decoder_layers(struct index_echo_context*);

#ifdef __cplusplus
}
#endif
