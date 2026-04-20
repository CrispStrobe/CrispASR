#pragma once

#ifdef __cplusplus
extern "C" {
#endif

struct moonshine_context;

struct moonshine_timing {
    double encode_ms; // encoder + cross-KV precompute
    double decode_ms; // decode loop
    int n_tokens;     // tokens decoded
    int n_samples;    // audio samples
};

struct moonshine_init_params {
    const char* model_path;
    const char* tokenizer_path; // NULL = auto-detect from model directory
    int n_threads;              // 0 = default (4)
};

struct moonshine_context* moonshine_init(const char* model_path);
struct moonshine_context* moonshine_init_with_params(struct moonshine_init_params params);
const char* moonshine_transcribe(struct moonshine_context* ctx, const float* audio, int n_samples);
// Run encoder conv stem. Caller must free(*out_features) when done.
int moonshine_encode(struct moonshine_context* ctx, const float* audio, int n_samples, float** out_features,
                     int* out_seq_len, int* out_hidden_dim);
void moonshine_free(struct moonshine_context* ctx);
void moonshine_print_model_info(struct moonshine_context* ctx);

void moonshine_set_n_threads(struct moonshine_context* ctx, int n_threads);
int moonshine_get_n_threads(struct moonshine_context* ctx);
int moonshine_get_timing(struct moonshine_context* ctx, struct moonshine_timing* timing);

#ifdef __cplusplus
}
#endif
