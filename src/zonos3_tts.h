// zonos3_tts.h — Zyphra ZONOS3-Mini TTS (issue #603).
//
// Backbone: 11-layer hybrid (9 Gated DeltaNet + 2 global GQA attention) with Full AttnRes residual routing and a
// latent 64-expert top-2 MoE, predicting 10 x 4096 codebooks at 15 Hz under a delay pattern. Codec: the mossdual
// iSTFT decoder (zac-keep2.gguf / zac.gguf) at 44.1 kHz. Both run on Zyphra's own GGUFs unchanged.
//
// The graph/codec code is vendored from github.com/Zyphra/ZONOS3-Mini servers/ggml (MIT, src/zonos3/LICENSE.zyphra)
// and diffed stage by stage against that upstream binary (tools/zonos3_diff.py).
#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct zonos3_tts_context;

struct zonos3_tts_context_params {
    int n_threads;
    int verbosity; // 0 = silent, 1 = normal, 2 = verbose
    bool use_gpu;
    float temperature; // < 0: registry default 0.75; 0 = greedy
    uint32_t seed;
    int max_frames;         // 15 Hz frames; 0 = default 450 (30 s)
    float cfg_scale;        // speaker CFG; 1 = off (upstream CLI default)
    const char* codec_path; // zac*.gguf; NULL = $CRISPASR_ZONOS3_CODEC, else zac-keep2.gguf / zac.gguf beside the model
};

struct zonos3_tts_context_params zonos3_tts_context_default_params(void);
struct zonos3_tts_context* zonos3_tts_init_from_file(const char* backbone_path,
                                                     struct zonos3_tts_context_params params);
void zonos3_tts_free(struct zonos3_tts_context* ctx);

// 192-d speaker embedding (gguf/voices/<Lang>/<name>.embedding.npy). NULL / n = 0 clears it (unconditioned voice).
int zonos3_tts_set_speaker_embedding(struct zonos3_tts_context* ctx, const float* emb, int n);
int zonos3_tts_load_speaker_embedding(struct zonos3_tts_context* ctx, const char* npy_path);
// Bucketed controls: name in {loudness (LUFS), fmax (Hz), speaking_rate (utf-8 bytes/s)}; NaN = unknown.
int zonos3_tts_set_condition(struct zonos3_tts_context* ctx, const char* name, double value);

void zonos3_tts_set_temperature(struct zonos3_tts_context* ctx, float temperature);
void zonos3_tts_set_seed(struct zonos3_tts_context* ctx, uint32_t seed);
int zonos3_tts_get_sample_rate(const struct zonos3_tts_context* ctx);

// Returns malloc'd 44.1 kHz mono PCM (free with zonos3_tts_free_audio), NULL on error.
float* zonos3_tts_synthesize(struct zonos3_tts_context* ctx, const char* text, int* out_n);
// Generate codes only: malloc'd [*out_frames, 10] int32 audio-stream codes (silence prefill + generated rows, still
// delayed), *out_eos = eos frame (audio-stream units) or -1.
int32_t* zonos3_tts_generate_codes(struct zonos3_tts_context* ctx, const char* text, int* out_frames, int* out_eos,
                                   int* out_prefill);
// Decode delayed codes [frames, 10] (as returned above) to 44.1 kHz PCM, prefill audio dropped, gain + limiter applied.
float* zonos3_tts_decode_codes(struct zonos3_tts_context* ctx, const int32_t* codes, int frames, int eos, int prefill,
                               int* out_n);
void zonos3_tts_free_audio(float* pcm);
void zonos3_tts_free_codes(int32_t* codes);

// Diagnostics (diff harness): run the prefill graph over the prompt for `text` and write every captured stage as
// <out_dir>/<name>.npy (same names/layout as upstream zonos3-cli --validate).
int zonos3_tts_dump_stages(struct zonos3_tts_context* ctx, const char* text, const char* out_dir);

#ifdef __cplusplus
}
#endif
