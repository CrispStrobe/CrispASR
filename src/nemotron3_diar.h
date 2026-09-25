// nemotron3_diar.h — NVIDIA Nemotron-3-Diarization (streaming Sortformer v3), #466.
//
// Frame-level speaker diarization: up to 8 speakers, ordered by first arrival,
// one probability per speaker every 10 ms. Loads the `sortformer` GGUF layout
// (NVIDIA's own Nemotron-3-Diarization.q8_0.gguf, or
// models/convert-nemotron3-diar-to-gguf.py output).
//
// Offline mode follows transformers' Nemotron3DiarizationForAudioFrameClassification:
// the recording is split into chunks of `chunk_len` encoder frames (80 ms each)
// with `chunk_right_context` look-ahead frames, and every chunk also attends to
// the Arrival-Order Speaker Cache + FIFO queue of earlier frames.
#pragma once

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct nemotron3_diar_context;

struct nemotron3_diar_params {
    int n_threads;
    bool use_gpu;
    int verbosity; // 0 silent, 1 normal, 2 verbose
};

struct nemotron3_diar_params nemotron3_diar_default_params(void);
struct nemotron3_diar_context* nemotron3_diar_init_from_file(const char* path, struct nemotron3_diar_params params);
void nemotron3_diar_free(struct nemotron3_diar_context* ctx);

int nemotron3_diar_n_speakers(struct nemotron3_diar_context* ctx);

// Chunk schedule: "offline" (default; transformers' offline forward, 340-frame
// chunks + 40 look-ahead) or a streaming preset of Nemotron3DiarizationProcessor,
// "low_latency" (1.04 s), "very_low_latency" (0.64 s) or "ultra_low_latency"
// (0.32 s). A streaming preset reproduces the model card's chunk-by-chunk
// session over the whole recording, speaker cache sized by streaming_config.
// Returns 0, or -1 for an unknown mode.
int nemotron3_diar_set_mode(struct nemotron3_diar_context* ctx, const char* mode);

// Live streaming session (transformers' streaming mode, chunk by chunk).
// mode: "low_latency" (default when NULL/empty), "very_low_latency" or
// "ultra_low_latency". push() buffers 16 kHz mono PCM and returns the
// probability rows [rows][S] of every chunk the audio completed (NULL, rows 0
// when none yet); end() flushes the last chunk. Rows continue one another, one
// per 10 ms from the start of the session. Returned arrays are caller-free()d.
// One session per stream; a context runs one session at a time.
struct nemotron3_diar_stream;
struct nemotron3_diar_stream* nemotron3_diar_stream_begin(struct nemotron3_diar_context* ctx, const char* mode);
float* nemotron3_diar_stream_push(struct nemotron3_diar_stream* st, const float* pcm, int n_samples, int* out_rows);
float* nemotron3_diar_stream_end(struct nemotron3_diar_stream* st, int* out_rows);
void nemotron3_diar_stream_free(struct nemotron3_diar_stream* st);
// Catch-up: when push() finds several complete chunks already buffered (the
// caller fell behind), run up to max_chunks of them as ONE forward - one step's
// compute - instead of one forward each. 1 (default) keeps the strict preset;
// the labels then match transformers exactly. Default from
// CRISPASR_SORTFORMER_CATCHUP.
void nemotron3_diar_stream_set_catchup(struct nemotron3_diar_stream* st, int max_chunks);

// Frames of an n_samples recording that hold audio. Offline: floor(n_samples /
// hop); the probability matrices have one more row (the centred STFT's last
// frame), which transformers' attention mask marks as padding, so segment/turn
// extraction must stop at this count. Streaming: the matrices' own row count.
int nemotron3_diar_n_valid_frames(struct nemotron3_diar_context* ctx, int n_samples);

// Speaker-activity probabilities for 16 kHz mono PCM: row-major [T][S], one row
// per 10 ms mel frame, S = n_speakers. Caller free()s. NULL on failure.
float* nemotron3_diar_probs(struct nemotron3_diar_context* ctx, const float* pcm, int n_samples, int* out_T,
                            int* out_S);

// Same, plus intermediate stages for the diff harness (each caller-free()d,
// any out pointer may be NULL): mel [T][n_mels], stacked-projection embeddings
// [Ne][d_model], and the pre-sigmoid logits [T][S] (what the function returns
// is sigmoid(logits)).
float* nemotron3_diar_probs_stages(struct nemotron3_diar_context* ctx, const float* pcm, int n_samples, int* out_T,
                                   int* out_S, float** out_mel, int* out_n_mels, float** out_embeds, int* out_Ne,
                                   int* out_d, float** out_logits);

#ifdef __cplusplus
}
#endif
