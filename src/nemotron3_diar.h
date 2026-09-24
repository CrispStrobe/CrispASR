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

// Frames of an n_samples recording that hold audio: floor(n_samples / hop).
// The probability matrices have one more row (the centred STFT's last frame);
// transformers' attention mask marks it, and every frame from here on, as
// padding, so segment/turn extraction must stop at this count.
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
