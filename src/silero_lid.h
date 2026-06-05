// silero_lid.h — native ggml runtime for Silero Language Classifier 95.
//
// Replaces the sherpa-onnx subprocess wrapper in crispasr_lid.cpp with a
// fully native inference path. The model takes raw 16 kHz mono audio and
// outputs log-probabilities over 95 languages.

#pragma once

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

struct silero_lid_context;

struct silero_lid_context* silero_lid_init(const char* gguf_path, int n_threads);
void silero_lid_free(struct silero_lid_context* ctx);

// Detect the language of raw 16 kHz mono PCM audio.
// Returns the ISO-style language string (e.g. "en, English") on success,
// or NULL on failure. Caller does NOT free the returned string (it points
// into the context's language table).
// `out_confidence` receives the log-probability of the top language.
const char* silero_lid_detect(struct silero_lid_context* ctx, const float* samples, int n_samples,
                              float* out_confidence);

// Number of supported languages.
int silero_lid_n_langs(struct silero_lid_context* ctx);

#ifdef __cplusplus
}
#endif
