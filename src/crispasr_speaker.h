// crispasr_speaker.h — local speaker playback via miniaudio.
//
// Cross-platform audio output using miniaudio's ma_device playback mode
// (Core Audio on macOS, ALSA/PulseAudio on Linux, WASAPI on Windows).
// The runtime symbols are already linked into libcrispasr because
// crispasr_audio.cpp defines MINIAUDIO_IMPLEMENTATION.
//
// Intended for CLI TTS (--tts-play) and future voice-loop use. Server
// mode routes audio to HTTP clients, not to local speakers.

#pragma once

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

struct crispasr_speaker;

// Open a playback device without starting it.
//
//   sample_rate:   ignored (kept for API symmetry; device-native rate is used).
//   channels:      ignored (kept for API symmetry; device-native channels used).
//   device_index:  -1 = default device; numeric index is a future addition.
//
// Returns nullptr on failure (no output device available).
// Call crispasr_speaker_close() when done.
struct crispasr_speaker* crispasr_speaker_open(int sample_rate, int channels, int device_index);

// Begin playing `pcm` (float32, interleaved, `n_samples` total samples at
// `app_sample_rate` Hz and `app_channels` channels). The implementation
// pre-resamples to the hardware-native rate/channel layout via linear
// interpolation, so the device callback is a straight memcpy (no runtime
// upsampling artifacts). Returns immediately — playback runs on the audio thread.
// The caller only needs to keep `pcm` valid until this function returns
// (the resampled copy is owned by the speaker).
// Returns 0 on success, non-zero on driver error.
int crispasr_speaker_play(struct crispasr_speaker* s, const float* pcm, int n_samples, int app_sample_rate,
                          int app_channels);

// Block until playback of the current buffer is complete. Returns 0 when
// playback finished naturally, -1 if interrupted by crispasr_speaker_stop().
int crispasr_speaker_wait(struct crispasr_speaker* s);

// Interrupt in-flight playback and unblock any waiting crispasr_speaker_wait()
// call. Idempotent. The audio thread may emit a few more frames while draining.
int crispasr_speaker_stop(struct crispasr_speaker* s);

// Release the device and free the handle. Implies stop(). Always call this.
void crispasr_speaker_close(struct crispasr_speaker* s);

// Human-readable name of the default playback device, or empty string if no
// output device is available. Returns a static buffer; not thread-safe.
const char* crispasr_speaker_default_device_name(void);

#ifdef __cplusplus
}
#endif
