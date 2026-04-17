// Minimal audio file decoder for the language wrappers.
//
// libwhisper callers (Dart, Python, Rust wrappers) need cross-platform
// decoding of WAV / MP3 / FLAC / WAVE-containerised OGG so they can hand
// `crispasr_session_transcribe` a clean 16-kHz mono float32 buffer
// regardless of the original input format.
//
// miniaudio (MIT-0) handles WAV, MP3 and FLAC out of the box and does
// resampling + channel down-mix internally via its `ma_decoder` stream.
// The only format it doesn't natively speak is Ogg Vorbis, which would
// need stb_vorbis alongside — skipped for now, add later if demand shows.

#define MINIAUDIO_IMPLEMENTATION
#define MA_NO_DEVICE_IO        // we don't need playback/capture devices
#define MA_NO_THREADING        // avoid pthreads coupling in the dylib
#define MA_NO_GENERATION       // skip synth helpers

#include "miniaudio.h"

#include <cstdlib>
#include <cstring>

#ifdef _WIN32
  #define CA_EXPORT extern "C" __declspec(dllexport)
#else
  #define CA_EXPORT extern "C" __attribute__((visibility("default")))
#endif

namespace {
constexpr int kTargetSampleRate = 16000;
constexpr int kTargetChannels   = 1;
} // namespace

/// Decode an audio file into float32 mono PCM at 16 kHz. Supports WAV,
/// MP3, and FLAC via miniaudio. The returned buffer is malloc-owned and
/// must be released with `crispasr_audio_free`.
///
/// Returns 0 on success and writes:
///   *out_pcm         → float * of `*out_samples` elements (mono)
///   *out_samples     → number of samples written
///   *out_sample_rate → 16000 (we always resample to this)
///
/// Negative return codes:
///   -1 bad args
///   -2 decoder init failed (unsupported format or read error)
///   -3 allocation failed
///   -4 decode of a chunk failed mid-stream
CA_EXPORT int crispasr_audio_load(const char * path,
                                  float **     out_pcm,
                                  int *        out_samples,
                                  int *        out_sample_rate) {
    if (!path || !out_pcm || !out_samples) return -1;
    *out_pcm = nullptr;
    *out_samples = 0;
    if (out_sample_rate) *out_sample_rate = 0;

    ma_decoder_config cfg = ma_decoder_config_init(
        ma_format_f32, kTargetChannels, kTargetSampleRate);
    ma_decoder decoder;
    if (ma_decoder_init_file(path, &cfg, &decoder) != MA_SUCCESS) {
        return -2;
    }

    // Decode in 1-second chunks. ma_decoder_get_length_in_pcm_frames can
    // fail on MP3 / streaming sources; chunked-read is what the CLI uses
    // and sidesteps that. The total allocation grows geometrically so we
    // don't re-alloc every chunk.
    constexpr ma_uint64 kChunkFrames = (ma_uint64) kTargetSampleRate; // 1 s
    float * buf = nullptr;
    size_t  capacity = 0;
    size_t  used = 0;

    for (;;) {
        if (capacity - used < kChunkFrames) {
            const size_t new_cap = capacity ? capacity * 2 : kChunkFrames * 8;
            float * nb = (float *) std::realloc(buf, new_cap * sizeof(float));
            if (!nb) {
                if (buf) std::free(buf);
                ma_decoder_uninit(&decoder);
                return -3;
            }
            buf = nb;
            capacity = new_cap;
        }

        ma_uint64 frames_read = 0;
        const ma_result rc = ma_decoder_read_pcm_frames(
            &decoder, buf + used, kChunkFrames, &frames_read);
        used += (size_t) frames_read;

        if (rc == MA_AT_END || frames_read == 0) break;
        if (rc != MA_SUCCESS) {
            std::free(buf);
            ma_decoder_uninit(&decoder);
            return -4;
        }
    }
    ma_decoder_uninit(&decoder);

    // Trim trailing capacity we didn't fill — keeps the allocation tight.
    if (used < capacity) {
        float * tb = (float *) std::realloc(buf, used * sizeof(float));
        if (tb) buf = tb;
    }

    *out_pcm = buf;
    *out_samples = (int) used;
    if (out_sample_rate) *out_sample_rate = kTargetSampleRate;
    return 0;
}

/// Release a buffer allocated by `crispasr_audio_load`.
CA_EXPORT void crispasr_audio_free(float * pcm) {
    if (pcm) std::free(pcm);
}
