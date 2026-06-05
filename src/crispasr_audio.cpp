// Minimal audio file decoder for the language wrappers.
//
// libwhisper callers (Dart, Python, Rust wrappers) need cross-platform
// decoding of WAV / MP3 / FLAC / WAVE-containerised OGG so they can hand
// `crispasr_session_transcribe` a clean 16-kHz mono float32 buffer
// regardless of the original input format.
//
// miniaudio (MIT-0) handles WAV, MP3 and FLAC out of the box and does
// resampling + channel down-mix internally via its `ma_decoder` stream.
// Ogg Vorbis is handled by stb_vorbis — include it header-only before
// miniaudio so MA_HAS_VORBIS is auto-defined.

// stb_vorbis lives in examples/ — use relative path from src/
#define STB_VORBIS_HEADER_ONLY
#include "../examples/stb_vorbis.c"

#define MINIAUDIO_IMPLEMENTATION
// Device IO (capture mode) is needed for `crispasr_mic_*` (PLAN #62d);
// MA_NO_DEVICE_IO would strip the ma_device_* symbols. Threading
// follows from device IO. MA_NO_GENERATION (no oscillators / synth
// helpers) is still safe to keep.
#define MA_NO_GENERATION

// On iOS / tvOS / watchOS / visionOS, miniaudio's CoreAudio backend
// pulls in AVFoundation Objective-C headers from a .cpp TU, which the
// C++ front-end can't parse (NSString, etc., need Objective-C++ →
// .mm), and visionOS additionally has no AVAudioSession at all. The
// static-lib build artifact for those platforms is consumed by host
// apps that handle mic capture themselves, so we drop device IO here.
// The crispasr_mic_* C ABI is stubbed out to return failure on the
// same platforms (see crispasr_mic.cpp). macOS keeps device IO.
//
// TARGET_OS_IPHONE is the catch-all for the iOS-family platforms
// (iOS, tvOS, watchOS, visionOS); macOS leaves it as 0.
#if defined(__APPLE__)
#include <TargetConditionals.h>
#if TARGET_OS_IPHONE
#define MA_NO_DEVICE_IO
#endif
#endif

#include "miniaudio.h"

#undef STB_VORBIS_HEADER_ONLY
#include "../examples/stb_vorbis.c"

#include <cstdlib>
#include <cstring>

#ifdef _WIN32
#define CA_EXPORT extern "C" __declspec(dllexport)
#else
#define CA_EXPORT extern "C" __attribute__((visibility("default")))
#endif

namespace {
constexpr int kTargetSampleRate = 16000;
constexpr int kTargetChannels = 1;
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
CA_EXPORT int crispasr_audio_load(const char* path, float** out_pcm, int* out_samples, int* out_sample_rate) {
    if (!path || !out_pcm || !out_samples)
        return -1;
    *out_pcm = nullptr;
    *out_samples = 0;
    if (out_sample_rate)
        *out_sample_rate = 0;

    ma_decoder_config cfg = ma_decoder_config_init(ma_format_f32, kTargetChannels, kTargetSampleRate);
    ma_decoder decoder;
    if (ma_decoder_init_file(path, &cfg, &decoder) != MA_SUCCESS) {
        return -2;
    }

    // Decode in 1-second chunks. ma_decoder_get_length_in_pcm_frames can
    // fail on MP3 / streaming sources; chunked-read is what the CLI uses
    // and sidesteps that. The total allocation grows geometrically so we
    // don't re-alloc every chunk.
    constexpr ma_uint64 kChunkFrames = (ma_uint64)kTargetSampleRate; // 1 s
    float* buf = nullptr;
    size_t capacity = 0;
    size_t used = 0;

    for (;;) {
        if (capacity - used < kChunkFrames) {
            const size_t new_cap = capacity ? capacity * 2 : kChunkFrames * 8;
            float* nb = (float*)std::realloc(buf, new_cap * sizeof(float));
            if (!nb) {
                if (buf)
                    std::free(buf);
                ma_decoder_uninit(&decoder);
                return -3;
            }
            buf = nb;
            capacity = new_cap;
        }

        ma_uint64 frames_read = 0;
        const ma_result rc = ma_decoder_read_pcm_frames(&decoder, buf + used, kChunkFrames, &frames_read);
        used += (size_t)frames_read;

        if (rc == MA_AT_END || frames_read == 0)
            break;
        if (rc != MA_SUCCESS) {
            std::free(buf);
            ma_decoder_uninit(&decoder);
            return -4;
        }
    }
    ma_decoder_uninit(&decoder);

    // Trim trailing capacity we didn't fill — keeps the allocation tight.
    if (used < capacity) {
        float* tb = (float*)std::realloc(buf, used * sizeof(float));
        if (tb)
            buf = tb;
    }

    *out_pcm = buf;
    *out_samples = (int)used;
    if (out_sample_rate)
        *out_sample_rate = kTargetSampleRate;
    return 0;
}

/// Release a buffer allocated by `crispasr_audio_load`.
CA_EXPORT void crispasr_audio_free(float* pcm) {
    if (pcm)
        std::free(pcm);
}
