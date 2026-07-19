// webrtc_vad.cpp — WebRTC VAD wrapper for CrispASR.
//
// Converts float32 16kHz PCM → int16 frames, runs WebRtcVad_Process per
// 30ms frame, then smooths per-frame decisions into speech segments using
// the same threshold/min_speech/min_silence contract as other VAD backends.

#include "webrtc_vad.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>

// WebRTC VAD C API (vendored in third_party/webrtc/)
#include <cstdio>
#include <cstdlib>

extern "C" {
#include "webrtc/common_audio/vad/include/webrtc_vad.h"

// Minimal rtc_FatalMessage required by the WebRTC C code's RTC_CHECK/RTC_DCHECK
// macros. In release builds (NDEBUG) the DCHECK paths are dead code, but the
// symbol must still exist for the linker.
#if !defined(RTC_FATAL_PROVIDED)
#define RTC_FATAL_PROVIDED
[[noreturn]] void rtc_FatalMessage(const char* file, int line, const char* msg) {
    fprintf(stderr, "WebRTC fatal: %s:%d: %s\n", file, line, msg);
    abort();
}
#endif
}

// Frame size: 30ms at 16kHz = 480 samples
static constexpr int WEBRTC_VAD_SAMPLE_RATE = 16000;
static constexpr int WEBRTC_VAD_FRAME_MS = 30;
static constexpr int WEBRTC_VAD_FRAME_SAMPLES = WEBRTC_VAD_SAMPLE_RATE * WEBRTC_VAD_FRAME_MS / 1000; // 480

static int get_webrtc_vad_mode(int requested_mode) {
    if (requested_mode >= 0 && requested_mode <= 3)
        return requested_mode;
    const char* env = getenv("CRISPASR_WEBRTC_VAD_MODE");
    if (env) {
        int m = atoi(env);
        if (m >= 0 && m <= 3)
            return m;
    }
    return 1; // default: moderate aggressiveness
}

int webrtc_vad_detect(const float* samples, int n_samples, struct webrtc_vad_segment** segments, int* n_segments,
                      float /*threshold*/, float min_speech_sec, float min_silence_sec, int mode) {
    if (!samples || n_samples <= 0 || !segments || !n_segments)
        return -1;

    *segments = nullptr;
    *n_segments = 0;

    // Create and init WebRTC VAD instance
    VadInst* vad = WebRtcVad_Create();
    if (!vad)
        return -1;
    if (WebRtcVad_Init(vad) != 0) {
        WebRtcVad_Free(vad);
        return -1;
    }

    int vad_mode = get_webrtc_vad_mode(mode);
    if (WebRtcVad_set_mode(vad, vad_mode) != 0) {
        WebRtcVad_Free(vad);
        return -1;
    }

    // Convert float32 [-1,1] to int16 and run per-frame VAD
    const int n_frames = n_samples / WEBRTC_VAD_FRAME_SAMPLES;
    if (n_frames == 0) {
        WebRtcVad_Free(vad);
        return 0;
    }

    std::vector<int> frame_decisions(n_frames, 0);
    std::vector<int16_t> frame_buf(WEBRTC_VAD_FRAME_SAMPLES);

    for (int f = 0; f < n_frames; f++) {
        const float* src = samples + f * WEBRTC_VAD_FRAME_SAMPLES;
        for (int i = 0; i < WEBRTC_VAD_FRAME_SAMPLES; i++) {
            float s = src[i] * 32767.0f;
            s = std::max(-32768.0f, std::min(32767.0f, s));
            frame_buf[i] = (int16_t)s;
        }
        int result = WebRtcVad_Process(vad, WEBRTC_VAD_SAMPLE_RATE, frame_buf.data(), WEBRTC_VAD_FRAME_SAMPLES);
        frame_decisions[f] = (result == 1) ? 1 : 0;
    }

    WebRtcVad_Free(vad);

    // Convert per-frame decisions to segments with min_speech / min_silence smoothing
    const float frame_sec = (float)WEBRTC_VAD_FRAME_MS / 1000.0f;
    const int min_speech_frames = std::max(1, (int)(min_speech_sec / frame_sec));
    const int min_silence_frames = std::max(1, (int)(min_silence_sec / frame_sec));

    // State machine: accumulate speech/silence runs
    std::vector<webrtc_vad_segment> result_segs;
    bool in_speech = false;
    int speech_start_frame = 0;
    int silence_count = 0;

    for (int f = 0; f < n_frames; f++) {
        if (!in_speech) {
            if (frame_decisions[f] == 1) {
                in_speech = true;
                speech_start_frame = f;
                silence_count = 0;
            }
        } else {
            if (frame_decisions[f] == 0) {
                silence_count++;
                if (silence_count >= min_silence_frames) {
                    // End of speech segment
                    int speech_end_frame = f - silence_count + 1;
                    int speech_len = speech_end_frame - speech_start_frame;
                    if (speech_len >= min_speech_frames) {
                        webrtc_vad_segment seg;
                        seg.start_sec = speech_start_frame * frame_sec;
                        seg.end_sec = speech_end_frame * frame_sec;
                        result_segs.push_back(seg);
                    }
                    in_speech = false;
                    silence_count = 0;
                }
            } else {
                silence_count = 0;
            }
        }
    }

    // Handle trailing speech
    if (in_speech) {
        int speech_end_frame = n_frames;
        int speech_len = speech_end_frame - speech_start_frame;
        if (speech_len >= min_speech_frames) {
            webrtc_vad_segment seg;
            seg.start_sec = speech_start_frame * frame_sec;
            seg.end_sec = speech_end_frame * frame_sec;
            result_segs.push_back(seg);
        }
    }

    // Copy to output
    if (!result_segs.empty()) {
        *n_segments = (int)result_segs.size();
        *segments = (webrtc_vad_segment*)malloc(sizeof(webrtc_vad_segment) * result_segs.size());
        if (!*segments) {
            *n_segments = 0;
            return -1;
        }
        memcpy(*segments, result_segs.data(), sizeof(webrtc_vad_segment) * result_segs.size());
    }

    return 0;
}
