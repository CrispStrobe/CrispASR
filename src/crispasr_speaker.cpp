// crispasr_speaker.cpp — local speaker playback via miniaudio.
//
// Thin wrapper around miniaudio's ma_device playback mode. The
// MINIAUDIO_IMPLEMENTATION lives in crispasr_audio.cpp, so this TU
// just consumes the API. Mirrors the structure of crispasr_mic.cpp.
//
// Device format strategy: configure sampleRate=0 / channels=0 so
// miniaudio picks the hardware-native rate (e.g. 96000 Hz on MacBook Air
// Speakers). In crispasr_speaker_play() we pre-resample the app's mono
// f32 buffer to the device's rate/channels using linear interpolation.
// The callback then becomes a straight memcpy (passthrough path), avoiding
// miniaudio's upsampler which produces audible artefacts at 4× ratios.

#include "crispasr_speaker.h"

#if defined(__APPLE__)
#include <TargetConditionals.h>
#if TARGET_OS_IPHONE
#define CRISPASR_SPEAKER_STUB_ONLY 1
#endif
#endif

#ifndef CRISPASR_SPEAKER_STUB_ONLY

#include "miniaudio.h"

#include <algorithm>
#include <atomic>
#include <cstring>
#include <new>
#include <vector>

struct crispasr_speaker {
    ma_device device;
    // Pre-resampled playback buffer: device-native sample rate, channel count,
    // f32 interleaved. Owned by crispasr_speaker_play(); the callback reads it.
    std::vector<float> buf;
    std::atomic<int> cursor_frames{0}; // cursor in device-native FRAMES
    std::atomic<bool> stopped{false};
    ma_event done_event;
    bool event_init = false;
    bool started = false;
};

namespace {
void speaker_data_cb(ma_device* dev, void* output, const void* /*input*/, ma_uint32 frames) {
    auto* s = (crispasr_speaker*)dev->pUserData;
    auto* out = (float*)output;
    const int nch = (int)dev->playback.internalChannels;
    const int cur = s->cursor_frames.load(std::memory_order_relaxed);
    const int tot = (int)(s->buf.size() / (size_t)nch);
    const int rem = tot - cur;
    const int cpy = (rem > 0 && !s->stopped.load(std::memory_order_relaxed)) ? std::min(rem, (int)frames) : 0;
    if (cpy > 0) {
        std::memcpy(out, s->buf.data() + cur * nch, (size_t)cpy * (size_t)nch * sizeof(float));
        s->cursor_frames.fetch_add(cpy, std::memory_order_relaxed);
    }
    if ((ma_uint32)cpy < frames)
        std::memset(out + (size_t)cpy * (size_t)nch, 0, ((size_t)frames - (size_t)cpy) * (size_t)nch * sizeof(float));
    if (cpy < (int)frames || s->stopped.load(std::memory_order_relaxed)) {
        s->stopped.store(true, std::memory_order_relaxed);
        if (s->event_init)
            ma_event_signal(&s->done_event);
    }
}
} // namespace

extern "C" struct crispasr_speaker* crispasr_speaker_open(int /*sample_rate*/, int /*channels*/, int device_index) {
    auto* s = new (std::nothrow) crispasr_speaker();
    if (!s)
        return nullptr;

    if (ma_event_init(&s->done_event) != MA_SUCCESS) {
        delete s;
        return nullptr;
    }
    s->event_init = true;

    ma_device_config cfg = ma_device_config_init(ma_device_type_playback);
    cfg.playback.format = ma_format_f32;
    cfg.playback.channels = 0; // device-native channel count
    cfg.sampleRate = 0;        // device-native sample rate
    cfg.dataCallback = speaker_data_cb;
    cfg.pUserData = s;

    (void)device_index; // numeric device selection is a future addition

    if (ma_device_init(nullptr, &cfg, &s->device) != MA_SUCCESS) {
        ma_event_uninit(&s->done_event);
        delete s;
        return nullptr;
    }
    return s;
}

extern "C" int crispasr_speaker_play(struct crispasr_speaker* s, const float* pcm, int n_samples, int app_sample_rate,
                                     int app_channels) {
    if (!s || !pcm || n_samples <= 0 || app_sample_rate <= 0 || app_channels < 1)
        return -1;
    if (s->started)
        return -2;

    const int dev_rate = (int)s->device.playback.internalSampleRate;
    const int dev_ch = (int)s->device.playback.internalChannels;

    // Pre-resample: linear interpolation from app (mono/stereo, app_rate) to
    // device (dev_ch channels, dev_rate). TTS backends always produce mono.
    const int n_app_frames = n_samples / app_channels;
    const int n_dev_frames = (int)((double)n_app_frames * dev_rate / app_sample_rate + 0.5);
    s->buf.resize((size_t)n_dev_frames * (size_t)dev_ch);

    const double ratio = (double)(n_app_frames - 1) / std::max(n_dev_frames - 1, 1);
    for (int fi = 0; fi < n_dev_frames; fi++) {
        double src_pos = fi * ratio;
        int i0 = (int)src_pos;
        int i1 = std::min(i0 + 1, n_app_frames - 1);
        float alpha = (float)(src_pos - (double)i0);
        // mix down to mono from app_channels, then expand to dev_ch
        float mono = 0.0f;
        for (int c = 0; c < app_channels; c++)
            mono += pcm[i0 * app_channels + c];
        mono /= (float)app_channels;
        float s1 = 0.0f;
        for (int c = 0; c < app_channels; c++)
            s1 += pcm[i1 * app_channels + c];
        s1 /= (float)app_channels;
        float sample = mono * (1.0f - alpha) + s1 * alpha;
        for (int c = 0; c < dev_ch; c++)
            s->buf[(size_t)fi * (size_t)dev_ch + (size_t)c] = sample;
    }

    s->cursor_frames.store(0, std::memory_order_relaxed);
    s->stopped.store(false, std::memory_order_relaxed);

    if (ma_device_start(&s->device) != MA_SUCCESS)
        return -3;
    s->started = true;
    return 0;
}

extern "C" int crispasr_speaker_wait(struct crispasr_speaker* s) {
    if (!s || !s->started)
        return -1;
    ma_event_wait(&s->done_event);
    ma_device_stop(&s->device);
    s->started = false;
    return s->stopped.load() ? 0 : -1;
}

extern "C" int crispasr_speaker_stop(struct crispasr_speaker* s) {
    if (!s)
        return -1;
    s->stopped.store(true, std::memory_order_relaxed);
    if (s->event_init)
        ma_event_signal(&s->done_event);
    if (s->started) {
        ma_device_stop(&s->device);
        s->started = false;
    }
    return 0;
}

extern "C" void crispasr_speaker_close(struct crispasr_speaker* s) {
    if (!s)
        return;
    if (s->started) {
        s->stopped.store(true, std::memory_order_relaxed);
        ma_device_stop(&s->device);
        s->started = false;
    }
    ma_device_uninit(&s->device);
    if (s->event_init)
        ma_event_uninit(&s->done_event);
    delete s;
}

extern "C" const char* crispasr_speaker_default_device_name(void) {
    static char name[MA_MAX_DEVICE_NAME_LENGTH + 1] = {0};
    ma_context ctx;
    if (ma_context_init(nullptr, 0, nullptr, &ctx) != MA_SUCCESS) {
        name[0] = '\0';
        return name;
    }
    ma_device_info* playback_infos = nullptr;
    ma_uint32 playback_count = 0;
    ma_device_info* capture_infos = nullptr;
    ma_uint32 capture_count = 0;
    if (ma_context_get_devices(&ctx, &playback_infos, &playback_count, &capture_infos, &capture_count) != MA_SUCCESS) {
        ma_context_uninit(&ctx);
        name[0] = '\0';
        return name;
    }
    name[0] = '\0';
    for (ma_uint32 i = 0; i < playback_count; i++) {
        if (playback_infos[i].isDefault) {
            std::strncpy(name, playback_infos[i].name, sizeof(name) - 1);
            break;
        }
    }
    if (name[0] == '\0' && playback_count > 0)
        std::strncpy(name, playback_infos[0].name, sizeof(name) - 1);
    ma_context_uninit(&ctx);
    return name;
}

#else // CRISPASR_SPEAKER_STUB_ONLY — iOS / tvOS / watchOS

struct crispasr_speaker {};

extern "C" struct crispasr_speaker* crispasr_speaker_open(int, int, int) {
    return nullptr;
}
extern "C" int crispasr_speaker_play(struct crispasr_speaker*, const float*, int, int, int) {
    return -1;
}
extern "C" int crispasr_speaker_wait(struct crispasr_speaker*) {
    return -1;
}
extern "C" int crispasr_speaker_stop(struct crispasr_speaker*) {
    return -1;
}
extern "C" void crispasr_speaker_close(struct crispasr_speaker*) {}
extern "C" const char* crispasr_speaker_default_device_name(void) {
    return "";
}

#endif // CRISPASR_SPEAKER_STUB_ONLY
