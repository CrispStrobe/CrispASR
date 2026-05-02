// crispasr_mic.cpp — library-level mic capture (PLAN #62d).
//
// Thin wrapper around miniaudio's ma_device capture mode. The
// MINIAUDIO_IMPLEMENTATION lives in crispasr_audio.cpp, so this TU
// just consumes the API.

#include "crispasr_mic.h"

// On iOS / tvOS / watchOS, miniaudio's CoreAudio backend pulls in
// AVFoundation Objective-C headers from a .cpp TU which the C++
// front-end can't parse (NSString etc. need .mm). The
// crispasr_audio.cpp companion already drops MA_NO_DEVICE_IO on these
// platforms, so the ma_device_* symbols don't exist in libcrispasr
// there. Stub out the C ABI to return failure / empty so language
// wrappers can still link.
#if defined(__APPLE__)
#include <TargetConditionals.h>
#if TARGET_OS_IOS || TARGET_OS_TV || TARGET_OS_WATCH
#define CRISPASR_MIC_STUB_ONLY 1
#endif
#endif

#ifndef CRISPASR_MIC_STUB_ONLY

#include "miniaudio.h"

#include <cstring>
#include <new>

struct crispasr_mic {
    ma_device device;
    crispasr_mic_callback cb = nullptr;
    void* userdata = nullptr;
    bool started = false;
};

namespace {
// miniaudio's data callback. Signature is fixed: pUserData is the
// device's opaque pointer (we set it to crispasr_mic*); pInput is
// the interleaved capture buffer; frameCount is per-channel.
void mic_data_cb(ma_device* pDevice, void* /*pOutput*/, const void* pInput, ma_uint32 frameCount) {
    auto* m = (crispasr_mic*)pDevice->pUserData;
    if (!m || !m->cb || !pInput)
        return;
    // miniaudio's capture buffer is float32 (we asked for it).
    // For mono (channels=1) frameCount == n_samples.
    // For stereo we hand the interleaved buffer + n_samples = frameCount * 2.
    const ma_uint32 n_per_call = frameCount * pDevice->capture.channels;
    m->cb((const float*)pInput, (int)n_per_call, m->userdata);
}
} // namespace

extern "C" struct crispasr_mic* crispasr_mic_open(int sample_rate, int channels, crispasr_mic_callback cb,
                                                  void* userdata) {
    if (!cb || sample_rate <= 0 || channels < 1 || channels > 2)
        return nullptr;

    auto* m = new (std::nothrow) crispasr_mic();
    if (!m)
        return nullptr;
    m->cb = cb;
    m->userdata = userdata;

    ma_device_config cfg = ma_device_config_init(ma_device_type_capture);
    cfg.capture.format = ma_format_f32;
    cfg.capture.channels = (ma_uint32)channels;
    cfg.sampleRate = (ma_uint32)sample_rate;
    cfg.dataCallback = mic_data_cb;
    cfg.pUserData = m;

    if (ma_device_init(nullptr, &cfg, &m->device) != MA_SUCCESS) {
        delete m;
        return nullptr;
    }
    return m;
}

extern "C" int crispasr_mic_start(struct crispasr_mic* m) {
    if (!m || m->started)
        return -1;
    if (ma_device_start(&m->device) != MA_SUCCESS)
        return -2;
    m->started = true;
    return 0;
}

extern "C" int crispasr_mic_stop(struct crispasr_mic* m) {
    if (!m)
        return -1;
    if (!m->started)
        return 0;
    ma_device_stop(&m->device);
    m->started = false;
    return 0;
}

extern "C" void crispasr_mic_close(struct crispasr_mic* m) {
    if (!m)
        return;
    if (m->started) {
        ma_device_stop(&m->device);
        m->started = false;
    }
    ma_device_uninit(&m->device);
    delete m;
}

extern "C" const char* crispasr_mic_default_device_name(void) {
    static char name[MA_MAX_DEVICE_NAME_LENGTH + 1] = {0};
    ma_context ctx;
    if (ma_context_init(nullptr, 0, nullptr, &ctx) != MA_SUCCESS) {
        name[0] = '\0';
        return name;
    }
    ma_device_info* capture_infos = nullptr;
    ma_uint32 capture_count = 0;
    ma_device_info* playback_infos = nullptr;
    ma_uint32 playback_count = 0;
    if (ma_context_get_devices(&ctx, &playback_infos, &playback_count, &capture_infos, &capture_count) != MA_SUCCESS) {
        ma_context_uninit(&ctx);
        name[0] = '\0';
        return name;
    }
    name[0] = '\0';
    for (ma_uint32 i = 0; i < capture_count; i++) {
        if (capture_infos[i].isDefault) {
            std::strncpy(name, capture_infos[i].name, sizeof(name) - 1);
            break;
        }
    }
    if (name[0] == '\0' && capture_count > 0) {
        std::strncpy(name, capture_infos[0].name, sizeof(name) - 1);
    }
    ma_context_uninit(&ctx);
    return name;
}

#else // CRISPASR_MIC_STUB_ONLY — iOS / tvOS / watchOS

// Stub the C ABI so libcrispasr links on iOS / tvOS / watchOS. Host
// apps that need mic capture on these platforms call AVAudioEngine
// directly and hand PCM into `crispasr_stream_feed`. The host has the
// AVAudioSession / privacy-prompt context the static lib wouldn't.
struct crispasr_mic {};

extern "C" struct crispasr_mic* crispasr_mic_open(int /*sample_rate*/, int /*channels*/, crispasr_mic_callback /*cb*/,
                                                  void* /*userdata*/) {
    return nullptr;
}
extern "C" int crispasr_mic_start(struct crispasr_mic* /*m*/) {
    return -1;
}
extern "C" int crispasr_mic_stop(struct crispasr_mic* /*m*/) {
    return -1;
}
extern "C" void crispasr_mic_close(struct crispasr_mic* /*m*/) {}
extern "C" const char* crispasr_mic_default_device_name(void) {
    return "";
}

#endif // CRISPASR_MIC_STUB_ONLY
