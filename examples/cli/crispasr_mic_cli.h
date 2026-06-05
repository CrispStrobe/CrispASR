// crispasr_mic_cli.h — CLI helpers for microphone capture.

#pragma once

#include "crispasr_mic.h"

#include <string>

#if defined(_WIN32)
inline std::string crispasr_windows_dshow_audio_arg_from_name(const char* name) {
    std::string device = (name && *name) ? name : "default";
    for (char& c : device) {
        if (c == '"')
            c = '\'';
    }
    return "audio=\"" + device + "\"";
}

inline std::string crispasr_windows_dshow_audio_arg() {
    return crispasr_windows_dshow_audio_arg_from_name(crispasr_mic_default_device_name());
}
#endif
