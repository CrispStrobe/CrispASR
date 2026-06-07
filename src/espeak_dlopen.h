// espeak_dlopen.h — runtime dlopen loader for libespeak-ng.
//
// Loads libespeak-ng at runtime via dlopen/LoadLibrary so the binary
// stays MIT-clean (no GPL code linked at build time). Falls back
// gracefully if the library isn't installed.
//
// Usage:
//   if (espeak_dl_load()) {
//       espeak_dl.Initialize(AUDIO_OUTPUT_SYNCHRONOUS, 0, path, flags);
//       espeak_dl.SetVoiceByName("en-us");
//       const char* phon = espeak_dl.TextToPhonemes(&tp, encoding, flags);
//       ...
//   }

#pragma once

#include <cstdint>
#include <cstdio>
#include <cstring>

// espeak-ng constants (from espeak-ng/speak_lib.h) — hardcoded to avoid
// needing the GPL header at compile time.
enum {
    CRISPASR_ESPEAK_AUDIO_OUTPUT_SYNCHRONOUS = 0x02,
    CRISPASR_ESPEAK_INITIALIZE_PHONEME_IPA = 0x0002,
    CRISPASR_ESPEAK_INITIALIZE_DONT_EXIT = 0x8000,
    CRISPASR_ESPEAK_CHARS_UTF8 = 1,
};

#ifdef _WIN32
#include <windows.h>
static inline void* espeak_dl_open(const char* name) {
    return (void*)LoadLibraryA(name);
}
static inline void* espeak_dl_sym(void* h, const char* sym) {
    return (void*)GetProcAddress((HMODULE)h, sym);
}
static inline void espeak_dl_close(void* h) {
    FreeLibrary((HMODULE)h);
}
#else
#include <dlfcn.h>
static inline void* espeak_dl_open(const char* name) {
    return dlopen(name, RTLD_LAZY | RTLD_LOCAL);
}
static inline void* espeak_dl_sym(void* h, const char* sym) {
    return dlsym(h, sym);
}
static inline void espeak_dl_close(void* h) {
    dlclose(h);
}
#endif

struct espeak_dl_api {
    void* handle = nullptr;
    bool loaded = false;

    // Function pointers matching espeak-ng's C API.
    int (*Initialize)(int output_type, int buf_length, const char* path, int options) = nullptr;
    int (*SetVoiceByName)(const char* name) = nullptr;
    const char* (*TextToPhonemes)(const void** textptr, int textmode, int phonememode) = nullptr;
    int (*GetSampleRate)(void) = nullptr;

    bool load() {
        if (loaded)
            return true;
        if (handle)
            return false; // already tried, failed

        // Try platform-specific library names.
        const char* names[] = {
#ifdef _WIN32
            "espeak-ng.dll", "libespeak-ng.dll",
#elif defined(__APPLE__)
            "libespeak-ng.1.dylib", "libespeak-ng.dylib",
            // Homebrew paths
            "/opt/homebrew/lib/libespeak-ng.1.dylib", "/usr/local/lib/libespeak-ng.1.dylib",
#else
            "libespeak-ng.so.1", "libespeak-ng.so",
            // Common Linux paths
            "/usr/lib/x86_64-linux-gnu/libespeak-ng.so.1", "/usr/lib/aarch64-linux-gnu/libespeak-ng.so.1",
            "/usr/local/lib/libespeak-ng.so.1",
#endif
            nullptr};

        for (int i = 0; names[i]; i++) {
            handle = espeak_dl_open(names[i]);
            if (handle)
                break;
        }

        if (!handle) {
            handle = (void*)(intptr_t)-1; // sentinel: tried and failed
            return false;
        }

        Initialize = (decltype(Initialize))espeak_dl_sym(handle, "espeak_Initialize");
        SetVoiceByName = (decltype(SetVoiceByName))espeak_dl_sym(handle, "espeak_SetVoiceByName");
        TextToPhonemes = (decltype(TextToPhonemes))espeak_dl_sym(handle, "espeak_TextToPhonemes");
        GetSampleRate = (decltype(GetSampleRate))espeak_dl_sym(handle, "espeak_ng_GetSampleRate");

        if (!Initialize || !SetVoiceByName || !TextToPhonemes) {
            fprintf(stderr, "espeak_dl: loaded library but missing symbols\n");
            espeak_dl_close(handle);
            handle = (void*)(intptr_t)-1;
            return false;
        }

        loaded = true;
        return true;
    }

    void unload() {
        if (loaded && handle && handle != (void*)(intptr_t)-1) {
            espeak_dl_close(handle);
        }
        handle = nullptr;
        loaded = false;
    }
};

// Global singleton — one per process, matches espeak-ng's global state.
inline espeak_dl_api& espeak_dl_get() {
    static espeak_dl_api instance;
    return instance;
}
