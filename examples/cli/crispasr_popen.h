// crispasr_popen.h — UTF-8-safe popen/pclose wrappers.
//
// Background: on Windows, the narrow `_popen(cmd, mode)` sends the
// command line through the CRT/cmd.exe path using whatever the active
// code page happens to be. Non-ASCII content in the command line —
// most commonly DirectShow device names like 麥克風 (zh) or Mikrofon
// (some de locales) appearing in `ffmpeg -f dshow -i audio="..."` —
// gets mangled before ffmpeg ever sees it, and the subprocess exits
// before delivering any PCM. Pre-existing `2>NUL` redirection then
// hides the failure entirely. See issue #70.
//
// The fix on Windows: keep the command as UTF-8, widen it via
// MultiByteToWideChar(CP_UTF8 → CP_ACP fallback), and call `_wpopen`
// with the wide string. POSIX paths are unchanged; UTF-8 is already
// the wire format.
//
// Header-only so any CLI translation unit can drop it in without a
// CMake change. Be sure to define NOMINMAX before including this
// header so windows.h doesn't shadow std::min / std::max — the
// existing project pattern (see examples/cli/cli.cpp) is reused here.

#pragma once

#include <cstdio>
#include <string>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace crispasr {

// Open a subprocess pipe. `cmd` is UTF-8; `mode` is "rb"/"wb"/"r"/"w"
// (the same alphabet `_popen` / `popen` accepts).
//
// Returns nullptr on failure. The caller closes via `crispasr_pclose`.
inline FILE* crispasr_popen(const std::string& cmd, const char* mode) {
#if defined(_WIN32)
    // Try CP_UTF8 first; fall back to CP_ACP if the conversion fails
    // (e.g. ill-formed UTF-8 from a caller that already in the active
    // code page). MB_ERR_INVALID_CHARS makes CP_UTF8 fail loudly so we
    // know to try the fallback rather than producing replacement chars.
    auto widen = [](UINT codepage, const std::string& in, std::wstring& out) -> bool {
        const int n =
            MultiByteToWideChar(codepage, MB_ERR_INVALID_CHARS, in.c_str(), static_cast<int>(in.size()), nullptr, 0);
        if (n <= 0)
            return false;
        out.resize(static_cast<size_t>(n));
        const int got =
            MultiByteToWideChar(codepage, MB_ERR_INVALID_CHARS, in.c_str(), static_cast<int>(in.size()), out.data(), n);
        if (got <= 0) {
            out.clear();
            return false;
        }
        return true;
    };
    std::wstring wcmd;
    if (!widen(CP_UTF8, cmd, wcmd) && !widen(CP_ACP, cmd, wcmd)) {
        // Last-ditch: treat the input as already in the system code
        // page and let _wpopen do its best.
        wcmd.assign(cmd.begin(), cmd.end());
    }
    std::wstring wmode;
    for (const char* p = mode; *p; ++p)
        wmode.push_back(static_cast<wchar_t>(*p));
    return _wpopen(wcmd.c_str(), wmode.c_str());
#else
    return ::popen(cmd.c_str(), mode);
#endif
}

inline int crispasr_pclose(FILE* pipe) {
#if defined(_WIN32)
    return _pclose(pipe);
#else
    return ::pclose(pipe);
#endif
}

} // namespace crispasr
