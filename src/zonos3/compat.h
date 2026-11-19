// Cross-platform shims for POSIX file calls used by the GGUF loaders.
// MSVC's CRT lacks fseeko/ftello/popen; map them to the _-prefixed equivalents.
// _fseeki64/_ftelli64 take/return 64-bit offsets -- required for the multi-GB
// GGUFs (a plain fseek would truncate on Win32, where long is 32-bit).
#pragma once

// MSVC only exposes M_PI and the other math constants from <cmath>/<math.h> when
// this is defined before they are included; harmless elsewhere. compat.h is the
// first include in each TU, so the math headers pick it up.
#define _USE_MATH_DEFINES

#ifdef _WIN32
#include <stdio.h>
#include <stdlib.h>
#define fseeko(stream, offset, whence) _fseeki64((stream), (long long)(offset), (whence))
#define ftello(stream) _ftelli64(stream)
#define popen _popen
#define pclose _pclose

// MSVC's CRT has no POSIX setenv; _putenv_s always overwrites, so emulate the
// overwrite flag by checking for an existing value first.
static inline int setenv(const char* name, const char* value, int overwrite) {
    if (!overwrite && getenv(name) != NULL)
        return 0;
    return _putenv_s(name, value);
}
#endif
