#pragma once

#include <cstddef>

namespace dia_debug {

// Diagnostic indices must be exact decimal values inside the replay history.
inline bool parse_dump_step(const char* value, size_t limit, size_t& step) {
    step = 0;
    if (!limit)
        return false;
    if (!value)
        return true;
    if (!*value)
        return false;
    for (const char* p = value; *p; ++p) {
        if (*p < '0' || *p > '9')
            return false;
        const size_t digit = (size_t)(*p - '0');
        if (digit >= limit || step > (limit - 1 - digit) / 10)
            return false;
        step = step * 10 + digit;
    }
    return true;
}

} // namespace dia_debug
