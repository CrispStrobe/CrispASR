#pragma once

#include <cstdlib>
#include <cstring>

// Numerical checks run in the normal unit tier. Repeated timings are opt-in
// and registered separately with CTest's bench label. A speed ratio is an
// observation on one machine, never a portable correctness assertion.
inline bool benchmark_timing_enabled() {
    const char* value = std::getenv("CRISPASR_RUN_PERF");
    return value && std::strcmp(value, "1") == 0;
}
