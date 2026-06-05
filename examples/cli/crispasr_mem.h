#pragma once
// crispasr_mem.h — memory usage reporting for verbose mode.
// Reads /proc/self/status on Linux, sysctl on macOS.

#include <cstdio>
#include <cstdint>

#ifdef __linux__
#include <fstream>
#include <string>
#endif

#ifdef __APPLE__
#include <mach/mach.h>
#endif

// Returns resident set size (RSS) in MB, or -1 if unavailable.
static inline double crispasr_rss_mb() {
#ifdef __linux__
    std::ifstream f("/proc/self/status");
    std::string line;
    while (std::getline(f, line)) {
        if (line.rfind("VmRSS:", 0) == 0) {
            // VmRSS:    123456 kB
            long kb = 0;
            sscanf(line.c_str(), "VmRSS: %ld", &kb);
            return kb / 1024.0;
        }
    }
    return -1;
#elif defined(__APPLE__)
    struct mach_task_basic_info info;
    mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO, (task_info_t)&info, &count) == KERN_SUCCESS)
        return info.resident_size / (1024.0 * 1024.0);
    return -1;
#else
    return -1;
#endif
}

// Log memory if verbose is true.
static inline void crispasr_log_mem(bool verbose, const char* label) {
    if (!verbose)
        return;
    double rss = crispasr_rss_mb();
    if (rss > 0)
        fprintf(stderr, "crispasr[mem]: %-30s RSS=%.0f MB\n", label, rss);
}

// Estimate memory needed for a given audio duration and backend.
// Returns estimated MB. Very rough — just prevents obvious OOM.
static inline double crispasr_estimate_mem_mb(double audio_seconds, const char* /*backend*/) {
    double T = audio_seconds * 50.0;
    double attn_mb = (T * T * 4.0) / (1024.0 * 1024.0);
    return attn_mb * 2.0;
}

// Check if audio is too long for the backend and warn.
// Returns true if audio should be chunked.
static inline bool crispasr_check_audio_length(double audio_seconds, int chunk_seconds, bool verbose) {
    if (audio_seconds > (double)chunk_seconds * 1.5) {
        if (verbose)
            fprintf(stderr, "crispasr[mem]: audio %.1fs exceeds chunk limit %ds — will auto-chunk\n", audio_seconds,
                    chunk_seconds);
        return true;
    }
    return false;
}
