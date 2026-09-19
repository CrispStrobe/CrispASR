#pragma once
// Available physical memory, for allocation decisions that must be made BEFORE
// the allocation is attempted.
//
// Why this exists (#441): parakeet's single-pass encoder estimates an O(T^2)
// relative-position bias and can refuse to take that path when it would not
// fit. The estimate and the switch were already implemented; what was missing
// was a budget, which defaulted to 0 = "policy disabled". On a 47.5-minute
// input that left a ~123 GB allocation to be attempted on a 15 GB machine,
// where Linux overcommit granted it about half the time and refused it the
// rest — and the refusal was dereferenced, so the failure mode was an
// intermittent SIGSEGV rather than a diagnosable error.
//
// MemAvailable is the right number on Linux, not MemFree: it is the kernel's
// own estimate of what a new allocation can obtain without swapping, and it
// already accounts for reclaimable page cache. MemFree would read as near-zero
// on any warm machine and make this guard fire constantly.

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <limits>

#ifdef __linux__
#include <cstring>
#endif

#ifdef __APPLE__
#include <mach/mach.h>
#include <sys/sysctl.h>
#endif

namespace core_sys_mem {

#ifdef __linux__
inline double read_number_file_mb(const char* path) {
    FILE* f = std::fopen(path, "r");
    if (!f)
        return -1.0;
    char value[128] = {0};
    const bool ok = std::fgets(value, sizeof(value), f) != nullptr;
    std::fclose(f);
    if (!ok || std::strncmp(value, "max", 3) == 0)
        return -1.0;
    char* end = nullptr;
    const double bytes = std::strtod(value, &end);
    return end != value && bytes >= 0.0 ? bytes / (1024.0 * 1024.0) : -1.0;
}

inline double cgroup_available_mb() {
    // cgroup v2, then v1. Some CI/container hosts expose the host's generous
    // MemAvailable while enforcing a much smaller cgroup limit; using only
    // /proc/meminfo would let an impossible encoder graph through (#441).
    double limit = read_number_file_mb("/sys/fs/cgroup/memory.max");
    double used = read_number_file_mb("/sys/fs/cgroup/memory.current");
    if (limit <= 0.0 || used < 0.0) {
        limit = read_number_file_mb("/sys/fs/cgroup/memory/memory.limit_in_bytes");
        used = read_number_file_mb("/sys/fs/cgroup/memory/memory.usage_in_bytes");
    }
    if (limit <= 0.0 || used < 0.0 || limit > 1024.0 * 1024.0 * 1024.0)
        return -1.0; // effectively unlimited sentinel
    return std::max(0.0, limit - used);
}
#endif

// Physical memory an allocation could plausibly obtain right now, in MiB.
// Returns -1 when it cannot be determined — callers MUST treat that as
// "unknown", never as "zero", or an unreadable /proc turns into a policy that
// refuses every input.
inline double available_mb() {
#ifdef __linux__
    FILE* f = std::fopen("/proc/meminfo", "r");
    if (!f)
        return -1.0;
    char line[256];
    double mb = -1.0;
    while (std::fgets(line, sizeof(line), f)) {
        long kb = 0;
        if (std::sscanf(line, "MemAvailable: %ld kB", &kb) == 1) {
            mb = (double)kb / 1024.0;
            break;
        }
    }
    std::fclose(f);
    const double cg = cgroup_available_mb();
    if (cg >= 0.0)
        return mb > 0.0 ? std::min(mb, cg) : cg;
    return mb;
#elif defined(__APPLE__)
    // free + inactive + purgeable is the closest analogue to MemAvailable.
    vm_size_t page = 0;
    mach_port_t host = mach_host_self();
    if (host_page_size(host, &page) != KERN_SUCCESS)
        return -1.0;
    vm_statistics64_data_t vm;
    mach_msg_type_number_t cnt = HOST_VM_INFO64_COUNT;
    if (host_statistics64(host, HOST_VM_INFO64, (host_info64_t)&vm, &cnt) != KERN_SUCCESS)
        return -1.0;
    const double bytes = (double)(vm.free_count + vm.inactive_count + vm.purgeable_count) * (double)page;
    return bytes / (1024.0 * 1024.0);
#else
    return -1.0;
#endif
}

} // namespace core_sys_mem
