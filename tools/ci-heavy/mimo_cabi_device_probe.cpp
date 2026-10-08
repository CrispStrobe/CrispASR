// Interpose only the native initializer; the real session C ABI still executes.
#include "mimo_asr.h"

static mimo_asr_context_params last;
static int calls;

extern "C" mimo_asr_context* mimo_asr_init_from_file(const char*, mimo_asr_context_params params) {
    last = params;
    ++calls;
    return nullptr; // No model allocation, and no fabricated live session.
}
extern "C" int mimo_probe_calls() {
    return calls;
}
extern "C" int mimo_probe_param(int field) {
    switch (field) {
    case 0:
        return last.n_threads;
    case 1:
        return last.use_gpu;
    case 2:
        return last.verbosity;
    default:
        return -1;
    }
}
