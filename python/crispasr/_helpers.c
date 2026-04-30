// Thin C helpers for Python ctypes binding.
// These wrap whisper.h functions that take structs by value,
// providing pointer-based alternatives that ctypes can call.

#include "crispasr.h"

// whisper_full takes params by value — this wrapper takes a pointer.
int whisper_full_ptr(struct whisper_context* ctx, const struct whisper_full_params* params, const float* samples,
                     int n_samples) {
    return whisper_full(ctx, *params, samples, n_samples);
}

// whisper_init_from_file_with_params takes params by value — pointer wrapper.
struct whisper_context* whisper_init_from_file_ptr(const char* path, const struct whisper_context_params* params) {
    return whisper_init_from_file_with_params(path, *params);
}
