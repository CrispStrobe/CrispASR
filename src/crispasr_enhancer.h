// crispasr_enhancer.h — selectable speech-enhancement (denoise) pre-pass.
//
// One object shared by the CLI (`--enhance`), the Session C ABI
// (`crispasr_session_set_enhance`), the standalone C ABI
// (`crispasr_enhance_audio_spec`) and the server. A *spec* string picks the
// method:
//
//   ""  / "none" / "off" / "0"  -> disabled (create returns nullptr, no error)
//   "rnnoise"                   -> built-in RNNoise (non-MSVC builds only)
//   any other string            -> path to an enhancement GGUF; the method is
//                                  chosen from its `general.architecture`
//                                  (see crispasr_enhancer_model_archs()).
//
// process() takes mono f32 PCM at ANY sample rate and returns exactly the same
// number of samples at that same rate: the enhancer resamples to its native
// rate (RNNoise 48 kHz via its 16 kHz wrapper, model GGUFs at their trained
// rate) and back. Enhancers are NOT thread-safe; use one per thread.

#pragma once

#include <string>
#include <vector>

struct CrispasrEnhancer;

/// True when `spec` means "no enhancement".
bool crispasr_enhance_spec_is_none(const char* spec);

/// Create an enhancer for `spec`. Returns nullptr either when the spec is
/// "none" (then `*err` stays empty) or on failure (then `*err` says why).
CrispasrEnhancer* crispasr_enhancer_create(const char* spec, int n_threads, bool verbose, std::string* err);

/// Run the enhancer. `out` is resized to `n_samples`. Returns false on failure.
bool crispasr_enhancer_process(CrispasrEnhancer* e, const float* in, int n_samples, int sample_rate,
                               std::vector<float>& out);

/// Short method name ("rnnoise", "frcrn-se", ...). Never null.
const char* crispasr_enhancer_name(const CrispasrEnhancer* e);

/// Native processing rate in Hz.
int crispasr_enhancer_native_rate(const CrispasrEnhancer* e);

void crispasr_enhancer_free(CrispasrEnhancer* e);

/// True when the built-in RNNoise method is compiled in.
bool crispasr_enhancer_has_rnnoise();

/// Comma-separated GGUF architectures accepted as enhancement models.
const char* crispasr_enhancer_model_archs();

// ---------------------------------------------------------------------------
// Model registration (internal). Each enhancement model port provides one
// table entry; crispasr_enhancer.cpp lists them.
// ---------------------------------------------------------------------------
struct CrispasrEnhanceModelOps {
    const char* arch; // GGUF general.architecture
    const char* name; // user-facing method name
    void* (*init)(const char* path, int n_threads, bool verbose);
    int (*native_rate)(void* ctx);
    // Enhance `n` samples at native rate; `out` must receive exactly `n`.
    bool (*process)(void* ctx, const float* in, int n, std::vector<float>& out);
    void (*free)(void* ctx);
};
