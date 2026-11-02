// ClearerVoice-Studio MossFormer2_SR_48K speech super-resolution (Apache-2.0).
//
// Native ggml port of the two-stage model in models/mossformer2_sr of
// modelscope/ClearerVoice-Studio @6b3774dc, converted by
// models/convert-mossformer2-sr-to-gguf.py, plus upstream's decode wrapper
// (utils/decode.py decode_one_audio_mossformer2_sr_48k):
//
//   48 kHz audio -> log-mel (get_mel: n_fft 1024, hop 256, periodic Hann,
//   (n_fft - hop) / 2 reflect padding, 80 Slaney bins over 0-8 kHz, magnitude,
//   log(clamp(x, 1e-5))) -> MossFormer2 MaskNet [T, 80] (single speaker; the
//   MossFormerM stack is mossformer2_blocks.{h,cpp}, UniDeepFsmn variant)
//   -> HiFi-GAN-style generator with Snake activations (x 8 8 2 2, 1024
//   channels, resblock kernels 3/7/11, dilations 1/3/5, tanh) -> 256 T
//   samples -> bandwidth_sub (utils/bandwidth_sub.py)
//
// bandwidth_sub detects the input's effective bandwidth (99.96 % cumulative
// energy of a scipy.signal.stft, 256-point Hann) and returns
// lowpass(input) + highpass(generated) at that cut-off (4th-order Butterworth,
// zero-phase filtfilt with odd padding, all in double precision as scipy), then
// cross-fades the first 100 ms from the plain input.  The band the input
// already has is kept verbatim; the model only fills in the band above it.
//
// One forward for inputs up to 20 s; beyond that 4-s windows at a 3-s stride
// with 0.5-s give-up edges and upstream's zero padding (mf2se::segment_plan).
// The generator runs in time chunks with a margin larger than its receptive
// field, so the chunked output equals the one-pass output (memory stays
// bounded; CRISPASR_MF2SR_GEN_CHUNK=<frames> overrides the chunk length,
// 0 = one pass).
//
// Deviations, for robustness only: upstream returns 256 * floor(n / 256)
// samples in the one-pass branch; the runtime zero-fills the < 256-sample tail
// so the output has exactly n samples.  Inputs shorter than one 1024-sample
// frame are zero-extended (upstream fails on them).  When the detected
// cut-off is 0 Hz or the Nyquist frequency (silence, or a full-band input),
// where scipy.signal.butter raises, the input is returned unchanged.  Inputs
// shorter than the 100-ms cross-fade use a truncated ramp (upstream fails).
#pragma once

#include <cstdint>
#include <string>
#include <vector>

struct CrispasrEnhanceModelOps;
struct mossformer2_sr_context;

mossformer2_sr_context* mossformer2_sr_init(const char* path, int n_threads, bool verbose);
void mossformer2_sr_free(mossformer2_sr_context* ctx);
int mossformer2_sr_sample_rate(const mossformer2_sr_context* ctx);
const char* mossformer2_sr_backend_name(const mossformer2_sr_context* ctx);

// Super-resolve `n` mono samples at 48 kHz (band-limited content, e.g. 8 or
// 16 kHz audio resampled to 48 kHz); `out` receives exactly `n` samples.
bool mossformer2_sr_process(mossformer2_sr_context* ctx, const float* in, int n, std::vector<float>& out);

// Stage-by-stage comparison against a tools/reference_backends/mossformer2_sr.py
// GGUF archive (crispasr-diff mossformer2-sr).  Every stage present in the
// reference is compared; F32/F16 need cosine > 0.999, quantized models > 0.99.
// Returns 0 when every stage passes.
int mossformer2_sr_diff(const char* model_gguf, const char* ref_gguf, const char* audio_wav, int verbosity);

// Registration for crispasr_enhancer (arch "mossformer2-sr"; output_native_rate).
const CrispasrEnhanceModelOps* mossformer2_sr_enhance_ops();

// --- model-free helpers (exposed for unit tests) ---------------------------
namespace mf2sr {
// get_mel() of `n` samples at 48 kHz: [T, 80] row-major, T = (n - 256) / 256 + 1
// (0 when n <= 384, where the reflect padding is undefined).
int log_mel(const float* x, int n, std::vector<float>& mel, int n_threads = 1);
// scipy.signal.butter(4, wn, btype='low' | 'high') (wn = cutoff / Nyquist in
// (0, 1)) as transfer-function coefficients (a[0] = 1).
struct Filter {
    std::vector<double> b, a;
};
Filter butter4(double wn, bool highpass);
// scipy.signal.lfilter_zi and filtfilt(b, a, x) (padtype 'odd', padlen
// 3 * max(len(a), len(b))).  filtfilt needs x.size() > padlen.
std::vector<double> lfilter_zi(const Filter& f);
std::vector<double> filtfilt(const Filter& f, const std::vector<double>& x);
// bandwidth_sub.detect_bandwidth's f_high in Hz; -1 for an all-zero signal.
double detect_bandwidth(const double* x, int64_t n, int fs, double threshold = 0.9996);
// bandwidth_sub(low, high, fs): returns min(n_low, n_high) samples (upstream
// requires n_high <= n_low).  See the header comment for the degenerate cases.
std::vector<double> bandwidth_sub(const std::vector<double>& low, const std::vector<double>& high, int fs,
                                  double threshold = 0.9996, double crossfade_ms = 100.0, double* f_high_out = nullptr);
} // namespace mf2sr
