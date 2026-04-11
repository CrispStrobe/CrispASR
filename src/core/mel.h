// src/core/mel.h — shared log-mel spectrogram computation.
//
// Replaces the 9 copy-pasted mel spectrogram implementations across the
// src/ model files (parakeet.cpp, canary.cpp, canary_ctc.cpp, cohere.cpp,
// qwen3_asr.cpp, voxtral.cpp, voxtral4b.cpp, granite_speech.cpp,
// whisper.cpp). Two algorithm clusters are supported via enums:
//
//   NeMo / Conformer style  — ln + per-mel z-score, (T, n_mels) output
//       used by parakeet, canary, canary_ctc, cohere
//
//   Whisper / HF style      — log10 + global clip (max-8+4)/4, (n_mels, T) output
//       used by whisper, qwen3, voxtral, voxtral4b, granite
//
// The function is parameterised rather than having two entry points
// because the STFT + mel projection steps are identical; only the log
// base, normalization, and output transpose differ. Keeping them in one
// code path means numerical differences between clusters stay localised
// to the post-processing step, not the heavy computation.
//
// Models continue to own their own FFT function — it's passed in as a
// function pointer so we don't have to unify the 9 near-identical
// Cooley-Tukey implementations in this first pass. (They can be
// consolidated in a follow-up; the win there is small compared to the
// mel extraction itself.)

#pragma once

#include <cstdint>
#include <vector>

namespace core_mel {

// Real-to-complex FFT callback signature. N is always a power of two.
// Output layout: interleaved (re, im) pairs, length 2*N floats.
// Each model passes its own FFT so we don't disturb numerical paths.
using FftR2C = void (*)(const float * in, int N, float * out);

enum class LogBase { Ln, Log10 };

enum class Normalization {
    // Per-mel band z-score across time: (x - mean) / sqrt(var + 1e-5).
    // Used by parakeet / canary / canary_ctc / cohere.
    PerFeatureZ,

    // Global clip-and-scale: y = (max(x, max(x)-8) + 4) / 4.
    // Used by whisper / qwen3 / voxtral / granite.
    GlobalClipMax,

    // Global clip with fixed ceiling (max-like value is baked into the
    // normalization rather than computed): y = (max(x, fixed_max-8) + 4) / 4.
    // Used by voxtral4b with fixed_max = 1.5.
    GlobalClipFixed,
};

enum class Layout {
    // Row-major (T, n_mels) — each frame's n_mels values contiguous.
    // Used by the NeMo cluster.
    TimeMels,

    // Row-major (n_mels, T) — each mel band's full time series contiguous.
    // Used by the HF/Whisper cluster.
    MelsTime,
};

enum class LogGuard {
    // log(x + log_eps): NeMo convention (parakeet, canary, canary_ctc, cohere).
    AddEpsilon,

    // log(max(x, log_eps)): HF / Whisper convention.
    MaxClip,
};

enum class MatmulPrecision {
    // Float32 accumulator. Fastest, matches NeMo numerical path.
    Float,

    // Float64 accumulator, promoted before multiply-add. Matches the
    // HF / Whisper / Qwen3 / Voxtral numerical path which explicitly
    // uses double for the mel projection.
    Double,
};

// Filterbank storage layout in memory. Both are row-major floats.
enum class FbLayout {
    // [n_mels, n_freqs]: fb[m * n_freqs + k]. NeMo cluster.
    MelsFreqs,

    // [n_freqs, n_mels]: fb[k * n_mels + m]. HF / Whisper cluster
    // (WhisperFeatureExtractor.mel_filters).
    FreqsMels,
};

struct Params {
    int n_fft       = 400;  // power-of-two FFT size
    int hop_length  = 160;  // frame stride in samples
    int win_length  = 400;  // window length, must be <= n_fft
    int n_mels      = 128;

    LogBase         log_base   = LogBase::Log10;
    LogGuard        log_guard  = LogGuard::AddEpsilon;
    Normalization   norm       = Normalization::GlobalClipMax;
    Layout          layout     = Layout::MelsTime;
    FbLayout        fb_layout  = FbLayout::MelsFreqs;
    MatmulPrecision matmul     = MatmulPrecision::Float;

    // Small positive constant used in the log guard:
    //   AddEpsilon -> log(x + log_eps)
    //   MaxClip    -> log(max(x, log_eps))
    // NeMo uses 2^-24; Whisper uses 1e-10. Pass what the model originally used.
    float log_eps = 1e-10f;

    // For Normalization::GlobalClipFixed: the fixed ceiling used in place
    // of the per-audio max. Ignored for other normalization modes.
    float fixed_max = 1.5f;

    // Apply symmetric zero-padding of n_fft/2 samples before/after the input
    // (matches torchaudio / NeMo center=True). Set false if the caller has
    // already padded the input.
    bool center_pad = true;

    // Drop the last STFT frame. Matches Whisper / HF feature extractor
    // convention that produces floor((n - n_fft) / hop + 1) - 1 frames
    // instead of the full count.
    bool drop_last_frame = false;

    // If (after drop_last_frame) the frame count is odd, also drop the
    // first frame so the caller sees an even T. Used by voxtral4b, which
    // feeds mel into a stride-2 conv and needs an even temporal length.
    bool drop_first_frame_if_odd = false;

    // Pad the mel output on the right so the final length is exactly this
    // many frames. 0 disables. Voxtral 3B pads to 3000 (= 30s at hop=160).
    //
    // Padding happens AFTER the log step but BEFORE normalization, so the
    // padded positions are filled with the log of log_eps (i.e. the value
    // the log guard would produce for a zero-energy frame) rather than
    // plain zero. This matches voxtral's behaviour where padded frames
    // participate in the global-max calculation at a sensible floor.
    int pad_to_T = 0;

    // Stack `stacked_frames` consecutive mel frames into a single wider
    // frame. With stacked_frames=2 the output goes from (T, n_mels) to
    // (T/2, n_mels * 2) where row i is [mel[2i, :], mel[2i+1, :]]. If the
    // frame count is not a multiple of stacked_frames the trailing
    // remainder is dropped. Used by granite-speech's audio encoder which
    // expects a (T/2, 160) input built from consecutive pairs of 80-mel
    // frames. Default 1 (no stacking). Currently only supported with
    // Layout::TimeMels; other layouts leave the knob ignored.
    int stacked_frames = 1;
};

// Compute log-mel spectrogram from raw PCM samples.
//
//   samples      : float32 PCM at the caller's sample rate (usually 16 kHz)
//   n_samples    : sample count
//   window       : float32[n_fft] Hann/Hamming window padded with zeros if
//                  win_length < n_fft (caller's responsibility). When the
//                  model stores only the win_length-sized window in its GGUF,
//                  this helper pads it inside compute().
//   mel_fb       : float32[n_mels * n_freqs] row-major filterbank with
//                  n_freqs = n_fft/2 + 1
//   fft          : model-specific FFT function pointer (see FftR2C above)
//   params       : configuration (see Params struct)
//   T_out [out]  : number of output frames
//
// Returns the flat log-mel buffer in the layout specified by params.layout.
std::vector<float> compute(
    const float * samples, int n_samples,
    const float * window,     // length win_length (we center-pad inside to n_fft)
    int           win_length,
    const float * mel_fb,     // [n_mels, n_freqs]
    int           n_freqs,
    FftR2C        fft,
    const Params & params,
    int         & T_out);

} // namespace core_mel
