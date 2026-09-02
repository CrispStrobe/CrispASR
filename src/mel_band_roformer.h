// src/mel_band_roformer.h — Mel-Band RoFormer source separation (§248).
//
// Frequency-band source separation (vocal / instrumental). STFT → mel-scale
// binary band split → per-band projection → alternating time/freq RoFormer
// blocks (RoPE + per-head output gating + value residuals) → per-band mask MLP
// (Tanh → GLU) → overlap-averaged complex mask → iSTFT.
//
// Blueprint: MIT lucidrains/BS-RoFormer (NOT Kim's unlicensed inference repo).
// Weights: KimberleyJSN/melbandroformer (MIT). Full op-by-op trace, band-layout
// verification, and the bs-roformer==0.3.10 reference pin are in
// docs/mel-band-roformer/PLAN.md.
//
// The C API deliberately mirrors src/htdemucs.h so the shared separation
// surface (src/core/separation_io.h, --separate) drives both backends through
// the same crispasr_separation_view shape.

#pragma once

#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

struct mel_band_roformer_context;

struct mel_band_roformer_params {
    int n_threads;  // 0 = auto
    bool use_gpu;   // attempt GPU acceleration (Metal/CUDA); CPU otherwise
    int gpu_device; // GPU device index
    // Change 176: segmentation toggles (defaults = validated 10 s Demucs-split).
    // Graph/fused/GPU are NOT caller fields — they resolve once in init via
    // mel_band_gates::resolve() (env AUTO/override, mirrors htdemucs #414).
    int segment_seconds; // Demucs-style split length; <=0 = default (10 s), 1 = per-second
    bool no_segment;     // 1 = whole-buffer forward even for long audio (A/B)
};

// Separation result: one waveform per source. For the vocals model the sources
// are {"vocals", "other"} (num_stems 1 emits the vocal estimate as stem 0 and
// the residual mix-minus-vocals as stem 1). Owns its storage; free with
// mel_band_roformer_result_free().
struct mel_band_roformer_result {
    int n_sources;
    int n_channels;  // 1 or 2 (matches the model's `stereo` flag)
    int n_samples;   // per-channel sample count
    int sample_rate; // model native rate (44100)
    // sources[s] points to n_channels * n_samples interleaved float32.
    float** sources;
    const char** source_names;
};

mel_band_roformer_params mel_band_roformer_default_params(void);

// Load a GGUF produced by models/convert-mel-band-roformer-to-gguf.py. Returns
// null on failure (bad magic, arch != "mel-band-roformer", missing tensors).
mel_band_roformer_context* mel_band_roformer_init_from_file(const char* model_path, mel_band_roformer_params params);
void mel_band_roformer_free(mel_band_roformer_context* ctx);

// Separate an interleaved audio signal at the model's native sample rate.
// `pcm` is n_samples-per-channel * n_channels interleaved float32. When the
// caller's channel count differs from the model's, the runtime up/downmixes.
// Returns null on failure. Caller frees with mel_band_roformer_result_free().
mel_band_roformer_result* mel_band_roformer_separate(mel_band_roformer_context* ctx, const float* pcm, int n_samples,
                                                     int in_channels);
void mel_band_roformer_result_free(mel_band_roformer_result* r);

int mel_band_roformer_sample_rate(const mel_band_roformer_context* ctx);
int mel_band_roformer_n_sources(const mel_band_roformer_context* ctx);
const char* mel_band_roformer_source_name(const mel_band_roformer_context* ctx, int idx);

// Per-stage diff harness runner (dots-tts / voxtral-tts pattern). Loads the
// runtime GGUF and the reference archive, runs the forward with the same input
// the Python dumper used, and compares each captured stage (freq_indices,
// stft_packed, band_gathered, band_split_out, layer{0,1,5}_{time,freq},
// mask_raw, output_vocals). Returns 0 if every stage passes cos threshold.
// verbosity: 0 = summary, 1 = per-stage, 2 = per-stage + magnitudes.
int mel_band_roformer_diff(const char* model_gguf, const char* ref_gguf, const char* audio_wav, int verbosity);

// Change 176 Phase 1: standalone band-split parity — the ggml-graph band-split
// vs the validated CPU reference on IDENTICAL input (a wav run through the same
// STFT/gather front-end). No Python fixture needed. Returns 0 iff the graph
// stage passes the cos threshold (>= 0.9995) with equal output sizes.
// verbosity: 1 = result, 2 = result + magnitudes.
int mel_band_roformer_parity(const char* model_gguf, const char* audio_wav, int verbosity);

#ifdef __cplusplus
}
#endif
