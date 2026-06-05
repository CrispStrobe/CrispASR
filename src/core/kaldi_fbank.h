// src/core/kaldi_fbank.h — Kaldi-compatible filterbank features.
//
// Matches `torchaudio.compliance.kaldi.fbank()` with default arguments
// for sample_frequency=16000, num_mel_bins=N, use_energy=False,
// htk_compat=False, use_log_fbank=True, use_power=True, dither=0,
// preemphasis_coefficient=0.97, remove_dc_offset=True, raw_energy=True,
// round_to_power_of_two=True, snip_edges=True, window_type='povey'.
//
// Useful for any speaker / VAD encoder trained against Kaldi's `fbank`
// pipeline. Currently consumed by the chatterbox CAMPPlus port (80-bin)
// and structurally identical to the `compute_fbank` helper inlined in
// `firered_asr.cpp` (which scales the audio to int16 range before the
// fbank — that path adds an `int16_scale` knob here).
//
// Output is row-major (T_frames, n_mels) float32. T_frames =
// (n_samples - win_samples) / hop_samples + 1 (kaldi snip_edges=True
// — drops trailing partial frames).

#pragma once

#include <cstdint>
#include <vector>

namespace core_kaldi {

enum class WindowType {
    Povey = 0,   // pow(hann, 0.85) — torchaudio fbank default
    Hamming = 1, // 0.54 - 0.46*cos(2πi/(N-1)) — FunASR WavFrontend default
};

struct FbankParams {
    int sample_rate = 16000;
    int n_mels = 80;
    int frame_length_ms = 25;        // → 400 samples @ 16 kHz
    int frame_shift_ms = 10;         // → 160 samples @ 16 kHz
    float low_freq = 20.0f;          // Hz
    float high_freq = 0.0f;          // 0 = Nyquist (sample_rate / 2)
    float preemph = 0.97f;           // pre-emphasis coefficient (0 = skip)
    float log_floor = 1.1920929e-7f; // FLT_EPSILON; below clamps log(x)
    bool remove_dc_offset = true;    // subtract per-frame mean before window
    bool int16_scale = false;        // multiply float input by 32768 first
                                     // (firered_asr / funasr trained on int16-scaled features;
                                     //  CAMPPlus / most modern speaker encoders consume raw [-1, 1])
    WindowType window_type = WindowType::Povey;
};

// Compute Kaldi-compatible filterbank features for the given 16 kHz mono
// PCM buffer. Returns (T_frames, n_mels) row-major float32 features.
//
// `T_frames_out` is set on return. If the audio is shorter than one
// window (n_samples < frame_length samples), returns empty + sets
// T_frames_out=0.
std::vector<float> compute_fbank(const float* pcm, int n_samples, const FbankParams& p, int& T_frames_out);

} // namespace core_kaldi
