// src/core/audio_resample.h — polyphase Kaiser-windowed sinc resampler.
//
// Used by chatterbox's atomic native voice-clone path to convert between
// 16 kHz (VE / S3Tokenizer / CAMPPlus input) and 24 kHz (prompt mel input).
// Documented separately because resampler quality affects perceptual
// output but isn't on the parity hot path — the diff harness's
// `prompt_feat_24k` stage feeds identical 24 kHz bytes to both python
// and C++ via `audio_24k_input`, so resampler drift doesn't muddy the
// parity numbers.
//
// Algorithm: standard polyphase upsample(L) → low-pass → downsample(M)
// where the sample rate ratio src/dst reduces to M/L. Filter is a
// Kaiser-windowed sinc (β=8.6, num_zeros=14) — same parameters
// `librosa.resample(res_type='kaiser_fast')` uses internally; output is
// not bit-equivalent to librosa (resampy implements a pre-computed
// polyphase table with a different precision knob) but acoustically
// indistinguishable.

#pragma once

#include <cstdint>
#include <vector>

namespace core_audio {

// Resample `n_in` mono float samples from `src_rate` Hz to `dst_rate` Hz.
// Returns `ceil(n_in * dst_rate / src_rate)` output samples. Both rates
// must be positive; if they're equal the input is copied verbatim.
//
// `num_zeros` controls filter sharpness (default 14 matches librosa's
// kaiser_fast). Higher values give a sharper transition band at the cost
// of compute.
std::vector<float> resample_polyphase(const float* in, int n_in, int src_rate, int dst_rate, int num_zeros = 14,
                                      float kaiser_beta = 8.6f);

} // namespace core_audio
