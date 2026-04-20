#pragma once

#include <cstdint>
#include <vector>

// Load a WAV file into float samples normalized to [-1, 1].
// Supports 16-bit PCM (format=1) and 32-bit float (format=3).
// Returns true on success.
bool moonshine_load_wav(const char* path, std::vector<float>& audio, int32_t* sample_rate);
