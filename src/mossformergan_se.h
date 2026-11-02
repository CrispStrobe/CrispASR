// MossFormerGAN_SE_16K speech enhancement (ClearerVoice-Studio, Apache-2.0).
//
// Native ggml port of the SyncANet generator plus upstream's decode wrapper
// (decode_one_audio_mossformergan_se_16k): RMS input normalisation, periodic
// Hamming STFT (n_fft 400, hop 100, centred, reflect pad), magnitude power
// compression (^0.3), the network, power uncompression, iSTFT and 10 s
// segmentation with 7.5 s stride for longer inputs. Mono 16 kHz in and out;
// the output has exactly the input length.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

struct CrispasrEnhanceModelOps;
struct mossformergan_se_context;

mossformergan_se_context* mossformergan_se_init(const char* path, int n_threads, bool verbose);
void mossformergan_se_free(mossformergan_se_context* ctx);
int mossformergan_se_sample_rate(const mossformergan_se_context* ctx);

// Enhance `n` mono samples at 16 kHz; `out` receives exactly `n` samples.
bool mossformergan_se_enhance(mossformergan_se_context* ctx, const float* in, int n, std::vector<float>& out);

// Parity/debug: one network forward on a power-compressed spectrogram in the
// upstream torch layout [2, T, 201]. Returns named stages in torch layout with
// the batch dimension dropped (names match tools/dump_mossformergan_se_reference.py).
struct mossformergan_se_stage {
    std::string name;
    std::vector<int64_t> shape; // torch order, outermost first
    std::vector<float> data;
};
bool mossformergan_se_forward_stages(mossformergan_se_context* ctx, const float* spec, int n_frames,
                                     std::vector<mossformergan_se_stage>& stages);

// Registration for crispasr_enhancer (arch "mossformergan-se").
const CrispasrEnhanceModelOps* mossformergan_se_enhance_ops();
