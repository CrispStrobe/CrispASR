// ClearerVoice-Studio MossFormer2_SE_48K speech enhancement (Apache-2.0).
//
// Native ggml port of TestNet -> MossFormer_MaskNet (models/mossformer2_se of
// modelscope/ClearerVoice-Studio @6b3774dc), converted by
// models/convert-mossformer2-se-to-gguf.py, plus upstream's decode wrapper
// (utils/decode.py decode_one_audio_mossformer2_se_48k):
//
//   audio * 32768 -> Kaldi fbank (60 log-mel, 40 ms / 8 ms Hamming frames)
//   + compute_deltas + delta-deltas -> [T, 180] -> network -> mask [T, 961]
//   -> mask * STFT(audio) (n_fft 1920, hop 384, symmetric Hamming, no
//   centring) -> iSTFT -> / 32768
//
// with one forward for inputs up to 20 s and 4-s windows at a 3-s stride
// (0.5-s give-up edges, upstream's zero padding) beyond that.  The shared
// MossFormerM stack is mossformer2_blocks.{h,cpp} (Gated_FSMN_Block, i.e.
// BlockHParams::fsmn_dilated = false).  Mono 48 kHz in and out; the output has
// exactly the input length.  Upstream's Kaldi dither (N(0, 1) at int16 scale
// per frame sample, from torch's unseeded global RNG) is kept but drawn from a
// fixed-seed per-frame stream, so runs are reproducible; it matters: without
// it, digital silence (e.g. the segmented path's end padding) yields
// log(FLT_EPSILON) features the network never saw, costing ~5 dB SNR on the
// last window.  CRISPASR_MF2SE_DITHER=<amplitude> overrides it (0 disables;
// crispasr-diff always uses 0 to match a dither-free reference).  Other
// deviations, for robustness: inputs shorter than one 1920-sample window are
// zero-extended instead of failing, and windows whose kept span lies entirely
// in the end padding are skipped (upstream truncates those samples anyway).
#pragma once

#include <cstdint>
#include <string>
#include <vector>

struct CrispasrEnhanceModelOps;
struct mossformer2_se_context;

mossformer2_se_context* mossformer2_se_init(const char* path, int n_threads, bool verbose);
void mossformer2_se_free(mossformer2_se_context* ctx);
int mossformer2_se_sample_rate(const mossformer2_se_context* ctx);
const char* mossformer2_se_backend_name(const mossformer2_se_context* ctx);

// Enhance `n` mono samples at 48 kHz; `out` receives exactly `n` samples.
bool mossformer2_se_enhance(mossformer2_se_context* ctx, const float* in, int n, std::vector<float>& out);

// Stage-by-stage comparison against a tools/reference_backends/mossformer2_se.py
// GGUF archive (crispasr-diff mossformer2-se).  Every stage present in the
// reference is compared; F32/F16 need cosine > 0.999, quantized models > 0.99.
// Returns 0 when every stage passes.
int mossformer2_se_diff(const char* model_gguf, const char* ref_gguf, const char* audio_wav, int verbosity);

// Registration for crispasr_enhancer (arch "mossformer2-se").
const CrispasrEnhanceModelOps* mossformer2_se_enhance_ops();

// --- model-free helpers (exposed for unit tests) ---------------------------
namespace mf2se {
// torchaudio.functional.compute_deltas over time (replicate padding) of a
// row-major [T, C] matrix.
std::vector<float> compute_deltas(const std::vector<float>& x, int T, int C, int win_length = 5);
// 60-bin Kaldi fbank (+ deltas, delta-deltas) of `n` samples at 48 kHz in
// [-1, 1], as decode_one_audio_mossformer2_se_48k computes it (`dither` at
// int16 scale; see core_kaldi::FbankParams::dither for the seeding).
// fbank [T, 60], feats [T, 180]; T = 0 for inputs shorter than one window.
int features(const float* in, int n, std::vector<float>& fbank, std::vector<float>& feats, float dither = 0.0f,
             uint64_t seed = 0);
// mask[T, 961] * STFT(x) -> iSTFT, torch.stft/istft(center=False, symmetric
// Hamming 1920, hop 384, length=n).  An all-ones mask reconstructs `x`.
std::vector<float> apply_mask(const float* x, int n, const float* mask, int n_frames, int n_threads = 1);
// The window starts decode_one_audio_mossformer2_se_48k runs for an `n`-sample
// input longer than its one-pass limit: padded length and window offsets.
struct SegmentPlan {
    int64_t padded = 0;
    int window = 0, stride = 0, give_up = 0;
    std::vector<int64_t> starts;
};
SegmentPlan segment_plan(int64_t n, int window, int stride);
} // namespace mf2se
