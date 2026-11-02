// Shared ggml building blocks of the ClearerVoice-Studio MossFormer2 family.
//
// MossFormer2_SS_16K (separation), MossFormer2_SE_48K (enhancement) and
// MossFormer2_SR_48K (super-resolution) all run the same "MossFormerM" stack
// from models/mossformer2_*/mossformer2_block.py (modelscope/ClearerVoice-Studio
// @6b3774dc): N x [FLASH_ShareA_FFConvM -> Gated-FSMN block] followed by
// LayerNorm(eps 1e-6), wrapped by Computation_Block (GroupNorm(1, D) + skip).
// The Gated-FSMN memory differs: SS uses Gated_FSMN_Block_Dilated
// (UniDeepFsmn_dilated: a 2-layer DilatedDenseNet), SE/SR use Gated_FSMN_Block
// (UniDeepFsmn: one k=2*lorder-1 depthwise conv with a residual); see
// BlockHParams::fsmn_dilated.  The encoder / mask head / decoder around that
// stack live in the per-model files (mossformer2_ss.cpp, mossformer2_se.cpp).
//
// Activation layout everywhere is ggml [C, T] (channels innermost), i.e. the
// PyTorch [B=1, T, C] sequence layout.  Depthwise time convolutions (dwconv)
// have two implementations: on the CPU backend a channel-vectorised custom op
// that works on [C, T] directly with tap-major kernels prepared at load
// (tap_kernel()); on every other backend ggml_ssm_conv over a transposed,
// padded copy, split into kernel widths all GPU backends implement, with
// dilation via zero-inserted kernels (dilate_kernel()).
#pragma once

#include "ggml.h"

#include <functional>
#include <map>
#include <string>
#include <vector>

namespace mf2 {

struct BlockHParams {
    int dim = 512;       // MossFormerM d_model
    int hidden = 2048;   // FLASH to_hidden width = int(dim * expansion_factor)
    int qk = 128;        // query_key_dim
    int group = 256;     // FLASH group_size
    int rotary_dim = 32; // RotaryEmbedding(dim=min(32, query_key_dim))
    float rotary_theta = 10000.0f;
    int fsmn_inner = 256; // Gated_FSMN_Block(_Dilated) inner_channels
    int fsmn_lorder = 20; // UniDeepFsmn(_dilated) lorder (kernel 2*lorder-1)
    // true: UniDeepFsmn_dilated (DilatedDenseNet, tensors fsmn.ddn.*; SS).
    // false: UniDeepFsmn (x + p + dwconv(p), tensor fsmn.mem.weight; SE/SR).
    bool fsmn_dilated = true;
    int conv_kernel = 17; // ConvModule depthwise kernel
    int n_layers = 24;
};

// Graph-building context.  `weight` must throw for a missing tensor; `derived`
// holds the load-time F32 kernels named by tap_name() (CPU) or dilated_name()
// (other backends); `capture` (optional) is called with stage names such as
// "l3_flash".
struct Ctx {
    ggml_context* g = nullptr;
    std::function<ggml_tensor*(const std::string&)> weight;
    const std::map<std::string, ggml_tensor*>* derived = nullptr;
    std::function<void(const std::string&, ggml_tensor*)> capture;
    ggml_tensor* positions = nullptr; // I32 [T], 0..T-1 (RoPE)
    // false: CPU custom-op dwconv (needs tap_name() kernels).  true: ggml_ssm_conv
    // split into the kernel widths every GPU backend implements (CUDA:
    // 3/4/5/9/15), dilation via dilated_name() kernels.
    bool portable_dwconv = false;
};

// --- generic ops ------------------------------------------------------------
ggml_tensor* as_f32(ggml_context* g, ggml_tensor* t); // cast non-F32 broadcast operands
ggml_tensor* linear(Ctx& c, ggml_tensor* x, const std::string& prefix, bool bias = true);
ggml_tensor* layer_norm(Ctx& c, ggml_tensor* x, const std::string& prefix, float eps); // over ne0
// nn.GroupNorm(1, C): statistics over the whole [C, T] tensor, per-channel affine.
ggml_tensor* global_group_norm(Ctx& c, ggml_tensor* x, const std::string& prefix, float eps);
// ScaleNorm: x / clamp(||x|| * dim^-0.5, eps) * g   (over ne0)
ggml_tensor* scale_norm(Ctx& c, ggml_tensor* x, const std::string& g_name, float eps = 1e-5f);
// PReLU: relu(x) + a * min(x, 0); `a` broadcasts (one slope, or one per row of ne0 / ne1).
ggml_tensor* prelu(ggml_context* g, ggml_tensor* x, ggml_tensor* a);
// Grouped depthwise conv over time with zero padding (PyTorch Conv1d/Conv2d
// with groups = C_out): x [C * gin, T]; output channel c reads input channels
// c * gin + j.  `kernel` names the GGUF tensor (PyTorch [C, gin, K(, 1)]).
// Returns [C, T + pad_l + pad_r - (K - 1) * dil].
ggml_tensor* dwconv(Ctx& c, ggml_tensor* x, const std::string& kernel, int gin, int dil, int pad_l, int pad_r);

// --- MossFormer2 modules ----------------------------------------------------
enum class Norm { Scale, Layer };
// FFConvM: norm -> Linear -> SiLU -> ConvModule (x + depthwise conv k=17).
ggml_tensor* ffconvm(Ctx& c, ggml_tensor* x, const std::string& prefix, Norm norm, const BlockHParams& hp);
// FLASH_ShareA_FFConvM (token shift, shared quadratic + linear attention,
// rotary qk, gated output) with residual.  prefix e.g. "blk.3.flash".
ggml_tensor* flash_layer(Ctx& c, ggml_tensor* x, const std::string& prefix, const BlockHParams& hp,
                         const std::string& stage = "");
// Gated_FSMN_Block(_Dilated) with residual.  prefix e.g. "blk.3.fsmn".
// Captures "<stage>_fsmn_ddn" (dilated) or "<stage>_fsmn_uni" (UniDeepFsmn).
ggml_tensor* gated_fsmn_block(Ctx& c, ggml_tensor* x, const std::string& prefix, const BlockHParams& hp,
                              const std::string& stage = "");
// MossformerBlock_GFSMN + final LayerNorm(eps 1e-6): `blk.{i}.{flash,fsmn}`.
// Captures "l{i}_flash" / "l{i}_fsmn" (and layer-0 sub-stages).
ggml_tensor* mossformer_m(Ctx& c, ggml_tensor* x, const BlockHParams& hp, const std::string& final_norm);
// Computation_Block: x + GroupNorm(MossFormerM(x)).
ggml_tensor* computation_block(Ctx& c, ggml_tensor* x, const BlockHParams& hp, const std::string& final_norm,
                               const std::string& group_norm);

// --- host-side helpers ------------------------------------------------------
// ScaledSinuEmbedding: [T, dim] row-major (ggml [dim, T]) = cat(sin, cos)(t * inv_freq) * scale.
std::vector<float> sinu_embedding(int T, const std::vector<float>& inv_freq, float scale);
// Derived kernels.  `w` is PyTorch order [C, gin, K].
// tap_kernel: ggml [C * gin, K] (row k holds every channel's tap k) for the CPU op.
std::vector<float> tap_kernel(const float* w, int C, int gin, int K);
// dilate_kernel: `gin` zero-inserted kernels of ggml [(K - 1) * dil + 1, C].
std::vector<std::vector<float>> dilate_kernel(const float* w, int C, int gin, int K, int dil);
std::string tap_name(const std::string& kernel);
std::string dilated_name(const std::string& kernel, int input);
// Every depthwise kernel of a MossFormerM stack: {name, gin, dil}.
struct DwKernel {
    std::string name;
    int gin, dil;
};
std::vector<DwKernel> dw_kernels(const BlockHParams& hp);
// Split K into ssm_conv kernel widths every backend implements.
std::vector<int> portable_kernel_split(int K);

} // namespace mf2
