> Research integration: use the standalone build below. Historical references
> to `crispasr-diff reuse` describe the frozen research branch; the product CLI
> does not register RE-USE. Existing numerical results and gates are unchanged.

# Optional cuFFT research frontend

`CRISPASR_REUSE_FRONTEND=cufft` selects an independent GPU FFT in the RE-USE
research diff. The model graph and inverse transform retain their existing CPU
implementation. This option does not register a product backend or establish
full CUDA model execution, quantization, long-audio support, or Metal support.
The default frontend remains `native`; `mkl` and `mkl-avx2` keep their existing
behavior.

The pinned upstream `models/stfts.py` creates `torch.hann_window(640)` on the
CPU and copies it to the input device. Accordingly, this implementation shares
the independent float32 periodic Hann generator from the optional oneMKL
frontend: float32 angles, VML high-accuracy cosine, then float32 scaling. It reads
no reference arrays, weights or PyTorch symbols. Replacing this generator with
double-precision `std::cos` on the same float32 angles changed 25 of 640 retained
Torch window values, so that approximation is not substituted.

The CPU builds reflected, centered, windowed frames with the original
640-point FFT and 80-sample hop. Public
[cuFFT host APIs](https://docs.nvidia.com/cuda/cufft/index.html#cufftplanmany)
execute a batched float32 R2C transform with explicit contiguous input/output
layouts. The returned complex values retain signed zeros. Host `atan2` returns
raw phase, and `log1p(hypot(real, imaginary))` returns magnitude. No phase
wrapping, canonicalization, reference conditioning, or CPU FFT fallback is
performed by this frontend. All CUDA/cuFFT operation errors fail the call, and
failed analysis preserves caller-owned output and diagnostics.

Both external SDKs are required only when the CUDA research option is enabled:

```sh
cmake -S tools/ci-heavy/issue579_reuse -B "$RUN_DIR/build" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCRISPASR_REUSE_MKL_FRONTEND=ON \
  -DCRISPASR_REUSE_CUDA_FRONTEND=ON \
  -DCRISPASR_REUSE_MKL_INCLUDE_DIR="$MKL_INCLUDE" \
  -DCRISPASR_REUSE_MKL_LIBRARY="$MKL_RUNTIME" \
  -DCRISPASR_REUSE_CUDA_RUNTIME_INCLUDE_DIR="$CUDA_INCLUDE" \
  -DCRISPASR_REUSE_CUFFT_INCLUDE_DIR="$CUFFT_INCLUDE" \
  -DCRISPASR_REUSE_CUDART_LIBRARY="$CUDART_LIBRARY" \
  -DCRISPASR_REUSE_CUFFT_LIBRARY="$CUFFT_LIBRARY"
cmake --build "$RUN_DIR/build" -j2
REUSE_CUDA_TEST_GPU=1 ctest --test-dir "$RUN_DIR/build" --output-on-failure
CRISPASR_REUSE_FRONTEND=cufft "$RUN_DIR/build/reuse-research-diff" model.gguf ref-cuda.gguf
```

Each path variable names the corresponding external SDK include directory or
shared library. A complete toolkit can instead be discovered through
`CUDAToolkit_ROOT` and the usual `MKLROOT` hints. Split wheel layouts may require
explicit paths: the CUDA runtime include directory must also contain
`crt/host_defines.h`. The implementation uses host APIs, so it does not require
`nvcc` or enable `GGML_CUDA`. No vendor libraries or headers are bundled. Only
the research diff and probes link these optional dependencies; the product
library and backend remain unchanged.

`reuse-frontend-cuda-probe input.wav output.f32` accepts mono PCM16 at 16 kHz,
400..16000 samples, divisible by 80. Its little-endian float32 layout matches
the MKL diagnostic: `window[640]`, then time-major real, imaginary, log-magnitude
and raw phase matrices, each `[1 + samples/80, 321]`. It prints cuFFT/runtime/
driver versions, GPU identity and CPU Hann library version. Missing devices,
unsupported input, SDK errors and failed writes return nonzero.

The model-free tests cover default dependency isolation, explicit missing-SDK
errors, required independent Hann dependency, rejection of hidden devices,
absence of fallback/capture on failure, and caller ownership for nonfinite and
invalid input. With `REUSE_CUDA_TEST_GPU=1`, tests require a real CUDA device,
check analytic amplitude/endpoint/silence controls, and compare the independently
computed Hann, every raw complex float32 bit and unwrapped phase with original
Torch CUDA STFT at 400, 1600 and 16000 samples. Repeated transforms must produce
identical captures. The full original-model intermediate and waveform gates
remain separate and retain their existing thresholds.

The preliminary Tesla T4 experiment (Torch 2.11/CUDA 13, cuFFT library major 11)
found direct cuFFT and Torch CUDA complex results bitwise identical on the
retained 0.1-second input. That experiment supplied Torch-generated windowed
frames and established the FFT hypothesis only. Acceptance of this independent
implementation additionally requires the physical-GPU tests above and an
unconditioned 71-stage comparison against the original CUDA capture. CPU AVX2
references have different raw phases; the chosen FFT environment must match
the intended original reference. Other SDK versions, GPUs and inputs require
qualification rather than an assumption of portable bitwise equality.

The Kaggle T4 run `issue579-reuse-cuda-profile-v2` tested source `1d889456a`
with physical GPU controls at 400, 1600 and 16000 samples; all six isolated
CTest targets passed. The integrated 1600-sample run independently generated
its window, magnitude and phase and passed all 71 unconditioned and 71
conditioned `crispasr-diff` checks against the retained original CUDA capture.
Unconditioned waveform relative RMS was 1.0843e-6. That original reference
used Mamba's unfused CUDA reference, not its fused kernel. These are bounded
frontend and one-clip model results, not qualification of a full GPU backend,
other hardware, longer recordings or speech quality.

Kaggle run `issue579-reuse-cuda-fused-v1` repeated the physical GPU tests and
compared this independently computed frontend and native CPU model graph with
an original **fused** Mamba 2.2.4 CUDA capture of the same 100 ms input. All
71 unconditioned and 71 conditioned checks passed; unconditioned waveform
relative RMS was 1.1407e-6. The original capture recorded 120 calls to each
fused selective-scan and causal-convolution extension. This qualifies one clip
on the tested T4, not long recordings, other devices, a GPU model graph or
speech quality.

The one-second follow-up `issue579-reuse-cuda-fused-1s-v1` also passed all
71 unconditioned and 71 conditioned comparisons against an original fused
Mamba CUDA capture on a T4; unconditioned waveform relative RMS was
6.5531e-7. All six physical-GPU CTest targets passed. This is one one-second
excerpt, not long-recording, other-device, full GPU model or quality coverage.
