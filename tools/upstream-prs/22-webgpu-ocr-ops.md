**Title:** `ggml-webgpu : add NORM, IM2COL, POOL_2D, CONV_TRANSPOSE_2D, UPSCALE, ARANGE`

**Status (2026-09-07):** Partially upstreamed. NORM, IM2COL and UPSCALE are present; regenerate this proposal for only POOL_2D, CONV_TRANSPOSE_2D and ARANGE.


---

The WebGPU backend covers the transformer op set but none of the ops that
CNN-style vision models need, so OCR/detection graphs (DBNet, ViT
recognizers with classic LayerNorm) either split to CPU per layer (sched)
or silently compute garbage (sched-less `ggml_backend_graph_compute` — the
encoder's `default:` returns nullopt, i.e. unhandled ops are dropped as
no-ops with no diagnostic).

Adds six kernels, all element-exact ports of the CPU reference semantics:

- `NORM` (classic LayerNorm) as a third variant of `row_norm.wgsl` — second
  workgroup accumulator for the mean, `var = E[x^2] - mean^2`, and the
  update helper gains a shift parameter (`dst = scale * (src + shift)`,
  shift = -mean; 0 for RMS/L2).
- `IM2COL` (2D, f32 input → f16/f32 columns) — one thread per dst element.
- `POOL_2D` (max/avg) — OOB taps skipped for max, zero-contribution for avg
  with full-kernel-area divisor (CPU semantics).
- `CONV_TRANSPOSE_2D` (f16 kernel × f32 input → f32) — gather formulation.
- `UPSCALE` (nearest + bilinear incl. align-corners, arbitrary sf0..sf3,
  non-contiguous src via stride_src0).
- `ARANGE` (f32).

Large convolution lowerings exceed the per-dimension workgroup limit, so
these kernels dispatch 2D with an `nwg_x` uniform and linearize in-shader
(helper `ggml_webgpu_wg2d`).

Also adds a stderr warning in the encoder's `default:` case — the silent
no-op skip cost us days of debugging a "0 detections" graph whose UPSCALE
nodes were simply dropped.

**Verification.** `test-backend-ops -o <OP> -b WebGPU` compiled with
emscripten (emdawnwebgpu, JSPI) and executed in headless Chromium on macOS
(Dawn/Metal): NORM 20/20, IM2COL 77/77, POOL_2D 128/128,
CONV_TRANSPOSE_2D 3/3, UPSCALE 11/11, ARANGE 2/2. End-to-end: a DBNet
(resnet18) + TrOCR pipeline in the browser produces region/text parity with
the CPU backend; detection stage drops from ~90 s (wasm SIMD CPU) to
~1.5 s on an M1. A ViT formula recognizer (pix2tex) gains ~1.4x from the
NORM kernel alone (~2.8x total vs wasm CPU).

Patch: `22-webgpu-ocr-ops.patch` (3 files modified, 5 shaders added,
+~950 lines). WGSL literal gotcha worth reviewer attention: `-3.4028235e38`
is rejected by WGSL as unrepresentable in f32 — `-FLT_MAX` is expressed as
`bitcast<f32>(0xff7fffffu)` in pool2d.

One caveat for the PR discussion: the fp16 `-sJSPI` browser path needs the
suspending exports listed in `-sJSPI_EXPORTS` by the embedding application;
nothing in this patch depends on it, but the browser verification setup
does.
