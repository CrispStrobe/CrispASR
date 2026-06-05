#include "conv-transpose-1d.cuh"
#include "convert.cuh"

template <typename src0_t>
static __global__ void conv_transpose_1d_kernel(
        const int s0, const int p0, const int d0, const int output_size,
        const int src0_ne0, const int src0_ne1, const int src0_ne2, const int src0_ne3,
        const int src1_ne0, const int src1_ne1, const int src1_ne2, const int src1_ne3,
        const int dst_ne0, const int dst_ne1, const int dst_ne2, const int dst_ne3,
        const src0_t * src0, const float * src1,  float * dst) {
    int global_index = threadIdx.x + blockIdx.x * blockDim.x;
    if (global_index >= output_size) {
        return;
    }

    int out_index = global_index / dst_ne0;

    float accumulator = 0;

    for (int c = 0; c < src0_ne2; c++) {
        int idx = global_index % dst_ne0;

        int kernel_offset = (src0_ne0 * src0_ne1 * c) + (out_index * src0_ne0);
        int input_offset = src1_ne0 * c;

        for (int i = 0; i < src1_ne0; i++) {
            if (!(idx >= i*s0 && idx < i*s0 + src0_ne0)) {
                continue;
            }
            int weight_idx = idx - i*s0;

            float kernel_weight = ggml_cuda_cast<float>(src0[kernel_offset + weight_idx]);
            float input_value =  src1[input_offset+i];

            accumulator += kernel_weight * input_value;
        }
    }
    dst[global_index] = accumulator;
    GGML_UNUSED_VARS(p0, d0, src0_ne3, src1_ne3, dst_ne3, src1_ne1, dst_ne1, src1_ne2, dst_ne2);
}

template <typename src0_t>
static void conv_transpose_1d_cuda(
        const int s0, const int p0, const int d0, const int output_size,
        const int src0_ne0, const int src0_ne1, const int src0_ne2, const int src0_ne3,
        const int src1_ne0, const int src1_ne1, const int src1_ne2, const int src1_ne3,
        const int dst_ne0, const int dst_ne1, const int dst_ne2, const int dst_ne3,
        const src0_t * src0, const float * src1,  float * dst,
        cudaStream_t stream) {

    const int num_blocks = (output_size + CUDA_CONV_TRANPOSE_1D_BLOCK_SIZE - 1) / CUDA_CONV_TRANPOSE_1D_BLOCK_SIZE;
    conv_transpose_1d_kernel<<<num_blocks,CUDA_CONV_TRANPOSE_1D_BLOCK_SIZE, 0, stream>>>(
        s0,p0,d0,output_size,
        src0_ne0, src0_ne1,  src0_ne2, src0_ne3,
        src1_ne0, src1_ne1,  src1_ne2, src1_ne3,
        dst_ne0,  dst_ne1,   dst_ne2,  dst_ne3,
        src0,src1, dst);
}

void ggml_cuda_op_conv_transpose_1d(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    const float * src1_d = (const float *)src1->data;

    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(src1));

    const int32_t * opts = (const int32_t *)dst->op_params;

    const int s0 = opts[0];
    const int p0 = 0;//opts[3];
    const int d0 = 1;//opts[4];

    const int64_t output_size = ggml_nelements(dst);

    if (src0->type == GGML_TYPE_F32) {
        conv_transpose_1d_cuda<float>(s0, p0, d0, output_size,
            src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3],
            src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3],
            dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
            (const float *)src0->data, src1_d, dst_d, stream);
    } else {
        conv_transpose_1d_cuda<half>(s0, p0, d0, output_size,
            src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3],
            src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3],
            dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
            (const half *)src0->data, src1_d, dst_d, stream);
    }
}
