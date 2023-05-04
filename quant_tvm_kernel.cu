#include <torch/all.h>
#include <torch/python.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "./src/kernels/gemm_kernel.cu"
#include "./src/kernels/gemv_kernel.cu"

void quant_kernel_3b_cuda(
    torch::Tensor mat_A,
    torch::Tensor mat_B,
    torch::Tensor mat_C,
    torch::Tensor scales,
    torch::Tensor zeros)
{
    int M = 1;
    for (int i = 0; i < mat_A.dim() - 1; i++)
        M *= mat_A.size(i);
    int N = mat_B.size(0);
    int K = mat_A.size(mat_A.dim() - 1);
    int K_Compressed = mat_B.size(1);

    if (M == 1)
    {
            const int num_warps = 4;
            const int warp_size = 32;
            dim3 blocks(
                (N + num_warps - 1) / num_warps, 1, 1);
            dim3 threads(warp_size, num_warps, 1);
            if (K == 9216)
                AT_DISPATCH_FLOATING_TYPES_AND_HALF(
                mat_A.type(), "tir_halfxint3_simt_bn4_k9216", ([&]
                                                               { tir_halfxint3_simt_bn4_k9216<<<blocks, threads>>>(
                                                                     (half *)mat_A.data_ptr(), (signed char *)mat_B.data_ptr(),
                                                                     (half *)scales.data_ptr(), (half *)zeros.data_ptr(), (half *)mat_C.data_ptr()); }));
        
            else if (K == 36864)
                AT_DISPATCH_FLOATING_TYPES_AND_HALF(
                    mat_A.type(), "tir_halfxint3_simt_bn4_k36864", ([&]
                                                                { tir_halfxint3_simt_bn4_k36864<<<blocks, threads>>>(
                                                                        (half *)mat_A.data_ptr(), (signed char *)mat_B.data_ptr(),
                                                                        (half *)scales.data_ptr(), (half *)zeros.data_ptr(), (half *)mat_C.data_ptr()); }));
    } else if (M == 16){
            const int warp_size = 32;
            const int warp_row = 1;
            const int warp_col = 4;
            const int raster = 0;
            const int BM = 16;
            const int BN = 64;
            const int BK = 64;
            dim3 blocks(
                (N + BN - 1) / BN, (M + BM - 1) / BM, 1);
            dim3 threads(warp_size, warp_row, warp_col);
            if (K == 9216)
                AT_DISPATCH_FLOATING_TYPES_AND_HALF(
                    mat_A.type(), "tir_halfxint3_tensorop_16x64x64x3_t0_y1z4_K9216_align8", ([&]
                                                                                             { tir_halfxint3_tensorop_16x64x64x3_t0_y1z4_K9216_align8<<<blocks, threads>>>(
                                                                                                   (half *)mat_A.data_ptr(), (signed char *)mat_B.data_ptr(),
                                                                                                   (half *)scales.data_ptr(), (half *)zeros.data_ptr(), (half *)mat_C.data_ptr()); }));
            else if (K == 36864)
                AT_DISPATCH_FLOATING_TYPES_AND_HALF(
                    mat_A.type(), "tir_halfxint3_tensorop_16x64x64x3_t0_y1z4_K36864_align8", ([&]
                                                                                              { tir_halfxint3_tensorop_16x64x64x3_t0_y1z4_K36864_align8<<<blocks, threads>>>(
                                                                                                    (half *)mat_A.data_ptr(), (signed char *)mat_B.data_ptr(),
                                                                                                    (half *)scales.data_ptr(), (half *)zeros.data_ptr(), (half *)mat_C.data_ptr()); }));
    } else if (M == 32){
            const int warp_size = 32;
            const int warp_row = 2;
            const int warp_col = 4;
            const int raster = 0;
            const int BM = 32;
            const int BN = 64;
            const int BK = 64;
            dim3 blocks(
                (N + BN - 1) / BN, (M + BM - 1) / BM, 1);
            dim3 threads(warp_size, warp_row, warp_col);
            if (K == 9216)
                AT_DISPATCH_FLOATING_TYPES_AND_HALF(
                    mat_A.type(), "tir_halfxint3_tensorop_32x64x64x2_t0_y2z4_K9216_align8", ([&]
                                                                                             { tir_halfxint3_tensorop_32x64x64x2_t0_y2z4_K9216_align8<<<blocks, threads>>>(
                                                                                                   (half *)mat_A.data_ptr(), (signed char *)mat_B.data_ptr(),
                                                                                                   (half *)scales.data_ptr(), (half *)zeros.data_ptr(), (half *)mat_C.data_ptr()); }));
            else if (K == 36864)
                AT_DISPATCH_FLOATING_TYPES_AND_HALF(
                    mat_A.type(), "tir_halfxint3_tensorop_32x64x64x2_t0_y2z4_K36864_align8", ([&]
                                                                                              { tir_halfxint3_tensorop_32x64x64x2_t0_y2z4_K36864_align8<<<blocks, threads>>>(
                                                                                                    (half *)mat_A.data_ptr(), (signed char *)mat_B.data_ptr(),
                                                                                                    (half *)scales.data_ptr(), (half *)zeros.data_ptr(), (half *)mat_C.data_ptr()); }));
    } else if (M == 64){
            const int warp_size = 32;
            const int warp_row = 2;
            const int warp_col = 4;
            const int raster = 0;
            const int BM = 64;
            const int BN = 64;
            const int BK = 64;
            dim3 blocks(
                (N + BN - 1) / BN, (M + BM - 1) / BM, 1);
            dim3 threads(warp_size, warp_row, warp_col);
            
            if (K == 9216)
                AT_DISPATCH_FLOATING_TYPES_AND_HALF(
                    mat_A.type(), "tir_halfxint3_tensorop_64x64x64x2_t0_y2z4_K9216_align8", ([&]
                                                                                             { tir_halfxint3_tensorop_64x64x64x2_t0_y2z4_K9216_align8<<<blocks, threads>>>(
                                                                                                   (half *)mat_A.data_ptr(), (signed char *)mat_B.data_ptr(),
                                                                                                   (half *)scales.data_ptr(), (half *)zeros.data_ptr(), (half *)mat_C.data_ptr()); }));
            else if (K == 36864)
                AT_DISPATCH_FLOATING_TYPES_AND_HALF(
                    mat_A.type(), "tir_halfxint3_tensorop_64x64x64x2_t0_y2z4_K36864_align8", ([&]
                                                                                              { tir_halfxint3_tensorop_64x64x64x2_t0_y2z4_K36864_align8<<<blocks, threads>>>(
                                                                                                    (half *)mat_A.data_ptr(), (signed char *)mat_B.data_ptr(),
                                                                                                    (half *)scales.data_ptr(), (half *)zeros.data_ptr(), (half *)mat_C.data_ptr()); }));
    } else if (M == 128){
            const int warp_size = 32;
            const int warp_row = 1;
            const int warp_col = 2;
            const int raster = 0;
            const int BM = 64;
            const int BN = 128;
            const int BK = 32;
            dim3 blocks(
                (N + BN - 1) / BN, (M + BM - 1) / BM, 1);
            dim3 threads(warp_size, warp_row, warp_col);
            if (K == 9216)
                AT_DISPATCH_FLOATING_TYPES_AND_HALF(
                    mat_A.type(), "tir_halfxint3_tensorop_64x128x32x3_t0_y1z2_K9216_align8", ([&]
                                                                                              { tir_halfxint3_tensorop_64x128x32x3_t0_y1z2_K9216_align8<<<blocks, threads>>>(
                                                                                                    (half *)mat_A.data_ptr(), (signed char *)mat_B.data_ptr(),
                                                                                                    (half *)scales.data_ptr(), (half *)zeros.data_ptr(), (half *)mat_C.data_ptr()); }));
            else if (K == 36864)
                AT_DISPATCH_FLOATING_TYPES_AND_HALF(
                    mat_A.type(), "tir_halfxint3_tensorop_64x128x32x3_t0_y1z2_K36864_align8", ([&]
                                                                                               { tir_halfxint3_tensorop_64x128x32x3_t0_y1z2_K36864_align8<<<blocks, threads>>>(
                                                                                                     (half *)mat_A.data_ptr(), (signed char *)mat_B.data_ptr(),
                                                                                                     (half *)scales.data_ptr(), (half *)zeros.data_ptr(), (half *)mat_C.data_ptr()); }));
    } else if (M >= 256){
            const int warp_size = 32;
            const int warp_row = 2;
            const int warp_col = 2;
            const int raster = 0;
            const int BM = 128;
            const int BN = 256;
            const int BK = 32;
            dim3 blocks(
                (N + BN - 1) / BN, (M + BM - 1) / BM, 1);
            dim3 threads(warp_size, warp_row, warp_col);
            if (K == 9216)
                AT_DISPATCH_FLOATING_TYPES_AND_HALF(
                    mat_A.type(), "tir_halfxint3_tensorop_128x256x32x1_t0_y2z2_K9216_align8", ([&]
                                                                                               { tir_halfxint3_tensorop_128x256x32x1_t0_y2z2_K9216_align8<<<blocks, threads>>>(
                                                                                                     (half *)mat_A.data_ptr(), (signed char *)mat_B.data_ptr(),
                                                                                                     (half *)scales.data_ptr(), (half *)zeros.data_ptr(), (half *)mat_C.data_ptr()); }));
            else if (K == 36864)
                AT_DISPATCH_FLOATING_TYPES_AND_HALF(
                    mat_A.type(), "tir_halfxint3_tensorop_128x256x32x1_t0_y2z2_K36864_align8", ([&]
                                                                                                { tir_halfxint3_tensorop_128x256x32x1_t0_y2z2_K36864_align8<<<blocks, threads>>>(
                                                                                                      (half *)mat_A.data_ptr(), (signed char *)mat_B.data_ptr(),
                                                                                                      (half *)scales.data_ptr(), (half *)zeros.data_ptr(), (half *)mat_C.data_ptr()); }));
    }
}