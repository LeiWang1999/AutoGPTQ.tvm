#include <cuda_fp16.h>
#include <mma.h>

#if (((__CUDACC_VER_MAJOR__ == 11) && (__CUDACC_VER_MINOR__ >= 4)) || \
     (__CUDACC_VER_MAJOR__ > 11))
#define TVM_ENABLE_L2_PREFETCH 1
#else
#define TVM_ENABLE_L2_PREFETCH 0
#endif

#ifdef _WIN32
using uint = unsigned int;
using uchar = unsigned char;
using ushort = unsigned short;
using int64_t = long long;
using uint64_t = unsigned long long;
#else
#define uint unsigned int
#define uchar unsigned char
#define ushort unsigned short
#define int64_t long long
#define uint64_t unsigned long long
#endif

extern "C" __global__ void __launch_bounds__(128) tir_halfxint3_simt_bn4_k9216(half *__restrict__ A, signed char *__restrict__ B, half *__restrict__ Scales, half *__restrict__ Zeros, half *__restrict__ C)
{
    half in_thread_C_local[1];
    half A_local[8];
    half red_buf0[1];
    in_thread_C_local[0] = __float2half_rn(0.000000e+00f);
    for (int k_0 = 0; k_0 < 36; ++k_0)
    {
        *(uint4 *)(A_local + 0) = *(uint4 *)(A + ((k_0 * 256) + (((int)threadIdx.x) * 8)));
        for (int k_2 = 0; k_2 < 8; ++k_2)
        {
            in_thread_C_local[0] = (in_thread_C_local[0] + (A_local[k_2] * ((((((k_2 * 3) & 7) <= 5) ? ((half)((((int)B[(((((((int)blockIdx.x) * 13824) + (((int)threadIdx.y) * 3456)) + (k_0 * 96)) + (((int)threadIdx.x) * 3)) + ((k_2 * 3) >> 3))]) >> ((k_2 * 3) & 7)) & 7)) : ((half)(((signed char)((((int)B[(((((((int)blockIdx.x) * 13824) + (((int)threadIdx.y) * 3456)) + (k_0 * 96)) + (((int)threadIdx.x) * 3)) + ((k_2 * 3) >> 3))]) >> ((k_2 * 3) & 7)) & ((1 << (8 - ((k_2 * 3) & 7))) - 1))) | ((signed char)(((((int)B[((((((((int)blockIdx.x) * 13824) + (((int)threadIdx.y) * 3456)) + (k_0 * 96)) + (((int)threadIdx.x) * 3)) + ((k_2 * 3) >> 3)) + 1)]) << (8 - ((k_2 * 3) & 7))) & (7 << (8 - ((k_2 * 3) & 7)))) & 7))))) * Scales[((((int)blockIdx.x) * 4) + ((int)threadIdx.y))]) - Zeros[((((int)blockIdx.x) * 4) + ((int)threadIdx.y))])));
        }
    }
    uint mask[1];
    half t0[1];
    red_buf0[0] = in_thread_C_local[0];
    mask[0] = (__activemask() & ((uint)(0 << (((int)threadIdx.y) * 32))));
    t0[0] = __shfl_down_sync(mask[0], red_buf0[0], 16, 32);
    red_buf0[0] = (red_buf0[0] + t0[0]);
    t0[0] = __shfl_down_sync(mask[0], red_buf0[0], 8, 32);
    red_buf0[0] = (red_buf0[0] + t0[0]);
    t0[0] = __shfl_down_sync(mask[0], red_buf0[0], 4, 32);
    red_buf0[0] = (red_buf0[0] + t0[0]);
    t0[0] = __shfl_down_sync(mask[0], red_buf0[0], 2, 32);
    red_buf0[0] = (red_buf0[0] + t0[0]);
    t0[0] = __shfl_down_sync(mask[0], red_buf0[0], 1, 32);
    red_buf0[0] = (red_buf0[0] + t0[0]);
    red_buf0[0] = __shfl_sync(mask[0], red_buf0[0], (((int)threadIdx.y) * 32), 32);
    C[((((int)blockIdx.x) * 4) + ((int)threadIdx.y))] = red_buf0[0];
}

extern "C" __global__ void __launch_bounds__(128) tir_halfxint3_simt_bn4_k36864(half *__restrict__ A, signed char *__restrict__ B, half *__restrict__ Scales, half *__restrict__ Zeros, half *__restrict__ C)
{
    half in_thread_C_local[1];
    half A_local[8];
    half red_buf0[1];
    in_thread_C_local[0] = __float2half_rn(0.000000e+00f);
    for (int k_0 = 0; k_0 < 144; ++k_0)
    {
        *(uint4 *)(A_local + 0) = *(uint4 *)(A + ((k_0 * 256) + (((int)threadIdx.x) * 8)));
        for (int k_2 = 0; k_2 < 8; ++k_2)
        {
            in_thread_C_local[0] = (in_thread_C_local[0] + (A_local[k_2] * ((((((k_2 * 3) & 7) <= 5) ? ((half)((((int)B[(((((((int)blockIdx.x) * 55296) + (((int)threadIdx.y) * 13824)) + (k_0 * 96)) + (((int)threadIdx.x) * 3)) + ((k_2 * 3) >> 3))]) >> ((k_2 * 3) & 7)) & 7)) : ((half)(((signed char)((((int)B[(((((((int)blockIdx.x) * 55296) + (((int)threadIdx.y) * 13824)) + (k_0 * 96)) + (((int)threadIdx.x) * 3)) + ((k_2 * 3) >> 3))]) >> ((k_2 * 3) & 7)) & ((1 << (8 - ((k_2 * 3) & 7))) - 1))) | ((signed char)(((((int)B[((((((((int)blockIdx.x) * 55296) + (((int)threadIdx.y) * 13824)) + (k_0 * 96)) + (((int)threadIdx.x) * 3)) + ((k_2 * 3) >> 3)) + 1)]) << (8 - ((k_2 * 3) & 7))) & (7 << (8 - ((k_2 * 3) & 7)))) & 7))))) * Scales[((((int)blockIdx.x) * 4) + ((int)threadIdx.y))]) - Zeros[((((int)blockIdx.x) * 4) + ((int)threadIdx.y))])));
        }
    }
    uint mask[1];
    half t0[1];
    red_buf0[0] = in_thread_C_local[0];
    mask[0] = (__activemask() & ((uint)(0 << (((int)threadIdx.y) * 32))));
    t0[0] = __shfl_down_sync(mask[0], red_buf0[0], 16, 32);
    red_buf0[0] = (red_buf0[0] + t0[0]);
    t0[0] = __shfl_down_sync(mask[0], red_buf0[0], 8, 32);
    red_buf0[0] = (red_buf0[0] + t0[0]);
    t0[0] = __shfl_down_sync(mask[0], red_buf0[0], 4, 32);
    red_buf0[0] = (red_buf0[0] + t0[0]);
    t0[0] = __shfl_down_sync(mask[0], red_buf0[0], 2, 32);
    red_buf0[0] = (red_buf0[0] + t0[0]);
    t0[0] = __shfl_down_sync(mask[0], red_buf0[0], 1, 32);
    red_buf0[0] = (red_buf0[0] + t0[0]);
    red_buf0[0] = __shfl_sync(mask[0], red_buf0[0], (((int)threadIdx.y) * 32), 32);
    C[((((int)blockIdx.x) * 4) + ((int)threadIdx.y))] = red_buf0[0];
}

