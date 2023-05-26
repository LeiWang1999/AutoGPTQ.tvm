#include <cuda_fp16.h>
extern "C" __global__ void __launch_bounds__(128) tir_halfxint4_simt_bn4_k6656(half* __restrict__ A, signed char* __restrict__ B, half* __restrict__ Scales, half* __restrict__ Zeros, half* __restrict__ C) {
  half in_thread_C_local[1];
  half A_local[8];
  half red_buf0[1];
  in_thread_C_local[0] = __float2half_rn(0.000000e+00f);
  for (int k_0 = 0; k_0 < 26; ++k_0) {
    *(uint4*)(A_local + 0) = *(uint4*)(A + ((k_0 * 256) + (((int)threadIdx.x) * 8)));
    for (int k_2 = 0; k_2 < 8; ++k_2) {
      in_thread_C_local[0] = (in_thread_C_local[0] + (A_local[k_2] * ((((half)((((int)B[(((((((int)blockIdx.x) * 13312) + (((int)threadIdx.y) * 3328)) + (k_0 * 128)) + (((int)threadIdx.x) * 4)) + (k_2 >> 1))]) >> ((k_2 & 1) * 4)) & 15)) * Scales[((((int)blockIdx.x) * 4) + ((int)threadIdx.y))]) - Zeros[((((int)blockIdx.x) * 4) + ((int)threadIdx.y))])));
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

