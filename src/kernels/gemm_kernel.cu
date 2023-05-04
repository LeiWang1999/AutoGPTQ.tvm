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

__global__ void __launch_bounds__(128) tir_halfxint3_tensorop_16x64x64x3_t0_y1z4_K9216_align8(half *__restrict__ A, signed char *__restrict__ B, half *__restrict__ Scales, half *__restrict__ Zeros, half *__restrict__ C)
{
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, half> C_wmma_accumulator[1];
    __shared__ half A_shared[3456];
    __shared__ half B_rescale_shared[13824];
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::row_major> A_shared_wmma_matrix_a[1];
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::col_major> B_rescale_shared_wmma_matrix_b[1];
    nvcuda::wmma::fill_fragment(C_wmma_accumulator[0], 0.000000e+00f);

    {
        unsigned int addr;
#if TVM_ENBALE_EFFICIENT_SMEM_PTR_CAST
        addr = static_cast<unsigned int>(__cvta_generic_to_shared((void *)(A_shared + (((((int)threadIdx.z) * 288) + ((((int)threadIdx.x) >> 3) * 72)) + ((((int)threadIdx.x) & 7) * 8)))));
#else
        __asm__ __volatile__(
            "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }\n"
            : "=r"(addr)
            : "l"((void *)(A_shared + (((((int)threadIdx.z) * 288) + ((((int)threadIdx.x) >> 3) * 72)) + ((((int)threadIdx.x) & 7) * 8)))));
#endif
        __asm__ __volatile__(
#if TVM_ENABLE_L2_PREFETCH
            "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
#else
            "cp.async.cg.shared.global [%0], [%1], %2;"
#endif
            ::"r"(addr),
            "l"((void *)(A + ((((((int)blockIdx.y) * 147456) + (((int)threadIdx.z) * 36864)) + ((((int)threadIdx.x) >> 3) * 9216)) + ((((int)threadIdx.x) & 7) * 8)))), "n"(16));
    }
    for (int ax0_ax1_fused_2 = 0; ax0_ax1_fused_2 < 32; ++ax0_ax1_fused_2)
    {
        B_rescale_shared[((((((int)threadIdx.z) * 1152) + ((ax0_ax1_fused_2 >> 1) * 72)) + ((ax0_ax1_fused_2 & 1) * 32)) + ((int)threadIdx.x))] = ((((((((int)threadIdx.x) * 3) & 7) <= 5) ? ((half)((((int)B[(((((((int)blockIdx.x) * 221184) + (((int)threadIdx.z) * 55296)) + ((ax0_ax1_fused_2 >> 1) * 3456)) + ((ax0_ax1_fused_2 & 1) * 12)) + ((((int)threadIdx.x) * 3) >> 3))]) >> ((((int)threadIdx.x) * 3) & 7)) & 7)) : ((half)(((signed char)((((int)B[(((((((int)blockIdx.x) * 221184) + (((int)threadIdx.z) * 55296)) + ((ax0_ax1_fused_2 >> 1) * 3456)) + ((ax0_ax1_fused_2 & 1) * 12)) + ((((int)threadIdx.x) * 3) >> 3))]) >> ((((int)threadIdx.x) * 3) & 7)) & ((1 << (8 - ((((int)threadIdx.x) * 3) & 7))) - 1))) | ((signed char)(((((int)B[((((((((int)blockIdx.x) * 221184) + (((int)threadIdx.z) * 55296)) + ((ax0_ax1_fused_2 >> 1) * 3456)) + ((ax0_ax1_fused_2 & 1) * 12)) + ((((int)threadIdx.x) * 3) >> 3)) + 1)]) << (8 - ((((int)threadIdx.x) * 3) & 7))) & (7 << (8 - ((((int)threadIdx.x) * 3) & 7)))) & 7))))) * Scales[(((((int)blockIdx.x) * 64) + (((int)threadIdx.z) * 16)) + (ax0_ax1_fused_2 >> 1))]) - Zeros[(((((int)blockIdx.x) * 64) + (((int)threadIdx.z) * 16)) + (ax0_ax1_fused_2 >> 1))]);
    }
    __asm__ __volatile__("cp.async.commit_group;");

    {
        unsigned int addr;
#if TVM_ENBALE_EFFICIENT_SMEM_PTR_CAST
        addr = static_cast<unsigned int>(__cvta_generic_to_shared((void *)(A_shared + ((((((int)threadIdx.z) * 288) + ((((int)threadIdx.x) >> 3) * 72)) + ((((int)threadIdx.x) & 7) * 8)) + 1152))));
#else
        __asm__ __volatile__(
            "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }\n"
            : "=r"(addr)
            : "l"((void *)(A_shared + ((((((int)threadIdx.z) * 288) + ((((int)threadIdx.x) >> 3) * 72)) + ((((int)threadIdx.x) & 7) * 8)) + 1152))));
#endif
        __asm__ __volatile__(
#if TVM_ENABLE_L2_PREFETCH
            "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
#else
            "cp.async.cg.shared.global [%0], [%1], %2;"
#endif
            ::"r"(addr),
            "l"((void *)(A + (((((((int)blockIdx.y) * 147456) + (((int)threadIdx.z) * 36864)) + ((((int)threadIdx.x) >> 3) * 9216)) + ((((int)threadIdx.x) & 7) * 8)) + 64))), "n"(16));
    }
    for (int ax0_ax1_fused_2_1 = 0; ax0_ax1_fused_2_1 < 32; ++ax0_ax1_fused_2_1)
    {
        B_rescale_shared[(((((((int)threadIdx.z) * 1152) + ((ax0_ax1_fused_2_1 >> 1) * 72)) + ((ax0_ax1_fused_2_1 & 1) * 32)) + ((int)threadIdx.x)) + 4608)] = ((((((((int)threadIdx.x) * 3) & 7) <= 5) ? ((half)((((int)B[((((((((int)blockIdx.x) * 221184) + (((int)threadIdx.z) * 55296)) + ((ax0_ax1_fused_2_1 >> 1) * 3456)) + ((ax0_ax1_fused_2_1 & 1) * 12)) + ((((int)threadIdx.x) * 3) >> 3)) + 24)]) >> ((((int)threadIdx.x) * 3) & 7)) & 7)) : ((half)(((signed char)((((int)B[((((((((int)blockIdx.x) * 221184) + (((int)threadIdx.z) * 55296)) + ((ax0_ax1_fused_2_1 >> 1) * 3456)) + ((ax0_ax1_fused_2_1 & 1) * 12)) + ((((int)threadIdx.x) * 3) >> 3)) + 24)]) >> ((((int)threadIdx.x) * 3) & 7)) & ((1 << (8 - ((((int)threadIdx.x) * 3) & 7))) - 1))) | ((signed char)(((((int)B[((((((((int)blockIdx.x) * 221184) + (((int)threadIdx.z) * 55296)) + ((ax0_ax1_fused_2_1 >> 1) * 3456)) + ((ax0_ax1_fused_2_1 & 1) * 12)) + ((((int)threadIdx.x) * 3) >> 3)) + 25)]) << (8 - ((((int)threadIdx.x) * 3) & 7))) & (7 << (8 - ((((int)threadIdx.x) * 3) & 7)))) & 7))))) * Scales[(((((int)blockIdx.x) * 64) + (((int)threadIdx.z) * 16)) + (ax0_ax1_fused_2_1 >> 1))]) - Zeros[(((((int)blockIdx.x) * 64) + (((int)threadIdx.z) * 16)) + (ax0_ax1_fused_2_1 >> 1))]);
    }
    __asm__ __volatile__("cp.async.commit_group;");

    for (int k_0_0 = 0; k_0_0 < 142; ++k_0_0)
    {
        __syncthreads();

        {
            unsigned int addr;
#if TVM_ENBALE_EFFICIENT_SMEM_PTR_CAST
            addr = static_cast<unsigned int>(__cvta_generic_to_shared((void *)(A_shared + ((((((k_0_0 + 2) % 3) * 1152) + (((int)threadIdx.z) * 288)) + ((((int)threadIdx.x) >> 3) * 72)) + ((((int)threadIdx.x) & 7) * 8)))));
#else
            __asm__ __volatile__(
                "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }\n"
                : "=r"(addr)
                : "l"((void *)(A_shared + ((((((k_0_0 + 2) % 3) * 1152) + (((int)threadIdx.z) * 288)) + ((((int)threadIdx.x) >> 3) * 72)) + ((((int)threadIdx.x) & 7) * 8)))));
#endif
            __asm__ __volatile__(
#if TVM_ENABLE_L2_PREFETCH
                "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
#else
                "cp.async.cg.shared.global [%0], [%1], %2;"
#endif
                ::"r"(addr),
                "l"((void *)(A + ((((((((int)blockIdx.y) * 147456) + (((int)threadIdx.z) * 36864)) + ((((int)threadIdx.x) >> 3) * 9216)) + (k_0_0 * 64)) + ((((int)threadIdx.x) & 7) * 8)) + 128))), "n"(16));
        }
        for (int ax0_ax1_fused_2_2 = 0; ax0_ax1_fused_2_2 < 32; ++ax0_ax1_fused_2_2)
        {
            B_rescale_shared[(((((((k_0_0 + 2) % 3) * 4608) + (((int)threadIdx.z) * 1152)) + ((ax0_ax1_fused_2_2 >> 1) * 72)) + ((ax0_ax1_fused_2_2 & 1) * 32)) + ((int)threadIdx.x))] = ((((((((int)threadIdx.x) * 3) & 7) <= 5) ? ((half)((((int)B[(((((((((int)blockIdx.x) * 221184) + (((int)threadIdx.z) * 55296)) + ((ax0_ax1_fused_2_2 >> 1) * 3456)) + (k_0_0 * 24)) + ((ax0_ax1_fused_2_2 & 1) * 12)) + ((((int)threadIdx.x) * 3) >> 3)) + 48)]) >> ((((int)threadIdx.x) * 3) & 7)) & 7)) : ((half)(((signed char)((((int)B[(((((((((int)blockIdx.x) * 221184) + (((int)threadIdx.z) * 55296)) + ((ax0_ax1_fused_2_2 >> 1) * 3456)) + (k_0_0 * 24)) + ((ax0_ax1_fused_2_2 & 1) * 12)) + ((((int)threadIdx.x) * 3) >> 3)) + 48)]) >> ((((int)threadIdx.x) * 3) & 7)) & ((1 << (8 - ((((int)threadIdx.x) * 3) & 7))) - 1))) | ((signed char)(((((int)B[(((((((((int)blockIdx.x) * 221184) + (((int)threadIdx.z) * 55296)) + ((ax0_ax1_fused_2_2 >> 1) * 3456)) + (k_0_0 * 24)) + ((ax0_ax1_fused_2_2 & 1) * 12)) + ((((int)threadIdx.x) * 3) >> 3)) + 49)]) << (8 - ((((int)threadIdx.x) * 3) & 7))) & (7 << (8 - ((((int)threadIdx.x) * 3) & 7)))) & 7))))) * Scales[(((((int)blockIdx.x) * 64) + (((int)threadIdx.z) * 16)) + (ax0_ax1_fused_2_2 >> 1))]) - Zeros[(((((int)blockIdx.x) * 64) + (((int)threadIdx.z) * 16)) + (ax0_ax1_fused_2_2 >> 1))]);
        }
        __asm__ __volatile__("cp.async.commit_group;");

        __asm__ __volatile__("cp.async.wait_group 2;");

        __syncthreads();
        for (int k_0_1 = 0; k_0_1 < 4; ++k_0_1)
        {
            nvcuda::wmma::load_matrix_sync(A_shared_wmma_matrix_a[0], (&(A_shared[(((k_0_0 % 3) * 1152) + (k_0_1 * 16))])), 72);
            nvcuda::wmma::load_matrix_sync(B_rescale_shared_wmma_matrix_b[0], (&(B_rescale_shared[((((k_0_0 % 3) * 4608) + (((int)threadIdx.z) * 1152)) + (k_0_1 * 16))])), 72);
            nvcuda::wmma::mma_sync(C_wmma_accumulator[0], A_shared_wmma_matrix_a[0], B_rescale_shared_wmma_matrix_b[0], C_wmma_accumulator[0]);
        }
    }
    __asm__ __volatile__("cp.async.wait_group 1;");

    __syncthreads();
    for (int k_0_1_1 = 0; k_0_1_1 < 4; ++k_0_1_1)
    {
        nvcuda::wmma::load_matrix_sync(A_shared_wmma_matrix_a[0], (&(A_shared[((k_0_1_1 * 16) + 1152)])), 72);
        nvcuda::wmma::load_matrix_sync(B_rescale_shared_wmma_matrix_b[0], (&(B_rescale_shared[(((((int)threadIdx.z) * 1152) + (k_0_1_1 * 16)) + 4608)])), 72);
        nvcuda::wmma::mma_sync(C_wmma_accumulator[0], A_shared_wmma_matrix_a[0], B_rescale_shared_wmma_matrix_b[0], C_wmma_accumulator[0]);
    }
    __asm__ __volatile__("cp.async.wait_group 0;");

    __syncthreads();
    for (int k_0_1_2 = 0; k_0_1_2 < 4; ++k_0_1_2)
    {
        nvcuda::wmma::load_matrix_sync(A_shared_wmma_matrix_a[0], (&(A_shared[((k_0_1_2 * 16) + 2304)])), 72);
        nvcuda::wmma::load_matrix_sync(B_rescale_shared_wmma_matrix_b[0], (&(B_rescale_shared[(((((int)threadIdx.z) * 1152) + (k_0_1_2 * 16)) + 9216)])), 72);
        nvcuda::wmma::mma_sync(C_wmma_accumulator[0], A_shared_wmma_matrix_a[0], B_rescale_shared_wmma_matrix_b[0], C_wmma_accumulator[0]);
    }
    nvcuda::wmma::store_matrix_sync((&(C[(((((int)blockIdx.y) * 147456) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.z) * 16))])), C_wmma_accumulator[0], 9216, nvcuda::wmma::mem_row_major);
}

__global__ void __launch_bounds__(256) tir_halfxint3_tensorop_32x64x64x2_t0_y2z4_K9216_align8(half *__restrict__ A, signed char *__restrict__ B, half *__restrict__ Scales, half *__restrict__ Zeros, half *__restrict__ C)
{
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, half> C_wmma_accumulator[1];
    __shared__ half A_shared[4608];
    __shared__ half B_rescale_shared[9216];
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::row_major> A_shared_wmma_matrix_a[1];
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::col_major> B_rescale_shared_wmma_matrix_b[1];
    nvcuda::wmma::fill_fragment(C_wmma_accumulator[0], 0.000000e+00f);

    {
        unsigned int addr;
#if TVM_ENBALE_EFFICIENT_SMEM_PTR_CAST
        addr = static_cast<unsigned int>(__cvta_generic_to_shared((void *)(A_shared + ((((((int)threadIdx.y) * 1152) + (((int)threadIdx.z) * 288)) + ((((int)threadIdx.x) >> 3) * 72)) + ((((int)threadIdx.x) & 7) * 8)))));
#else
        __asm__ __volatile__(
            "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }\n"
            : "=r"(addr)
            : "l"((void *)(A_shared + ((((((int)threadIdx.y) * 1152) + (((int)threadIdx.z) * 288)) + ((((int)threadIdx.x) >> 3) * 72)) + ((((int)threadIdx.x) & 7) * 8)))));
#endif
        __asm__ __volatile__(
#if TVM_ENABLE_L2_PREFETCH
            "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
#else
            "cp.async.cg.shared.global [%0], [%1], %2;"
#endif
            ::"r"(addr),
            "l"((void *)(A + (((((((int)blockIdx.y) * 294912) + (((int)threadIdx.y) * 147456)) + (((int)threadIdx.z) * 36864)) + ((((int)threadIdx.x) >> 3) * 9216)) + ((((int)threadIdx.x) & 7) * 8)))), "n"(16));
    }
    for (int ax0_ax1_fused_2 = 0; ax0_ax1_fused_2 < 16; ++ax0_ax1_fused_2)
    {
        B_rescale_shared[(((((((int)threadIdx.y) * 2304) + (((int)threadIdx.z) * 576)) + ((ax0_ax1_fused_2 >> 1) * 72)) + ((ax0_ax1_fused_2 & 1) * 32)) + ((int)threadIdx.x))] = ((((((((int)threadIdx.x) * 3) & 7) <= 5) ? ((half)((((int)B[((((((((int)blockIdx.x) * 221184) + (((int)threadIdx.y) * 110592)) + (((int)threadIdx.z) * 27648)) + ((ax0_ax1_fused_2 >> 1) * 3456)) + ((ax0_ax1_fused_2 & 1) * 12)) + ((((int)threadIdx.x) * 3) >> 3))]) >> ((((int)threadIdx.x) * 3) & 7)) & 7)) : ((half)(((signed char)((((int)B[((((((((int)blockIdx.x) * 221184) + (((int)threadIdx.y) * 110592)) + (((int)threadIdx.z) * 27648)) + ((ax0_ax1_fused_2 >> 1) * 3456)) + ((ax0_ax1_fused_2 & 1) * 12)) + ((((int)threadIdx.x) * 3) >> 3))]) >> ((((int)threadIdx.x) * 3) & 7)) & ((1 << (8 - ((((int)threadIdx.x) * 3) & 7))) - 1))) | ((signed char)(((((int)B[(((((((((int)blockIdx.x) * 221184) + (((int)threadIdx.y) * 110592)) + (((int)threadIdx.z) * 27648)) + ((ax0_ax1_fused_2 >> 1) * 3456)) + ((ax0_ax1_fused_2 & 1) * 12)) + ((((int)threadIdx.x) * 3) >> 3)) + 1)]) << (8 - ((((int)threadIdx.x) * 3) & 7))) & (7 << (8 - ((((int)threadIdx.x) * 3) & 7)))) & 7))))) * Scales[((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.z) * 8)) + (ax0_ax1_fused_2 >> 1))]) - Zeros[((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.z) * 8)) + (ax0_ax1_fused_2 >> 1))]);
    }
    __asm__ __volatile__("cp.async.commit_group;");

    for (int k_0_0 = 0; k_0_0 < 143; ++k_0_0)
    {
        __syncthreads();

        {
            unsigned int addr;
#if TVM_ENBALE_EFFICIENT_SMEM_PTR_CAST
            addr = static_cast<unsigned int>(__cvta_generic_to_shared((void *)(A_shared + (((((((k_0_0 + 1) & 1) * 2304) + (((int)threadIdx.y) * 1152)) + (((int)threadIdx.z) * 288)) + ((((int)threadIdx.x) >> 3) * 72)) + ((((int)threadIdx.x) & 7) * 8)))));
#else
            __asm__ __volatile__(
                "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }\n"
                : "=r"(addr)
                : "l"((void *)(A_shared + (((((((k_0_0 + 1) & 1) * 2304) + (((int)threadIdx.y) * 1152)) + (((int)threadIdx.z) * 288)) + ((((int)threadIdx.x) >> 3) * 72)) + ((((int)threadIdx.x) & 7) * 8)))));
#endif
            __asm__ __volatile__(
#if TVM_ENABLE_L2_PREFETCH
                "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
#else
                "cp.async.cg.shared.global [%0], [%1], %2;"
#endif
                ::"r"(addr),
                "l"((void *)(A + (((((((((int)blockIdx.y) * 294912) + (((int)threadIdx.y) * 147456)) + (((int)threadIdx.z) * 36864)) + ((((int)threadIdx.x) >> 3) * 9216)) + (k_0_0 * 64)) + ((((int)threadIdx.x) & 7) * 8)) + 64))), "n"(16));
        }
        for (int ax0_ax1_fused_2_1 = 0; ax0_ax1_fused_2_1 < 16; ++ax0_ax1_fused_2_1)
        {
            B_rescale_shared[((((((((k_0_0 + 1) & 1) * 4608) + (((int)threadIdx.y) * 2304)) + (((int)threadIdx.z) * 576)) + ((ax0_ax1_fused_2_1 >> 1) * 72)) + ((ax0_ax1_fused_2_1 & 1) * 32)) + ((int)threadIdx.x))] = ((((((((int)threadIdx.x) * 3) & 7) <= 5) ? ((half)((((int)B[((((((((((int)blockIdx.x) * 221184) + (((int)threadIdx.y) * 110592)) + (((int)threadIdx.z) * 27648)) + ((ax0_ax1_fused_2_1 >> 1) * 3456)) + (k_0_0 * 24)) + ((ax0_ax1_fused_2_1 & 1) * 12)) + ((((int)threadIdx.x) * 3) >> 3)) + 24)]) >> ((((int)threadIdx.x) * 3) & 7)) & 7)) : ((half)(((signed char)((((int)B[((((((((((int)blockIdx.x) * 221184) + (((int)threadIdx.y) * 110592)) + (((int)threadIdx.z) * 27648)) + ((ax0_ax1_fused_2_1 >> 1) * 3456)) + (k_0_0 * 24)) + ((ax0_ax1_fused_2_1 & 1) * 12)) + ((((int)threadIdx.x) * 3) >> 3)) + 24)]) >> ((((int)threadIdx.x) * 3) & 7)) & ((1 << (8 - ((((int)threadIdx.x) * 3) & 7))) - 1))) | ((signed char)(((((int)B[((((((((((int)blockIdx.x) * 221184) + (((int)threadIdx.y) * 110592)) + (((int)threadIdx.z) * 27648)) + ((ax0_ax1_fused_2_1 >> 1) * 3456)) + (k_0_0 * 24)) + ((ax0_ax1_fused_2_1 & 1) * 12)) + ((((int)threadIdx.x) * 3) >> 3)) + 25)]) << (8 - ((((int)threadIdx.x) * 3) & 7))) & (7 << (8 - ((((int)threadIdx.x) * 3) & 7)))) & 7))))) * Scales[((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.z) * 8)) + (ax0_ax1_fused_2_1 >> 1))]) - Zeros[((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.z) * 8)) + (ax0_ax1_fused_2_1 >> 1))]);
        }
        __asm__ __volatile__("cp.async.commit_group;");

        __asm__ __volatile__("cp.async.wait_group 1;");

        __syncthreads();
        for (int k_0_1 = 0; k_0_1 < 4; ++k_0_1)
        {
            nvcuda::wmma::load_matrix_sync(A_shared_wmma_matrix_a[0], (&(A_shared[((((k_0_0 & 1) * 2304) + (((int)threadIdx.y) * 1152)) + (k_0_1 * 16))])), 72);
            nvcuda::wmma::load_matrix_sync(B_rescale_shared_wmma_matrix_b[0], (&(B_rescale_shared[((((k_0_0 & 1) * 4608) + (((int)threadIdx.z) * 1152)) + (k_0_1 * 16))])), 72);
            nvcuda::wmma::mma_sync(C_wmma_accumulator[0], A_shared_wmma_matrix_a[0], B_rescale_shared_wmma_matrix_b[0], C_wmma_accumulator[0]);
        }
    }
    __asm__ __volatile__("cp.async.wait_group 0;");

    __syncthreads();
    for (int k_0_1_1 = 0; k_0_1_1 < 4; ++k_0_1_1)
    {
        nvcuda::wmma::load_matrix_sync(A_shared_wmma_matrix_a[0], (&(A_shared[(((((int)threadIdx.y) * 1152) + (k_0_1_1 * 16)) + 2304)])), 72);
        nvcuda::wmma::load_matrix_sync(B_rescale_shared_wmma_matrix_b[0], (&(B_rescale_shared[(((((int)threadIdx.z) * 1152) + (k_0_1_1 * 16)) + 4608)])), 72);
        nvcuda::wmma::mma_sync(C_wmma_accumulator[0], A_shared_wmma_matrix_a[0], B_rescale_shared_wmma_matrix_b[0], C_wmma_accumulator[0]);
    }
    nvcuda::wmma::store_matrix_sync((&(C[((((((int)blockIdx.y) * 294912) + (((int)threadIdx.y) * 147456)) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.z) * 16))])), C_wmma_accumulator[0], 9216, nvcuda::wmma::mem_row_major);
}

__global__ void __launch_bounds__(256) tir_halfxint3_tensorop_64x64x64x2_t0_y2z4_K9216_align8(half *__restrict__ A, signed char *__restrict__ B, half *__restrict__ Scales, half *__restrict__ Zeros, half *__restrict__ C)
{
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, half> C_wmma_accumulator[2];
    __shared__ half A_shared[9216];
    __shared__ half B_rescale_shared[9216];
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::row_major> A_shared_wmma_matrix_a[2];
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::col_major> B_rescale_shared_wmma_matrix_b[1];
    for (int i_0_2_init = 0; i_0_2_init < 2; ++i_0_2_init)
    {
        nvcuda::wmma::fill_fragment(C_wmma_accumulator[i_0_2_init], 0.000000e+00f);
    }
    for (int ax0_ax1_fused_2 = 0; ax0_ax1_fused_2 < 2; ++ax0_ax1_fused_2)
    {

        {
            unsigned int addr;
#if TVM_ENBALE_EFFICIENT_SMEM_PTR_CAST
            addr = static_cast<unsigned int>(__cvta_generic_to_shared((void *)(A_shared + (((((((int)threadIdx.y) * 2304) + (((int)threadIdx.z) * 576)) + (ax0_ax1_fused_2 * 288)) + ((((int)threadIdx.x) >> 3) * 72)) + ((((int)threadIdx.x) & 7) * 8)))));
#else
            __asm__ __volatile__(
                "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }\n"
                : "=r"(addr)
                : "l"((void *)(A_shared + (((((((int)threadIdx.y) * 2304) + (((int)threadIdx.z) * 576)) + (ax0_ax1_fused_2 * 288)) + ((((int)threadIdx.x) >> 3) * 72)) + ((((int)threadIdx.x) & 7) * 8)))));
#endif
            __asm__ __volatile__(
#if TVM_ENABLE_L2_PREFETCH
                "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
#else
                "cp.async.cg.shared.global [%0], [%1], %2;"
#endif
                ::"r"(addr),
                "l"((void *)(A + ((((((((int)blockIdx.y) * 589824) + (((int)threadIdx.y) * 294912)) + (((int)threadIdx.z) * 73728)) + (ax0_ax1_fused_2 * 36864)) + ((((int)threadIdx.x) >> 3) * 9216)) + ((((int)threadIdx.x) & 7) * 8)))), "n"(16));
        }
    }
    for (int ax0_ax1_fused_2_1 = 0; ax0_ax1_fused_2_1 < 16; ++ax0_ax1_fused_2_1)
    {
        B_rescale_shared[(((((((int)threadIdx.y) * 2304) + (((int)threadIdx.z) * 576)) + ((ax0_ax1_fused_2_1 >> 1) * 72)) + ((ax0_ax1_fused_2_1 & 1) * 32)) + ((int)threadIdx.x))] = ((((((((int)threadIdx.x) * 3) & 7) <= 5) ? ((half)((((int)B[((((((((int)blockIdx.x) * 221184) + (((int)threadIdx.y) * 110592)) + (((int)threadIdx.z) * 27648)) + ((ax0_ax1_fused_2_1 >> 1) * 3456)) + ((ax0_ax1_fused_2_1 & 1) * 12)) + ((((int)threadIdx.x) * 3) >> 3))]) >> ((((int)threadIdx.x) * 3) & 7)) & 7)) : ((half)(((signed char)((((int)B[((((((((int)blockIdx.x) * 221184) + (((int)threadIdx.y) * 110592)) + (((int)threadIdx.z) * 27648)) + ((ax0_ax1_fused_2_1 >> 1) * 3456)) + ((ax0_ax1_fused_2_1 & 1) * 12)) + ((((int)threadIdx.x) * 3) >> 3))]) >> ((((int)threadIdx.x) * 3) & 7)) & ((1 << (8 - ((((int)threadIdx.x) * 3) & 7))) - 1))) | ((signed char)(((((int)B[(((((((((int)blockIdx.x) * 221184) + (((int)threadIdx.y) * 110592)) + (((int)threadIdx.z) * 27648)) + ((ax0_ax1_fused_2_1 >> 1) * 3456)) + ((ax0_ax1_fused_2_1 & 1) * 12)) + ((((int)threadIdx.x) * 3) >> 3)) + 1)]) << (8 - ((((int)threadIdx.x) * 3) & 7))) & (7 << (8 - ((((int)threadIdx.x) * 3) & 7)))) & 7))))) * Scales[((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.z) * 8)) + (ax0_ax1_fused_2_1 >> 1))]) - Zeros[((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.z) * 8)) + (ax0_ax1_fused_2_1 >> 1))]);
    }
    __asm__ __volatile__("cp.async.commit_group;");

    for (int k_0_0 = 0; k_0_0 < 143; ++k_0_0)
    {
        __syncthreads();
        for (int ax0_ax1_fused_2_2 = 0; ax0_ax1_fused_2_2 < 2; ++ax0_ax1_fused_2_2)
        {

            {
                unsigned int addr;
#if TVM_ENBALE_EFFICIENT_SMEM_PTR_CAST
                addr = static_cast<unsigned int>(__cvta_generic_to_shared((void *)(A_shared + ((((((((k_0_0 + 1) & 1) * 4608) + (((int)threadIdx.y) * 2304)) + (((int)threadIdx.z) * 576)) + (ax0_ax1_fused_2_2 * 288)) + ((((int)threadIdx.x) >> 3) * 72)) + ((((int)threadIdx.x) & 7) * 8)))));
#else
                __asm__ __volatile__(
                    "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }\n"
                    : "=r"(addr)
                    : "l"((void *)(A_shared + ((((((((k_0_0 + 1) & 1) * 4608) + (((int)threadIdx.y) * 2304)) + (((int)threadIdx.z) * 576)) + (ax0_ax1_fused_2_2 * 288)) + ((((int)threadIdx.x) >> 3) * 72)) + ((((int)threadIdx.x) & 7) * 8)))));
#endif
                __asm__ __volatile__(
#if TVM_ENABLE_L2_PREFETCH
                    "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
#else
                    "cp.async.cg.shared.global [%0], [%1], %2;"
#endif
                    ::"r"(addr),
                    "l"((void *)(A + ((((((((((int)blockIdx.y) * 589824) + (((int)threadIdx.y) * 294912)) + (((int)threadIdx.z) * 73728)) + (ax0_ax1_fused_2_2 * 36864)) + ((((int)threadIdx.x) >> 3) * 9216)) + (k_0_0 * 64)) + ((((int)threadIdx.x) & 7) * 8)) + 64))), "n"(16));
            }
        }
        for (int ax0_ax1_fused_2_3 = 0; ax0_ax1_fused_2_3 < 16; ++ax0_ax1_fused_2_3)
        {
            B_rescale_shared[((((((((k_0_0 + 1) & 1) * 4608) + (((int)threadIdx.y) * 2304)) + (((int)threadIdx.z) * 576)) + ((ax0_ax1_fused_2_3 >> 1) * 72)) + ((ax0_ax1_fused_2_3 & 1) * 32)) + ((int)threadIdx.x))] = ((((((((int)threadIdx.x) * 3) & 7) <= 5) ? ((half)((((int)B[((((((((((int)blockIdx.x) * 221184) + (((int)threadIdx.y) * 110592)) + (((int)threadIdx.z) * 27648)) + ((ax0_ax1_fused_2_3 >> 1) * 3456)) + (k_0_0 * 24)) + ((ax0_ax1_fused_2_3 & 1) * 12)) + ((((int)threadIdx.x) * 3) >> 3)) + 24)]) >> ((((int)threadIdx.x) * 3) & 7)) & 7)) : ((half)(((signed char)((((int)B[((((((((((int)blockIdx.x) * 221184) + (((int)threadIdx.y) * 110592)) + (((int)threadIdx.z) * 27648)) + ((ax0_ax1_fused_2_3 >> 1) * 3456)) + (k_0_0 * 24)) + ((ax0_ax1_fused_2_3 & 1) * 12)) + ((((int)threadIdx.x) * 3) >> 3)) + 24)]) >> ((((int)threadIdx.x) * 3) & 7)) & ((1 << (8 - ((((int)threadIdx.x) * 3) & 7))) - 1))) | ((signed char)(((((int)B[((((((((((int)blockIdx.x) * 221184) + (((int)threadIdx.y) * 110592)) + (((int)threadIdx.z) * 27648)) + ((ax0_ax1_fused_2_3 >> 1) * 3456)) + (k_0_0 * 24)) + ((ax0_ax1_fused_2_3 & 1) * 12)) + ((((int)threadIdx.x) * 3) >> 3)) + 25)]) << (8 - ((((int)threadIdx.x) * 3) & 7))) & (7 << (8 - ((((int)threadIdx.x) * 3) & 7)))) & 7))))) * Scales[((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.z) * 8)) + (ax0_ax1_fused_2_3 >> 1))]) - Zeros[((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.z) * 8)) + (ax0_ax1_fused_2_3 >> 1))]);
        }
        __asm__ __volatile__("cp.async.commit_group;");

        __asm__ __volatile__("cp.async.wait_group 1;");

        __syncthreads();
        for (int k_0_1 = 0; k_0_1 < 4; ++k_0_1)
        {
            for (int ax0_0 = 0; ax0_0 < 2; ++ax0_0)
            {
                nvcuda::wmma::load_matrix_sync(A_shared_wmma_matrix_a[ax0_0], (&(A_shared[(((((k_0_0 & 1) * 4608) + (((int)threadIdx.y) * 2304)) + (ax0_0 * 1152)) + (k_0_1 * 16))])), 72);
            }
            nvcuda::wmma::load_matrix_sync(B_rescale_shared_wmma_matrix_b[0], (&(B_rescale_shared[((((k_0_0 & 1) * 4608) + (((int)threadIdx.z) * 1152)) + (k_0_1 * 16))])), 72);
            for (int i_0_2 = 0; i_0_2 < 2; ++i_0_2)
            {
                nvcuda::wmma::mma_sync(C_wmma_accumulator[i_0_2], A_shared_wmma_matrix_a[i_0_2], B_rescale_shared_wmma_matrix_b[0], C_wmma_accumulator[i_0_2]);
            }
        }
    }
    __asm__ __volatile__("cp.async.wait_group 0;");

    __syncthreads();
    for (int k_0_1_1 = 0; k_0_1_1 < 4; ++k_0_1_1)
    {
        for (int ax0_0_1 = 0; ax0_0_1 < 2; ++ax0_0_1)
        {
            nvcuda::wmma::load_matrix_sync(A_shared_wmma_matrix_a[ax0_0_1], (&(A_shared[((((((int)threadIdx.y) * 2304) + (ax0_0_1 * 1152)) + (k_0_1_1 * 16)) + 4608)])), 72);
        }
        nvcuda::wmma::load_matrix_sync(B_rescale_shared_wmma_matrix_b[0], (&(B_rescale_shared[(((((int)threadIdx.z) * 1152) + (k_0_1_1 * 16)) + 4608)])), 72);
        for (int i_0_2_1 = 0; i_0_2_1 < 2; ++i_0_2_1)
        {
            nvcuda::wmma::mma_sync(C_wmma_accumulator[i_0_2_1], A_shared_wmma_matrix_a[i_0_2_1], B_rescale_shared_wmma_matrix_b[0], C_wmma_accumulator[i_0_2_1]);
        }
    }
    for (int ax0_0_2 = 0; ax0_0_2 < 2; ++ax0_0_2)
    {
        nvcuda::wmma::store_matrix_sync((&(C[(((((((int)blockIdx.y) * 589824) + (((int)threadIdx.y) * 294912)) + (ax0_0_2 * 147456)) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.z) * 16))])), C_wmma_accumulator[ax0_0_2], 9216, nvcuda::wmma::mem_row_major);
    }
}

__global__ void __launch_bounds__(64) tir_halfxint3_tensorop_64x128x32x3_t0_y1z2_K9216_align8(half *__restrict__ A, signed char *__restrict__ B, half *__restrict__ Scales, half *__restrict__ Zeros, half *__restrict__ C)
{
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, half> C_wmma_accumulator[16];
    __shared__ half A_shared[7680];
    __shared__ half B_rescale_shared[15360];
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::row_major> A_shared_wmma_matrix_a[4];
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::col_major> B_rescale_shared_wmma_matrix_b[4];
    for (int i_0_2_init = 0; i_0_2_init < 4; ++i_0_2_init)
    {
        for (int j_0_2_init = 0; j_0_2_init < 4; ++j_0_2_init)
        {
            nvcuda::wmma::fill_fragment(C_wmma_accumulator[((i_0_2_init * 4) + j_0_2_init)], 0.000000e+00f);
        }
    }
    for (int ax0_ax1_fused_2 = 0; ax0_ax1_fused_2 < 4; ++ax0_ax1_fused_2)
    {

        {
            unsigned int addr;
#if TVM_ENBALE_EFFICIENT_SMEM_PTR_CAST
            addr = static_cast<unsigned int>(__cvta_generic_to_shared((void *)(A_shared + ((((((int)threadIdx.z) * 1280) + (ax0_ax1_fused_2 * 320)) + ((((int)threadIdx.x) >> 2) * 40)) + ((((int)threadIdx.x) & 3) * 8)))));
#else
            __asm__ __volatile__(
                "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }\n"
                : "=r"(addr)
                : "l"((void *)(A_shared + ((((((int)threadIdx.z) * 1280) + (ax0_ax1_fused_2 * 320)) + ((((int)threadIdx.x) >> 2) * 40)) + ((((int)threadIdx.x) & 3) * 8)))));
#endif
            __asm__ __volatile__(
#if TVM_ENABLE_L2_PREFETCH
                "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
#else
                "cp.async.cg.shared.global [%0], [%1], %2;"
#endif
                ::"r"(addr),
                "l"((void *)(A + (((((((int)blockIdx.y) * 589824) + (((int)threadIdx.z) * 294912)) + (ax0_ax1_fused_2 * 73728)) + ((((int)threadIdx.x) >> 2) * 9216)) + ((((int)threadIdx.x) & 3) * 8)))), "n"(16));
        }
    }
    for (int ax0_ax1_fused_2_1 = 0; ax0_ax1_fused_2_1 < 64; ++ax0_ax1_fused_2_1)
    {
        B_rescale_shared[(((((int)threadIdx.z) * 2560) + (ax0_ax1_fused_2_1 * 40)) + ((int)threadIdx.x))] = ((((((((int)threadIdx.x) * 3) & 7) <= 5) ? ((half)((((int)B[((((((int)blockIdx.x) * 442368) + (((int)threadIdx.z) * 221184)) + (ax0_ax1_fused_2_1 * 3456)) + ((((int)threadIdx.x) * 3) >> 3))]) >> ((((int)threadIdx.x) * 3) & 7)) & 7)) : ((half)(((signed char)((((int)B[((((((int)blockIdx.x) * 442368) + (((int)threadIdx.z) * 221184)) + (ax0_ax1_fused_2_1 * 3456)) + ((((int)threadIdx.x) * 3) >> 3))]) >> ((((int)threadIdx.x) * 3) & 7)) & ((1 << (8 - ((((int)threadIdx.x) * 3) & 7))) - 1))) | ((signed char)(((((int)B[(((((((int)blockIdx.x) * 442368) + (((int)threadIdx.z) * 221184)) + (ax0_ax1_fused_2_1 * 3456)) + ((((int)threadIdx.x) * 3) >> 3)) + 1)]) << (8 - ((((int)threadIdx.x) * 3) & 7))) & (7 << (8 - ((((int)threadIdx.x) * 3) & 7)))) & 7))))) * Scales[(((((int)blockIdx.x) * 128) + (((int)threadIdx.z) * 64)) + ax0_ax1_fused_2_1)]) - Zeros[(((((int)blockIdx.x) * 128) + (((int)threadIdx.z) * 64)) + ax0_ax1_fused_2_1)]);
    }
    __asm__ __volatile__("cp.async.commit_group;");

    for (int ax0_ax1_fused_2_2 = 0; ax0_ax1_fused_2_2 < 4; ++ax0_ax1_fused_2_2)
    {

        {
            unsigned int addr;
#if TVM_ENBALE_EFFICIENT_SMEM_PTR_CAST
            addr = static_cast<unsigned int>(__cvta_generic_to_shared((void *)(A_shared + (((((((int)threadIdx.z) * 1280) + (ax0_ax1_fused_2_2 * 320)) + ((((int)threadIdx.x) >> 2) * 40)) + ((((int)threadIdx.x) & 3) * 8)) + 2560))));
#else
            __asm__ __volatile__(
                "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }\n"
                : "=r"(addr)
                : "l"((void *)(A_shared + (((((((int)threadIdx.z) * 1280) + (ax0_ax1_fused_2_2 * 320)) + ((((int)threadIdx.x) >> 2) * 40)) + ((((int)threadIdx.x) & 3) * 8)) + 2560))));
#endif
            __asm__ __volatile__(
#if TVM_ENABLE_L2_PREFETCH
                "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
#else
                "cp.async.cg.shared.global [%0], [%1], %2;"
#endif
                ::"r"(addr),
                "l"((void *)(A + ((((((((int)blockIdx.y) * 589824) + (((int)threadIdx.z) * 294912)) + (ax0_ax1_fused_2_2 * 73728)) + ((((int)threadIdx.x) >> 2) * 9216)) + ((((int)threadIdx.x) & 3) * 8)) + 32))), "n"(16));
        }
    }
    for (int ax0_ax1_fused_2_3 = 0; ax0_ax1_fused_2_3 < 64; ++ax0_ax1_fused_2_3)
    {
        B_rescale_shared[((((((int)threadIdx.z) * 2560) + (ax0_ax1_fused_2_3 * 40)) + ((int)threadIdx.x)) + 5120)] = ((((((((int)threadIdx.x) * 3) & 7) <= 5) ? ((half)((((int)B[(((((((int)blockIdx.x) * 442368) + (((int)threadIdx.z) * 221184)) + (ax0_ax1_fused_2_3 * 3456)) + ((((int)threadIdx.x) * 3) >> 3)) + 12)]) >> ((((int)threadIdx.x) * 3) & 7)) & 7)) : ((half)(((signed char)((((int)B[(((((((int)blockIdx.x) * 442368) + (((int)threadIdx.z) * 221184)) + (ax0_ax1_fused_2_3 * 3456)) + ((((int)threadIdx.x) * 3) >> 3)) + 12)]) >> ((((int)threadIdx.x) * 3) & 7)) & ((1 << (8 - ((((int)threadIdx.x) * 3) & 7))) - 1))) | ((signed char)(((((int)B[(((((((int)blockIdx.x) * 442368) + (((int)threadIdx.z) * 221184)) + (ax0_ax1_fused_2_3 * 3456)) + ((((int)threadIdx.x) * 3) >> 3)) + 13)]) << (8 - ((((int)threadIdx.x) * 3) & 7))) & (7 << (8 - ((((int)threadIdx.x) * 3) & 7)))) & 7))))) * Scales[(((((int)blockIdx.x) * 128) + (((int)threadIdx.z) * 64)) + ax0_ax1_fused_2_3)]) - Zeros[(((((int)blockIdx.x) * 128) + (((int)threadIdx.z) * 64)) + ax0_ax1_fused_2_3)]);
    }
    __asm__ __volatile__("cp.async.commit_group;");

    for (int k_0_0 = 0; k_0_0 < 286; ++k_0_0)
    {
        __syncthreads();
        for (int ax0_ax1_fused_2_4 = 0; ax0_ax1_fused_2_4 < 4; ++ax0_ax1_fused_2_4)
        {

            {
                unsigned int addr;
#if TVM_ENBALE_EFFICIENT_SMEM_PTR_CAST
                addr = static_cast<unsigned int>(__cvta_generic_to_shared((void *)(A_shared + (((((((k_0_0 + 2) % 3) * 2560) + (((int)threadIdx.z) * 1280)) + (ax0_ax1_fused_2_4 * 320)) + ((((int)threadIdx.x) >> 2) * 40)) + ((((int)threadIdx.x) & 3) * 8)))));
#else
                __asm__ __volatile__(
                    "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }\n"
                    : "=r"(addr)
                    : "l"((void *)(A_shared + (((((((k_0_0 + 2) % 3) * 2560) + (((int)threadIdx.z) * 1280)) + (ax0_ax1_fused_2_4 * 320)) + ((((int)threadIdx.x) >> 2) * 40)) + ((((int)threadIdx.x) & 3) * 8)))));
#endif
                __asm__ __volatile__(
#if TVM_ENABLE_L2_PREFETCH
                    "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
#else
                    "cp.async.cg.shared.global [%0], [%1], %2;"
#endif
                    ::"r"(addr),
                    "l"((void *)(A + (((((((((int)blockIdx.y) * 589824) + (((int)threadIdx.z) * 294912)) + (ax0_ax1_fused_2_4 * 73728)) + ((((int)threadIdx.x) >> 2) * 9216)) + (k_0_0 * 32)) + ((((int)threadIdx.x) & 3) * 8)) + 64))), "n"(16));
            }
        }
        for (int ax0_ax1_fused_2_5 = 0; ax0_ax1_fused_2_5 < 64; ++ax0_ax1_fused_2_5)
        {
            B_rescale_shared[((((((k_0_0 + 2) % 3) * 5120) + (((int)threadIdx.z) * 2560)) + (ax0_ax1_fused_2_5 * 40)) + ((int)threadIdx.x))] = ((((((((int)threadIdx.x) * 3) & 7) <= 5) ? ((half)((((int)B[((((((((int)blockIdx.x) * 442368) + (((int)threadIdx.z) * 221184)) + (ax0_ax1_fused_2_5 * 3456)) + (k_0_0 * 12)) + ((((int)threadIdx.x) * 3) >> 3)) + 24)]) >> ((((int)threadIdx.x) * 3) & 7)) & 7)) : ((half)(((signed char)((((int)B[((((((((int)blockIdx.x) * 442368) + (((int)threadIdx.z) * 221184)) + (ax0_ax1_fused_2_5 * 3456)) + (k_0_0 * 12)) + ((((int)threadIdx.x) * 3) >> 3)) + 24)]) >> ((((int)threadIdx.x) * 3) & 7)) & ((1 << (8 - ((((int)threadIdx.x) * 3) & 7))) - 1))) | ((signed char)(((((int)B[((((((((int)blockIdx.x) * 442368) + (((int)threadIdx.z) * 221184)) + (ax0_ax1_fused_2_5 * 3456)) + (k_0_0 * 12)) + ((((int)threadIdx.x) * 3) >> 3)) + 25)]) << (8 - ((((int)threadIdx.x) * 3) & 7))) & (7 << (8 - ((((int)threadIdx.x) * 3) & 7)))) & 7))))) * Scales[(((((int)blockIdx.x) * 128) + (((int)threadIdx.z) * 64)) + ax0_ax1_fused_2_5)]) - Zeros[(((((int)blockIdx.x) * 128) + (((int)threadIdx.z) * 64)) + ax0_ax1_fused_2_5)]);
        }
        __asm__ __volatile__("cp.async.commit_group;");

        __asm__ __volatile__("cp.async.wait_group 2;");

        __syncthreads();
        for (int k_0_1 = 0; k_0_1 < 2; ++k_0_1)
        {
            for (int ax0_0 = 0; ax0_0 < 4; ++ax0_0)
            {
                nvcuda::wmma::load_matrix_sync(A_shared_wmma_matrix_a[ax0_0], (&(A_shared[((((k_0_0 % 3) * 2560) + (ax0_0 * 640)) + (k_0_1 * 16))])), 40);
            }
            for (int ax0_0_1 = 0; ax0_0_1 < 4; ++ax0_0_1)
            {
                nvcuda::wmma::load_matrix_sync(B_rescale_shared_wmma_matrix_b[ax0_0_1], (&(B_rescale_shared[(((((k_0_0 % 3) * 5120) + (((int)threadIdx.z) * 2560)) + (ax0_0_1 * 640)) + (k_0_1 * 16))])), 40);
            }
            for (int i_0_2 = 0; i_0_2 < 4; ++i_0_2)
            {
                for (int j_0_2 = 0; j_0_2 < 4; ++j_0_2)
                {
                    nvcuda::wmma::mma_sync(C_wmma_accumulator[((i_0_2 * 4) + j_0_2)], A_shared_wmma_matrix_a[i_0_2], B_rescale_shared_wmma_matrix_b[j_0_2], C_wmma_accumulator[((i_0_2 * 4) + j_0_2)]);
                }
            }
        }
    }
    __asm__ __volatile__("cp.async.wait_group 1;");

    __syncthreads();
    for (int k_0_1_1 = 0; k_0_1_1 < 2; ++k_0_1_1)
    {
        for (int ax0_0_2 = 0; ax0_0_2 < 4; ++ax0_0_2)
        {
            nvcuda::wmma::load_matrix_sync(A_shared_wmma_matrix_a[ax0_0_2], (&(A_shared[(((ax0_0_2 * 640) + (k_0_1_1 * 16)) + 2560)])), 40);
        }
        for (int ax0_0_3 = 0; ax0_0_3 < 4; ++ax0_0_3)
        {
            nvcuda::wmma::load_matrix_sync(B_rescale_shared_wmma_matrix_b[ax0_0_3], (&(B_rescale_shared[((((((int)threadIdx.z) * 2560) + (ax0_0_3 * 640)) + (k_0_1_1 * 16)) + 5120)])), 40);
        }
        for (int i_0_2_1 = 0; i_0_2_1 < 4; ++i_0_2_1)
        {
            for (int j_0_2_1 = 0; j_0_2_1 < 4; ++j_0_2_1)
            {
                nvcuda::wmma::mma_sync(C_wmma_accumulator[((i_0_2_1 * 4) + j_0_2_1)], A_shared_wmma_matrix_a[i_0_2_1], B_rescale_shared_wmma_matrix_b[j_0_2_1], C_wmma_accumulator[((i_0_2_1 * 4) + j_0_2_1)]);
            }
        }
    }
    __asm__ __volatile__("cp.async.wait_group 0;");

    __syncthreads();
    for (int k_0_1_2 = 0; k_0_1_2 < 2; ++k_0_1_2)
    {
        for (int ax0_0_4 = 0; ax0_0_4 < 4; ++ax0_0_4)
        {
            nvcuda::wmma::load_matrix_sync(A_shared_wmma_matrix_a[ax0_0_4], (&(A_shared[(((ax0_0_4 * 640) + (k_0_1_2 * 16)) + 5120)])), 40);
        }
        for (int ax0_0_5 = 0; ax0_0_5 < 4; ++ax0_0_5)
        {
            nvcuda::wmma::load_matrix_sync(B_rescale_shared_wmma_matrix_b[ax0_0_5], (&(B_rescale_shared[((((((int)threadIdx.z) * 2560) + (ax0_0_5 * 640)) + (k_0_1_2 * 16)) + 10240)])), 40);
        }
        for (int i_0_2_2 = 0; i_0_2_2 < 4; ++i_0_2_2)
        {
            for (int j_0_2_2 = 0; j_0_2_2 < 4; ++j_0_2_2)
            {
                nvcuda::wmma::mma_sync(C_wmma_accumulator[((i_0_2_2 * 4) + j_0_2_2)], A_shared_wmma_matrix_a[i_0_2_2], B_rescale_shared_wmma_matrix_b[j_0_2_2], C_wmma_accumulator[((i_0_2_2 * 4) + j_0_2_2)]);
            }
        }
    }
    for (int ax0_0_6 = 0; ax0_0_6 < 4; ++ax0_0_6)
    {
        for (int ax1_0 = 0; ax1_0 < 4; ++ax1_0)
        {
            nvcuda::wmma::store_matrix_sync((&(C[(((((((int)blockIdx.y) * 589824) + (ax0_0_6 * 147456)) + (((int)blockIdx.x) * 128)) + (((int)threadIdx.z) * 64)) + (ax1_0 * 16))])), C_wmma_accumulator[((ax0_0_6 * 4) + ax1_0)], 9216, nvcuda::wmma::mem_row_major);
        }
    }
}

__global__ void __launch_bounds__(128) tir_halfxint3_tensorop_128x256x32x1_t0_y2z2_K9216_align8(half *__restrict__ A, signed char *__restrict__ B, half *__restrict__ Scales, half *__restrict__ Zeros, half *__restrict__ C)
{
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, half> C_wmma_accumulator[32];
    __shared__ half A_shared[5120];
    __shared__ half B_rescale_shared[10240];
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::row_major> A_shared_wmma_matrix_a[4];
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::col_major> B_rescale_shared_wmma_matrix_b[8];
    for (int i_0_2_init = 0; i_0_2_init < 4; ++i_0_2_init)
    {
        for (int j_0_2_init = 0; j_0_2_init < 8; ++j_0_2_init)
        {
            nvcuda::wmma::fill_fragment(C_wmma_accumulator[((i_0_2_init * 8) + j_0_2_init)], 0.000000e+00f);
        }
    }
    for (int k_0_0 = 0; k_0_0 < 288; ++k_0_0)
    {
        __syncthreads();
        for (int ax0_ax1_fused_2 = 0; ax0_ax1_fused_2 < 4; ++ax0_ax1_fused_2)
        {
            *(uint4 *)(A_shared + (((((((int)threadIdx.y) * 2560) + (((int)threadIdx.z) * 1280)) + (ax0_ax1_fused_2 * 320)) + ((((int)threadIdx.x) >> 2) * 40)) + ((((int)threadIdx.x) & 3) * 8))) = *(uint4 *)(A + (((((((((int)blockIdx.y) * 1179648) + (((int)threadIdx.y) * 589824)) + (((int)threadIdx.z) * 294912)) + (ax0_ax1_fused_2 * 73728)) + ((((int)threadIdx.x) >> 2) * 9216)) + (k_0_0 * 32)) + ((((int)threadIdx.x) & 3) * 8)));
        }
        for (int ax0_ax1_fused_2_1 = 0; ax0_ax1_fused_2_1 < 64; ++ax0_ax1_fused_2_1)
        {
            B_rescale_shared[((((((int)threadIdx.y) * 5120) + (((int)threadIdx.z) * 2560)) + (ax0_ax1_fused_2_1 * 40)) + ((int)threadIdx.x))] = ((((((((int)threadIdx.x) * 3) & 7) <= 5) ? ((half)((((int)B[((((((((int)blockIdx.x) * 884736) + (((int)threadIdx.y) * 442368)) + (((int)threadIdx.z) * 221184)) + (ax0_ax1_fused_2_1 * 3456)) + (k_0_0 * 12)) + ((((int)threadIdx.x) * 3) >> 3))]) >> ((((int)threadIdx.x) * 3) & 7)) & 7)) : ((half)(((signed char)((((int)B[((((((((int)blockIdx.x) * 884736) + (((int)threadIdx.y) * 442368)) + (((int)threadIdx.z) * 221184)) + (ax0_ax1_fused_2_1 * 3456)) + (k_0_0 * 12)) + ((((int)threadIdx.x) * 3) >> 3))]) >> ((((int)threadIdx.x) * 3) & 7)) & ((1 << (8 - ((((int)threadIdx.x) * 3) & 7))) - 1))) | ((signed char)(((((int)B[(((((((((int)blockIdx.x) * 884736) + (((int)threadIdx.y) * 442368)) + (((int)threadIdx.z) * 221184)) + (ax0_ax1_fused_2_1 * 3456)) + (k_0_0 * 12)) + ((((int)threadIdx.x) * 3) >> 3)) + 1)]) << (8 - ((((int)threadIdx.x) * 3) & 7))) & (7 << (8 - ((((int)threadIdx.x) * 3) & 7)))) & 7))))) * Scales[((((((int)blockIdx.x) * 256) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.z) * 64)) + ax0_ax1_fused_2_1)]) - Zeros[((((((int)blockIdx.x) * 256) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.z) * 64)) + ax0_ax1_fused_2_1)]);
        }
        __syncthreads();
        for (int k_0_1 = 0; k_0_1 < 2; ++k_0_1)
        {
            for (int ax0_0 = 0; ax0_0 < 4; ++ax0_0)
            {
                nvcuda::wmma::load_matrix_sync(A_shared_wmma_matrix_a[ax0_0], (&(A_shared[(((((int)threadIdx.y) * 2560) + (ax0_0 * 640)) + (k_0_1 * 16))])), 40);
            }
            for (int ax0_0_1 = 0; ax0_0_1 < 8; ++ax0_0_1)
            {
                nvcuda::wmma::load_matrix_sync(B_rescale_shared_wmma_matrix_b[ax0_0_1], (&(B_rescale_shared[(((((int)threadIdx.z) * 5120) + (ax0_0_1 * 640)) + (k_0_1 * 16))])), 40);
            }
            for (int i_0_2 = 0; i_0_2 < 4; ++i_0_2)
            {
                for (int j_0_2 = 0; j_0_2 < 8; ++j_0_2)
                {
                    nvcuda::wmma::mma_sync(C_wmma_accumulator[((i_0_2 * 8) + j_0_2)], A_shared_wmma_matrix_a[i_0_2], B_rescale_shared_wmma_matrix_b[j_0_2], C_wmma_accumulator[((i_0_2 * 8) + j_0_2)]);
                }
            }
        }
    }
    for (int ax0_0_2 = 0; ax0_0_2 < 4; ++ax0_0_2)
    {
        for (int ax1_0 = 0; ax1_0 < 8; ++ax1_0)
        {
            nvcuda::wmma::store_matrix_sync((&(C[((((((((int)blockIdx.y) * 1179648) + (((int)threadIdx.y) * 589824)) + (ax0_0_2 * 147456)) + (((int)blockIdx.x) * 256)) + (((int)threadIdx.z) * 128)) + (ax1_0 * 16))])), C_wmma_accumulator[((ax0_0_2 * 8) + ax1_0)], 9216, nvcuda::wmma::mem_row_major);
        }
    }
}

__global__ void __launch_bounds__(128) tir_halfxint3_tensorop_16x64x64x3_t0_y1z4_K36864_align8(half *__restrict__ A, signed char *__restrict__ B, half *__restrict__ Scales, half *__restrict__ Zeros, half *__restrict__ C)
{
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, half> C_wmma_accumulator[1];
    __shared__ half A_shared[3456];
    __shared__ half B_rescale_shared[13824];
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::row_major> A_shared_wmma_matrix_a[1];
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::col_major> B_rescale_shared_wmma_matrix_b[1];
    nvcuda::wmma::fill_fragment(C_wmma_accumulator[0], 0.000000e+00f);

    {
        unsigned int addr;
#if TVM_ENBALE_EFFICIENT_SMEM_PTR_CAST
        addr = static_cast<unsigned int>(__cvta_generic_to_shared((void *)(A_shared + (((((int)threadIdx.z) * 288) + ((((int)threadIdx.x) >> 3) * 72)) + ((((int)threadIdx.x) & 7) * 8)))));
#else
        __asm__ __volatile__(
            "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }\n"
            : "=r"(addr)
            : "l"((void *)(A_shared + (((((int)threadIdx.z) * 288) + ((((int)threadIdx.x) >> 3) * 72)) + ((((int)threadIdx.x) & 7) * 8)))));
#endif
        __asm__ __volatile__(
#if TVM_ENABLE_L2_PREFETCH
            "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
#else
            "cp.async.cg.shared.global [%0], [%1], %2;"
#endif
            ::"r"(addr),
            "l"((void *)(A + ((((((int)blockIdx.y) * 589824) + (((int)threadIdx.z) * 147456)) + ((((int)threadIdx.x) >> 3) * 36864)) + ((((int)threadIdx.x) & 7) * 8)))), "n"(16));
    }
    for (int ax0_ax1_fused_2 = 0; ax0_ax1_fused_2 < 32; ++ax0_ax1_fused_2)
    {
        B_rescale_shared[((((((int)threadIdx.z) * 1152) + ((ax0_ax1_fused_2 >> 1) * 72)) + ((ax0_ax1_fused_2 & 1) * 32)) + ((int)threadIdx.x))] = ((((((((int)threadIdx.x) * 3) & 7) <= 5) ? ((half)((((int)B[(((((((int)blockIdx.x) * 884736) + (((int)threadIdx.z) * 221184)) + ((ax0_ax1_fused_2 >> 1) * 13824)) + ((ax0_ax1_fused_2 & 1) * 12)) + ((((int)threadIdx.x) * 3) >> 3))]) >> ((((int)threadIdx.x) * 3) & 7)) & 7)) : ((half)(((signed char)((((int)B[(((((((int)blockIdx.x) * 884736) + (((int)threadIdx.z) * 221184)) + ((ax0_ax1_fused_2 >> 1) * 13824)) + ((ax0_ax1_fused_2 & 1) * 12)) + ((((int)threadIdx.x) * 3) >> 3))]) >> ((((int)threadIdx.x) * 3) & 7)) & ((1 << (8 - ((((int)threadIdx.x) * 3) & 7))) - 1))) | ((signed char)(((((int)B[((((((((int)blockIdx.x) * 884736) + (((int)threadIdx.z) * 221184)) + ((ax0_ax1_fused_2 >> 1) * 13824)) + ((ax0_ax1_fused_2 & 1) * 12)) + ((((int)threadIdx.x) * 3) >> 3)) + 1)]) << (8 - ((((int)threadIdx.x) * 3) & 7))) & (7 << (8 - ((((int)threadIdx.x) * 3) & 7)))) & 7))))) * Scales[(((((int)blockIdx.x) * 64) + (((int)threadIdx.z) * 16)) + (ax0_ax1_fused_2 >> 1))]) - Zeros[(((((int)blockIdx.x) * 64) + (((int)threadIdx.z) * 16)) + (ax0_ax1_fused_2 >> 1))]);
    }
    __asm__ __volatile__("cp.async.commit_group;");

    {
        unsigned int addr;
#if TVM_ENBALE_EFFICIENT_SMEM_PTR_CAST
        addr = static_cast<unsigned int>(__cvta_generic_to_shared((void *)(A_shared + ((((((int)threadIdx.z) * 288) + ((((int)threadIdx.x) >> 3) * 72)) + ((((int)threadIdx.x) & 7) * 8)) + 1152))));
#else
        __asm__ __volatile__(
            "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }\n"
            : "=r"(addr)
            : "l"((void *)(A_shared + ((((((int)threadIdx.z) * 288) + ((((int)threadIdx.x) >> 3) * 72)) + ((((int)threadIdx.x) & 7) * 8)) + 1152))));
#endif
        __asm__ __volatile__(
#if TVM_ENABLE_L2_PREFETCH
            "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
#else
            "cp.async.cg.shared.global [%0], [%1], %2;"
#endif
            ::"r"(addr),
            "l"((void *)(A + (((((((int)blockIdx.y) * 589824) + (((int)threadIdx.z) * 147456)) + ((((int)threadIdx.x) >> 3) * 36864)) + ((((int)threadIdx.x) & 7) * 8)) + 64))), "n"(16));
    }
    for (int ax0_ax1_fused_2_1 = 0; ax0_ax1_fused_2_1 < 32; ++ax0_ax1_fused_2_1)
    {
        B_rescale_shared[(((((((int)threadIdx.z) * 1152) + ((ax0_ax1_fused_2_1 >> 1) * 72)) + ((ax0_ax1_fused_2_1 & 1) * 32)) + ((int)threadIdx.x)) + 4608)] = ((((((((int)threadIdx.x) * 3) & 7) <= 5) ? ((half)((((int)B[((((((((int)blockIdx.x) * 884736) + (((int)threadIdx.z) * 221184)) + ((ax0_ax1_fused_2_1 >> 1) * 13824)) + ((ax0_ax1_fused_2_1 & 1) * 12)) + ((((int)threadIdx.x) * 3) >> 3)) + 24)]) >> ((((int)threadIdx.x) * 3) & 7)) & 7)) : ((half)(((signed char)((((int)B[((((((((int)blockIdx.x) * 884736) + (((int)threadIdx.z) * 221184)) + ((ax0_ax1_fused_2_1 >> 1) * 13824)) + ((ax0_ax1_fused_2_1 & 1) * 12)) + ((((int)threadIdx.x) * 3) >> 3)) + 24)]) >> ((((int)threadIdx.x) * 3) & 7)) & ((1 << (8 - ((((int)threadIdx.x) * 3) & 7))) - 1))) | ((signed char)(((((int)B[((((((((int)blockIdx.x) * 884736) + (((int)threadIdx.z) * 221184)) + ((ax0_ax1_fused_2_1 >> 1) * 13824)) + ((ax0_ax1_fused_2_1 & 1) * 12)) + ((((int)threadIdx.x) * 3) >> 3)) + 25)]) << (8 - ((((int)threadIdx.x) * 3) & 7))) & (7 << (8 - ((((int)threadIdx.x) * 3) & 7)))) & 7))))) * Scales[(((((int)blockIdx.x) * 64) + (((int)threadIdx.z) * 16)) + (ax0_ax1_fused_2_1 >> 1))]) - Zeros[(((((int)blockIdx.x) * 64) + (((int)threadIdx.z) * 16)) + (ax0_ax1_fused_2_1 >> 1))]);
    }
    __asm__ __volatile__("cp.async.commit_group;");

    for (int k_0_0 = 0; k_0_0 < 574; ++k_0_0)
    {
        __syncthreads();

        {
            unsigned int addr;
#if TVM_ENBALE_EFFICIENT_SMEM_PTR_CAST
            addr = static_cast<unsigned int>(__cvta_generic_to_shared((void *)(A_shared + ((((((k_0_0 + 2) % 3) * 1152) + (((int)threadIdx.z) * 288)) + ((((int)threadIdx.x) >> 3) * 72)) + ((((int)threadIdx.x) & 7) * 8)))));
#else
            __asm__ __volatile__(
                "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }\n"
                : "=r"(addr)
                : "l"((void *)(A_shared + ((((((k_0_0 + 2) % 3) * 1152) + (((int)threadIdx.z) * 288)) + ((((int)threadIdx.x) >> 3) * 72)) + ((((int)threadIdx.x) & 7) * 8)))));
#endif
            __asm__ __volatile__(
#if TVM_ENABLE_L2_PREFETCH
                "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
#else
                "cp.async.cg.shared.global [%0], [%1], %2;"
#endif
                ::"r"(addr),
                "l"((void *)(A + ((((((((int)blockIdx.y) * 589824) + (((int)threadIdx.z) * 147456)) + ((((int)threadIdx.x) >> 3) * 36864)) + (k_0_0 * 64)) + ((((int)threadIdx.x) & 7) * 8)) + 128))), "n"(16));
        }
        for (int ax0_ax1_fused_2_2 = 0; ax0_ax1_fused_2_2 < 32; ++ax0_ax1_fused_2_2)
        {
            B_rescale_shared[(((((((k_0_0 + 2) % 3) * 4608) + (((int)threadIdx.z) * 1152)) + ((ax0_ax1_fused_2_2 >> 1) * 72)) + ((ax0_ax1_fused_2_2 & 1) * 32)) + ((int)threadIdx.x))] = ((((((((int)threadIdx.x) * 3) & 7) <= 5) ? ((half)((((int)B[(((((((((int)blockIdx.x) * 884736) + (((int)threadIdx.z) * 221184)) + ((ax0_ax1_fused_2_2 >> 1) * 13824)) + (k_0_0 * 24)) + ((ax0_ax1_fused_2_2 & 1) * 12)) + ((((int)threadIdx.x) * 3) >> 3)) + 48)]) >> ((((int)threadIdx.x) * 3) & 7)) & 7)) : ((half)(((signed char)((((int)B[(((((((((int)blockIdx.x) * 884736) + (((int)threadIdx.z) * 221184)) + ((ax0_ax1_fused_2_2 >> 1) * 13824)) + (k_0_0 * 24)) + ((ax0_ax1_fused_2_2 & 1) * 12)) + ((((int)threadIdx.x) * 3) >> 3)) + 48)]) >> ((((int)threadIdx.x) * 3) & 7)) & ((1 << (8 - ((((int)threadIdx.x) * 3) & 7))) - 1))) | ((signed char)(((((int)B[(((((((((int)blockIdx.x) * 884736) + (((int)threadIdx.z) * 221184)) + ((ax0_ax1_fused_2_2 >> 1) * 13824)) + (k_0_0 * 24)) + ((ax0_ax1_fused_2_2 & 1) * 12)) + ((((int)threadIdx.x) * 3) >> 3)) + 49)]) << (8 - ((((int)threadIdx.x) * 3) & 7))) & (7 << (8 - ((((int)threadIdx.x) * 3) & 7)))) & 7))))) * Scales[(((((int)blockIdx.x) * 64) + (((int)threadIdx.z) * 16)) + (ax0_ax1_fused_2_2 >> 1))]) - Zeros[(((((int)blockIdx.x) * 64) + (((int)threadIdx.z) * 16)) + (ax0_ax1_fused_2_2 >> 1))]);
        }
        __asm__ __volatile__("cp.async.commit_group;");

        __asm__ __volatile__("cp.async.wait_group 2;");

        __syncthreads();
        for (int k_0_1 = 0; k_0_1 < 4; ++k_0_1)
        {
            nvcuda::wmma::load_matrix_sync(A_shared_wmma_matrix_a[0], (&(A_shared[(((k_0_0 % 3) * 1152) + (k_0_1 * 16))])), 72);
            nvcuda::wmma::load_matrix_sync(B_rescale_shared_wmma_matrix_b[0], (&(B_rescale_shared[((((k_0_0 % 3) * 4608) + (((int)threadIdx.z) * 1152)) + (k_0_1 * 16))])), 72);
            nvcuda::wmma::mma_sync(C_wmma_accumulator[0], A_shared_wmma_matrix_a[0], B_rescale_shared_wmma_matrix_b[0], C_wmma_accumulator[0]);
        }
    }
    __asm__ __volatile__("cp.async.wait_group 1;");

    __syncthreads();
    for (int k_0_1_1 = 0; k_0_1_1 < 4; ++k_0_1_1)
    {
        nvcuda::wmma::load_matrix_sync(A_shared_wmma_matrix_a[0], (&(A_shared[((k_0_1_1 * 16) + 1152)])), 72);
        nvcuda::wmma::load_matrix_sync(B_rescale_shared_wmma_matrix_b[0], (&(B_rescale_shared[(((((int)threadIdx.z) * 1152) + (k_0_1_1 * 16)) + 4608)])), 72);
        nvcuda::wmma::mma_sync(C_wmma_accumulator[0], A_shared_wmma_matrix_a[0], B_rescale_shared_wmma_matrix_b[0], C_wmma_accumulator[0]);
    }
    __asm__ __volatile__("cp.async.wait_group 0;");

    __syncthreads();
    for (int k_0_1_2 = 0; k_0_1_2 < 4; ++k_0_1_2)
    {
        nvcuda::wmma::load_matrix_sync(A_shared_wmma_matrix_a[0], (&(A_shared[((k_0_1_2 * 16) + 2304)])), 72);
        nvcuda::wmma::load_matrix_sync(B_rescale_shared_wmma_matrix_b[0], (&(B_rescale_shared[(((((int)threadIdx.z) * 1152) + (k_0_1_2 * 16)) + 9216)])), 72);
        nvcuda::wmma::mma_sync(C_wmma_accumulator[0], A_shared_wmma_matrix_a[0], B_rescale_shared_wmma_matrix_b[0], C_wmma_accumulator[0]);
    }
    nvcuda::wmma::store_matrix_sync((&(C[(((((int)blockIdx.y) * 147456) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.z) * 16))])), C_wmma_accumulator[0], 9216, nvcuda::wmma::mem_row_major);
}

__global__ void __launch_bounds__(256) tir_halfxint3_tensorop_32x64x64x2_t0_y2z4_K36864_align8(half *__restrict__ A, signed char *__restrict__ B, half *__restrict__ Scales, half *__restrict__ Zeros, half *__restrict__ C)
{
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, half> C_wmma_accumulator[1];
    __shared__ half A_shared[4608];
    __shared__ half B_rescale_shared[9216];
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::row_major> A_shared_wmma_matrix_a[1];
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::col_major> B_rescale_shared_wmma_matrix_b[1];
    nvcuda::wmma::fill_fragment(C_wmma_accumulator[0], 0.000000e+00f);

    {
        unsigned int addr;
#if TVM_ENBALE_EFFICIENT_SMEM_PTR_CAST
        addr = static_cast<unsigned int>(__cvta_generic_to_shared((void *)(A_shared + ((((((int)threadIdx.y) * 1152) + (((int)threadIdx.z) * 288)) + ((((int)threadIdx.x) >> 3) * 72)) + ((((int)threadIdx.x) & 7) * 8)))));
#else
        __asm__ __volatile__(
            "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }\n"
            : "=r"(addr)
            : "l"((void *)(A_shared + ((((((int)threadIdx.y) * 1152) + (((int)threadIdx.z) * 288)) + ((((int)threadIdx.x) >> 3) * 72)) + ((((int)threadIdx.x) & 7) * 8)))));
#endif
        __asm__ __volatile__(
#if TVM_ENABLE_L2_PREFETCH
            "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
#else
            "cp.async.cg.shared.global [%0], [%1], %2;"
#endif
            ::"r"(addr),
            "l"((void *)(A + (((((((int)blockIdx.y) * 1179648) + (((int)threadIdx.y) * 589824)) + (((int)threadIdx.z) * 147456)) + ((((int)threadIdx.x) >> 3) * 36864)) + ((((int)threadIdx.x) & 7) * 8)))), "n"(16));
    }
    for (int ax0_ax1_fused_2 = 0; ax0_ax1_fused_2 < 16; ++ax0_ax1_fused_2)
    {
        B_rescale_shared[(((((((int)threadIdx.y) * 2304) + (((int)threadIdx.z) * 576)) + ((ax0_ax1_fused_2 >> 1) * 72)) + ((ax0_ax1_fused_2 & 1) * 32)) + ((int)threadIdx.x))] = ((((((((int)threadIdx.x) * 3) & 7) <= 5) ? ((half)((((int)B[((((((((int)blockIdx.x) * 884736) + (((int)threadIdx.y) * 442368)) + (((int)threadIdx.z) * 110592)) + ((ax0_ax1_fused_2 >> 1) * 13824)) + ((ax0_ax1_fused_2 & 1) * 12)) + ((((int)threadIdx.x) * 3) >> 3))]) >> ((((int)threadIdx.x) * 3) & 7)) & 7)) : ((half)(((signed char)((((int)B[((((((((int)blockIdx.x) * 884736) + (((int)threadIdx.y) * 442368)) + (((int)threadIdx.z) * 110592)) + ((ax0_ax1_fused_2 >> 1) * 13824)) + ((ax0_ax1_fused_2 & 1) * 12)) + ((((int)threadIdx.x) * 3) >> 3))]) >> ((((int)threadIdx.x) * 3) & 7)) & ((1 << (8 - ((((int)threadIdx.x) * 3) & 7))) - 1))) | ((signed char)(((((int)B[(((((((((int)blockIdx.x) * 884736) + (((int)threadIdx.y) * 442368)) + (((int)threadIdx.z) * 110592)) + ((ax0_ax1_fused_2 >> 1) * 13824)) + ((ax0_ax1_fused_2 & 1) * 12)) + ((((int)threadIdx.x) * 3) >> 3)) + 1)]) << (8 - ((((int)threadIdx.x) * 3) & 7))) & (7 << (8 - ((((int)threadIdx.x) * 3) & 7)))) & 7))))) * Scales[((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.z) * 8)) + (ax0_ax1_fused_2 >> 1))]) - Zeros[((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.z) * 8)) + (ax0_ax1_fused_2 >> 1))]);
    }
    __asm__ __volatile__("cp.async.commit_group;");

    for (int k_0_0 = 0; k_0_0 < 575; ++k_0_0)
    {
        __syncthreads();

        {
            unsigned int addr;
#if TVM_ENBALE_EFFICIENT_SMEM_PTR_CAST
            addr = static_cast<unsigned int>(__cvta_generic_to_shared((void *)(A_shared + (((((((k_0_0 + 1) & 1) * 2304) + (((int)threadIdx.y) * 1152)) + (((int)threadIdx.z) * 288)) + ((((int)threadIdx.x) >> 3) * 72)) + ((((int)threadIdx.x) & 7) * 8)))));
#else
            __asm__ __volatile__(
                "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }\n"
                : "=r"(addr)
                : "l"((void *)(A_shared + (((((((k_0_0 + 1) & 1) * 2304) + (((int)threadIdx.y) * 1152)) + (((int)threadIdx.z) * 288)) + ((((int)threadIdx.x) >> 3) * 72)) + ((((int)threadIdx.x) & 7) * 8)))));
#endif
            __asm__ __volatile__(
#if TVM_ENABLE_L2_PREFETCH
                "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
#else
                "cp.async.cg.shared.global [%0], [%1], %2;"
#endif
                ::"r"(addr),
                "l"((void *)(A + (((((((((int)blockIdx.y) * 1179648) + (((int)threadIdx.y) * 589824)) + (((int)threadIdx.z) * 147456)) + ((((int)threadIdx.x) >> 3) * 36864)) + (k_0_0 * 64)) + ((((int)threadIdx.x) & 7) * 8)) + 64))), "n"(16));
        }
        for (int ax0_ax1_fused_2_1 = 0; ax0_ax1_fused_2_1 < 16; ++ax0_ax1_fused_2_1)
        {
            B_rescale_shared[((((((((k_0_0 + 1) & 1) * 4608) + (((int)threadIdx.y) * 2304)) + (((int)threadIdx.z) * 576)) + ((ax0_ax1_fused_2_1 >> 1) * 72)) + ((ax0_ax1_fused_2_1 & 1) * 32)) + ((int)threadIdx.x))] = ((((((((int)threadIdx.x) * 3) & 7) <= 5) ? ((half)((((int)B[((((((((((int)blockIdx.x) * 884736) + (((int)threadIdx.y) * 442368)) + (((int)threadIdx.z) * 110592)) + ((ax0_ax1_fused_2_1 >> 1) * 13824)) + (k_0_0 * 24)) + ((ax0_ax1_fused_2_1 & 1) * 12)) + ((((int)threadIdx.x) * 3) >> 3)) + 24)]) >> ((((int)threadIdx.x) * 3) & 7)) & 7)) : ((half)(((signed char)((((int)B[((((((((((int)blockIdx.x) * 884736) + (((int)threadIdx.y) * 442368)) + (((int)threadIdx.z) * 110592)) + ((ax0_ax1_fused_2_1 >> 1) * 13824)) + (k_0_0 * 24)) + ((ax0_ax1_fused_2_1 & 1) * 12)) + ((((int)threadIdx.x) * 3) >> 3)) + 24)]) >> ((((int)threadIdx.x) * 3) & 7)) & ((1 << (8 - ((((int)threadIdx.x) * 3) & 7))) - 1))) | ((signed char)(((((int)B[((((((((((int)blockIdx.x) * 884736) + (((int)threadIdx.y) * 442368)) + (((int)threadIdx.z) * 110592)) + ((ax0_ax1_fused_2_1 >> 1) * 13824)) + (k_0_0 * 24)) + ((ax0_ax1_fused_2_1 & 1) * 12)) + ((((int)threadIdx.x) * 3) >> 3)) + 25)]) << (8 - ((((int)threadIdx.x) * 3) & 7))) & (7 << (8 - ((((int)threadIdx.x) * 3) & 7)))) & 7))))) * Scales[((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.z) * 8)) + (ax0_ax1_fused_2_1 >> 1))]) - Zeros[((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.z) * 8)) + (ax0_ax1_fused_2_1 >> 1))]);
        }
        __asm__ __volatile__("cp.async.commit_group;");

        __asm__ __volatile__("cp.async.wait_group 1;");

        __syncthreads();
        for (int k_0_1 = 0; k_0_1 < 4; ++k_0_1)
        {
            nvcuda::wmma::load_matrix_sync(A_shared_wmma_matrix_a[0], (&(A_shared[((((k_0_0 & 1) * 2304) + (((int)threadIdx.y) * 1152)) + (k_0_1 * 16))])), 72);
            nvcuda::wmma::load_matrix_sync(B_rescale_shared_wmma_matrix_b[0], (&(B_rescale_shared[((((k_0_0 & 1) * 4608) + (((int)threadIdx.z) * 1152)) + (k_0_1 * 16))])), 72);
            nvcuda::wmma::mma_sync(C_wmma_accumulator[0], A_shared_wmma_matrix_a[0], B_rescale_shared_wmma_matrix_b[0], C_wmma_accumulator[0]);
        }
    }
    __asm__ __volatile__("cp.async.wait_group 0;");

    __syncthreads();
    for (int k_0_1_1 = 0; k_0_1_1 < 4; ++k_0_1_1)
    {
        nvcuda::wmma::load_matrix_sync(A_shared_wmma_matrix_a[0], (&(A_shared[(((((int)threadIdx.y) * 1152) + (k_0_1_1 * 16)) + 2304)])), 72);
        nvcuda::wmma::load_matrix_sync(B_rescale_shared_wmma_matrix_b[0], (&(B_rescale_shared[(((((int)threadIdx.z) * 1152) + (k_0_1_1 * 16)) + 4608)])), 72);
        nvcuda::wmma::mma_sync(C_wmma_accumulator[0], A_shared_wmma_matrix_a[0], B_rescale_shared_wmma_matrix_b[0], C_wmma_accumulator[0]);
    }
    nvcuda::wmma::store_matrix_sync((&(C[((((((int)blockIdx.y) * 294912) + (((int)threadIdx.y) * 147456)) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.z) * 16))])), C_wmma_accumulator[0], 9216, nvcuda::wmma::mem_row_major);
}

__global__ void __launch_bounds__(256) tir_halfxint3_tensorop_64x64x64x2_t0_y2z4_K36864_align8(half *__restrict__ A, signed char *__restrict__ B, half *__restrict__ Scales, half *__restrict__ Zeros, half *__restrict__ C)
{
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, half> C_wmma_accumulator[2];
    __shared__ half A_shared[9216];
    __shared__ half B_rescale_shared[9216];
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::row_major> A_shared_wmma_matrix_a[2];
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::col_major> B_rescale_shared_wmma_matrix_b[1];
    for (int i_0_2_init = 0; i_0_2_init < 2; ++i_0_2_init)
    {
        nvcuda::wmma::fill_fragment(C_wmma_accumulator[i_0_2_init], 0.000000e+00f);
    }
    for (int ax0_ax1_fused_2 = 0; ax0_ax1_fused_2 < 2; ++ax0_ax1_fused_2)
    {

        {
            unsigned int addr;
#if TVM_ENBALE_EFFICIENT_SMEM_PTR_CAST
            addr = static_cast<unsigned int>(__cvta_generic_to_shared((void *)(A_shared + (((((((int)threadIdx.y) * 2304) + (((int)threadIdx.z) * 576)) + (ax0_ax1_fused_2 * 288)) + ((((int)threadIdx.x) >> 3) * 72)) + ((((int)threadIdx.x) & 7) * 8)))));
#else
            __asm__ __volatile__(
                "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }\n"
                : "=r"(addr)
                : "l"((void *)(A_shared + (((((((int)threadIdx.y) * 2304) + (((int)threadIdx.z) * 576)) + (ax0_ax1_fused_2 * 288)) + ((((int)threadIdx.x) >> 3) * 72)) + ((((int)threadIdx.x) & 7) * 8)))));
#endif
            __asm__ __volatile__(
#if TVM_ENABLE_L2_PREFETCH
                "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
#else
                "cp.async.cg.shared.global [%0], [%1], %2;"
#endif
                ::"r"(addr),
                "l"((void *)(A + ((((((((int)blockIdx.y) * 2359296) + (((int)threadIdx.y) * 1179648)) + (((int)threadIdx.z) * 294912)) + (ax0_ax1_fused_2 * 147456)) + ((((int)threadIdx.x) >> 3) * 36864)) + ((((int)threadIdx.x) & 7) * 8)))), "n"(16));
        }
    }
    for (int ax0_ax1_fused_2_1 = 0; ax0_ax1_fused_2_1 < 16; ++ax0_ax1_fused_2_1)
    {
        B_rescale_shared[(((((((int)threadIdx.y) * 2304) + (((int)threadIdx.z) * 576)) + ((ax0_ax1_fused_2_1 >> 1) * 72)) + ((ax0_ax1_fused_2_1 & 1) * 32)) + ((int)threadIdx.x))] = ((((((((int)threadIdx.x) * 3) & 7) <= 5) ? ((half)((((int)B[((((((((int)blockIdx.x) * 884736) + (((int)threadIdx.y) * 442368)) + (((int)threadIdx.z) * 110592)) + ((ax0_ax1_fused_2_1 >> 1) * 13824)) + ((ax0_ax1_fused_2_1 & 1) * 12)) + ((((int)threadIdx.x) * 3) >> 3))]) >> ((((int)threadIdx.x) * 3) & 7)) & 7)) : ((half)(((signed char)((((int)B[((((((((int)blockIdx.x) * 884736) + (((int)threadIdx.y) * 442368)) + (((int)threadIdx.z) * 110592)) + ((ax0_ax1_fused_2_1 >> 1) * 13824)) + ((ax0_ax1_fused_2_1 & 1) * 12)) + ((((int)threadIdx.x) * 3) >> 3))]) >> ((((int)threadIdx.x) * 3) & 7)) & ((1 << (8 - ((((int)threadIdx.x) * 3) & 7))) - 1))) | ((signed char)(((((int)B[(((((((((int)blockIdx.x) * 884736) + (((int)threadIdx.y) * 442368)) + (((int)threadIdx.z) * 110592)) + ((ax0_ax1_fused_2_1 >> 1) * 13824)) + ((ax0_ax1_fused_2_1 & 1) * 12)) + ((((int)threadIdx.x) * 3) >> 3)) + 1)]) << (8 - ((((int)threadIdx.x) * 3) & 7))) & (7 << (8 - ((((int)threadIdx.x) * 3) & 7)))) & 7))))) * Scales[((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.z) * 8)) + (ax0_ax1_fused_2_1 >> 1))]) - Zeros[((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.z) * 8)) + (ax0_ax1_fused_2_1 >> 1))]);
    }
    __asm__ __volatile__("cp.async.commit_group;");

    for (int k_0_0 = 0; k_0_0 < 575; ++k_0_0)
    {
        __syncthreads();
        for (int ax0_ax1_fused_2_2 = 0; ax0_ax1_fused_2_2 < 2; ++ax0_ax1_fused_2_2)
        {

            {
                unsigned int addr;
#if TVM_ENBALE_EFFICIENT_SMEM_PTR_CAST
                addr = static_cast<unsigned int>(__cvta_generic_to_shared((void *)(A_shared + ((((((((k_0_0 + 1) & 1) * 4608) + (((int)threadIdx.y) * 2304)) + (((int)threadIdx.z) * 576)) + (ax0_ax1_fused_2_2 * 288)) + ((((int)threadIdx.x) >> 3) * 72)) + ((((int)threadIdx.x) & 7) * 8)))));
#else
                __asm__ __volatile__(
                    "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }\n"
                    : "=r"(addr)
                    : "l"((void *)(A_shared + ((((((((k_0_0 + 1) & 1) * 4608) + (((int)threadIdx.y) * 2304)) + (((int)threadIdx.z) * 576)) + (ax0_ax1_fused_2_2 * 288)) + ((((int)threadIdx.x) >> 3) * 72)) + ((((int)threadIdx.x) & 7) * 8)))));
#endif
                __asm__ __volatile__(
#if TVM_ENABLE_L2_PREFETCH
                    "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
#else
                    "cp.async.cg.shared.global [%0], [%1], %2;"
#endif
                    ::"r"(addr),
                    "l"((void *)(A + ((((((((((int)blockIdx.y) * 2359296) + (((int)threadIdx.y) * 1179648)) + (((int)threadIdx.z) * 294912)) + (ax0_ax1_fused_2_2 * 147456)) + ((((int)threadIdx.x) >> 3) * 36864)) + (k_0_0 * 64)) + ((((int)threadIdx.x) & 7) * 8)) + 64))), "n"(16));
            }
        }
        for (int ax0_ax1_fused_2_3 = 0; ax0_ax1_fused_2_3 < 16; ++ax0_ax1_fused_2_3)
        {
            B_rescale_shared[((((((((k_0_0 + 1) & 1) * 4608) + (((int)threadIdx.y) * 2304)) + (((int)threadIdx.z) * 576)) + ((ax0_ax1_fused_2_3 >> 1) * 72)) + ((ax0_ax1_fused_2_3 & 1) * 32)) + ((int)threadIdx.x))] = ((((((((int)threadIdx.x) * 3) & 7) <= 5) ? ((half)((((int)B[((((((((((int)blockIdx.x) * 884736) + (((int)threadIdx.y) * 442368)) + (((int)threadIdx.z) * 110592)) + ((ax0_ax1_fused_2_3 >> 1) * 13824)) + (k_0_0 * 24)) + ((ax0_ax1_fused_2_3 & 1) * 12)) + ((((int)threadIdx.x) * 3) >> 3)) + 24)]) >> ((((int)threadIdx.x) * 3) & 7)) & 7)) : ((half)(((signed char)((((int)B[((((((((((int)blockIdx.x) * 884736) + (((int)threadIdx.y) * 442368)) + (((int)threadIdx.z) * 110592)) + ((ax0_ax1_fused_2_3 >> 1) * 13824)) + (k_0_0 * 24)) + ((ax0_ax1_fused_2_3 & 1) * 12)) + ((((int)threadIdx.x) * 3) >> 3)) + 24)]) >> ((((int)threadIdx.x) * 3) & 7)) & ((1 << (8 - ((((int)threadIdx.x) * 3) & 7))) - 1))) | ((signed char)(((((int)B[((((((((((int)blockIdx.x) * 884736) + (((int)threadIdx.y) * 442368)) + (((int)threadIdx.z) * 110592)) + ((ax0_ax1_fused_2_3 >> 1) * 13824)) + (k_0_0 * 24)) + ((ax0_ax1_fused_2_3 & 1) * 12)) + ((((int)threadIdx.x) * 3) >> 3)) + 25)]) << (8 - ((((int)threadIdx.x) * 3) & 7))) & (7 << (8 - ((((int)threadIdx.x) * 3) & 7)))) & 7))))) * Scales[((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.z) * 8)) + (ax0_ax1_fused_2_3 >> 1))]) - Zeros[((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.z) * 8)) + (ax0_ax1_fused_2_3 >> 1))]);
        }
        __asm__ __volatile__("cp.async.commit_group;");

        __asm__ __volatile__("cp.async.wait_group 1;");

        __syncthreads();
        for (int k_0_1 = 0; k_0_1 < 4; ++k_0_1)
        {
            for (int ax0_0 = 0; ax0_0 < 2; ++ax0_0)
            {
                nvcuda::wmma::load_matrix_sync(A_shared_wmma_matrix_a[ax0_0], (&(A_shared[(((((k_0_0 & 1) * 4608) + (((int)threadIdx.y) * 2304)) + (ax0_0 * 1152)) + (k_0_1 * 16))])), 72);
            }
            nvcuda::wmma::load_matrix_sync(B_rescale_shared_wmma_matrix_b[0], (&(B_rescale_shared[((((k_0_0 & 1) * 4608) + (((int)threadIdx.z) * 1152)) + (k_0_1 * 16))])), 72);
            for (int i_0_2 = 0; i_0_2 < 2; ++i_0_2)
            {
                nvcuda::wmma::mma_sync(C_wmma_accumulator[i_0_2], A_shared_wmma_matrix_a[i_0_2], B_rescale_shared_wmma_matrix_b[0], C_wmma_accumulator[i_0_2]);
            }
        }
    }
    __asm__ __volatile__("cp.async.wait_group 0;");

    __syncthreads();
    for (int k_0_1_1 = 0; k_0_1_1 < 4; ++k_0_1_1)
    {
        for (int ax0_0_1 = 0; ax0_0_1 < 2; ++ax0_0_1)
        {
            nvcuda::wmma::load_matrix_sync(A_shared_wmma_matrix_a[ax0_0_1], (&(A_shared[((((((int)threadIdx.y) * 2304) + (ax0_0_1 * 1152)) + (k_0_1_1 * 16)) + 4608)])), 72);
        }
        nvcuda::wmma::load_matrix_sync(B_rescale_shared_wmma_matrix_b[0], (&(B_rescale_shared[(((((int)threadIdx.z) * 1152) + (k_0_1_1 * 16)) + 4608)])), 72);
        for (int i_0_2_1 = 0; i_0_2_1 < 2; ++i_0_2_1)
        {
            nvcuda::wmma::mma_sync(C_wmma_accumulator[i_0_2_1], A_shared_wmma_matrix_a[i_0_2_1], B_rescale_shared_wmma_matrix_b[0], C_wmma_accumulator[i_0_2_1]);
        }
    }
    for (int ax0_0_2 = 0; ax0_0_2 < 2; ++ax0_0_2)
    {
        nvcuda::wmma::store_matrix_sync((&(C[(((((((int)blockIdx.y) * 589824) + (((int)threadIdx.y) * 294912)) + (ax0_0_2 * 147456)) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.z) * 16))])), C_wmma_accumulator[ax0_0_2], 9216, nvcuda::wmma::mem_row_major);
    }
}

__global__ void __launch_bounds__(64) tir_halfxint3_tensorop_64x128x32x3_t0_y1z2_K36864_align8(half *__restrict__ A, signed char *__restrict__ B, half *__restrict__ Scales, half *__restrict__ Zeros, half *__restrict__ C)
{
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, half> C_wmma_accumulator[16];
    __shared__ half A_shared[7680];
    __shared__ half B_rescale_shared[15360];
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::row_major> A_shared_wmma_matrix_a[4];
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::col_major> B_rescale_shared_wmma_matrix_b[4];
    for (int i_0_2_init = 0; i_0_2_init < 4; ++i_0_2_init)
    {
        for (int j_0_2_init = 0; j_0_2_init < 4; ++j_0_2_init)
        {
            nvcuda::wmma::fill_fragment(C_wmma_accumulator[((i_0_2_init * 4) + j_0_2_init)], 0.000000e+00f);
        }
    }
    for (int ax0_ax1_fused_2 = 0; ax0_ax1_fused_2 < 4; ++ax0_ax1_fused_2)
    {

        {
            unsigned int addr;
#if TVM_ENBALE_EFFICIENT_SMEM_PTR_CAST
            addr = static_cast<unsigned int>(__cvta_generic_to_shared((void *)(A_shared + ((((((int)threadIdx.z) * 1280) + (ax0_ax1_fused_2 * 320)) + ((((int)threadIdx.x) >> 2) * 40)) + ((((int)threadIdx.x) & 3) * 8)))));
#else
            __asm__ __volatile__(
                "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }\n"
                : "=r"(addr)
                : "l"((void *)(A_shared + ((((((int)threadIdx.z) * 1280) + (ax0_ax1_fused_2 * 320)) + ((((int)threadIdx.x) >> 2) * 40)) + ((((int)threadIdx.x) & 3) * 8)))));
#endif
            __asm__ __volatile__(
#if TVM_ENABLE_L2_PREFETCH
                "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
#else
                "cp.async.cg.shared.global [%0], [%1], %2;"
#endif
                ::"r"(addr),
                "l"((void *)(A + (((((((int)blockIdx.y) * 2359296) + (((int)threadIdx.z) * 1179648)) + (ax0_ax1_fused_2 * 294912)) + ((((int)threadIdx.x) >> 2) * 36864)) + ((((int)threadIdx.x) & 3) * 8)))), "n"(16));
        }
    }
    for (int ax0_ax1_fused_2_1 = 0; ax0_ax1_fused_2_1 < 64; ++ax0_ax1_fused_2_1)
    {
        B_rescale_shared[(((((int)threadIdx.z) * 2560) + (ax0_ax1_fused_2_1 * 40)) + ((int)threadIdx.x))] = ((((((((int)threadIdx.x) * 3) & 7) <= 5) ? ((half)((((int)B[((((((int)blockIdx.x) * 1769472) + (((int)threadIdx.z) * 884736)) + (ax0_ax1_fused_2_1 * 13824)) + ((((int)threadIdx.x) * 3) >> 3))]) >> ((((int)threadIdx.x) * 3) & 7)) & 7)) : ((half)(((signed char)((((int)B[((((((int)blockIdx.x) * 1769472) + (((int)threadIdx.z) * 884736)) + (ax0_ax1_fused_2_1 * 13824)) + ((((int)threadIdx.x) * 3) >> 3))]) >> ((((int)threadIdx.x) * 3) & 7)) & ((1 << (8 - ((((int)threadIdx.x) * 3) & 7))) - 1))) | ((signed char)(((((int)B[(((((((int)blockIdx.x) * 1769472) + (((int)threadIdx.z) * 884736)) + (ax0_ax1_fused_2_1 * 13824)) + ((((int)threadIdx.x) * 3) >> 3)) + 1)]) << (8 - ((((int)threadIdx.x) * 3) & 7))) & (7 << (8 - ((((int)threadIdx.x) * 3) & 7)))) & 7))))) * Scales[(((((int)blockIdx.x) * 128) + (((int)threadIdx.z) * 64)) + ax0_ax1_fused_2_1)]) - Zeros[(((((int)blockIdx.x) * 128) + (((int)threadIdx.z) * 64)) + ax0_ax1_fused_2_1)]);
    }
    __asm__ __volatile__("cp.async.commit_group;");

    for (int ax0_ax1_fused_2_2 = 0; ax0_ax1_fused_2_2 < 4; ++ax0_ax1_fused_2_2)
    {

        {
            unsigned int addr;
#if TVM_ENBALE_EFFICIENT_SMEM_PTR_CAST
            addr = static_cast<unsigned int>(__cvta_generic_to_shared((void *)(A_shared + (((((((int)threadIdx.z) * 1280) + (ax0_ax1_fused_2_2 * 320)) + ((((int)threadIdx.x) >> 2) * 40)) + ((((int)threadIdx.x) & 3) * 8)) + 2560))));
#else
            __asm__ __volatile__(
                "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }\n"
                : "=r"(addr)
                : "l"((void *)(A_shared + (((((((int)threadIdx.z) * 1280) + (ax0_ax1_fused_2_2 * 320)) + ((((int)threadIdx.x) >> 2) * 40)) + ((((int)threadIdx.x) & 3) * 8)) + 2560))));
#endif
            __asm__ __volatile__(
#if TVM_ENABLE_L2_PREFETCH
                "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
#else
                "cp.async.cg.shared.global [%0], [%1], %2;"
#endif
                ::"r"(addr),
                "l"((void *)(A + ((((((((int)blockIdx.y) * 2359296) + (((int)threadIdx.z) * 1179648)) + (ax0_ax1_fused_2_2 * 294912)) + ((((int)threadIdx.x) >> 2) * 36864)) + ((((int)threadIdx.x) & 3) * 8)) + 32))), "n"(16));
        }
    }
    for (int ax0_ax1_fused_2_3 = 0; ax0_ax1_fused_2_3 < 64; ++ax0_ax1_fused_2_3)
    {
        B_rescale_shared[((((((int)threadIdx.z) * 2560) + (ax0_ax1_fused_2_3 * 40)) + ((int)threadIdx.x)) + 5120)] = ((((((((int)threadIdx.x) * 3) & 7) <= 5) ? ((half)((((int)B[(((((((int)blockIdx.x) * 1769472) + (((int)threadIdx.z) * 884736)) + (ax0_ax1_fused_2_3 * 13824)) + ((((int)threadIdx.x) * 3) >> 3)) + 12)]) >> ((((int)threadIdx.x) * 3) & 7)) & 7)) : ((half)(((signed char)((((int)B[(((((((int)blockIdx.x) * 1769472) + (((int)threadIdx.z) * 884736)) + (ax0_ax1_fused_2_3 * 13824)) + ((((int)threadIdx.x) * 3) >> 3)) + 12)]) >> ((((int)threadIdx.x) * 3) & 7)) & ((1 << (8 - ((((int)threadIdx.x) * 3) & 7))) - 1))) | ((signed char)(((((int)B[(((((((int)blockIdx.x) * 1769472) + (((int)threadIdx.z) * 884736)) + (ax0_ax1_fused_2_3 * 13824)) + ((((int)threadIdx.x) * 3) >> 3)) + 13)]) << (8 - ((((int)threadIdx.x) * 3) & 7))) & (7 << (8 - ((((int)threadIdx.x) * 3) & 7)))) & 7))))) * Scales[(((((int)blockIdx.x) * 128) + (((int)threadIdx.z) * 64)) + ax0_ax1_fused_2_3)]) - Zeros[(((((int)blockIdx.x) * 128) + (((int)threadIdx.z) * 64)) + ax0_ax1_fused_2_3)]);
    }
    __asm__ __volatile__("cp.async.commit_group;");

    for (int k_0_0 = 0; k_0_0 < 1150; ++k_0_0)
    {
        __syncthreads();
        for (int ax0_ax1_fused_2_4 = 0; ax0_ax1_fused_2_4 < 4; ++ax0_ax1_fused_2_4)
        {

            {
                unsigned int addr;
#if TVM_ENBALE_EFFICIENT_SMEM_PTR_CAST
                addr = static_cast<unsigned int>(__cvta_generic_to_shared((void *)(A_shared + (((((((k_0_0 + 2) % 3) * 2560) + (((int)threadIdx.z) * 1280)) + (ax0_ax1_fused_2_4 * 320)) + ((((int)threadIdx.x) >> 2) * 40)) + ((((int)threadIdx.x) & 3) * 8)))));
#else
                __asm__ __volatile__(
                    "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }\n"
                    : "=r"(addr)
                    : "l"((void *)(A_shared + (((((((k_0_0 + 2) % 3) * 2560) + (((int)threadIdx.z) * 1280)) + (ax0_ax1_fused_2_4 * 320)) + ((((int)threadIdx.x) >> 2) * 40)) + ((((int)threadIdx.x) & 3) * 8)))));
#endif
                __asm__ __volatile__(
#if TVM_ENABLE_L2_PREFETCH
                    "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
#else
                    "cp.async.cg.shared.global [%0], [%1], %2;"
#endif
                    ::"r"(addr),
                    "l"((void *)(A + (((((((((int)blockIdx.y) * 2359296) + (((int)threadIdx.z) * 1179648)) + (ax0_ax1_fused_2_4 * 294912)) + ((((int)threadIdx.x) >> 2) * 36864)) + (k_0_0 * 32)) + ((((int)threadIdx.x) & 3) * 8)) + 64))), "n"(16));
            }
        }
        for (int ax0_ax1_fused_2_5 = 0; ax0_ax1_fused_2_5 < 64; ++ax0_ax1_fused_2_5)
        {
            B_rescale_shared[((((((k_0_0 + 2) % 3) * 5120) + (((int)threadIdx.z) * 2560)) + (ax0_ax1_fused_2_5 * 40)) + ((int)threadIdx.x))] = ((((((((int)threadIdx.x) * 3) & 7) <= 5) ? ((half)((((int)B[((((((((int)blockIdx.x) * 1769472) + (((int)threadIdx.z) * 884736)) + (ax0_ax1_fused_2_5 * 13824)) + (k_0_0 * 12)) + ((((int)threadIdx.x) * 3) >> 3)) + 24)]) >> ((((int)threadIdx.x) * 3) & 7)) & 7)) : ((half)(((signed char)((((int)B[((((((((int)blockIdx.x) * 1769472) + (((int)threadIdx.z) * 884736)) + (ax0_ax1_fused_2_5 * 13824)) + (k_0_0 * 12)) + ((((int)threadIdx.x) * 3) >> 3)) + 24)]) >> ((((int)threadIdx.x) * 3) & 7)) & ((1 << (8 - ((((int)threadIdx.x) * 3) & 7))) - 1))) | ((signed char)(((((int)B[((((((((int)blockIdx.x) * 1769472) + (((int)threadIdx.z) * 884736)) + (ax0_ax1_fused_2_5 * 13824)) + (k_0_0 * 12)) + ((((int)threadIdx.x) * 3) >> 3)) + 25)]) << (8 - ((((int)threadIdx.x) * 3) & 7))) & (7 << (8 - ((((int)threadIdx.x) * 3) & 7)))) & 7))))) * Scales[(((((int)blockIdx.x) * 128) + (((int)threadIdx.z) * 64)) + ax0_ax1_fused_2_5)]) - Zeros[(((((int)blockIdx.x) * 128) + (((int)threadIdx.z) * 64)) + ax0_ax1_fused_2_5)]);
        }
        __asm__ __volatile__("cp.async.commit_group;");

        __asm__ __volatile__("cp.async.wait_group 2;");

        __syncthreads();
        for (int k_0_1 = 0; k_0_1 < 2; ++k_0_1)
        {
            for (int ax0_0 = 0; ax0_0 < 4; ++ax0_0)
            {
                nvcuda::wmma::load_matrix_sync(A_shared_wmma_matrix_a[ax0_0], (&(A_shared[((((k_0_0 % 3) * 2560) + (ax0_0 * 640)) + (k_0_1 * 16))])), 40);
            }
            for (int ax0_0_1 = 0; ax0_0_1 < 4; ++ax0_0_1)
            {
                nvcuda::wmma::load_matrix_sync(B_rescale_shared_wmma_matrix_b[ax0_0_1], (&(B_rescale_shared[(((((k_0_0 % 3) * 5120) + (((int)threadIdx.z) * 2560)) + (ax0_0_1 * 640)) + (k_0_1 * 16))])), 40);
            }
            for (int i_0_2 = 0; i_0_2 < 4; ++i_0_2)
            {
                for (int j_0_2 = 0; j_0_2 < 4; ++j_0_2)
                {
                    nvcuda::wmma::mma_sync(C_wmma_accumulator[((i_0_2 * 4) + j_0_2)], A_shared_wmma_matrix_a[i_0_2], B_rescale_shared_wmma_matrix_b[j_0_2], C_wmma_accumulator[((i_0_2 * 4) + j_0_2)]);
                }
            }
        }
    }
    __asm__ __volatile__("cp.async.wait_group 1;");

    __syncthreads();
    for (int k_0_1_1 = 0; k_0_1_1 < 2; ++k_0_1_1)
    {
        for (int ax0_0_2 = 0; ax0_0_2 < 4; ++ax0_0_2)
        {
            nvcuda::wmma::load_matrix_sync(A_shared_wmma_matrix_a[ax0_0_2], (&(A_shared[(((ax0_0_2 * 640) + (k_0_1_1 * 16)) + 2560)])), 40);
        }
        for (int ax0_0_3 = 0; ax0_0_3 < 4; ++ax0_0_3)
        {
            nvcuda::wmma::load_matrix_sync(B_rescale_shared_wmma_matrix_b[ax0_0_3], (&(B_rescale_shared[((((((int)threadIdx.z) * 2560) + (ax0_0_3 * 640)) + (k_0_1_1 * 16)) + 5120)])), 40);
        }
        for (int i_0_2_1 = 0; i_0_2_1 < 4; ++i_0_2_1)
        {
            for (int j_0_2_1 = 0; j_0_2_1 < 4; ++j_0_2_1)
            {
                nvcuda::wmma::mma_sync(C_wmma_accumulator[((i_0_2_1 * 4) + j_0_2_1)], A_shared_wmma_matrix_a[i_0_2_1], B_rescale_shared_wmma_matrix_b[j_0_2_1], C_wmma_accumulator[((i_0_2_1 * 4) + j_0_2_1)]);
            }
        }
    }
    __asm__ __volatile__("cp.async.wait_group 0;");

    __syncthreads();
    for (int k_0_1_2 = 0; k_0_1_2 < 2; ++k_0_1_2)
    {
        for (int ax0_0_4 = 0; ax0_0_4 < 4; ++ax0_0_4)
        {
            nvcuda::wmma::load_matrix_sync(A_shared_wmma_matrix_a[ax0_0_4], (&(A_shared[(((ax0_0_4 * 640) + (k_0_1_2 * 16)) + 5120)])), 40);
        }
        for (int ax0_0_5 = 0; ax0_0_5 < 4; ++ax0_0_5)
        {
            nvcuda::wmma::load_matrix_sync(B_rescale_shared_wmma_matrix_b[ax0_0_5], (&(B_rescale_shared[((((((int)threadIdx.z) * 2560) + (ax0_0_5 * 640)) + (k_0_1_2 * 16)) + 10240)])), 40);
        }
        for (int i_0_2_2 = 0; i_0_2_2 < 4; ++i_0_2_2)
        {
            for (int j_0_2_2 = 0; j_0_2_2 < 4; ++j_0_2_2)
            {
                nvcuda::wmma::mma_sync(C_wmma_accumulator[((i_0_2_2 * 4) + j_0_2_2)], A_shared_wmma_matrix_a[i_0_2_2], B_rescale_shared_wmma_matrix_b[j_0_2_2], C_wmma_accumulator[((i_0_2_2 * 4) + j_0_2_2)]);
            }
        }
    }
    for (int ax0_0_6 = 0; ax0_0_6 < 4; ++ax0_0_6)
    {
        for (int ax1_0 = 0; ax1_0 < 4; ++ax1_0)
        {
            nvcuda::wmma::store_matrix_sync((&(C[(((((((int)blockIdx.y) * 589824) + (ax0_0_6 * 147456)) + (((int)blockIdx.x) * 128)) + (((int)threadIdx.z) * 64)) + (ax1_0 * 16))])), C_wmma_accumulator[((ax0_0_6 * 4) + ax1_0)], 9216, nvcuda::wmma::mem_row_major);
        }
    }
}

__global__ void __launch_bounds__(128) tir_halfxint3_tensorop_128x256x32x1_t0_y2z2_K36864_align8(half *__restrict__ A, signed char *__restrict__ B, half *__restrict__ Scales, half *__restrict__ Zeros, half *__restrict__ C)
{
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, half> C_wmma_accumulator[32];
    __shared__ half A_shared[5120];
    __shared__ half B_rescale_shared[10240];
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::row_major> A_shared_wmma_matrix_a[4];
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::col_major> B_rescale_shared_wmma_matrix_b[8];
    for (int i_0_2_init = 0; i_0_2_init < 4; ++i_0_2_init)
    {
        for (int j_0_2_init = 0; j_0_2_init < 8; ++j_0_2_init)
        {
            nvcuda::wmma::fill_fragment(C_wmma_accumulator[((i_0_2_init * 8) + j_0_2_init)], 0.000000e+00f);
        }
    }
    for (int k_0_0 = 0; k_0_0 < 1152; ++k_0_0)
    {
        __syncthreads();
        for (int ax0_ax1_fused_2 = 0; ax0_ax1_fused_2 < 4; ++ax0_ax1_fused_2)
        {
            *(uint4 *)(A_shared + (((((((int)threadIdx.y) * 2560) + (((int)threadIdx.z) * 1280)) + (ax0_ax1_fused_2 * 320)) + ((((int)threadIdx.x) >> 2) * 40)) + ((((int)threadIdx.x) & 3) * 8))) = *(uint4 *)(A + (((((((((int)blockIdx.y) * 4718592) + (((int)threadIdx.y) * 2359296)) + (((int)threadIdx.z) * 1179648)) + (ax0_ax1_fused_2 * 294912)) + ((((int)threadIdx.x) >> 2) * 36864)) + (k_0_0 * 32)) + ((((int)threadIdx.x) & 3) * 8)));
        }
        for (int ax0_ax1_fused_2_1 = 0; ax0_ax1_fused_2_1 < 64; ++ax0_ax1_fused_2_1)
        {
            B_rescale_shared[((((((int)threadIdx.y) * 5120) + (((int)threadIdx.z) * 2560)) + (ax0_ax1_fused_2_1 * 40)) + ((int)threadIdx.x))] = ((((((((int)threadIdx.x) * 3) & 7) <= 5) ? ((half)((((int)B[((((((((int)blockIdx.x) * 3538944) + (((int)threadIdx.y) * 1769472)) + (((int)threadIdx.z) * 884736)) + (ax0_ax1_fused_2_1 * 13824)) + (k_0_0 * 12)) + ((((int)threadIdx.x) * 3) >> 3))]) >> ((((int)threadIdx.x) * 3) & 7)) & 7)) : ((half)(((signed char)((((int)B[((((((((int)blockIdx.x) * 3538944) + (((int)threadIdx.y) * 1769472)) + (((int)threadIdx.z) * 884736)) + (ax0_ax1_fused_2_1 * 13824)) + (k_0_0 * 12)) + ((((int)threadIdx.x) * 3) >> 3))]) >> ((((int)threadIdx.x) * 3) & 7)) & ((1 << (8 - ((((int)threadIdx.x) * 3) & 7))) - 1))) | ((signed char)(((((int)B[(((((((((int)blockIdx.x) * 3538944) + (((int)threadIdx.y) * 1769472)) + (((int)threadIdx.z) * 884736)) + (ax0_ax1_fused_2_1 * 13824)) + (k_0_0 * 12)) + ((((int)threadIdx.x) * 3) >> 3)) + 1)]) << (8 - ((((int)threadIdx.x) * 3) & 7))) & (7 << (8 - ((((int)threadIdx.x) * 3) & 7)))) & 7))))) * Scales[((((((int)blockIdx.x) * 256) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.z) * 64)) + ax0_ax1_fused_2_1)]) - Zeros[((((((int)blockIdx.x) * 256) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.z) * 64)) + ax0_ax1_fused_2_1)]);
        }
        __syncthreads();
        for (int k_0_1 = 0; k_0_1 < 2; ++k_0_1)
        {
            for (int ax0_0 = 0; ax0_0 < 4; ++ax0_0)
            {
                nvcuda::wmma::load_matrix_sync(A_shared_wmma_matrix_a[ax0_0], (&(A_shared[(((((int)threadIdx.y) * 2560) + (ax0_0 * 640)) + (k_0_1 * 16))])), 40);
            }
            for (int ax0_0_1 = 0; ax0_0_1 < 8; ++ax0_0_1)
            {
                nvcuda::wmma::load_matrix_sync(B_rescale_shared_wmma_matrix_b[ax0_0_1], (&(B_rescale_shared[(((((int)threadIdx.z) * 5120) + (ax0_0_1 * 640)) + (k_0_1 * 16))])), 40);
            }
            for (int i_0_2 = 0; i_0_2 < 4; ++i_0_2)
            {
                for (int j_0_2 = 0; j_0_2 < 8; ++j_0_2)
                {
                    nvcuda::wmma::mma_sync(C_wmma_accumulator[((i_0_2 * 8) + j_0_2)], A_shared_wmma_matrix_a[i_0_2], B_rescale_shared_wmma_matrix_b[j_0_2], C_wmma_accumulator[((i_0_2 * 8) + j_0_2)]);
                }
            }
        }
    }
    for (int ax0_0_2 = 0; ax0_0_2 < 4; ++ax0_0_2)
    {
        for (int ax1_0 = 0; ax1_0 < 8; ++ax1_0)
        {
            nvcuda::wmma::store_matrix_sync((&(C[((((((((int)blockIdx.y) * 1179648) + (((int)threadIdx.y) * 589824)) + (ax0_0_2 * 147456)) + (((int)blockIdx.x) * 256)) + (((int)threadIdx.z) * 128)) + (ax1_0 * 16))])), C_wmma_accumulator[((ax0_0_2 * 8) + ax1_0)], 9216, nvcuda::wmma::mem_row_major);
        }
    }
}
