import tvm
import numpy as np
import tvm.testing
from tvm.script import tir as T
import os

def get_gemm_workloads(bits, N, K):
    M = 512
    N = N
    K = K
    group_stride = 32 * bits // 8
    mask = (1 << bits) - 1
    @tvm.script.ir_module
    class MyModule:
        @T.prim_func
        def main(a: T.handle, b: T.handle, c: T.handle, scales: T.handle, zeros: T.handle):
            T.func_attr({"global_symbol": "main", "tir.noalias": True})
            A = T.match_buffer(a, [M, K], dtype="float16")
            B = T.match_buffer(b, [N, K // 8 * bits], dtype="int8")
            C = T.match_buffer(c, [M, N], dtype="float16")
            Scales = T.match_buffer(scales, [N], dtype="float16")
            Zeros = T.match_buffer(zeros, [N], dtype="float16")

            B_decompress = T.alloc_buffer([N, K], dtype="float16")
            B_rescale = T.alloc_buffer([N, K], dtype="float16")

            for i, j in T.grid(N, K):
                with T.block("B_decompress"):
                    vi, vj = T.axis.remap("SS", [i, j])
                    B_decompress[vi, vj] = T.Select(((vj % 32) * bits) % 8 <= 5, ((B[vi, (vj // 32) * group_stride + (vj % 32) * bits // 8] >> (((vj % 32) * bits) % 8) & mask)).astype("float16"), (((B[vi, (vj // 32) * group_stride + (vj % 32) * bits // 8] >> ((vj % 32) * bits) % 8) & (
                        1 << (8 - ((vj % 32) * bits) % 8)) - 1).astype("int8") | ((B[vi, (vj // 32) * group_stride + (vj % 32) * bits // 8 + 1] << (8 - ((vj % 32) * bits) % 8)) & (mask << (8 - ((vj % 32) * bits) % 8)) & mask).astype("int8")).astype("float16"))
            
            for i, j in T.grid(N, K):
                with T.block("B_rescale"):
                    vi, vj = T.axis.remap("SS", [i, j])
                    B_rescale[vi, vj] = B_decompress[vi, vj] * \
                        Scales[vi].astype('float16') - Zeros[vi].astype('float16')
                        
            for i, j, k in T.grid(M, N, K):
                with T.block("B"):
                    vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                    with T.init():
                        C[vi, vj] = T.float16(0)
                    C[vi, vj] = C[vi, vj] + \
                        A[vi, vk].astype("float16") * \
                        B_rescale[vj, vk].astype("float16")
    return MyModule

def get_gemv_workloads(bits, K):
    
    group_stride = 32 * bits // 8
    M = 1
    N = 9216
    K = K
    mask = (1 << bits) - 1
    vec = 8 if bits == 3 else 8
    num_warps = 4
    warp_size = 32

    @tvm.script.ir_module
    class MyModule:
        @T.prim_func
        def main(a: T.handle, b: T.handle, c: T.handle, scales: T.handle, zeros: T.handle):
            T.func_attr({"global_symbol": "main", "tir.noalias": True})
            A = T.match_buffer(a, [M, K], dtype="float16")
            B = T.match_buffer(b, [N, K // 8 * bits], dtype="int8")
            C = T.match_buffer(c, [M, N], dtype="float16")
            Scales = T.match_buffer(scales, [N], dtype="float16")
            Zeros = T.match_buffer(zeros, [N], dtype="float16")
            
            B_decompress = T.alloc_buffer([N, K], dtype="float16")
            B_rescale= T.alloc_buffer([N, K], dtype="float16")

            for i, j  in T.grid(N, K):
                with T.block("B_decompress"):
                    vi, vj = T.axis.remap("SS", [i, j])
                    B_decompress[vi, vj] = T.Select(((vj % 32) * bits) % 8 <= 5, ((B[vi, (vj // 32) * group_stride +  (vj % 32)* bits // 8] >> (((vj % 32) * bits) % 8) & mask)).astype("float16"), (((B[vi, (vj // 32) * group_stride +  (vj % 32)* bits // 8] >> ((vj % 32) * bits) % 8) & (1 << (8 - ((vj % 32) * bits) % 8)) - 1).astype("int8") | ((B[vi, (vj // 32) * group_stride +  (vj % 32)* bits // 8 + 1] << (8 - ((vj % 32) * bits) % 8)) & (mask << (8 - ((vj % 32) * bits) % 8)) & mask).astype("int8")).astype("float16")) 
            
            for i, j in T.grid(N, K):
                with T.block("B_rescale"):
                    vi, vj = T.axis.remap("SS", [i, j])
                    B_rescale[vi, vj] = B_decompress[vi, vj] * Scales[vi].astype('float16') - Zeros[vi].astype('float16') 
            
            for i, j, k in T.grid(M, N, K):
                with T.block("B"):
                    vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                    with T.init():
                        C[vi, vj] = 0.0
                    C[vi, vj] = C[vi, vj] + \
                        A[vi, vk].astype("float16") * \
                        B_rescale[vj, vk].astype("float16")

    return MyModule