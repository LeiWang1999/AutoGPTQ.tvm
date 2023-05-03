# from tvm.script import tir as T

@T.prim_func
def main(A: T.Buffer((16384, 16384), "float16"), B: T.Buffer((16384, 8192), "int8"), C: T.Buffer((16384, 16384), "float16")):
    T.func_attr({"global_symbol": "main", "tir.noalias": True})
    # with T.block("root"):
    A_shared = T.alloc_buffer((16384, 16384), "float16", scope="shared")
    A_shared_wmma_matrix_a = T.alloc_buffer((16384, 16384), "float16", scope="wmma.matrix_a")
    B_decompress_shared = T.alloc_buffer((16384, 16384), "float16", scope="shared")
    B_decompress_shared_wmma_matrix_b = T.alloc_buffer((16384, 16384), "float16", scope="wmma.matrix_b")
    C_wmma_accumulator = T.alloc_buffer((16384, 16384), "float16", scope="wmma.accumulator")
    for j_0_0_1 in T.thread_binding(8, thread="blockIdx.z"):
        for i_0_0 in T.thread_binding(1024, thread="blockIdx.y"):
            for j_0_0_0 in T.thread_binding(16, thread="blockIdx.x"):
                for i_0_1 in T.thread_binding(1, thread="threadIdx.y"):
                    for j_0_1 in T.thread_binding(1, thread="threadIdx.z"):
                        for i_0_2_init, j_0_2_init, i_1_init, j_1_init in T.grid(1, 8, 16, 16):
                            with T.block("B_init"):
                                vi = T.axis.spatial(16384, i_0_0 * 16 + i_0_1 * 16 + i_0_2_init * 16 + i_1_init)
                                vj = T.axis.spatial(16384, j_0_0_0 * 1024 + j_0_0_1 * 128 + j_0_1 * 128 + j_0_2_init * 16 + j_1_init)
                                T.reads()
                                T.writes(C_wmma_accumulator[vi, vj])
                                C_wmma_accumulator[vi, vj] = T.float16(0)
                        for k_0_0 in range(256):
                            for ax0_ax1_fused_0 in T.thread_binding(1, thread="threadIdx.y"):
                                for ax0_ax1_fused_1 in T.thread_binding(1, thread="threadIdx.z"):
                                    for ax0_ax1_fused_2 in range(4):
                                        for ax0_ax1_fused_3 in T.thread_binding(32, thread="threadIdx.x"):
                                            for ax0_ax1_fused_4 in T.vectorized(8):
                                                with T.block("A_shared"):
                                                    v0 = T.axis.spatial(16384, i_0_0 * 16 + (ax0_ax1_fused_0 * 1024 + ax0_ax1_fused_1 * 1024 + ax0_ax1_fused_2 * 256 + ax0_ax1_fused_3 * 8 + ax0_ax1_fused_4) // 64)
                                                    v1 = T.axis.spatial(16384, k_0_0 * 64 + (ax0_ax1_fused_0 * 1024 + ax0_ax1_fused_1 * 1024 + ax0_ax1_fused_2 * 256 + ax0_ax1_fused_3 * 8 + ax0_ax1_fused_4) % 64)
                                                    T.reads(A[v0, v1])
                                                    T.writes(A_shared[v0, v1])
                                                    A_shared[v0, v1] = A[v0, v1]
                            for ax0_ax1_fused_0 in T.thread_binding(1, thread="threadIdx.y"):
                                for ax0_ax1_fused_1 in T.thread_binding(1, thread="threadIdx.z"):
                                    for ax0_ax1_fused_2 in range(64):
                                        for ax0_ax1_fused_3 in T.thread_binding(32, thread="threadIdx.x"):
                                            for ax0_ax1_fused_4 in T.vectorized(4):
                                                with T.block("B_decompress_shared"):
                                                    v0 = T.axis.spatial(16384, j_0_0_0 * 1024 + j_0_0_1 * 128 + (ax0_ax1_fused_0 * 8192 + ax0_ax1_fused_1 * 8192 + ax0_ax1_fused_2 * 128 + ax0_ax1_fused_3 * 4 + ax0_ax1_fused_4) // 64)
                                                    v1 = T.axis.spatial(16384, k_0_0 * 64 + (ax0_ax1_fused_0 * 8192 + ax0_ax1_fused_1 * 8192 + ax0_ax1_fused_2 * 128 + ax0_ax1_fused_3 * 4 + ax0_ax1_fused_4) % 64)
                                                    T.reads(B[v0, v1 // 2:v1 // 2 + 2])
                                                    T.writes(B_decompress_shared[v0, v1])
                                                    B_decompress_shared[v0, v1] = T.Select(v1 % 32 * 4 % 8 <= 5, T.Cast("float16", T.bitwise_and(T.shift_right(T.Cast("int32", B[v0, v1 // 32 * 16 + v1 % 32 * 4 // 8]), v1 % 32 * 4 % 8), 15)), T.Cast("float16", T.bitwise_or(T.Cast("int8", T.bitwise_and(T.shift_right(T.Cast("int32", B[v0, v1 // 32 * 16 + v1 % 32 * 4 // 8]), v1 % 32 * 4 % 8), T.shift_left(1, 8 - v1 % 32 * 4 % 8) - 1)), T.Cast("int8", T.bitwise_and(T.bitwise_and(T.shift_left(T.Cast("int32", B[v0, v1 // 32 * 16 + v1 % 32 * 4 // 8 + 1]), 8 - v1 % 32 * 4 % 8), T.shift_left(15, 8 - v1 % 32 * 4 % 8)), 15)))))
                            for k_0_1 in range(4):
                                for ax0_0, ax1_0, ax0_1, ax1_1 in T.grid(1, 1, 16, 16):
                                    with T.block("A_shared_wmma.matrix_a"):
                                        v0 = T.axis.spatial(16384, i_0_0 * 16 + ax0_0 * 16 + ax0_1)
                                        v1 = T.axis.spatial(16384, k_0_0 * 64 + k_0_1 * 16 + ax1_0 * 16 + ax1_1)
                                        T.reads(A_shared[v0, v1])
                                        T.writes(A_shared_wmma_matrix_a[v0, v1])
                                        A_shared_wmma_matrix_a[v0, v1] = A_shared[v0, v1]
                                for ax0_0, ax1_0, ax0_1, ax1_1 in T.grid(8, 1, 16, 16):
                                    with T.block("B_decompress_shared_wmma.matrix_b"):
                                        v0 = T.axis.spatial(16384, j_0_0_0 * 1024 + j_0_0_1 * 128 + ax0_0 * 16 + ax0_1)
                                        v1 = T.axis.spatial(16384, k_0_0 * 64 + k_0_1 * 16 + ax1_0 * 16 + ax1_1)
                                        T.reads(B_decompress_shared[v0, v1])
                                        T.writes(B_decompress_shared_wmma_matrix_b[v0, v1])
                                        B_decompress_shared_wmma_matrix_b[v0, v1] = B_decompress_shared[v0, v1]
                                for i_0_2, j_0_2, i_1, j_1, k_1 in T.grid(1, 8, 16, 16, 16):
                                    with T.block("B_update"):
                                        vi = T.axis.spatial(16384, i_0_0 * 16 + i_0_1 * 16 + i_0_2 * 16 + i_1)
                                        vj = T.axis.spatial(16384, j_0_0_0 * 1024 + j_0_0_1 * 128 + j_0_1 * 128 + j_0_2 * 16 + j_1)
                                        vk = T.axis.reduce(16384, k_0_0 * 64 + k_0_1 * 16 + k_1)
                                        T.reads(C_wmma_accumulator[vi, vj], A_shared_wmma_matrix_a[vi, vk], B_decompress_shared_wmma_matrix_b[vj, vk])
                                        T.writes(C_wmma_accumulator[vi, vj])
                                        C_wmma_accumulator[vi, vj] = C_wmma_accumulator[vi, vj] + A_shared_wmma_matrix_a[vi, vk] * B_decompress_shared_wmma_matrix_b[vj, vk]
                        for ax0_0, ax1_0, ax0_1, ax1_1 in T.grid(1, 8, 16, 16):
                            with T.block("C_wmma.accumulator"):
                                v0 = T.axis.spatial(16384, i_0_0 * 16 + ax0_0 * 16 + ax0_1)
                                v1 = T.axis.spatial(16384, j_0_0_0 * 1024 + j_0_0_1 * 128 + ax1_0 * 16 + ax1_1)
                                T.reads(C_wmma_accumulator[v0, v1])
                                T.writes(C[v0, v1])
                                C[v0, v1] = C_wmma_accumulator[v0, v1]