# from tvm.script import tir as T

@T.prim_func
def main(A: T.Buffer((32, 8192), "float16"), B: T.Buffer((8192, 8192), "float16"), C: T.Buffer((32, 8192), "float16")):
    T.func_attr({"global_symbol": "main", "tir.noalias": True})
    # with T.block("root"):
    A_shared = T.alloc_buffer((32, 8192), "float16", scope="shared")
    A_shared_wmma_matrix_a = T.alloc_buffer((32, 8192), "float16", scope="wmma.matrix_a")
    B_shared = T.alloc_buffer((8192, 8192), "float16", scope="shared")
    B_shared_wmma_matrix_b = T.alloc_buffer((8192, 8192), "float16", scope="wmma.matrix_b")
    C_wmma_accumulator = T.alloc_buffer((32, 8192), "float16", scope="wmma.accumulator")
    for j_0_0_1 in T.thread_binding(8, thread="blockIdx.z"):
        for i_0_0 in T.thread_binding(1, thread="blockIdx.y"):
            for j_0_0_0 in T.thread_binding(8, thread="blockIdx.x"):
                for i_0_1 in T.thread_binding(2, thread="threadIdx.y"):
                    for j_0_1 in T.thread_binding(1, thread="threadIdx.z"):
                        for i_0_2_init, j_0_2_init, i_1_init, j_1_init in T.grid(1, 8, 16, 16):
                            with T.block("B_init"):
                                vi = T.axis.spatial(32, i_0_0 * 32 + i_0_1 * 16 + i_0_2_init * 16 + i_1_init)
                                vj = T.axis.spatial(8192, j_0_0_0 * 1024 + j_0_0_1 * 128 + j_0_1 * 128 + j_0_2_init * 16 + j_1_init)
                                T.reads()
                                T.writes(C_wmma_accumulator[vi, vj])
                                C_wmma_accumulator[vi, vj] = T.float16(0)
                        for k_0_0 in range(128):
                            for ax0_ax1_fused_0 in T.thread_binding(2, thread="threadIdx.y"):
                                for ax0_ax1_fused_1 in T.thread_binding(1, thread="threadIdx.z"):
                                    for ax0_ax1_fused_2 in range(4):
                                        for ax0_ax1_fused_3 in T.thread_binding(32, thread="threadIdx.x"):
                                            for ax0_ax1_fused_4 in T.vectorized(8):
                                                with T.block("A_shared"):
                                                    v0 = T.axis.spatial(32, (ax0_ax1_fused_0 * 1024 + ax0_ax1_fused_1 * 1024 + ax0_ax1_fused_2 * 256 + ax0_ax1_fused_3 * 8 + ax0_ax1_fused_4) // 64)
                                                    v1 = T.axis.spatial(8192, k_0_0 * 64 + (ax0_ax1_fused_0 * 1024 + ax0_ax1_fused_1 * 1024 + ax0_ax1_fused_2 * 256 + ax0_ax1_fused_3 * 8 + ax0_ax1_fused_4) % 64)
                                                    T.reads(A[v0, v1])
                                                    T.writes(A_shared[v0, v1])
                                                    A_shared[v0, v1] = A[v0, v1]
                            for ax0_ax1_fused_0 in T.thread_binding(2, thread="threadIdx.y"):
                                for ax0_ax1_fused_1 in T.thread_binding(1, thread="threadIdx.z"):
                                    for ax0_ax1_fused_2 in range(16):
                                        for ax0_ax1_fused_3 in T.thread_binding(32, thread="threadIdx.x"):
                                            for ax0_ax1_fused_4 in T.vectorized(8):
                                                with T.block("B_shared"):
                                                    v0 = T.axis.spatial(8192, j_0_0_0 * 1024 + j_0_0_1 * 128 + (ax0_ax1_fused_0 * 4096 + ax0_ax1_fused_1 * 4096 + ax0_ax1_fused_2 * 256 + ax0_ax1_fused_3 * 8 + ax0_ax1_fused_4) // 64)
                                                    v1 = T.axis.spatial(8192, k_0_0 * 64 + (ax0_ax1_fused_0 * 4096 + ax0_ax1_fused_1 * 4096 + ax0_ax1_fused_2 * 256 + ax0_ax1_fused_3 * 8 + ax0_ax1_fused_4) % 64)
                                                    T.reads(B[v0, v1])
                                                    T.writes(B_shared[v0, v1])
                                                    B_shared[v0, v1] = B[v0, v1]
                            for k_0_1 in range(4):
                                for ax0_0, ax1_0, ax0_1, ax1_1 in T.grid(1, 1, 16, 16):
                                    with T.block("A_shared_wmma.matrix_a"):
                                        v0 = T.axis.spatial(32, i_0_1 * 16 + ax0_0 * 16 + ax0_1)
                                        v1 = T.axis.spatial(8192, k_0_0 * 64 + k_0_1 * 16 + ax1_0 * 16 + ax1_1)
                                        T.reads(A_shared[v0, v1])
                                        T.writes(A_shared_wmma_matrix_a[v0, v1])
                                        A_shared_wmma_matrix_a[v0, v1] = A_shared[v0, v1]
                                for ax0_0, ax1_0, ax0_1, ax1_1 in T.grid(8, 1, 16, 16):
                                    with T.block("B_shared_wmma.matrix_b"):
                                        v0 = T.axis.spatial(8192, j_0_0_0 * 1024 + j_0_0_1 * 128 + ax0_0 * 16 + ax0_1)
                                        v1 = T.axis.spatial(8192, k_0_0 * 64 + k_0_1 * 16 + ax1_0 * 16 + ax1_1)
                                        T.reads(B_shared[v0, v1])
                                        T.writes(B_shared_wmma_matrix_b[v0, v1])
                                        B_shared_wmma_matrix_b[v0, v1] = B_shared[v0, v1]
                                for i_0_2, j_0_2, i_1, j_1, k_1 in T.grid(1, 8, 16, 16, 16):
                                    with T.block("B_update"):
                                        vi = T.axis.spatial(32, i_0_0 * 32 + i_0_1 * 16 + i_0_2 * 16 + i_1)
                                        vj = T.axis.spatial(8192, j_0_0_0 * 1024 + j_0_0_1 * 128 + j_0_1 * 128 + j_0_2 * 16 + j_1)
                                        vk = T.axis.reduce(8192, k_0_0 * 64 + k_0_1 * 16 + k_1)
                                        T.reads(C_wmma_accumulator[vi, vj], A_shared_wmma_matrix_a[vi, vk], B_shared_wmma_matrix_b[vj, vk])
                                        T.writes(C_wmma_accumulator[vi, vj])
                                        C_wmma_accumulator[vi, vj] = C_wmma_accumulator[vi, vj] + A_shared_wmma_matrix_a[vi, vk] * B_shared_wmma_matrix_b[vj, vk]
                        for ax0_0, ax1_0, ax0_1, ax1_1 in T.grid(1, 8, 16, 16):
                            with T.block("C_wmma.accumulator"):
                                v0 = T.axis.spatial(32, i_0_1 * 16 + ax0_0 * 16 + ax0_1)
                                v1 = T.axis.spatial(8192, j_0_0_0 * 1024 + j_0_0_1 * 128 + ax1_0 * 16 + ax1_1)
                                T.reads(C_wmma_accumulator[v0, v1])
                                T.writes(C[v0, v1])
                                C[v0, v1] = C_wmma_accumulator[v0, v1]