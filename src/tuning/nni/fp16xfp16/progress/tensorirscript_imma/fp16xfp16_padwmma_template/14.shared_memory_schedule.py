# from tvm.script import tir as T

@T.prim_func
def main(A: T.Buffer((16384, 16384), "float16"), B: T.Buffer((16384, 16384), "float16"), C: T.Buffer((16384, 16384), "float16")):
    T.func_attr({"global_symbol": "main", "tir.noalias": True})
    # with T.block("root"):
    A_shared = T.alloc_buffer((16384, 16384), "float16", scope="shared")
    B_shared = T.alloc_buffer((16384, 16384), "float16", scope="shared")
    for i_0 in T.thread_binding(256, thread="blockIdx.y"):
        for j_0 in T.thread_binding(256, thread="blockIdx.x"):
            for i_1_0 in T.thread_binding(4, thread="threadIdx.z"):
                for j_1_0 in T.thread_binding(16, thread="threadIdx.y"):
                    for k_0 in range(1024):
                        for ax0_ax1_fused_0 in range(1):
                            for ax0_ax1_fused_1 in T.thread_binding(4, thread="threadIdx.z"):
                                for ax0_ax1_fused_2 in T.thread_binding(16, thread="threadIdx.y"):
                                    for ax0_ax1_fused_3 in T.thread_binding(32, thread="threadIdx.x"):
                                        for ax0_ax1_fused_4 in T.vectorized(8):
                                            with T.block("A_shared"):
                                                v0 = T.axis.spatial(16384, i_0 * 64 + (ax0_ax1_fused_0 * 16384 + ax0_ax1_fused_1 * 4096 + ax0_ax1_fused_2 * 256 + ax0_ax1_fused_3 * 8 + ax0_ax1_fused_4) // 16)
                                                v1 = T.axis.spatial(16384, k_0 * 16 + (ax0_ax1_fused_0 * 16384 + ax0_ax1_fused_1 * 4096 + ax0_ax1_fused_2 * 256 + ax0_ax1_fused_3 * 8 + ax0_ax1_fused_4) % 16)
                                                T.where((((ax0_ax1_fused_0 * 4 + ax0_ax1_fused_1) * 16 + ax0_ax1_fused_2) * 32 + ax0_ax1_fused_3) * 8 + ax0_ax1_fused_4 < 1024)
                                                T.reads(A[v0, v1])
                                                T.writes(A_shared[v0, v1])
                                                T.block_attr({"buffer_dim_align": [[0, 0, 32, 8]]})
                                                A_shared[v0, v1] = A[v0, v1]
                        for ax0_ax1_fused_0 in range(1):
                            for ax0_ax1_fused_1 in T.thread_binding(4, thread="threadIdx.z"):
                                for ax0_ax1_fused_2 in T.thread_binding(16, thread="threadIdx.y"):
                                    for ax0_ax1_fused_3 in T.thread_binding(32, thread="threadIdx.x"):
                                        for ax0_ax1_fused_4 in T.vectorized(8):
                                            with T.block("B_shared"):
                                                v0 = T.axis.spatial(16384, j_0 * 64 + (ax0_ax1_fused_0 * 16384 + ax0_ax1_fused_1 * 4096 + ax0_ax1_fused_2 * 256 + ax0_ax1_fused_3 * 8 + ax0_ax1_fused_4) // 16)
                                                v1 = T.axis.spatial(16384, k_0 * 16 + (ax0_ax1_fused_0 * 16384 + ax0_ax1_fused_1 * 4096 + ax0_ax1_fused_2 * 256 + ax0_ax1_fused_3 * 8 + ax0_ax1_fused_4) % 16)
                                                T.where((((ax0_ax1_fused_0 * 4 + ax0_ax1_fused_1) * 16 + ax0_ax1_fused_2) * 32 + ax0_ax1_fused_3) * 8 + ax0_ax1_fused_4 < 1024)
                                                T.reads(B[v0, v1])
                                                T.writes(B_shared[v0, v1])
                                                T.block_attr({"buffer_dim_align": [[0, 0, 32, 8]]})
                                                B_shared[v0, v1] = B[v0, v1]
                        for k_1, i_1_1, j_1_1 in T.grid(16, 16, 4):
                            with T.block("B"):
                                vi = T.axis.spatial(16384, i_0 * 64 + i_1_0 * 16 + i_1_1)
                                vj = T.axis.spatial(16384, j_0 * 64 + j_1_0 * 4 + j_1_1)
                                vk = T.axis.reduce(16384, k_0 * 16 + k_1)
                                T.reads(A_shared[vi, vk], B_shared[vj, vk])
                                T.writes(C[vi, vj])
                                with T.init():
                                    C[vi, vj] = T.float16(0)
                                C[vi, vj] = C[vi, vj] + A_shared[vi, vk] * B_shared[vj, vk]