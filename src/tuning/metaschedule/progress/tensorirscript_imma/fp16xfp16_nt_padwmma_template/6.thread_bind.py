# from tvm.script import tir as T

@T.prim_func
def main(A: T.Buffer((16384, 16384), "float16"), B: T.Buffer((16384, 16384), "float16"), C: T.Buffer((16384, 16384), "float16")):
    T.func_attr({"global_symbol": "main", "tir.noalias": True})
    # with T.block("root"):
    A_shared = T.alloc_buffer((16384, 16384), "float16", scope="shared")
    A_shared_wmma_matrix_a = T.alloc_buffer((16384, 16384), "float16", scope="wmma.matrix_a")
    B_shared = T.alloc_buffer((16384, 16384), "float16", scope="shared")
    B_shared_wmma_matrix_b = T.alloc_buffer((16384, 16384), "float16", scope="wmma.matrix_b")
    C_wmma_accumulator = T.alloc_buffer((16384, 16384), "float16", scope="wmma.accumulator")
    for ax0, ax1 in T.grid(16384, 16384):
        with T.block("B_shared"):
            v0, v1 = T.axis.remap("SS", [ax0, ax1])
            T.reads(B[v0, v1])
            T.writes(B_shared[v0, v1])
            B_shared[v0, v1] = B[v0, v1]
    for ax0, ax1 in T.grid(16384, 16384):
        with T.block("A_shared"):
            v0, v1 = T.axis.remap("SS", [ax0, ax1])
            T.reads(A[v0, v1])
            T.writes(A_shared[v0, v1])
            A_shared[v0, v1] = A[v0, v1]
    for ax0, ax1 in T.grid(16384, 16384):
        with T.block("A_shared_wmma.matrix_a"):
            v0, v1 = T.axis.remap("SS", [ax0, ax1])
            T.reads(A_shared[v0, v1])
            T.writes(A_shared_wmma_matrix_a[v0, v1])
            A_shared_wmma_matrix_a[v0, v1] = A_shared[v0, v1]
    for ax0, ax1 in T.grid(16384, 16384):
        with T.block("B_shared_wmma.matrix_b"):
            v0, v1 = T.axis.remap("SS", [ax0, ax1])
            T.reads(B_shared[v0, v1])
            T.writes(B_shared_wmma_matrix_b[v0, v1])
            B_shared_wmma_matrix_b[v0, v1] = B_shared[v0, v1]
    for j_0_0_1 in T.thread_binding(16, thread="blockIdx.z"):
        for i_0_0 in T.thread_binding(128, thread="blockIdx.y"):
            for j_0_0_0 in T.thread_binding(4, thread="blockIdx.x"):
                for i_0_1 in T.thread_binding(2, thread="threadIdx.y"):
                    for j_0_1 in T.thread_binding(2, thread="threadIdx.z"):
                        for k_0_0, k_0_1, i_0_2, j_0_2, i_1, j_1, k_1 in T.grid(512, 2, 4, 8, 16, 16, 16):
                            with T.block("B"):
                                vi = T.axis.spatial(16384, i_0_0 * 128 + i_0_1 * 64 + i_0_2 * 16 + i_1)
                                vj = T.axis.spatial(16384, j_0_0_0 * 4096 + j_0_0_1 * 256 + j_0_1 * 128 + j_0_2 * 16 + j_1)
                                vk = T.axis.reduce(16384, k_0_0 * 32 + k_0_1 * 16 + k_1)
                                T.reads(A_shared_wmma_matrix_a[vi, vk], B_shared_wmma_matrix_b[vj, vk])
                                T.writes(C_wmma_accumulator[vi, vj])
                                with T.init():
                                    C_wmma_accumulator[vi, vj] = T.float16(0)
                                C_wmma_accumulator[vi, vj] = C_wmma_accumulator[vi, vj] + A_shared_wmma_matrix_a[vi, vk] * B_shared_wmma_matrix_b[vj, vk]
    for ax0, ax1 in T.grid(16384, 16384):
        with T.block("C_wmma.accumulator"):
            v0, v1 = T.axis.remap("SS", [ax0, ax1])
            T.reads(C_wmma_accumulator[v0, v1])
            T.writes(C[v0, v1])
            C[v0, v1] = C_wmma_accumulator[v0, v1]