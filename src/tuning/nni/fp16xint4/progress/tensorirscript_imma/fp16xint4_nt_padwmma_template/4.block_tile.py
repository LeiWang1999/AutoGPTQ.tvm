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
        with T.block("B_decompress_shared"):
            v0, v1 = T.axis.remap("SS", [ax0, ax1])
            T.reads(B[v0, v1 // 2:v1 // 2 + 2])
            T.writes(B_decompress_shared[v0, v1])
            B_decompress_shared[v0, v1] = T.Select(v1 % 32 * 4 % 8 <= 5, T.Cast("float16", T.bitwise_and(T.shift_right(T.Cast("int32", B[v0, v1 // 32 * 16 + v1 % 32 * 4 // 8]), v1 % 32 * 4 % 8), 15)), T.Cast("float16", T.bitwise_or(T.Cast("int8", T.bitwise_and(T.shift_right(T.Cast("int32", B[v0, v1 // 32 * 16 + v1 % 32 * 4 // 8]), v1 % 32 * 4 % 8), T.shift_left(1, 8 - v1 % 32 * 4 % 8) - 1)), T.Cast("int8", T.bitwise_and(T.bitwise_and(T.shift_left(T.Cast("int32", B[v0, v1 // 32 * 16 + v1 % 32 * 4 // 8 + 1]), 8 - v1 % 32 * 4 % 8), T.shift_left(15, 8 - v1 % 32 * 4 % 8)), 15)))))
    for ax0, ax1 in T.grid(16384, 16384):
        with T.block("B_decompress_shared_wmma.matrix_b"):
            v0, v1 = T.axis.remap("SS", [ax0, ax1])
            T.reads(B_decompress_shared[v0, v1])
            T.writes(B_decompress_shared_wmma_matrix_b[v0, v1])
            B_decompress_shared_wmma_matrix_b[v0, v1] = B_decompress_shared[v0, v1]
    for j_0_0_1, i_0_0, j_0_0_0, i_0_1, j_0_1, k_0_0, k_0_1, i_0_2, j_0_2, i_1, j_1, k_1 in T.grid(8, 1024, 16, 1, 1, 256, 4, 1, 8, 16, 16, 16):
        with T.block("B"):
            vi = T.axis.spatial(16384, i_0_0 * 16 + i_0_1 * 16 + i_0_2 * 16 + i_1)
            vj = T.axis.spatial(16384, j_0_0_0 * 1024 + j_0_0_1 * 128 + j_0_1 * 128 + j_0_2 * 16 + j_1)
            vk = T.axis.reduce(16384, k_0_0 * 64 + k_0_1 * 16 + k_1)
            T.reads(A_shared_wmma_matrix_a[vi, vk], B_decompress_shared_wmma_matrix_b[vj, vk])
            T.writes(C_wmma_accumulator[vi, vj])
            with T.init():
                C_wmma_accumulator[vi, vj] = T.float16(0)
            C_wmma_accumulator[vi, vj] = C_wmma_accumulator[vi, vj] + A_shared_wmma_matrix_a[vi, vk] * B_decompress_shared_wmma_matrix_b[vj, vk]
    for ax0, ax1 in T.grid(16384, 16384):
        with T.block("C_wmma.accumulator"):
            v0, v1 = T.axis.remap("SS", [ax0, ax1])
            T.reads(C_wmma_accumulator[v0, v1])
            T.writes(C[v0, v1])
            C[v0, v1] = C_wmma_accumulator[v0, v1]