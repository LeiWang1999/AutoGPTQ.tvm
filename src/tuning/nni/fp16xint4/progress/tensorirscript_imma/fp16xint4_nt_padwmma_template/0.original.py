# from tvm.script import tir as T

@T.prim_func
def main(A: T.Buffer((16384, 16384), "float16"), B: T.Buffer((16384, 8192), "int8"), C: T.Buffer((16384, 16384), "float16")):
    T.func_attr({"global_symbol": "main", "tir.noalias": True})
    # with T.block("root"):
    B_decompress = T.alloc_buffer((16384, 16384), "float16")
    for i, j in T.grid(16384, 16384):
        with T.block("B_decompress"):
            vi, vj = T.axis.remap("SS", [i, j])
            T.reads(B[vi, vj // 2:vj // 2 + 2])
            T.writes(B_decompress[vi, vj])
            B_decompress[vi, vj] = T.Select(vj % 32 * 4 % 8 <= 5, T.Cast("float16", T.bitwise_and(T.shift_right(T.Cast("int32", B[vi, vj // 32 * 16 + vj % 32 * 4 // 8]), vj % 32 * 4 % 8), 15)), T.Cast("float16", T.bitwise_or(T.Cast("int8", T.bitwise_and(T.shift_right(T.Cast("int32", B[vi, vj // 32 * 16 + vj % 32 * 4 // 8]), vj % 32 * 4 % 8), T.shift_left(1, 8 - vj % 32 * 4 % 8) - 1)), T.Cast("int8", T.bitwise_and(T.bitwise_and(T.shift_left(T.Cast("int32", B[vi, vj // 32 * 16 + vj % 32 * 4 // 8 + 1]), 8 - vj % 32 * 4 % 8), T.shift_left(15, 8 - vj % 32 * 4 % 8)), 15)))))
    for i, j, k in T.grid(16384, 16384, 16384):
        with T.block("B"):
            vi, vj, vk = T.axis.remap("SSR", [i, j, k])
            T.reads(A[vi, vk], B_decompress[vj, vk])
            T.writes(C[vi, vj])
            with T.init():
                C[vi, vj] = T.float16(0)
            C[vi, vj] = C[vi, vj] + A[vi, vk] * B_decompress[vj, vk]