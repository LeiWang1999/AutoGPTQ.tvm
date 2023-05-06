# from tvm.script import tir as T

@T.prim_func
def main(A: T.Buffer((16384, 16384), "float16"), B: T.Buffer((16384, 16384), "float16"), C: T.Buffer((16384, 16384), "float16")):
    T.func_attr({"global_symbol": "main", "tir.noalias": True})
    # with T.block("root"):
    for i_0 in T.thread_binding(256, thread="blockIdx.y"):
        for j_0 in T.thread_binding(256, thread="blockIdx.x"):
            for i_1_0 in T.thread_binding(4, thread="threadIdx.z"):
                for j_1_0 in T.thread_binding(16, thread="threadIdx.y"):
                    for k_0, k_1, i_1_1, j_1_1 in T.grid(1024, 16, 16, 4):
                        with T.block("B"):
                            vi = T.axis.spatial(16384, i_0 * 64 + i_1_0 * 16 + i_1_1)
                            vj = T.axis.spatial(16384, j_0 * 64 + j_1_0 * 4 + j_1_1)
                            vk = T.axis.reduce(16384, k_0 * 16 + k_1)
                            T.reads(A[vi, vk], B[vj, vk])
                            T.writes(C[vi, vj])
                            with T.init():
                                C[vi, vj] = T.float16(0)
                            C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vj, vk]