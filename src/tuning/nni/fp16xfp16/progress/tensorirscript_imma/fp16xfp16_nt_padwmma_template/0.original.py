# from tvm.script import tir as T

@T.prim_func
def main(A: T.Buffer((128, 8192), "float16"), B: T.Buffer((8192, 8192), "float16"), C: T.Buffer((128, 8192), "float16")):
    T.func_attr({"global_symbol": "main", "tir.noalias": True})
    # with T.block("root"):
    for i, j, k in T.grid(128, 8192, 8192):
        with T.block("B"):
            vi, vj, vk = T.axis.remap("SSR", [i, j, k])
            T.reads(A[vi, vk], B[vj, vk])
            T.writes(C[vi, vj])
            with T.init():
                C[vi, vj] = T.float16(0)
            C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vj, vk]