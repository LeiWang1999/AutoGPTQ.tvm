import tvm
from tvm.script import tir as T
from tvm import meta_schedule as ms
import os
from tvm.tir.tensor_intrin.cuda import get_wmma_intrin_group


bit = 4
group_stride = 32 * bit // 8
mask = (1 << bit) - 1

M = 16384
N = 16384
K = 16384

# tuning config
target = tvm.target.Target("nvidia/geforce-rtx-3090")
trails = 1000

def cuda_build(mod, target, _params):
    from tvm.driver import build as tvm_build

    with tvm.transform.PassContext(config={"tir.predicate_opt": True}):
        return tvm_build(mod, target=target)

@tvm.script.ir_module
class MyModule:
    @T.prim_func
    def main(a: T.handle, b: T.handle, c: T.handle):
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        A = T.match_buffer(a, [M, K], dtype="float16")
        B = T.match_buffer(b, [N, K // 8 * bit], dtype="int8")
        C = T.match_buffer(c, [M, N], dtype="float16")
        B_decompress = T.alloc_buffer([N, K], dtype="float16")
        
        for i, j  in T.grid(N, K):
            with T.block("B_decompress"):
                vi, vj = T.axis.remap("SS", [i, j])
                B_decompress[vi, vj] = T.Select(((vj % 32) * bit) % 8 <= 5, ((B[vi, (vj // 32) * group_stride +  (vj % 32)* bit // 8] >> (((vj % 32) * bit) % 8) & mask)).astype("float16"), (((B[vi, (vj // 32) * group_stride +  (vj % 32)* bit // 8] >> ((vj % 32) * bit) % 8) & (1 << (8 - ((vj % 32) * bit) % 8)) - 1).astype("int8") | ((B[vi, (vj // 32) * group_stride +  (vj % 32)* bit // 8 + 1] << (8 - ((vj % 32) * bit) % 8)) & (mask << (8 - ((vj % 32) * bit) % 8)) & mask).astype("int8")).astype("float16")) 

        for i, j, k in T.grid(M, N, K):
            with T.block("B"):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                with T.init():
                    C[vi, vj] = T.float16(0)
                C[vi, vj] = C[vi, vj] + \
                    A[vi, vk].astype("float16") * B_decompress[vj, vk].astype("float16")

workdir = "./logs/gemm_M16N16384K16384_4b"

database = ms.database.JSONDatabase(work_dir=workdir)
mod = MyModule
sch = ms.tir_integration.compile_tir(database=database, mod=mod, target=target)
if sch is None:
    print("No valid schedule found!")
    exit()
