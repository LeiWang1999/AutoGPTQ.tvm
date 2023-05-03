"""
    Considering a gemm problem, in this part we try to leverage the ldmatrix, mma, and stmatrix to do the computation.
    The ldmatrix and stmatrix are used to load and store the data from global memory to shared memory.
    The mma is used to do the computation.
    thread_x will be set into 32, which represents the number of threads in a warp.
    thread_y and thread_z will be set into value which represents the array of warps. 
"""
import tvm
from tvm.script import tir as T
import numpy as np
import os
from tvm.tir.tensor_intrin.cuda import (
    LDMATRIX_16x16_A_INTRIN,
    LDMATRIX_16x16_B_INTRIN,
    LDMATRIX_16x16_B_TRANS_INTRIN,
    LDMATRIX_16x32_A_INTRIN,
    LDMATRIX_32x16_B_INTRIN,
    LDMATRIX_16x32_B_TRANS_INTRIN,
    MMA_f16f16f32_INTRIN,
    MMA_f16f16f32_TRANS_INTRIN,
    MMA_f16f16f16_INTRIN,
    MMA_f16f16f16_TRANS_INTRIN,
    MMA_i8i8i32_INTRIN,
    MMA_i8i8i32_TRANS_INTRIN,
    MMA_fill_16x16_f32_INTRIN,
    MMA_fill_16x16_f16_INTRIN,
    MMA_fill_16x16_i32_INTRIN,
    MMA_store_16x16_f32_global_INTRIN,
    MMA_store_16x16_f16_global_INTRIN,
    MMA_store_16x16_i32_global_INTRIN,
    shared_16x16_to_ldmatrix_32x8_layout,
    shared_32x16_to_ldmatrix_32x16_layout,
    shared_16x32_to_ldmatrix_32x16_layout,
)
import nni

# get file name and remove the suffix
fname = os.path.basename(__file__)
fname = os.path.splitext(fname)[0]
# create log path
log_path = "progress/tensorirscript_imma/4bit_gemm/" + fname
count = 0
def write_code(code, path, fname):
    global count
    # if path not exist, create it
    fname = str(count) + "." + fname
    count += 1
    if not os.path.exists(path):
        os.makedirs(path)
    # join path and fname
    fname = os.path.join(path, fname)
    with open(fname, "w") as f:
        f.write(code)

def write_sch(sch, path, fname):
    py_fname = fname + ".py"
    write_code(sch.mod["main"].script(), path, py_fname)
    cu_fname = fname + ".cu"
    write_code(sch.mod.astext(), path, cu_fname)


VERIFY = False

bit = 4
group_stride = 32 * bit // 8
mask = (1 << bit) - 1

M = 16384
N = 16384
K = 16384
if VERIFY:
    M = 256
    N = 256
    K = 256

params = nni.get_next_parameters()
BM = params['BM']
BN = params['BN']
BK = params['BK']
block_row_warps = params['block_row_warps']
block_col_warps = params['block_col_warps']
raster = params['raster']
warp_size = 32

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


ir_module = MyModule
print(ir_module)
sch = tvm.tir.Schedule(ir_module, debug_mask="all")
write_sch(sch, log_path, "original")

block_b = sch.get_block("B")
block_b_decompress = sch.get_block("B_decompress")

write_sch(sch, log_path, "cache_related")

(i, j, k) = sch.get_loops(block_b)
by, i = sch.split(i, factors=[None, BM])
bx, j = sch.split(j, factors=[None, BN])
bk, k = sch.split(k, factors=[None, BK])

write_sch(sch, log_path, "split_inner_loops")

sch.reorder(by, bx, bk, i, j, k)
write_sch(sch, log_path, "reorder_inner_loops")

sch.bind(bx, "blockIdx.x")
sch.bind(by, "blockIdx.y")

write_sch(sch, log_path, "block_bind")

block_b_tz, block_b_inner_i = sch.split(
    i, factors=[block_row_warps, None])

block_b_ty, block_b_inner_j = sch.split(
    j, factors=[block_col_warps, None])

sch.reorder(block_b_tz, block_b_ty, bk, block_b_inner_i, block_b_inner_j, k)

write_sch(sch, log_path, "split_outer_loops")

sch.bind(block_b_tz, "threadIdx.z")
sch.bind(block_b_ty, "threadIdx.y")

write_sch(sch, log_path, "thread_bind")

# schdule the shared memory

def fetch_to_shared(block, idx, vector_size=4):
    block_read = sch.cache_read(block, idx, "shared")
    sch.compute_at(block_read, bk)
    fused = sch.fuse(*sch.get_loops(block_read)[-2:])
    _, f_0, f_1, f_2, f_3 = sch.split(
        fused, factors=[None, block_row_warps, block_col_warps, warp_size, vector_size])
    sch.bind(f_2, "threadIdx.x")
    sch.bind(f_1, "threadIdx.y")
    sch.bind(f_0, "threadIdx.z")
    sch.vectorize(f_3)
    offset = 8
    sch.storage_align(block_read, 0, axis=-2, factor=32, offset=offset)

# schedule A
fetch_to_shared(block_b, 0, 8)
# schedule B
fetch_to_shared(block_b, 1, 4)
write_sch(sch, log_path, "shared_memory_schedule")

sch.compute_inline(block_b_decompress)

# blockize for mma tensorize

mma_m = 16
mma_n = 16
mma_k = 16

block_b_inner_i, block_b_inner_i_tc = sch.split(
    block_b_inner_i, factors=[None, mma_m])
block_b_inner_j, block_b_inner_j_tc = sch.split(
    block_b_inner_j, factors=[None, mma_n])
k, k_tc = sch.split(k, factors=[None, mma_k])

sch.reorder(block_b_inner_i, block_b_inner_j,
            k, block_b_inner_i_tc, block_b_inner_j_tc, k_tc)

write_sch(sch, log_path, "mma_tile")

# block_inner = sch.blockize(block_b_inner_i_tc)
# block_outer, block_inner = block_inner, block_b
write_sch(sch, log_path, "blockize")

A_warp = sch.cache_read(block_b, 0, "warp")
B_warp = sch.cache_read(block_b, 1, "warp")
sch.compute_at(A_warp, k)
sch.compute_at(B_warp, k)
C_warp = sch.cache_write(block_b, 0, "warp")
sch.reverse_compute_at(C_warp, block_b_ty)
write_sch(sch, log_path, "cache_read_write_warp")

ii, jj = sch.get_loops(C_warp)[-2:]
io, ii = sch.split(ii, factors=[None, mma_m])
jo, ji = sch.split(jj, factors=[None, mma_n])
sch.reorder(io, jo, ii, ji)


def tile_wmma_fragment(block_read, height, width):
    i, j = sch.get_loops(block_read)[-2:]
    # i0, i1 = sch.split(i, factors=[None, height])
    # j0, j1 = sch.split(j, factors=[None, width])
    # sch.reorder(i0, j0, i1, j1)
    return i

loop_a = tile_wmma_fragment(A_warp, mma_m, mma_k)

loop_b = tile_wmma_fragment(B_warp, mma_n, mma_k)

write_sch(sch, log_path, "tile_fragment")


block_init_c = sch.decompose_reduction(
    block_b, bk)
write_sch(sch, log_path, "decompose_reduction")

def index_map_A(i, j):
    return (
            i // 16,
            j // 16,
            *shared_16x16_to_ldmatrix_32x8_layout(i % 16, j % 16),
        )

def index_map_B(i, j):
    return (
            i // 16,
            j // 16,
            *shared_16x16_to_ldmatrix_32x8_layout(i % 16, j % 16),
        )

def index_map_C(i, j):
    return (
            i // 16,
            j // 16,
            *shared_16x16_to_ldmatrix_32x8_layout(i % 16, j % 16),
        )



sch.transform_layout(A_warp, ("write", 0), index_map_A)
sch.transform_layout(B_warp, ("write", 0), index_map_A)
sch.transform_layout(C_warp, ("read", 0), index_map_C)

write_sch(sch, log_path, "transform_layout")

init_block_b_i, init_block_b_j = sch.get_loops(block_init_c)[-4:-2]
sch.tensorize(loop_a, LDMATRIX_16x16_A_INTRIN)
sch.tensorize(loop_b, LDMATRIX_16x16_B_TRANS_INTRIN)
write_sch(sch, log_path, "tensorize_ldmatrix")

sch.tensorize(block_b_inner_i_tc, MMA_f16f16f16_TRANS_INTRIN)
write_sch(sch, log_path, "tensorize_mma_sync")

sch.tensorize(sch.get_loops(block_init_c)[-2], MMA_fill_16x16_f16_INTRIN)
write_sch(sch, log_path, "tensorize_mma_fill")

sch.tensorize(sch.get_loops(C_warp)[-2], MMA_store_16x16_f16_global_INTRIN)
write_sch(sch, log_path, "tensorize_store")

if raster > 0:
    sch.annotate(init_block_b_i,
                 ann_key="thread_rasterization", ann_val=raster)

@tvm.register_func
def tvm_callback_cuda_postproc(code):
    code = code.replace("#define TVM_ENBALE_EFFICIENT_SMEM_PTR_CAST 1", "#define TVM_ENBALE_EFFICIENT_SMEM_PTR_CAST 0")
    # print(code)
    return code

ctx = tvm.cuda(0)
cuda_mod = tvm.build(sch.mod, target="cuda")

write_code(cuda_mod.imported_modules[0].get_source(), log_path, "tmp.cu")

def bit_compress(x, bits, axis):
    # given a tensor x (M, K), which only the low bits bits have value, we can compress it to (M, K // 8 * bits)
    shape = x.shape
    compress_shape = shape[:axis] + (shape[axis] // 8 * bits,) + shape[axis + 1:]
    _compressed = np.zeros(compress_shape).astype("int8")
    mask = (1 << bits) - 1
    for i in range(shape[axis]):
        val = (x[..., i] & mask).astype("int8")
        # _compressed is a int8 dtype which can store 1 or more low bits value, like if we have 4 bits, we can store 2 values in one int8
        _compressed[..., i // (8 // bits)] |= (val << ((i % (8 // bits)) * bits)).astype("int8")
    return _compressed

np.random.seed(0)
a_raw = (np.random.rand(M, K)).astype("float16")
b_raw = (np.arange(N * K) % 4).reshape((N, K)).astype("int8")
b_compressed = bit_compress(b_raw, bit, 1)

cuda_a = tvm.nd.array((a_raw).astype("float16"), ctx)
cuda_b = tvm.nd.array((b_compressed).astype("int8"), ctx)
cuda_c = tvm.nd.array((np.zeros((M, N))).astype("float16"), ctx)
cuda_mod(cuda_a, cuda_b, cuda_c)

if VERIFY:
    cuda_mod(cuda_a, cuda_b, cuda_c)
    c_np = np.matmul(a_raw, b_raw.T)
    tvm_c_np = cuda_c.numpy()
    print(c_np[0:10])
    print(tvm_c_np[0:10])
    # np.testing.assert_allclose(
    #     c_np, tvm_c_np, rtol=1e-1, atol=1e-1
    # )

num_flops = 2 * M * K * N
num_runs = 3
timer_cuda_mod = cuda_mod.time_evaluator(
    cuda_mod.entry_name, ctx, number=num_runs)

t = timer_cuda_mod(cuda_a, cuda_b, cuda_c).mean

GFLOPS = num_flops / (t * 1e3) / 1e6
print("average time cost of %d runs = %g ms, %g GFLOPS." %
      (num_runs, t * 1e3, GFLOPS))

nni.report_final_result(t * 1e3)
