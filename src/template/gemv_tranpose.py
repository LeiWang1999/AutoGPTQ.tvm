import tvm
import numpy as np
import tvm.testing
from tvm.script import tir as T
import os

bit = 3

# get file name and remove the suffix
fname = os.path.basename(__file__)
fname = os.path.splitext(fname)[0]
# create log path
log_path = f"progress/tensorirscript_simt/halfx{bit}bit_gemv/" + fname
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


group_stride = 32 * bit // 8
M = 1
N = 9216
K = 36864
mask = (1 << bit) - 1
vec = 8 if bit == 3 else 8
num_warps = 4
warp_size = 32


@tvm.script.ir_module
class MyModule:
    @T.prim_func
    def main(a: T.handle, b: T.handle, c: T.handle, scales: T.handle, zeros: T.handle):
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        A = T.match_buffer(a, [M, K], dtype="float16")
        B = T.match_buffer(b, [N, K // 8 * bit], dtype="int8")
        C = T.match_buffer(c, [M, N], dtype="float16")
        Scales = T.match_buffer(scales, [N], dtype="float16")
        Zeros = T.match_buffer(zeros, [N], dtype="float16")
        
        B_decompress = T.alloc_buffer([N, K], dtype="float16")
        B_rescale= T.alloc_buffer([N, K], dtype="float16")

        for i, j  in T.grid(N, K):
            with T.block("B_decompress"):
                vi, vj = T.axis.remap("SS", [i, j])
                B_decompress[vi, vj] = T.Select(((vj % 32) * bit) % 8 <= 5, ((B[vi, (vj // 32) * group_stride +  (vj % 32)* bit // 8] >> (((vj % 32) * bit) % 8) & mask)).astype("float16"), (((B[vi, (vj // 32) * group_stride +  (vj % 32)* bit // 8] >> ((vj % 32) * bit) % 8) & (1 << (8 - ((vj % 32) * bit) % 8)) - 1).astype("int8") | ((B[vi, (vj // 32) * group_stride +  (vj % 32)* bit // 8 + 1] << (8 - ((vj % 32) * bit) % 8)) & (mask << (8 - ((vj % 32) * bit) % 8)) & mask).astype("int8")).astype("float16")) 
        
        for i, j in T.grid(N, K):
            with T.block("B_rescale"):
                vi, vj = T.axis.remap("SS", [i, j])
                B_rescale[vi, vj] = B_decompress[vi, vj] * Scales[vi].astype('float16') - Zeros[vi].astype('float16') 
        
        for i, j, k in T.grid(M, N, K):
            with T.block("B"):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                with T.init():
                    C[vi, vj] = 0.0
                C[vi, vj] = C[vi, vj] + \
                    A[vi, vk].astype("float16") * \
                    B_rescale[vj, vk].astype("float16")
                    

ir_module = MyModule
sch = tvm.tir.Schedule(ir_module, debug_mask="all")
print(ir_module)

block_b = sch.get_block("B")
block_b_decompress = sch.get_block("B_decompress")
block_b_rescale = sch.get_block("B_rescale")
sch.compute_inline(block_b_decompress)
sch.compute_inline(block_b_rescale)

i, j, k = sch.get_loops(block_b)
block_shared_local_A = sch.cache_read(block_b, 0, "local")
# block_shared_local_B = sch.cache_read(block_b, 1, "local")
block_local_C = sch.cache_write(block_b, 0, "local")
write_sch(sch, log_path, "cache_related")

bx, j = sch.split(
    j, factors=[None, num_warps])
k, tx, vk = sch.split(k, factors=[None, warp_size, vec])
sch.reorder(bx, j, i, k, tx)

sch.bind(bx, "blockIdx.x")
sch.bind(tx, "threadIdx.x")
sch.bind(j, "threadIdx.y")
write_sch(sch, log_path, "do_split")

sch.compute_at(block_shared_local_A, tx, preserve_unit_loops=True)
# sch.compute_at(block_shared_local_B, tx, preserve_unit_loops=True)
sch.reverse_compute_at(block_local_C, j, preserve_unit_loops=True)
write_sch(sch, log_path, "compute_at_related")


block_local_a_v = sch.get_loops(block_shared_local_A)[-1]
sch.vectorize(block_local_a_v)

# block_local_b_v = sch.get_loops(block_shared_local_B)[-1]
# sch.vectorize(block_local_b_v)

# sch.decompose_reduction(block_b, k)
write_sch(sch, log_path, "decompose_reduction")

ctx = tvm.cuda(0)
cuda_mod = tvm.build(sch.mod, target="cuda")

code = cuda_mod.imported_modules[0].get_source()
code = code.replace(
    "main_kernel0", f"tir_halfxint{bit}_simt_bn{num_warps}_k{K}")

write_code(code, log_path, "tmp.cu")

def bit_compress(x:np.ndarray, bits, axis):
    # given a tensor x (M, K), which only the low bits bits have value, we can compress it to (M, K // 8 * bits)
    
    shape = x.shape
    compress_shape = shape[:axis] + (shape[axis] // 8 * bits,) + shape[axis + 1:]
    if bits == 4:
        _compressed = np.zeros(compress_shape).astype("int8")
        mask = (1 << bits) - 1
        for i in range(shape[axis]):
            val = (x[..., i] & mask).astype("int8")
            # _compressed is a int8 dtype which can store 1 or more low bits value, like if we have 4 bits, we can store 2 values in one int8
            _compressed[..., i // (8 // bits)] |= (val << ((i % (8 // bits)) * bits)).astype("int8")
    elif bits == 3:
        intweight = x.T.astype(np.uint32)
        qweight = np.zeros(
           (intweight.shape[0] // 32 * 3, intweight.shape[1]), dtype=np.uint32
        )
        
        i = 0
        row = 0
        while row < qweight.shape[0]:
            for j in range(i, i + 10):
                qweight[row] |= intweight[j] << (3 * (j - i))
            i += 10
            qweight[row] |= intweight[i] << 30
            row += 1
            qweight[row] |= (intweight[i] >> 2) & 1
            i += 1
            for j in range(i, i + 10):
                qweight[row] |= intweight[j] << (3 * (j - i) + 1)
            i += 10
            qweight[row] |= intweight[i] << 31
            row += 1
            qweight[row] |= (intweight[i] >> 1) & 0x3
            i += 1
            for j in range(i, i + 10):
                qweight[row] |= intweight[j] << (3 * (j - i) + 2)
            i += 10
            row += 1

        qweight = qweight.astype(np.int32)
        _compressed = np.ascontiguousarray(qweight.transpose())
        _compressed = _compressed.view(np.int8)
        
    return _compressed

    
# a_raw = (np.arange(M * K) % 16).reshape((M, K)).astype("float16")
# a_raw = np.random.rand(M, K).astype("float16")
a_raw = np.ones((M, K)).astype("float16")
b_raw = np.ones((N, K)).astype("int8")
# b_raw = (np.arange(N * K) % 4).reshape((N, K)).astype("int8")
scales = np.ones(N).astype("float16")
zeros = np.zeros(N).astype("float16")

c_np = np.matmul(a_raw, b_raw.T)
print(a_raw)
b_compressed = bit_compress(b_raw, bit, 1)
# b_compressed = np.random.randn(N, K // 8 * bit).astype("int8")
print(b_compressed.shape)

cuda_a = tvm.nd.array((a_raw).astype("float16"), ctx)
cuda_b = tvm.nd.array((b_compressed).astype("int8"), ctx)
cuda_s = tvm.nd.array((scales).astype("float16"), ctx)
cuda_z = tvm.nd.array((zeros).astype("float16"), ctx)
cuda_c = tvm.nd.array((np.zeros((M, N))).astype("float16"), ctx)
cuda_mod(cuda_a, cuda_b, cuda_c, cuda_s, cuda_z)

print("np c is ", c_np)
print("tvm c is ", cuda_c)



num_runs = 3
timer_cuda_mod = cuda_mod.time_evaluator(
    cuda_mod.entry_name, ctx, number=num_runs)

t = timer_cuda_mod(cuda_a, cuda_b, cuda_c, cuda_s, cuda_z).mean

print("average time cost of %d runs = %g ms." %
      (num_runs, t * 1e3))
