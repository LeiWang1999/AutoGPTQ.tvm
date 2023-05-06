import tvm
import numpy as np
import tvm.testing
from tvm.script import tir as T
import os
from tvm.tir.tensor_intrin.cuda import (
    WMMA_FILL_16x16x16_F16_INTRIN,
    WMMA_LOAD_16x16x16_F16_A_INTRIN,
    WMMA_LOAD_16x16x16_F16_A_DYN_INTRIN,
    WMMA_LOAD_16x16x16_F16_B_TRANS_INTRIN,
    WMMA_LOAD_16x16x16_F16_B_TRANS_DYN_INTRIN,
    WMMA_SYNC_16x16x16_f16f16f16_TRANS_INTRIN,
    WMMA_STORE_16x16x16_F16_SHARED_INTRIN,
    WMMA_STORE_16x16x16_F16_SHARED_DYN_INTRIN,
    WMMA_STORE_16x16x16_F16_GLOBAL_INTRIN,
)

configurations = [
    {
        "M": 16,
        "block_row_warps": 1,
        "block_col_warps": 4,
        "BM": 16,
        "BN": 64,
        "BK": 64,
        "raster": 0,
        "stage": 3
    },
    # {
    #     "M": 32,
    #     "block_row_warps": 2,
    #     "block_col_warps": 4,
    #     "BM": 32,
    #     "BN": 64,
    #     "BK": 64,
    #     "raster": 0,
    #     "stage": 2
    # }, 
    # {
    #     "M": 64,
    #     "block_row_warps": 2,
    #     "block_col_warps": 4,
    #     "BM": 64,
    #     "BN": 64,
    #     "BK": 64,
    #     "raster": 0,
    #     "stage": 2
    # },
    # {
    #     "M": 128,
    #     "block_row_warps": 1,
    #     "block_col_warps": 2,
    #     "BM": 64,
    #     "BN": 128,
    #     "BK": 32,
    #     "raster": 0,
    #     "stage": 3
    # },
    # {
    #     "M": 256,
    #     "block_row_warps": 2,
    #     "block_col_warps": 2,
    #     "BM": 128,
    #     "BN": 256,
    #     "BK": 32,
    #     "raster": 0,
    #     "stage": 1
    # }
]


for params in configurations:

    VERIFY = True

    bit = 3
    group_stride = 32 * bit // 8
    mask = (1 << bit) - 1
    M = 512
    N = 9216
    K = 768
    if VERIFY:
        M = 512
        N = 1024
        K = 768

    warp_size = 32
    wmma_m = 16
    wmma_n = 16
    wmma_k = 16

    BM = params['BM']
    BN = params['BN']
    BK = params['BK']
    block_row_warps = params['block_row_warps']
    block_col_warps = params['block_col_warps']
    raster = params['raster']
    stage = params['stage']
    warp_row_tiles = BM // (wmma_m * block_row_warps)
    warp_col_tiles = BN // (wmma_n * block_col_warps)
    chunk = BK // (wmma_k)
    vec = 8
    shared_pad = 8

    # get file name and remove the suffix
    fname = f"gemm_fp16xint3_M{params['M']}K{K}_{params['BM']}x{params['BN']}x{params['BK']}"
    # create log path
    log_path = "progress/tensorirscript_imma/" + fname
    kernel_path = f"progress/tensorirscript_imma/3bit_gemm_K{K}/"
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

    def append_code(code, path, fname):
        global count
        # if path not exist, create it
        fname = str(count) + "." + fname
        count += 1
        if not os.path.exists(path):
            os.makedirs(path)
        # join path and fname
        fname = os.path.join(path, fname)
        # append code to fname
        with open(fname, "a") as f:
            f.write(code)

    def write_sch(sch, path, fname):
        py_fname = fname + ".py"
        write_code(sch.mod["main"].script(), path, py_fname)
        cu_fname = fname + ".cu"
        write_code(sch.mod.astext(), path, cu_fname)

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
            B_rescale = T.alloc_buffer([N, K], dtype="float16")

            for i, j in T.grid(N, K):
                with T.block("B_decompress"):
                    vi, vj = T.axis.remap("SS", [i, j])
                    B_decompress[vi, vj] = T.Select(((vj % 32) * bit) % 8 <= 5, ((B[vi, (vj // 32) * group_stride + (vj % 32) * bit // 8] >> (((vj % 32) * bit) % 8) & mask)).astype("float16"), (((B[vi, (vj // 32) * group_stride + (vj % 32) * bit // 8] >> ((vj % 32) * bit) % 8) & (
                        1 << (8 - ((vj % 32) * bit) % 8)) - 1).astype("int8") | ((B[vi, (vj // 32) * group_stride + (vj % 32) * bit // 8 + 1] << (8 - ((vj % 32) * bit) % 8)) & (mask << (8 - ((vj % 32) * bit) % 8)) & mask).astype("int8")).astype("float16"))
            
            for i, j in T.grid(N, K):
                with T.block("B_rescale"):
                    vi, vj = T.axis.remap("SS", [i, j])
                    B_rescale[vi, vj] = B_decompress[vi, vj] * \
                        Scales[vi].astype('float16') - Zeros[vi].astype('float16')
                        
            for i, j, k in T.grid(M, N, K):
                with T.block("B"):
                    vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                    with T.init():
                        C[vi, vj] = T.float16(0)
                    C[vi, vj] = C[vi, vj] + \
                        A[vi, vk].astype("float16") * \
                        B_rescale[vj, vk].astype("float16")


    ir_module = MyModule
    sch = tvm.tir.Schedule(ir_module, debug_mask="all")

    print(type(ir_module))
    print(ir_module.script())

    write_sch(sch, log_path, "original")
    block_b = sch.get_block("B")
    block_b_decompress = sch.get_block("B_decompress")
    block_b_rescale = sch.get_block("B_rescale")

    block_shared_A = sch.cache_read(block_b, 0, "shared")
    block_shared_local_A = sch.cache_read(block_b, 0, "wmma.matrix_a")
    block_shared_B = sch.cache_read(block_b, 1, "shared")
    block_shared_local_B = sch.cache_read(block_b, 1, "wmma.matrix_b")
    block_local_C = sch.cache_write(block_b, 0, "wmma.accumulator")

    sch.compute_inline(block_b_decompress)
    sch.compute_inline(block_b_rescale)

    write_sch(sch, log_path, "cache_related")

    (i, j, k) = sch.get_loops(block_b)
    i, kernel_i = sch.split(i, factors=[None, wmma_m])
    j, kernel_j = sch.split(j, factors=[None, wmma_n])
    k, kernel_k = sch.split(k, factors=[None, wmma_k])
    block_i, i, ii = sch.split(i, factors=[None, block_row_warps, warp_row_tiles])
    block_j, j, jj = sch.split(j, factors=[None, block_col_warps, warp_col_tiles])
    if raster > 0:
        block_j, block_k = sch.split(block_j, factors=[None, raster])
        ko, ki = sch.split(k, factors=[None, chunk])
        sch.reorder(block_k, block_i, block_j, i, j, ko, ki,
                    ii, jj, kernel_i, kernel_j, kernel_k)
        sch.bind(block_k, "blockIdx.z")
        sch.bind(block_i, "blockIdx.y")
        sch.bind(block_j, "blockIdx.x")
        sch.bind(i, "threadIdx.y")
        sch.bind(j, "threadIdx.z")
    else:
        ko, ki = sch.split(k, factors=[None, chunk])
        sch.reorder(block_i, block_j, i, j, ko, ki, ii,
                    jj, kernel_i, kernel_j, kernel_k)
        sch.bind(block_i, "blockIdx.y")
        sch.bind(block_j, "blockIdx.x")
        sch.bind(i, "threadIdx.y")
        sch.bind(j, "threadIdx.z")

    write_sch(sch, log_path, "block_tile")

    write_sch(sch, log_path, "thread_bind")

    # cache read A from global memory to shared_memory
    sch.compute_at(block_shared_local_A, ki)
    sch.compute_at(block_shared_A, ko, preserve_unit_loops=True)
    sch.compute_at(block_shared_local_B, ki)
    sch.compute_at(block_shared_B, ko, preserve_unit_loops=True)
    sch.reverse_compute_at(block_local_C, j)
    write_sch(sch, log_path, "cache_read_compute_at")


    A_shared_fused = sch.fuse(*sch.get_loops(block_shared_A)[-2:])
    A_shared_ty, A_shared_tz, A_shared_inner, A_shared_tx, A_shared_vi = sch.split(
        A_shared_fused, factors=[block_row_warps, block_col_warps, None, warp_size, vec])
    sch.vectorize(A_shared_vi)
    sch.bind(A_shared_tx, "threadIdx.x")
    sch.bind(A_shared_ty, "threadIdx.y")
    sch.bind(A_shared_tz, "threadIdx.z")
    sch.storage_align(block_shared_A, 0, axis=-2, factor=32, offset=shared_pad)
    write_sch(sch, log_path, "schedule_A_shared")

    B_shared_fused = sch.fuse(*sch.get_loops(block_shared_B)[-2:])
    B_shared_ty, B_shared_tz, B_shared_inner, B_shared_tx, B_shared_vi = sch.split(
        B_shared_fused, factors=[block_row_warps, block_col_warps, None, warp_size, 1])
    sch.vectorize(B_shared_vi)
    sch.bind(B_shared_tx, "threadIdx.x")
    sch.bind(B_shared_ty, "threadIdx.y")
    sch.bind(B_shared_tz, "threadIdx.z")
    sch.storage_align(block_shared_B, 0, axis=-2, factor=32, offset=shared_pad)
    write_sch(sch, log_path, "schedule_B_shared")


    A_local_i, A_local_j = sch.get_loops(block_shared_local_A)[-2:]
    A_local_i, A_local_kernel_i = sch.split(A_local_i, factors=[None, wmma_m])
    A_local_j, A_local_kernel_j = sch.split(A_local_j, factors=[None, wmma_k])
    sch.reorder(A_local_i, A_local_j, A_local_kernel_i, A_local_kernel_j)

    B_local_i, B_local_j = sch.get_loops(block_shared_local_B)[-2:]
    B_local_i, B_local_kernel_i = sch.split(B_local_i, factors=[None, wmma_n])
    B_local_j, B_local_kernel_j = sch.split(B_local_j, factors=[None, wmma_k])
    sch.reorder(B_local_i, B_local_j, B_local_kernel_i, B_local_kernel_j)

    C_local_i, C_local_j = sch.get_loops(block_local_C)[-2:]
    C_local_i, C_local_kernel_i = sch.split(C_local_i, factors=[None, wmma_m])
    C_local_j, C_local_kernel_j = sch.split(C_local_j, factors=[None, wmma_n])
    sch.reorder(C_local_i, C_local_j, C_local_kernel_i, C_local_kernel_j)

    # decompose reduction
    init_block_b = sch.decompose_reduction(block_b, ko)
    write_sch(sch, log_path, "decompose_reduction")

    # transpose layout

    init_block_b_i, init_block_b_j = sch.get_loops(init_block_b)[-4:-2]
    sch.tensorize(sch.get_loops(init_block_b)[-2], WMMA_FILL_16x16x16_F16_INTRIN)
    write_sch(sch, log_path,
            "tensorize_fill")
    block_shared_local_A_i, block_shared_local_A_j = sch.get_loops(
        block_shared_local_A)[-4:-2]
    sch.tensorize(sch.get_loops(block_shared_local_A)
                [-2], WMMA_LOAD_16x16x16_F16_A_INTRIN)
    write_sch(sch, log_path,
            "tensorize_load")
    block_shared_local_B_i, block_shared_local_B_j = sch.get_loops(
        block_shared_local_B)[-4:-2]
    sch.tensorize(sch.get_loops(block_shared_local_B)
                [-2], WMMA_LOAD_16x16x16_F16_B_TRANS_INTRIN)
    sch.tensorize(kernel_i, WMMA_SYNC_16x16x16_f16f16f16_TRANS_INTRIN)

    sch.tensorize(sch.get_loops(block_local_C)
                [-2], WMMA_STORE_16x16x16_F16_GLOBAL_INTRIN)
    write_sch(sch, log_path,
            "tensorize")

    # unroll
    # sch.unroll(init_block_b_i)
    # sch.unroll(init_block_b_j)
    # sch.unroll(block_shared_local_A_i)
    # sch.unroll(block_shared_local_A_j)
    # sch.unroll(block_shared_local_B_i)
    # sch.unroll(block_shared_local_B_j)
    # sch.unroll(ii)
    # sch.unroll(jj)
    # sch.unroll(A_shared_inner)
    # sch.unroll(B_shared_inner)

    if stage > 1:

        sch.annotate(ki, ann_key="software_pipeline_stage", ann_val=[0, 0, 0])
        sch.annotate(ki, ann_key="software_pipeline_order", ann_val=[0, 1, 2])

        sch.annotate(ko, ann_key="software_pipeline_stage",
                    ann_val=[0, 0, 0, stage - 1, 0])
        sch.annotate(ko, ann_key="software_pipeline_order",
                    ann_val=[0, 1, 3, 2, 4])
        sch.annotate(ko, ann_key="software_pipeline_async_stages", ann_val=[0])

    write_sch(sch, log_path,
            "do_unroll")


    ctx = tvm.cuda(0)
    with tvm.transform.PassContext(config={"tir.use_async_copy": 1}):
        cuda_mod = tvm.build(sch.mod, target="cuda")

    code = cuda_mod.imported_modules[0].get_source()
    code = code.replace(
        "main_kernel0", f"tir_halfxint3_tensorop_{BM}x{BN}x{BK}x{stage}_t{raster}_y{block_row_warps}z{block_col_warps}_K{K}_align{vec}")
    write_code(code, log_path, "tmp.cu")

    def bit_compress(x, bits, axis):
        if bits == 3:
            # given a tensor x (M, K), which only the low bits bits have value, we can compress it to (M, K // 8 * bits)
            shape = x.shape
            qshape = shape[:axis] + (shape[axis] // 32 * bits,) + shape[axis + 1:]
            qweight = np.zeros(qshape).astype("int32")
            mask = (1 << bits) - 1
            for row in range(qweight.shape[0]):
                # print("compressing: ", row)
                weight = x[row]
                # compress consective 32 weight 32-bit(actually is only 3bit value) integers into 3 32-bit integers
                i = 0
                col = 0
                while col < qweight.shape[1]:
                    for j in range(i, i + 10):
                        qweight[row, col] |= weight[j] << (3 * (j - i))
                    i += 10
                    qweight[row, col] |= weight[i] << 30
                    col += 1
                    qweight[row, col] |= (weight[i] >> 2) & 1
                    i += 1
                    for j in range(i, i + 10):
                        qweight[row, col] |= weight[j] << (3 * (j - i) + 1)
                    i += 10
                    qweight[row, col] |= weight[i] << 31
                    col += 1
                    qweight[row, col] |= (weight[i] >> 1) & 0x3
                    i += 1
                    for j in range(i, i + 10):
                        qweight[row, col] |= weight[j] << (3 * (j - i) + 2)
                    i += 10
                    col += 1
            # convert to int8 in meomory view
            qweight = qweight.view("int8")
            return qweight
        elif bits == 4:
            # given a tensor x (M, K), which only the low bits bits have value, we can compress it to (M, K // 8 * bits)
            shape = x.shape
            compress_shape = shape[:axis] + \
                (shape[axis] // 8 * bits,) + shape[axis + 1:]
            _compressed = np.zeros(compress_shape).astype("int8")
            mask = (1 << bits) - 1
            for i in range(shape[axis]):
                val = (x[..., i] & mask).astype("int8")
                # _compressed is a int8 dtype which can store 1 or more low bits value, like if we have 4 bits, we can store 2 values in one int8
                _compressed[..., i // (8 // bits)] |= (val <<
                                                    ((i % (8 // bits)) * bits)).astype("int8")
            return _compressed


    np.random.seed(0)
    a_raw = (np.random.rand(M, K)).astype("float16")
    b_raw = (np.arange(N * K) % 4).reshape((N, K)).astype("int8")
    # b_compressed = bit_compress(b_raw, bit, 1)
    b_compressed = np.random.randint(0, 4, (N, K // 8 * bit)).astype("int8")
    # b_raw = (np.arange(N * K) % 4).reshape((N, K)).astype("int8")
    scales = np.ones(N).astype("float16")
    zeros = np.zeros(N).astype("float16")

    cuda_a = tvm.nd.array((a_raw).astype("float16"), ctx)
    cuda_b = tvm.nd.array((b_compressed).astype("int8"), ctx)
    cuda_s = tvm.nd.array((scales).astype("float16"), ctx)
    cuda_z = tvm.nd.array((zeros).astype("float16"), ctx)
    cuda_c = tvm.nd.array((np.zeros((M, N))).astype("float16"), ctx)
    cuda_mod(cuda_a, cuda_b, cuda_c, cuda_s, cuda_z)

    if VERIFY:
        cuda_mod(cuda_a, cuda_b, cuda_c, cuda_s, cuda_z)
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

    t = timer_cuda_mod(cuda_a, cuda_b, cuda_c, cuda_s, cuda_z).mean

    GFLOPS = num_flops / (t * 1e3) / 1e6
    print("average time cost of %d runs = %g ms, %g GFLOPS." %
        (num_runs, t * 1e3, GFLOPS))
    
    code = code.split("extern \"C\"")[1]
    append_code(code, kernel_path, "kernel.cu")
    
