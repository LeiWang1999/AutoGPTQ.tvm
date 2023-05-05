from typing import Any
import torch
import pycuda
import pycuda.autoprimaryctx
import pycuda.driver as cuda
import numpy as np
import re
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray
from .workloads import get_gemm_workloads, get_gemv_workloads
import tvm
import numpy as np
from tvm.script import tir as T

class TensorHolder(pycuda.driver.PointerHolderBase):
    def __init__(self, t):
        super(TensorHolder, self).__init__()
        self.t = t
        self.gpudata = t.data_ptr()
    
    def get_pointer(self):
        return self.t.data_ptr()

class TVMExecutable(object):
    def __init__(self, src, name):
        __doc__ = 'Initialize FastMLM object'
        super(TVMExecutable, self).__init__()
        self.source_code : str = src
        self.func_name: str = name
        self.kernel_func = self._get_kernel(self.source_code, self.func_name)


    def __call__(self, input, qweight, output, scales, zeros, grid:tuple, block:tuple) -> Any:
        self.kernel_func(TensorHolder(input), TensorHolder(qweight), TensorHolder(scales), TensorHolder(zeros), TensorHolder(output), grid=grid, block=block)
        pass

    def _get_kernel(self, src_code, name):
        src = torch.cuda.ByteTensor(8)
        mod = SourceModule(src_code, no_extern_c=True)
        return mod.get_function(name)
    

class TVMHandler(object):
    def __init__(self, n:int, k:int, bits:int):
        __doc__ = 'Initialize FastMLM object'
        super(TVMHandler, self).__init__()
        self.k = k
        self.n = n
        self.bits = bits
        self.configurations = {
            'm1':
                {
                'num_warps': 4
                },
            'm16': {
                "block_row_warps": 1,
                "block_col_warps": 4,
                "BM": 16,
                "BN": 64,
                "BK": 64,
                "raster": 0,
                "stage": 3
            },
            'm32': {
                "block_row_warps": 2,
                "block_col_warps": 4,
                "BM": 32,
                "BN": 64,
                "BK": 64,
                "raster": 0,
                "stage": 2
            }, 
            'm64': {
                "block_row_warps": 2,
                "block_col_warps": 4,
                "BM": 64,
                "BN": 64,
                "BK": 64,
                "raster": 0,
                "stage": 2
            },
            'm128': {
                "block_row_warps": 1,
                "block_col_warps": 2,
                "BM": 64,
                "BN": 128,
                "BK": 32,
                "raster": 0,
                "stage": 3
            },
            'm256': {
                "block_row_warps": 2,
                "block_col_warps": 2,
                "BM": 128,
                "BN": 256,
                "BK": 32,
                "raster": 0,
                "stage": 1
            }
        }
        self.m1: TVMExecutable = self._get_executable_m1(bits, k)
        self.m16: TVMExecutable = self._get_executable_mx(bits, n, k, 'm16')
        self.m32: TVMExecutable = self._get_executable_mx(bits, n, k, 'm32')
        self.m64: TVMExecutable = self._get_executable_mx(bits, n, k, 'm64')
        self.m128: TVMExecutable = self._get_executable_mx(bits, n, k, 'm128')
        self.m256: TVMExecutable = self._get_executable_mx(bits, n, k, 'm256')
    
    def __call__(self, input, qweight, output, scales, zeros) -> Any:
        assert len(output.shape) == 2, "output should be 2D"
        M = output.shape[0]
        N = output.shape[1]
        if M == 1:
            block = (32, self.configurations['m1']['num_warps'], 1)
            grid = (N // self.configurations['m1']['num_warps'], 1, 1)
            self.m1(input, qweight, output, scales, zeros, block=block, grid=grid)
        elif M == 16:
            mx_config = self.configurations['m16']
            block = (32, mx_config['block_row_warps'], mx_config['block_col_warps'])
            grid = (N // mx_config['BN'], M // mx_config['BM'], 1)
            self.m16(input, qweight, output, scales, zeros, block=block, grid=grid)
        elif M == 32:
            mx_config = self.configurations['m32']
            block = (32, mx_config['block_row_warps'], mx_config['block_col_warps'])
            grid = (N // mx_config['BN'], M // mx_config['BM'], 1)
            self.m32(input, qweight, output, scales, zeros, block=block, grid=grid)
        elif M == 64:
            mx_config = self.configurations['m64']
            block = (32, mx_config['block_row_warps'], mx_config['block_col_warps'])
            grid = (N // mx_config['BN'], M // mx_config['BM'], 1)
            self.m64(input, qweight, output, scales, zeros, block=block, grid=grid)
        elif M == 128:
            mx_config = self.configurations['m128']
            block = (32, mx_config['block_row_warps'], mx_config['block_col_warps'])
            grid = (N // mx_config['BN'], M // mx_config['BM'], 1)
            self.m128(input, qweight, output, scales, zeros, block=block, grid=grid)
        else:
            mx_config = self.configurations['m256']
            block = (32, mx_config['block_row_warps'], mx_config['block_col_warps'])
            grid = (N // mx_config['BN'], M // mx_config['BM'], 1)
            self.m256(input, qweight, output, scales, zeros, block=block, grid=grid)

    
    def _get_executable_m1(self, bits:int, k:int):
        # get src code
        m1_module = get_gemv_workloads(bits, k)
        num_warps = self.configurations['m1']['num_warps']
        m1_mod = self._apply_gemv_schedule(m1_module, bits, k, num_warps)
        code = m1_mod.imported_modules[0].get_source()
        name = f"tir_halfxint{bits}_simt_bn{num_warps}_k{k}"
        code = code.replace(
                    "main_kernel0", name)
        code = code.split("extern \"C\"")[1]
        code = "extern \"C\"" + code
        code = "#include <cuda_fp16.h>\n" + code
        return TVMExecutable(code, name)
            
    def _get_executable_mx(self, bits:int, n:int, k:int, mx:str='m16'):
        mx_module = get_gemm_workloads(bits, n, k)
        mx_config = self.configurations[mx]
        mx_mod = self._apply_gemm_schedule(mx_module, bits, k, mx_config)
        name = f"tir_halfxint3_tensorop_{mx_config['BM']}x{mx_config['BN']}x{mx_config['BK']}x{mx_config['stage']}_t{mx_config['raster']}_y{mx_config['block_row_warps']}z{mx_config['block_col_warps']}_K{k}_align8"
        code = mx_mod.imported_modules[0].get_source()
        code = code.replace(
                    "main_kernel0", name)
        code = code.split("extern \"C\"")[1]
        code = "extern \"C\"" + code
        code = "#include <mma.h>\n" + code
        code = "#include <cuda_fp16.h>\n" + code
        return TVMExecutable(code, name)

        
        
    def _apply_gemv_schedule(self, ir_module, bits, K, num_warps=4):
        num_warps = num_warps
        warp_size = 32
        vec = 8
        sch = tvm.tir.Schedule(ir_module, debug_mask="all")
        block_b = sch.get_block("B")
        block_b_decompress = sch.get_block("B_decompress")
        block_b_rescale = sch.get_block("B_rescale")
        sch.compute_inline(block_b_decompress)
        sch.compute_inline(block_b_rescale)
        i, j, k = sch.get_loops(block_b)
        block_shared_local_A = sch.cache_read(block_b, 0, "local")
        block_local_C = sch.cache_write(block_b, 0, "local")
        bx, j = sch.split(
            j, factors=[None, num_warps])
        k, tx, vk = sch.split(k, factors=[None, warp_size, vec])
        sch.reorder(bx, j, i, k, tx)

        sch.bind(bx, "blockIdx.x")
        sch.bind(tx, "threadIdx.x")
        sch.bind(j, "threadIdx.y")

        sch.compute_at(block_shared_local_A, tx, preserve_unit_loops=True)
        sch.reverse_compute_at(block_local_C, j, preserve_unit_loops=True)
        block_local_a_v = sch.get_loops(block_shared_local_A)[-1]
        sch.vectorize(block_local_a_v)
        ctx = tvm.cuda(0)
        cuda_mod = tvm.build(sch.mod, target="cuda")
        return cuda_mod
    
    def _apply_gemm_schedule(self, ir_module, bits, K, config):
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

        wmma_m = 16
        wmma_n = 16
        wmma_k = 16
        warp_size = 32
        BM = config['BM']
        BN = config['BN']
        BK = config['BK']
        block_row_warps = config['block_row_warps']
        block_col_warps = config['block_col_warps']
        raster = config['raster']
        stage = config['stage']
        warp_row_tiles = BM // (wmma_m * block_row_warps)
        warp_col_tiles = BN // (wmma_n * block_col_warps)
        chunk = BK // (wmma_k)
        vec = 8
        shared_pad = 8
        sch = tvm.tir.Schedule(ir_module, debug_mask="all")
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


        # cache read A from global memory to shared_memory
        sch.compute_at(block_shared_local_A, ki)
        sch.compute_at(block_shared_A, ko, preserve_unit_loops=True)
        sch.compute_at(block_shared_local_B, ki)
        sch.compute_at(block_shared_B, ko, preserve_unit_loops=True)
        sch.reverse_compute_at(block_local_C, j)


        A_shared_fused = sch.fuse(*sch.get_loops(block_shared_A)[-2:])
        A_shared_ty, A_shared_tz, A_shared_inner, A_shared_tx, A_shared_vi = sch.split(
            A_shared_fused, factors=[block_row_warps, block_col_warps, None, warp_size, vec])
        sch.vectorize(A_shared_vi)
        sch.bind(A_shared_tx, "threadIdx.x")
        sch.bind(A_shared_ty, "threadIdx.y")
        sch.bind(A_shared_tz, "threadIdx.z")
        sch.storage_align(block_shared_A, 0, axis=-2, factor=32, offset=shared_pad)

        B_shared_fused = sch.fuse(*sch.get_loops(block_shared_B)[-2:])
        B_shared_ty, B_shared_tz, B_shared_inner, B_shared_tx, B_shared_vi = sch.split(
            B_shared_fused, factors=[block_row_warps, block_col_warps, None, warp_size, 1])
        sch.vectorize(B_shared_vi)
        sch.bind(B_shared_tx, "threadIdx.x")
        sch.bind(B_shared_ty, "threadIdx.y")
        sch.bind(B_shared_tz, "threadIdx.z")
        sch.storage_align(block_shared_B, 0, axis=-2, factor=32, offset=shared_pad)



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

        # transpose layout

        init_block_b_i, init_block_b_j = sch.get_loops(init_block_b)[-4:-2]
        sch.tensorize(sch.get_loops(init_block_b)[-2], WMMA_FILL_16x16x16_F16_INTRIN)

        block_shared_local_A_i, block_shared_local_A_j = sch.get_loops(
            block_shared_local_A)[-4:-2]
        sch.tensorize(sch.get_loops(block_shared_local_A)
                    [-2], WMMA_LOAD_16x16x16_F16_A_INTRIN)

        block_shared_local_B_i, block_shared_local_B_j = sch.get_loops(
            block_shared_local_B)[-4:-2]
        sch.tensorize(sch.get_loops(block_shared_local_B)
                    [-2], WMMA_LOAD_16x16x16_F16_B_TRANS_INTRIN)
        sch.tensorize(kernel_i, WMMA_SYNC_16x16x16_f16f16f16_TRANS_INTRIN)

        sch.tensorize(sch.get_loops(block_local_C)
                    [-2], WMMA_STORE_16x16x16_F16_GLOBAL_INTRIN)

        if stage > 1:

            sch.annotate(ki, ann_key="software_pipeline_stage", ann_val=[0, 0, 0])
            sch.annotate(ki, ann_key="software_pipeline_order", ann_val=[0, 1, 2])

            sch.annotate(ko, ann_key="software_pipeline_stage",
                        ann_val=[0, 0, 0, stage - 1, 0])
            sch.annotate(ko, ann_key="software_pipeline_order",
                        ann_val=[0, 1, 3, 2, 4])
            sch.annotate(ko, ann_key="software_pipeline_async_stages", ann_val=[0])


        ctx = tvm.cuda(0)
        with tvm.transform.PassContext(config={"tir.use_async_copy": 1}):
            cuda_mod = tvm.build(sch.mod, target="cuda")
        
        
        
        return cuda_mod


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
        
if __name__ == '__main__':
    # test for 3x3 kernel
    M = 16
    N = 1024
    K = 768
    handler = TVMHandler(bits=3, n=N, k=K)
    x = torch.ones((M, K), dtype=torch.float16).cuda()
    w = (np.arange(N * K) % 4).reshape((N, K)).astype("int8")
    qw = bit_compress(w, 3, 1)
    print(np.matmul(x.cpu().numpy(), w.T))
    w = torch.from_numpy(w).cuda()
    qw = torch.from_numpy(qw).cuda()
    scales = torch.ones(N, dtype=torch.float16).cuda()
    zeros = torch.zeros(N, dtype=torch.float16).cuda()
    y = torch.zeros((M, N), dtype=torch.float16).cuda()
    handler(x, qw, y, scales, zeros)
    print(y)
    
    