import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))  
from auto_gptq.nn_modules.tvm_untils import workloads, pycuda_warpper

# get shapes from models
bits = 4
shapes = [
    # N K
    # (3072, 768),
    # (768, 3072),
    # (768, 768),
    (6656, 6656)
]

configurations = {
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
    }
}

m_candidates = [16]
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

handler_database = {}
for n, k in shapes:
    key = f"b{bits}n{n}k{k}"
    handler_database[key] = {}
    for m in m_candidates:
        mx = f'm{m}'
        handler_database[key][mx] = {}
        handler_database[key][mx]['params'] = configurations[mx]
        if m == 1:
            m1_module = workloads.get_gemv_workloads(bits, n, k)
            num_warps = configurations['m1']['num_warps']
            m1_mod = workloads._apply_gemv_schedule(m1_module, bits, k, num_warps)
            code = m1_mod.imported_modules[0].get_source()
            name = f"tir_halfxint{bits}_simt_bn{num_warps}_k{k}"
            code = code.replace(
                "main_kernel0", name)
            code = code.split("extern \"C\"")[1]
            code = "extern \"C\"" + code
            code = "#include <cuda_fp16.h>\n" + code
        else:
            mx = f'm{m}'
            mx_config = configurations[mx]
            mx_mod = workloads._apply_dynamic_gemm_schedule(bits, m, n, k, mx_config)
            # test 
            import torch
            import numpy as np
            import tvm
            pm = 6
            x = torch.rand((pm, k), dtype=torch.float16).cuda()
            w = (np.arange(n * k) % 4).reshape((n, k)).astype("int8")
            # qw = bit_compress(w, 3, 1)
            qw = np.random.randint(0, 8, (n, k // 8 * bits)).astype("int8")
            print(np.matmul(x.cpu().numpy(), w.T))
            w = torch.from_numpy(w).cuda()
            qw = torch.from_numpy(qw).cuda()
            scales = torch.ones(n, dtype=torch.float16).cuda()
            zeros = torch.zeros(n, dtype=torch.float16).cuda()
            y = torch.zeros((pm, n), dtype=torch.float16).cuda() 
            ctx = tvm.cuda(0)
            cuda_x = tvm.nd.array(x.cpu().numpy(), ctx)
            cuda_qw = tvm.nd.array(qw.cpu().numpy(), ctx)
            cuda_scales = tvm.nd.array(scales.cpu().numpy(), ctx)
            cuda_zeros = tvm.nd.array(zeros.cpu().numpy(), ctx)
            cuda_y = tvm.nd.array(torch.zeros((m, n), dtype=torch.float16), ctx)
            mx_mod(cuda_x, cuda_qw, cuda_y, cuda_scales, cuda_zeros, pm)
            print(cuda_y)      
            name = f"tir_halfxint3_tensorop_{mx_config['BM']}x{mx_config['BN']}x{mx_config['BK']}x{mx_config['stage']}_t{mx_config['raster']}_y{mx_config['block_row_warps']}z{mx_config['block_col_warps']}_K{k}_align8"
            
            code = mx_mod.imported_modules[0].get_source()
            
            code = code.replace(
                "main_kernel0", name)
            code = code.split("extern \"C\"")[1]
            code = "extern \"C\"" + code
            code = '''
                static inline __device__ __host__ unsigned
                __pack_half2(const half x, const half y) {
                unsigned v0 = *((unsigned short *)&x);
                unsigned v1 = *((unsigned short *)&y);
                return (v1 << 16) | v0;
            }\n''' + code
            code = "#include <mma.h>\n" + code
            code = "#include <cuda_fp16.h>\n" + code
            block = (32, mx_config['block_row_warps'],
            mx_config['block_col_warps'])
            grid = (n // mx_config['BN'], m // mx_config['BM'], 1)
            _y = torch.zeros((m, n), dtype=torch.float16).cuda()
            _e = pycuda_warpper.TVMExecutable(code, name)
            print(grid, block)
            _e(x, qw, _y, scales, zeros, pm, grid=grid, block=block)
            print(_y)

            
            
        handler_database[key][mx]['code'] = code
        handler_database[key][mx]['func_name'] = name
        print(code)