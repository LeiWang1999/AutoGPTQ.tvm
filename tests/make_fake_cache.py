import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))  
from auto_gptq.nn_modules.tvm_untils import workloads, pycuda_warpper

# get shapes from models
bits = 3
shapes = [
    # N K
    (3072, 768),
    (768, 3072),
    (768, 768),
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

m_candidates = [1, 16, 32, 64, 128, 256]

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
            }''' + code
            code = "#include <mma.h>\n" + code
            code = "#include <cuda_fp16.h>\n" + code

            
            
        handler_database[key][mx]['code'] = code
        handler_database[key][mx]['func_name'] = name

# print(handler_database)
# with open('./.cache/handler_database.json', 'w') as f:
#     import json
#     json.dump(handler_database, f, indent=4)