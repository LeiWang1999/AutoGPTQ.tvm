import tvm
from tvm.script import tir as T
from tvm import meta_schedule as ms
import os
from tvm.tir.tensor_intrin.cuda import get_wmma_intrin_group

def gemm_workloads(align_m:int = 16, bit:int = 3):
    bit = 4
    group_stride = 32 * bit // 8
    mask = (1 << bit) - 1
    ...
    