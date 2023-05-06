import torch
import torch.nn as nn
# include path
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import auto_gptq
from auto_gptq.nn_modules.qlinear_tvm import QuantLinear
from quantization.quant import quantize, Quantizer
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

print('Benchmarking OPT-175B FC2 matvec ...')

DEV = torch.device('cuda:0')

torch.manual_seed(0)

configs = [
    # M N K
    [16, 9216, 9216],
    [3, 9216, 9216],
    [32, 9216, 9216],
    [22, 9216, 9216],
    [64, 9216, 9216],
    [62, 9216, 9216],
    [128, 9216, 9216],
    [122, 9216, 9216],
    [256, 9216, 9216],
    [252, 9216, 9216],
    [1024, 9216, 9216],
    [1022, 9216, 9216],
    [16, 9216, 36864],
    [2, 9216, 36864],
    [32, 9216, 36864],
    [22, 9216, 36864],
    [64, 9216, 36864],
    [62, 9216, 36864],
    [128, 9216, 36864],
    [122, 9216, 36864],
    [256, 9216, 36864],
    [252, 9216, 36864],
    [1024, 9216, 36864],
    [1022, 9216, 36864],
]

bits = 3
for M, N, K in configs:
    print(f'Verifiying kernel correctness of M {M} N {N} K {K} ...')

    layer = nn.Linear(K, N)
    layer = layer.to(DEV).half()

    vec = torch.randn((M, K)).to(DEV)
    # vec = torch.ones((M, K)).to(DEV)
    layer = layer.to('cpu')

    quantizer = Quantizer()
    quantizer.configure(bits, perchannel=True, sym=False, mse=False)
    quantizer.find_params(layer.weight.data, weight=True)
    layer.weight.data = quantize(
        layer.weight.data, quantizer.scale, quantizer.zero, quantizer.maxq
    )
    layer.weight.data = layer.weight.data.half()

    qlayer = QuantLinear(bits, -1, layer.in_features, layer.out_features, True)
    qlayer.pack(layer, quantizer.scale, quantizer.zero, -1)

    qlayer = qlayer.to(DEV)
    layer = layer.to(DEV)

    with torch.no_grad():
        print('Simu:', layer.to(DEV)(vec.half()))
        print('Kern:', qlayer(vec.half()))