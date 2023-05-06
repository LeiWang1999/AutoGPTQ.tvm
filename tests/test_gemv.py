import torch
import torch.nn as nn

import quant_tvm

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

DEV = torch.device('cuda:0')
print('Verifiying kernel correctness ...')
torch.manual_seed(0)

M = 9216
N = 9216

layer = nn.Linear(M, N)
layer = layer.to(DEV).half()

vec = torch.randn(M).to(DEV).reshape((1, 1, M))
layer = layer.to('cpu')

from quantization.quant import *
quantizer = Quantizer()
quantizer.configure(3, perchannel=True, sym=False, mse=False)
quantizer.find_params(layer.weight.data, weight=True)
layer.weight.data = quantize(
    layer.weight.data, quantizer.scale, quantizer.zero, quantizer.maxq
)
layer.weight.data = layer.weight.data.half()
# layer.weight.data = torch.zeros(layer.weight.data.shape, device=layer.weight.device, dtype=layer.weight.dtype)
# layer.bias.data = torch.zeros(layer.bias.data.shape, device=layer.bias.device, dtype=layer.bias.dtype)
# print('Quantizer:', layer.bias.data)

qlayer = Quant3Linear(layer.in_features, layer.out_features)
qlayer.pack(layer, quantizer.scale, quantizer.zero)

qlayer = qlayer.to(DEV)
layer = layer.to(DEV)

with torch.no_grad():
    print('Simu:', layer.to(DEV)(vec.half()))
    print('Kern:', qlayer(vec.half()))