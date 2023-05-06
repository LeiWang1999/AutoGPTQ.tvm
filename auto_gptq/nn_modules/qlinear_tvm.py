import numpy as np
import torch
import torch.nn as nn
from .tvm_untils import database

def quantize(x, scale, zero, maxq):
    if maxq < 0:
        return (x > scale / 2).float() * scale + (x < zero / 2).float() * zero
    q = torch.clamp(torch.round(x / scale) + zero, 0, maxq)
    return scale * (q - zero)

class Quantizer(nn.Module):

    def __init__(self, shape=1):
        super(Quantizer, self).__init__()
        self.register_buffer('maxq', torch.tensor(0))
        self.register_buffer('scale', torch.zeros(shape))
        self.register_buffer('zero', torch.zeros(shape))

    def configure(
        self,
        bits, perchannel=False, sym=True, 
        mse=False, norm=2.4, grid=100, maxshrink=.8,
        trits=False
    ):
        self.maxq = torch.tensor(2 ** bits - 1)
        self.perchannel = perchannel
        self.sym = sym
        self.mse = mse
        self.norm = norm
        self.grid = grid
        self.maxshrink = maxshrink 
        if trits:
            self.maxq = torch.tensor(-1) 

    def find_params(self, x, weight=False):
        dev = x.device
        self.maxq = self.maxq.to(dev)

        shape = x.shape
        if self.perchannel:
            if weight:
                x = x.flatten(1)
            else:
                if len(shape) == 4:
                    x = x.permute([1, 0, 2, 3])
                    x = x.flatten(1)
                if len(shape) == 3:
                    x = x.reshape((-1, shape[-1])).t()
                if len(shape) == 2:
                    x = x.t()
        else:
            x = x.flatten().unsqueeze(0)

        tmp = torch.zeros(x.shape[0], device=dev)
        xmin = torch.minimum(x.min(1)[0], tmp)
        xmax = torch.maximum(x.max(1)[0], tmp)

        if self.sym:
            xmax = torch.maximum(torch.abs(xmin), xmax)
            tmp = xmin < 0
            if torch.any(tmp):
                xmin[tmp] = -xmax[tmp]
        tmp = (xmin == 0) & (xmax == 0)
        xmin[tmp] = -1
        xmax[tmp] = +1

        if self.maxq < 0:
          self.scale = xmax
          self.zero = xmin
        else:
          self.scale = (xmax - xmin) / self.maxq
          if self.sym:
              self.zero = torch.full_like(self.scale, (self.maxq + 1) / 2)
          else:
              self.zero = torch.round(-xmin / self.scale)

        if self.mse:
            best = torch.full([x.shape[0]], float('inf'), device=dev)
            for i in range(int(self.maxshrink * self.grid)):
                p = 1 - i / self.grid 
                xmin1 = p * xmin
                xmax1 = p * xmax
                scale1 = (xmax1 - xmin1) / self.maxq
                zero1 = torch.round(-xmin1 / scale1) if not self.sym else self.zero
                q = quantize(x, scale1.unsqueeze(1), zero1.unsqueeze(1), self.maxq)
                q -= x
                q.abs_()
                q.pow_(self.norm)
                err = torch.sum(q, 1)
                tmp = err < best
                if torch.any(tmp):
                    best[tmp] = err[tmp]
                    self.scale[tmp] = scale1[tmp]
                    self.zero[tmp] = zero1[tmp]
        if not self.perchannel:
            if weight:
                tmp = shape[0]
            else:
                tmp = shape[1] if len(shape) != 3 else shape[2]
            self.scale = self.scale.repeat(tmp)
            self.zero = self.zero.repeat(tmp)

        if weight:
            shape = [-1] + [1] * (len(shape) - 1)
            self.scale = self.scale.reshape(shape)
            self.zero = self.zero.reshape(shape)
            return
        if len(shape) == 4:
            self.scale = self.scale.reshape((1, -1, 1, 1))
            self.zero = self.zero.reshape((1, -1, 1, 1))
        if len(shape) == 3:
            self.scale = self.scale.reshape((1, 1, -1))
            self.zero = self.zero.reshape((1, 1, -1)) 
        if len(shape) == 2:
            self.scale = self.scale.unsqueeze(0)
            self.zero = self.zero.unsqueeze(0)

    def quantize(self, x):
        if self.ready():
            return quantize(x, self.scale, self.zero, self.maxq)
        return x

    def enabled(self):
        return self.maxq > 0

    def ready(self):
        return torch.all(self.scale != 0)


try:
    import quant_tvm
except:
    print('TVM extension not installed.')

# Assumes layer is perfectly divisible into 1024 * 1024 blocks
class QuantLinear(nn.Module): 

    def __init__(
        self,
        bits,
        groupsize,
        infeatures,
        outfeatures,
        bias,
        ):
        super().__init__()
        if bits not in [2, 3, 4, 8]:
            raise NotImplementedError("Only 2,3,4,8 bits are supported.")
        self.register_buffer('zeros', torch.zeros((outfeatures, 1), dtype=torch.float16))
        self.register_buffer('scales', torch.zeros((outfeatures, 1), dtype=torch.float16))
        if bias:
            self.register_buffer('bias', torch.zeros((outfeatures), dtype=torch.float16))
        else:
            self.bias = None
        self.groupsize = groupsize if groupsize != -1 else infeatures
        self.register_buffer(
            'qweight', torch.zeros((outfeatures, infeatures // 8 * 3), dtype=torch.int8)
        )
        self.tvm_handler = database.get_handler(n=outfeatures, k=infeatures, bits=bits)

    def pack(self, linear, scales, zeros, g_idx):
        self.zeros = zeros * scales
        self.scales = scales.clone()
        self.bias = linear.bias.clone()
        
        intweight = torch.round((linear.weight.data + self.zeros) / self.scales).to(torch.int)
        intweight = intweight.t().contiguous()
        intweight = intweight.numpy().astype(np.uint32)
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
        qweight = np.ascontiguousarray(qweight.T)
        qweight = qweight.view(dtype=np.int8)
        self.qweight = torch.from_numpy(qweight) 
        self.scales = self.scales.half()
        self.zeros = self.zeros.half()
        self.bias = self.bias.half()
        

    def forward(self, x):
        # print('QuantLinear forward, xshape is ', x.shape)
        # print(x)
        dtype = x.dtype
        x = x.half()
        M = 1
        for i in range(len(x.shape) - 1):
            M *= x.shape[i]
        if x.shape[-1] == x.numel():
            outshape = list(x.shape)
            outshape[-1] = self.bias.numel()
            y = torch.zeros(outshape, dtype=x.dtype, device=x.device)
            self.tvm_handler(x, self.qweight, y, self.scales, self.zeros)
            y = y.reshape(outshape)
            y += self.bias
            # print(y)
            return y 
        elif 1 < M <= 16:
            outshape = list(x.shape)
            outshape[-1] = self.bias.numel()
            # if M <= 16, we need to pad it to 16
            x = x.reshape((M, -1))
            if M % 16 != 0:
                pad = 16 - x.shape[0] % 16
                x = torch.nn.functional.pad(x, (0, 0, 0, pad))
        elif 16 < M <= 32:
            outshape = list(x.shape)
            outshape[-1] = self.bias.numel()
            x = x.reshape((M, -1))
            if x.shape[0] % 32 != 0:
                pad = 32 - x.shape[0] % 32
                x = torch.nn.functional.pad(x, (0, 0, 0, pad))
        elif 32 < M <= 64:
            outshape = list(x.shape)
            outshape[-1] = self.bias.numel()
            x = x.reshape((M, -1))
            if x.shape[0] % 64 != 0:
                pad = 64 - x.shape[0] % 64
                x = torch.nn.functional.pad(x, (0, 0, 0, pad))
        elif 64 < M <= 128:
            outshape = list(x.shape)
            outshape[-1] = self.bias.numel()
            x = x.reshape((M, -1))
            if x.shape[0] % 128 != 0:
                pad = 128 - x.shape[0] % 128
                x = torch.nn.functional.pad(x, (0, 0, 0, pad))
        else:
            outshape = list(x.shape)
            outshape[-1] = self.bias.numel()
            x = x.reshape((M, -1))
            if x.shape[0] % 256 != 0:
                pad = 256 - x.shape[0] % 256
                x = torch.nn.functional.pad(x, (0, 0, 0, pad))

        y_pad = torch.zeros((x.shape[0], self.bias.numel()), dtype=x.dtype, device=x.device)
        # quant_tvm.quant_kernel_3b(x, self.qweight, y_pad, self.scales, self.zeros)
        self.tvm_handler(x, self.qweight, y_pad, self.scales, self.zeros)
        # recover y_pad to y
        y = torch.zeros(outshape, dtype=dtype, device=x.device)
        y[:M] = y_pad[:M]
        y += self.bias
        y.to(dtype)
        # print(y)
        return y

        