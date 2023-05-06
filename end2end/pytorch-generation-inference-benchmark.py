import time

import torch
import torch.nn as nn

from quantization.gptq import *

DEV = 'cuda'

def get_opt(model):
    import torch
    def skip(*args, **kwargs):
        pass
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    from transformers import OPTForCausalLM
    model = OPTForCausalLM.from_pretrained(model, torch_dtype='auto')
    model.seqlen = model.config.max_position_embeddings
    return model

def benchmark(model, input_ids, check=False):
    input_ids = input_ids.to(model.gpus[0] if hasattr(model, 'gpus') else DEV)
    torch.cuda.synchronize()

    cache = {'past': None}
    def clear_past(i):
        def tmp(layer, inp, out):
            if cache['past']:
                cache['past'][i] = None
        return tmp
    for i, layer in enumerate(model.model.decoder.layers):
        layer.register_forward_hook(clear_past(i))

    print('Benchmarking ...')
    iterations = 100
    def sync():
        if hasattr(model, 'gpus'):
            for gpu in model.gpus:
                torch.cuda.synchronize(gpu)
        else:
            torch.cuda.synchronize()
    with torch.no_grad():
        times = []
        for i in range(iterations):
            tick = time.time()
            out = model(
                input_ids
            )
            sync()
            times.append(time.time() - tick)
            print(i, times[-1])
        sync()
        import numpy as np
        print('Median:', np.median(times) * 1000, 'ms')
        print("Output is ", out.logits)


import argparse

model = "/workspace/v-leiwang3/lowbit_model/opt_3bit_first_layer/checkpoint"
model = get_opt(model)
model.eval()
model = model.cuda()
 
# fill input_ids with random data
input_ids = torch.ones((1, 1), dtype=torch.int64, device="cuda")
benchmark(model, input_ids, check=False)
