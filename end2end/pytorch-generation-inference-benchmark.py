import time

import torch
import torch.nn as nn

from transformers import AutoTokenizer, TextGenerationPipeline, AutoModelForCausalLM

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

def benchmark(model, inputs):
    torch.cuda.synchronize()
    print('Benchmarking ...')
    print("Input is ", inputs)
    iterations = 10
    pipeline = TextGenerationPipeline(model=model, tokenizer=tokenizer)
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
            out = pipeline(inputs)
            sync()
            times.append(time.time() - tick)
            print(i, times[-1])
        sync()
        import numpy as np
        print('Median:', np.median(times) * 1000, 'ms')
        print("Output is ", out)


import argparse

# model = "facebook/opt-66b"
# model = get_opt(model)
# model.eval()
# model = model.cuda()
pretrained_model_dir = "/workspace/v-leiwang3/lowbit_model/oa_llama_30b/oasst-sft-6-llama-30b"

model = AutoModelForCausalLM.from_pretrained(pretrained_model_dir, torch_dtype='auto')
# fill input_ids with random data
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_dir, use_fast=True)
# fill input_ids with random data
# inputs = tokenizer("auto_gptq is an interesting", return_tensors="pt").to(model.device)
inputs = "auto_gptq is an interesting"
benchmark(model, inputs)
