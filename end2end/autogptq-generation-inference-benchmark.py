import time

import torch
import torch.nn as nn

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from transformers import AutoTokenizer, TextGenerationPipeline
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

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
    pipeline = TextGenerationPipeline(model=model, tokenizer=tokenizer)
    iterations = 10
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
            # print(i, times[-1])
        sync()
        import numpy as np
        print('Median Inference Time:', np.median(times) * 1000, 'ms')
        print("Output is ", out)


import argparse

# pretrained_model_dir = "facebook/opt-125m"
# quantized_model_dir = "quantization/models/opt-125m-3bit"

pretrained_model_dir = "/workspace/v-leiwang3/lowbit_model/oa_llama_30b/oasst-sft-6-llama-30b"
quantized_model_dir = "quantization/models/huggingchat-30b-4bit"

model = AutoGPTQForCausalLM.from_quantized(quantized_model_dir, device="cuda:0", use_tvm=True)

model.eval()
model = model.cuda()
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_dir, use_fast=True)
# fill input_ids with random data
# inputs = tokenizer("auto_gptq is an interesting", return_tensors="pt").to(model.device)
inputs = "gptq is an opensource project"

print(inputs)
benchmark(model, inputs)
