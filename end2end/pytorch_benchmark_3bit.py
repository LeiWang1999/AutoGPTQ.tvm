import time
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import torch
import torch.nn as nn
from quantization.gptq import *
from quantization.modelutils import *
from quantization.quant import *

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

def load_quant3(model, checkpoint):
    from transformers import OPTConfig, OPTForCausalLM 
    config = OPTConfig.from_pretrained(model)
    def noop(*args, **kwargs):
        pass
    torch.nn.init.kaiming_uniform_ = noop 
    torch.nn.init.uniform_ = noop 
    torch.nn.init.normal_ = noop 

    torch.set_default_dtype(torch.half)
    transformers.modeling_utils._init_weights = False
    torch.set_default_dtype(torch.half)
    model = OPTForCausalLM(config)
    torch.set_default_dtype(torch.float)
    model = model.eval()
    layers = find_layers(model)
    for name in ['model.decoder.project_out', 'model.decoder.project_in', 'lm_head']:
        if name in layers:
            del layers[name]
    make_quant3(model, layers, faster=True)

    print('Loading model ...')
    model.load_state_dict(torch.load(checkpoint))
    model.seqlen = model.config.max_position_embeddings
    print('Done.')

    return model

def benchmark(model, input_ids, check=False):
    input_ids = input_ids.to(model.gpus[0] if hasattr(model, 'gpus') else DEV)
    torch.cuda.synchronize()

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
        # print("Output is ", out)


# model = "/workspace/v-leiwang3/lowbit_model/opt_3bit_first_layer/checkpoint"
# load = "/workspace/v-leiwang3/lowbit_workspace/GPTQ-tvm/quantization/models/opt_layer1_3bit_faster.pt"

model = "facebook/opt-66b"
load = "/workspace/v-leiwang3/lowbit_workspace/GPTQ-tvm/quantization/models/opt_66b_3bit_faster.pt"

DEV = 'cuda'

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("facebook/opt-66b")

model = load_quant3(model, load).cuda().half()


# gpus = [torch.device('cuda:%d' % i) for i in range(torch.cuda.device_count())]
# model = model.to(DEV)
# model.eval()
# input_ids = torch.ones((1, 1), dtype=torch.int64, device="cuda")
# input_ids[0] = 128
# benchmark(model, input_ids, False)

prompt = "Hello, I am conscious and"
inputs = tokenizer(prompt, return_tensors="pt").to(DEV)

# print inputs token
print("Input Tokens: ", tokenizer.convert_ids_to_tokens(inputs["input_ids"][0]))

print("Input IDs: ", inputs["input_ids"])

# Generate output
generated_output = model.generate(inputs["input_ids"], max_new_tokens=50)

# Decode the generated token IDs
# decoded_output = tokenizer.decode(generated_output.cpu().squeeze().tolist())
print(tokenizer.batch_decode(generated_output, skip_special_tokens=True))

# # print inputs token
# # print("Input Tokens: ", tokenizer.convert_ids_to_tokens(inputs["input_ids"][0]))

# # print(inputs["input_ids"][0])
# # benchmark(model, inputs["input_ids"][0], check=False)
# print(inputs["input_ids"].shape)
# outputs = model(**inputs, labels=inputs["input_ids"])
# print("logits:", outputs.logits)