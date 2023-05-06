import time
import torch
import torch.nn as nn
import math

DEV = 'cuda'

def opt_multigpu(model, gpus):
    model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(gpus[0])
    model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(gpus[0])
    if hasattr(model.model.decoder, 'project_in') and model.model.decoder.project_in:
        model.model.decoder.project_in = model.model.decoder.project_in.to(gpus[0])
    if hasattr(model.model.decoder, 'project_out') and model.model.decoder.project_out:
        model.model.decoder.project_out = model.model.decoder.project_out.to(gpus[-1])
    if hasattr(model.model.decoder, 'final_layer_norm') and model.model.decoder.final_layer_norm:
        model.model.decoder.final_layer_norm = model.model.decoder.final_layer_norm.to(gpus[-1])
    import copy
    model.lm_head = copy.deepcopy(model.lm_head).to(gpus[-1])

    cache = {'mask': None}

    class MoveModule(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
            self.dev = next(iter(self.module.parameters())).device
        def forward(self, *inp, **kwargs):
            inp = list(inp)
            if inp[0].device != self.dev:
                inp[0] = inp[0].to(self.dev)
            if cache['mask'] is None or cache['mask'].device != self.dev:
                cache['mask'] = kwargs['attention_mask'].to(self.dev)
            kwargs['attention_mask'] = cache['mask']
            tmp = self.module(*inp, **kwargs)
            return tmp

    layers = model.model.decoder.layers
    pergpu = math.ceil(len(layers) / len(gpus))
    for i in range(len(layers)):
        layers[i] = MoveModule(layers[i].to(gpus[i // pergpu]))

    model.gpus = gpus

def get_opt(model):
    import torch
    def skip(*args, **kwargs):
        pass
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    from transformers import OPTForCausalLM
    model = OPTForCausalLM.from_pretrained(model, torch_dtype='auto', device_map='auto')
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
    iterations = 1
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

# model = "/workspace/v-leiwang3/lowbit_model/opt_3bit_first_layer/checkpoint"

model = "facebook/opt-66b"
model = get_opt(model)

gpus = []
# gpus = [torch.device('cuda:%d' % i) for i in range(torch.cuda.device_count())]
print('Using GPUs:', gpus)
if len(gpus) > 1:
    opt_multigpu(model, gpus)
else:
    model = model
# fill input_ids with random data
# input_ids = torch.ones((1, 1), dtype=torch.int64, device="cuda")
# benchmark(model, input_ids, check=False)

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("facebook/opt-66b")

prompt = "Hello, I am conscious and"
inputs = tokenizer(prompt, return_tensors="pt")
# print inputs token
print("Input Tokens: ", tokenizer.convert_ids_to_tokens(inputs["input_ids"][0]))

# print(inputs["input_ids"][0])
# benchmark(model, inputs["input_ids"][0], check=False)

# outputs = model(**inputs, labels=inputs["input_ids"])
# print("logits:", outputs.logits)

# Get the predicted token IDs
# predicted_token_ids = outputs.logits.argmax(dim=-1)

print("Input IDs: ", inputs["input_ids"])
input_ids = inputs["input_ids"]
input_ids = input_ids.to(model.gpus[0] if hasattr(model, 'gpus') else DEV)

# Generate output using model.generate()
generated_output = model.generate(input_ids, max_new_tokens=50)

# Decode the generated token IDs
decoded_output = tokenizer.decode(generated_output.cpu().squeeze().tolist())
print(decoded_output.strip())