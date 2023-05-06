import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from transformers import AutoTokenizer, TextGenerationPipeline
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig


pretrained_model_dir = "facebook/opt-125m"
quantized_model_dir = "opt-125m-4bit"

tokenizer = AutoTokenizer.from_pretrained(pretrained_model_dir, use_fast=True)

# load quantized model, currently only support cpu or single gpu
model = AutoGPTQForCausalLM.from_quantized(quantized_model_dir, device="cuda:0", use_triton=False, use_tvm=True).half()

generated = model.generate(**tokenizer("auto-gptq is ", return_tensors="pt").to("cuda:0"))[0]

# Decode the generated token IDs
decoded_output = tokenizer.decode(generated.cpu().squeeze().tolist())
print(decoded_output.strip())