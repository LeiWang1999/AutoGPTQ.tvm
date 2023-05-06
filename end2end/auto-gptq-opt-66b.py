import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from transformers import AutoTokenizer, TextGenerationPipeline
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig


pretrained_model_dir = "facebook/opt-66b"
quantized_model_dir = "quantization/models/opt-66b-3bit"

tokenizer = AutoTokenizer.from_pretrained(pretrained_model_dir, use_fast=True)

# load quantized model, currently only support cpu or single gpu
model = AutoGPTQForCausalLM.from_quantized(quantized_model_dir, device="cuda:0", use_triton=False, use_tvm=True).half()

# or you can also use pipeline
pipeline = TextGenerationPipeline(model=model, tokenizer=tokenizer)
print(pipeline("auto-gptq is")[0]["generated_text"])
