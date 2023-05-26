import os
import sys
os.environ["TOKENIZERS_PARALLELISM"] = "false"
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from transformers import AutoTokenizer, TextGenerationPipeline
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig


pretrained_model_dir = "/workspace/v-leiwang3/lowbit_model/oa_llama_30b/oasst-sft-6-llama-30b"
quantized_model_dir = "quantization/models/huggingchat-30b-4bit"


# # load quantized model to the first GPU
model = AutoGPTQForCausalLM.from_quantized(quantized_model_dir, device="cuda:0", use_triton=False, use_tvm=True)
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_dir, use_fast=True)



# inference with model.generate
inputs = tokenizer(["Please Write a short story about a time traveler who accidentally alters history."], return_tensors="pt").to(model.device)

if "token_type_ids" in inputs:
    del inputs["token_type_ids"]

# outputs = model.generate(**inputs,
#     max_new_tokens=1024)
# print(outputs)
# print(tokenizer.decode(outputs[0], skip_special_tokens=True))

# or you can also use pipeline
pipeline = TextGenerationPipeline(model=model, tokenizer=tokenizer)
print(pipeline("Write a short story about a time traveler who accidentally alters history.", max_new_tokens=1024, temperature=0.8)[0]["generated_text"])

while True:
    text = input(">>> ")
    print(pipeline(text, max_new_tokens=1024)[0]["generated_text"])
