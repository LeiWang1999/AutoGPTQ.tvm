import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from transformers import AutoTokenizer, TextGenerationPipeline
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

from auto_gptq.nn_modules import qlinear_tvm
qlinear_tvm.is_mlc_llm = True
from auto_gptq.nn_modules.tvm_untils import cache, pycuda_warpper
cache.cache_dir = "/workspace/v-leiwang3/lowbit_workspace/GPTQ-tvm/.cache"
pycuda_warpper.nni_database_path = "/workspace/v-leiwang3/lowbit_workspace/GPTQ-tvm/.nnidatabase"

bits = 4
pretrained_model_dir = "/workspace/v-leiwang3/lowbit_workspace/mlc-llm/dist/models/llama-7b-hf"
quantized_model_dir = f"quantization/models/llama-7b-{bits}bit-mlc-llm"

# pretrained_model_dir = "facebook/opt-125m"
# quantized_model_dir = f"quantization/models/opt-125m-{bits}bit"

tokenizer = AutoTokenizer.from_pretrained(pretrained_model_dir, use_fast=True)
examples = [
    tokenizer(
        "auto-gptq is an easy-to-use model quantization library with user-friendly apis, based on GPTQ algorithm."
    )
]

quantize_config = BaseQuantizeConfig(
    bits=bits,  # quantize model to 3-bit
    # group_size=-1,
    sym=False,
    desc_act=True,    
)

# load un-quantized model, the model will always be force loaded into cpu
model = AutoGPTQForCausalLM.from_pretrained(pretrained_model_dir, quantize_config)

# quantize model, the examples should be list of dict whose keys can only be "input_ids" and "attention_mask" 
# with value under torch.LongTensor type.
model.quantize(examples, use_triton=False, use_tvm=True)

# save quantized model
model.save_quantized(quantized_model_dir)


# load quantized model to the first GPU
model = AutoGPTQForCausalLM.from_quantized(quantized_model_dir, device="cuda:0", use_triton=False, use_tvm=True)
for name, tensor in model.state_dict().items():
    print("name: ", name, "tensor: ", tensor.shape, tensor.dtype)
# # or you can also use pipeline
# pipeline = TextGenerationPipeline(model=model, tokenizer=tokenizer)
# print(pipeline("auto-gptq is")[0]["generated_text"])
