import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))  
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from transformers import AutoTokenizer, TextGenerationPipeline
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig


# pretrained_model_dir = "facebook/opt-125m"
# quantized_model_dir = "quantization/models/opt-125m-3bit"

# pretrained_model_dir = "/workspace/v-leiwang3/lowbit_model/opt_3bit_first_layer/checkpoint/"
# quantized_model_dir = "quantization/models/opt-66b-1layer-3bit"


pretrained_model_dir = "/workspace/v-leiwang3/lowbit_model/oa_llama_30b/oasst-sft-6-llama-30b"
quantized_model_dir = "quantization/models/huggingchat-30b-4bit"

enable_quantize = False

tokenizer = AutoTokenizer.from_pretrained(pretrained_model_dir, use_fast=True)
examples = [
    tokenizer(
        "auto-gptq is an easy-to-use model quantization library with user-friendly apis, based on GPTQ algorithm."
    )
]


if enable_quantize:
    quantize_config = BaseQuantizeConfig(
        bits=3,  # quantize model to 3-bit
        # desc_act=False,  # disable activation description
        # group_size=128,  # disable group quantization
        desc_act=True
    )
    # load un-quantized model, the model will always be force loaded into cpu
    model = AutoGPTQForCausalLM.from_pretrained(pretrained_model_dir, quantize_config)

    # quantize model, the examples should be list of dict whose keys can only be "input_ids" and "attention_mask" 
    # with value under torch.LongTensor type.
    model.quantize(examples, use_tvm=True)

    # save quantized model
    model.save_quantized(quantized_model_dir)

# load quantized model to the first GPU
model = AutoGPTQForCausalLM.from_quantized(quantized_model_dir, device="cuda:0", use_tvm=True)

# or you can also use pipeline
pipeline = TextGenerationPipeline(model=model, tokenizer=tokenizer)
print(pipeline("auto_gptq is an interesting")[0]["generated_text"])


'''
output is: auto-gptq is auto-gptq is a easy way to use tool command freeto use, easy to use
'''