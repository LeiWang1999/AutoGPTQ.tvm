# GPTQ-tvm [MLC-LLM supports load auto-gptq prequantized checkpoints now, please move to this [pr](https://github.com/mlc-ai/mlc-llm/pull/694)]

This is a toy project to implement GPTQ Kernel with TVM, which supports diverse lowbit dtypes in kernel de-compression tuning and trikcy support for dynamic shape.

we also interagte this codegen stage into auto-gptq, which is an easy-to-use pakage for GPTQ.

## Text Generation Usage

```python
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from transformers import AutoTokenizer, TextGenerationPipeline
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig


pretrained_model_dir = "facebook/opt-66b"
quantized_model_dir = "opt-66b-3bit"


tokenizer = AutoTokenizer.from_pretrained(pretrained_model_dir, use_fast=True)
examples = [
    tokenizer(
        "auto-gptq is an easy-to-use model quantization library with user-friendly apis, based on GPTQ algorithm."
    )
]

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
print(pipeline("auto-gptq is")[0]["generated_text"])

'''
output is: auto-gptq is auto-gptq is a easy way to use tool command freeto use, easy to use
'''

```

## Reference

- [Apache TVM](https://github.com/apache/tvm)
- GPTQ Series
    - [GPTQ](https://github.com/IST-DASLab/gptq)
    - [GPTQ-triton](https://github.com/fpgaminer/GPTQ-triton)
    - [AutoGPTQ](https://github.com/PanQiWei/AutoGPTQ)
