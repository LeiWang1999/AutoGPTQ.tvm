import os
import sys
os.environ['PYTHONPATH'] = r'/workspace/v-leiwang3/transformers_origin/src'
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['STATIC_GRAPH'] = 'True'

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from transformers import AutoTokenizer, TextGenerationPipeline, AutoModelForCausalLM
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
import torch
import time

enable_quantize = False
export_nnfusion = True
use_triton = False

assert not (export_nnfusion and use_triton)

pretrained_model_dir = "/workspace/v-leiwang3/AutoGPTQ_nf4/models/Llama-2-70b-hf"
quantized_model_dir = f"/workspace/v-leiwang3/AutoGPTQ.tvm/quantization/models/llama2-70b_nf4_g128"

tokenizer = AutoTokenizer.from_pretrained(pretrained_model_dir, use_fast=True)

examples = [
    tokenizer(
        "auto-gptq is an easy-to-use model quantization library with user-friendly apis, based on GPTQ algorithm."
    )
]

if enable_quantize:
    quantize_config = BaseQuantizeConfig(
        bits=4,  # quantize bits
        # desc_act=False,  # disable activation description
        # group_size=128,  # disable group quantization
        desc_act=True
    )

    # load un-quantized model, the model will always be force loaded into cpu
    model = AutoGPTQForCausalLM.from_pretrained(pretrained_model_dir, quantize_config)
    model.model.model.layers = model.model.model.layers[:1]
    
    # quantize model, the examples should be list of dict whose keys can only be "input_ids" and "attention_mask" 
    # with value under torch.LongTensor type.
    model.quantize(examples, use_triton=use_triton, export_nnfusion=export_nnfusion)

    # save quantized model
    model.save_quantized(quantized_model_dir)
    # save tokenizer
    tokenizer.save_pretrained(quantized_model_dir)
    
# load quantized model to the first GPU
model = AutoGPTQForCausalLM.from_quantized(quantized_model_dir, device="cuda:0", use_triton=use_triton, export_nnfusion=export_nnfusion).half().cuda()
# or you can also use pipeline
# pipeline = TextGenerationPipeline(model=model, tokenizer=tokenizer)
# print(pipeline("auto-gptq is")[0]["generated_text"])

# export 2 onnx
batch_size = 1
seq_length = 1
input_shape = (batch_size, seq_length)
onnx_name = f"qmodel_b{batch_size}s{seq_length}.onnx"
output_path = os.path.join(quantized_model_dir, f"qmodel_b{batch_size}s{seq_length}", onnx_name)
if not os.path.exists(os.path.dirname(output_path)):
    os.makedirs(os.path.dirname(output_path))
# "input_ids": torch.ones([batch, 1], dtype=torch.long),
# "attention_mask": torch.ones([batch, seq_single], dtype=torch.long),
# "past_key_values": tuple(
#     (torch.rand([batch, head, seq_single, hid]), torch.rand([batch, head, seq_single, hid])) for _ in range(layer)),
# "cur_index": torch.ones([1], dtype=torch.long)
input_ids = torch.ones(input_shape, dtype=torch.long, device="cuda:0")
attention_mask = torch.ones(input_shape, dtype=torch.long, device="cuda:0")
past_key_values = tuple(
    (torch.rand([batch_size, 12, seq_length, 64], device="cuda:0"), torch.rand([batch_size, 12, seq_length, 64], device="cuda:0")) for _ in range(1))
cur_index = torch.ones([1], dtype=torch.long, device="cuda:0")


class WrapperModelSingle(torch.nn.Module):
    def __init__(self, model):
        super(WrapperModelSingle, self).__init__()
        self._model = model

    def forward(self, input_ids, attention_mask, past_key_values, cur_index):
        out = self._model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=None,
            return_dict=True,
            output_attentions=False,
            output_hidden_states=False,
            cur_index=cur_index
            )
        return out

if not export_nnfusion:
    start = time.time()
    for i in range(100):
        outputs = model(input_ids=input_ids)
    end = time.time()
    print("time: ", (end - start) / 100 * 1000, "ms")
    print(outputs.logits)
else:
    model = model.half().cuda()
    wrapper_model = WrapperModelSingle(model)
    torch.onnx.export(      
        wrapper_model,  
        (input_ids, 
        attention_mask,
        past_key_values,
        cur_index),
        f=output_path,  
        opset_version=11, 
    )     
    import onnx
    from onnxsim import simplify
    model = onnx.load(output_path)
    model_simp, check = simplify(model)
    assert check, "Simplified ONNX model could not be validated"
    onnx.save(model_simp, output_path, save_as_external_data=True)
    print("export onnx done")