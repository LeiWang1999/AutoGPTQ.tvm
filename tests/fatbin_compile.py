import os
import json

kernel_path = "/workspace/v-leiwang3/lowbit_workspace/GPTQ-tvm/.cache/A6000.json"
output_path = "/workspace/v-leiwang3/lowbit_workspace/GPTQ-tvm/tests/rtx-a6000"
sm_arch = 'sm_86'
# load kernel json
with open(kernel_path, 'r') as f:
    kernel = json.load(f)

# if output path not exist, create it
if not os.path.exists(output_path):
    os.makedirs(output_path)
    
f_map = {}
for k in kernel:
    print(k)
    for m in kernel[k]:
        kernels = kernel[k][m]
        code = kernels['code']
        func_name = kernels['func_name']
        bin_name = f"{k}.{m}.fatbin"
        bin_output_path = os.path.join(output_path, bin_name)
        # write code into a tmp file
        tmp_code_path = os.path.join(output_path, f"{k}.{m}.cu")
        with open(tmp_code_path, 'w') as f:
            f.write(code)
        
        # use nvcc to compile fatbin
        cmd = f"nvcc -arch={sm_arch} -fatbin -o {bin_output_path} {tmp_code_path}"
        os.system(cmd)
        print(cmd)
        f_map[bin_name] = {}
        # write the map from bin_name to func_name
        f_map[bin_name]['func_name'] = func_name
        f_map[bin_name]['params'] = kernels['params']

# write the map into a json file
with open(os.path.join(output_path, 'func_map.json'), 'w') as f:
    json.dump(f_map, f, indent=4)
    