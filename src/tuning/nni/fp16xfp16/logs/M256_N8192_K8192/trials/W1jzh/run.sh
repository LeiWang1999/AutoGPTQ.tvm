#!/bin/bash
export NNI_PLATFORM='local'
export NNI_EXP_ID='M256_N8192_K8192'
export NNI_SYS_DIR='/workspace/v-leiwang3/GPTQ-tvm/src/tuning/nni/fp16xfp16/logs/M256_N8192_K8192/trials/W1jzh'
export NNI_TRIAL_JOB_ID='W1jzh'
export NNI_OUTPUT_DIR='/workspace/v-leiwang3/GPTQ-tvm/src/tuning/nni/fp16xfp16/logs/M256_N8192_K8192/trials/W1jzh'
export NNI_TRIAL_SEQ_ID='698'
export NNI_CODE_DIR='/workspace/v-leiwang3/GPTQ-tvm/src/tuning/nni/fp16xfp16'
export CUDA_VISIBLE_DEVICES='1'
cd $NNI_CODE_DIR
eval 'python3 ./fp16xfp16_nt_padwmma_template.py --M 256 --N 8192 --K 8192' 1>/workspace/v-leiwang3/GPTQ-tvm/src/tuning/nni/fp16xfp16/logs/M256_N8192_K8192/trials/W1jzh/stdout 2>/workspace/v-leiwang3/GPTQ-tvm/src/tuning/nni/fp16xfp16/logs/M256_N8192_K8192/trials/W1jzh/stderr
echo $? `date +%s%3N` >'/workspace/v-leiwang3/GPTQ-tvm/src/tuning/nni/fp16xfp16/logs/M256_N8192_K8192/trials/W1jzh/.nni/state'