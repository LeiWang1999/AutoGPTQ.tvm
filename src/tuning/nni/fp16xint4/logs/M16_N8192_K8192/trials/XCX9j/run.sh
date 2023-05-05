#!/bin/bash
export NNI_PLATFORM='local'
export NNI_EXP_ID='M16_N8192_K8192'
export NNI_SYS_DIR='/workspace/v-leiwang3/GPTQ-tvm/src/tuning/nni/fp16xint4/logs/M16_N8192_K8192/trials/XCX9j'
export NNI_TRIAL_JOB_ID='XCX9j'
export NNI_OUTPUT_DIR='/workspace/v-leiwang3/GPTQ-tvm/src/tuning/nni/fp16xint4/logs/M16_N8192_K8192/trials/XCX9j'
export NNI_TRIAL_SEQ_ID='15'
export NNI_CODE_DIR='/workspace/v-leiwang3/GPTQ-tvm/src/tuning/nni/fp16xint4'
export CUDA_VISIBLE_DEVICES='1'
cd $NNI_CODE_DIR
eval 'python3 ./fp16xint4_nt_padwmma_template.py --M 16 --N 8192 --K 8192' 1>/workspace/v-leiwang3/GPTQ-tvm/src/tuning/nni/fp16xint4/logs/M16_N8192_K8192/trials/XCX9j/stdout 2>/workspace/v-leiwang3/GPTQ-tvm/src/tuning/nni/fp16xint4/logs/M16_N8192_K8192/trials/XCX9j/stderr
echo $? `date +%s%3N` >'/workspace/v-leiwang3/GPTQ-tvm/src/tuning/nni/fp16xint4/logs/M16_N8192_K8192/trials/XCX9j/.nni/state'