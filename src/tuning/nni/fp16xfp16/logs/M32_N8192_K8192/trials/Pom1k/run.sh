#!/bin/bash
export NNI_PLATFORM='local'
export NNI_EXP_ID='M32_N8192_K8192'
export NNI_SYS_DIR='/workspace/v-leiwang3/GPTQ-tvm/src/tuning/nni/fp16xfp16/logs/M32_N8192_K8192/trials/Pom1k'
export NNI_TRIAL_JOB_ID='Pom1k'
export NNI_OUTPUT_DIR='/workspace/v-leiwang3/GPTQ-tvm/src/tuning/nni/fp16xfp16/logs/M32_N8192_K8192/trials/Pom1k'
export NNI_TRIAL_SEQ_ID='646'
export NNI_CODE_DIR='/workspace/v-leiwang3/GPTQ-tvm/src/tuning/nni/fp16xfp16'
export CUDA_VISIBLE_DEVICES='1'
cd $NNI_CODE_DIR
eval 'python3 ./fp16xfp16_nt_padwmma_template.py --M 32 --N 8192 --K 8192' 1>/workspace/v-leiwang3/GPTQ-tvm/src/tuning/nni/fp16xfp16/logs/M32_N8192_K8192/trials/Pom1k/stdout 2>/workspace/v-leiwang3/GPTQ-tvm/src/tuning/nni/fp16xfp16/logs/M32_N8192_K8192/trials/Pom1k/stderr
echo $? `date +%s%3N` >'/workspace/v-leiwang3/GPTQ-tvm/src/tuning/nni/fp16xfp16/logs/M32_N8192_K8192/trials/Pom1k/.nni/state'