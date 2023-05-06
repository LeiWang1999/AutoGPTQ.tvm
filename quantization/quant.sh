# CUDA_VISIBLE_DEVICES=0 python ./opt.py /workspace/v-leiwang3/lowbit_model/opt_3bit_first_layer/checkpoint c4 --wbits 3 --faster-kernel --save  ./models/opt_layer1_3bit_faster.pt | tee quant.log

CUDA_VISIBLE_DEVICES=3 python ./opt.py facebook/opt-66b c4 --wbits 3 --faster-kernel --save  ./models/opt_66b_3bit_faster.pt | tee quant.log
