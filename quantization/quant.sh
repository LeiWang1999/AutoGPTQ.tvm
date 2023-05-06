CUDA_VISIBLE_DEVICES=3 python ./opt.py facebook/opt-66b c4 --wbits 3 --faster-kernel --save  ./models/opt_66b_3bit_faster.pt | tee quant.log
