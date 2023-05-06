export PYTHONPATH=/root/tvm_mma/python
export CUDA_VISIBLE_DEVICES=3

python tune_compress.py | tee run.log
