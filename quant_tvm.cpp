#include <torch/all.h>
#include <torch/python.h>
#include <c10/cuda/CUDAGuard.h>

void quant_kernel_3b_cuda(
    torch::Tensor mat_A, torch::Tensor mat_B, torch::Tensor mat_C,
    torch::Tensor scales, torch::Tensor zeros);

void quant_kernel_3b(
    torch::Tensor mat_A, torch::Tensor mat_B, torch::Tensor mat_C,
    torch::Tensor scales, torch::Tensor zeros)
{
  const at::cuda::OptionalCUDAGuard device_guard(device_of(mat_A));
  quant_kernel_3b_cuda(mat_A, mat_B, mat_C, scales, zeros);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("quant_kernel_3b", &quant_kernel_3b, "3-bit Quantized Matrix Multiplication (CUDA)");
}
