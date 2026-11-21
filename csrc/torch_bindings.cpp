// SPDX-License-Identifier: Apache-2.0
#include <ATen/cuda/CUDAContext.h>
#include <pybind11/pybind11.h>
#include <torch/library.h>

#include "ops.h"

namespace {
at::Tensor ouro_mlp(const at::Tensor& hidden, const at::Tensor& gate_weight,
                    const at::Tensor& up_weight, const at::Tensor& down_weight) {
  // Keep HF's GEMM geometry and accumulation order. A single wider projection
  // can pick another cuBLAS reduction and change BF16 bits before activation.
  auto gate = at::linear(hidden, gate_weight);
  auto up = at::linear(hidden, up_weight);
  // The fused activation was qualified over the entire BF16 input domain on SM80.
  const auto* device = at::cuda::getDeviceProperties(gate.get_device());
  const bool fused = gate.scalar_type() == at::kBFloat16 && device->major == 8 && device->minor == 0;
  auto activation = fused
      ? vllm_rlt::silu_and_mul_cuda(gate, up)
      : at::mul(at::silu(gate), up);
  return at::linear(activation, down_weight);
}
}  // namespace

TORCH_LIBRARY(vllm_rlt, ops) {
  ops.def("ouro_mlp(Tensor hidden, Tensor gate_weight, Tensor up_weight, "
          "Tensor down_weight) -> Tensor");
  ops.def("silu_and_mul(Tensor gate, Tensor up) -> Tensor");
}

TORCH_LIBRARY_IMPL(vllm_rlt, CUDA, ops) {
  ops.impl("ouro_mlp", &ouro_mlp);
  ops.impl("silu_and_mul", &vllm_rlt::silu_and_mul_cuda);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {}
