// SPDX-License-Identifier: Apache-2.0
#include <pybind11/pybind11.h>
#include <torch/library.h>

#include "ops.h"

TORCH_LIBRARY(vllm_rlt, ops) {
  ops.def("packed_silu_and_mul(Tensor input) -> Tensor");
  ops.def("silu_and_mul(Tensor gate, Tensor up) -> Tensor");
}

TORCH_LIBRARY_IMPL(vllm_rlt, CUDA, ops) {
  ops.impl("packed_silu_and_mul", &vllm_rlt::packed_silu_and_mul_cuda);
  ops.impl("silu_and_mul", &vllm_rlt::silu_and_mul_cuda);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {}
