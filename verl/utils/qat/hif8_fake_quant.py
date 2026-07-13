# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""HiF8 per-element fake quantization for QAT (Quantization-Aware Training).

HiF8 is Huawei Ascend's native 8-bit floating point format. Each element
is independently quantized without any external scale or shared exponent.

The fake quant simulates the precision loss of the roundtrip:
    x_hif8 = x.to(hifloat8)    # quantize to 8-bit float
    x_dequant = x_hif8.to(x_dtype)  # dequantize back
"""

import torch

__all__ = [
    "HIF8FakeQuantFunction",
    "hif8_per_element_fake_quantize",
]


def _get_hif8_dtype() -> torch.dtype | None:
    """Try to get the real torch_npu HiF8 dtype, fall back to float8_e4m3fn."""
    try:
        import torch_npu
        hifloat8 = getattr(torch_npu, "hifloat8", None)
        if hifloat8 is not None:
            return hifloat8
    except ImportError:
        pass
    # Fallback: float8_e4m3fn is the closest standard 8-bit float
    return torch.float8_e4m3fn


def hif8_per_element_fake_quantize(tensor: torch.Tensor) -> torch.Tensor:
    """Apply per-element HiF8 fake quantization: simulate .to(hifloat8).to(original_dtype).

    Each element is independently quantized — no external scale, no shared
    exponent. This is the native HiF8 quantization mode.

    Uses real torch_npu.hifloat8 dtype if available (NPU), otherwise falls
    back to float8_e4m3fn for approximate simulation on CPU/GPU.

    Args:
        tensor: Input tensor in any floating dtype (bf16/fp16/fp32).

    Returns:
        Dequantized tensor in the original dtype, simulating HiF8 per-element quantization.
    """
    original_dtype = tensor.dtype
    hif8_dtype = _get_hif8_dtype()

    return tensor.to(hif8_dtype).to(original_dtype)


class HIF8FakeQuantFunction(torch.autograd.Function):
    """Straight-Through Estimator for per-element HiF8 fake quantization.

    Forward:  x → .to(hifloat8) → .to(original_dtype)   (simulate precision loss)
    Backward: STE — gradient passes through unchanged.
    """

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        tensor: torch.Tensor,
    ) -> torch.Tensor:
        return hif8_per_element_fake_quantize(tensor)

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_output: torch.Tensor,
    ) -> tuple:
        return grad_output,
