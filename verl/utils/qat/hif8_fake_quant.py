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

"""HiF8 fake quantization for QAT (Quantization-Aware Training).

HiF8 is Huawei Ascend's 8-bit floating point format with variable bit allocation.
It has two variants:
  - HIF8_15: max value 15.0, used for forward weights and activations
  - HIF8_224: max value 224.0, used for backward gradients

Quantization granularity aligns with MindSpeed's delayed_hif8_pertensor recipe
and vllm-ascend's dynamic quantization pattern:
  - Weight: per-channel (each output channel gets its own shared exponent)
  - Activation: per-token dynamic (each token independently quantized)

The shared-exponent formula matches torch_npu's HiF8 quantization:
  shared_exp = ceil(log2(amax / hif8_max))
  quantized = round(x / 2^shared_exp)
"""

import torch
import torch.nn.functional as F

__all__ = [
    "HIF8FakeQuantFunction",
    "HIF8_15_MAX",
    "HIF8_224_MAX",
    "hif8_per_channel_fake_quantize",
    "hif8_per_token_fake_quantize",
]

# HiF8 format constants (matching MindSpeed's FormatEnum)
# HIF8_15: forward weights/activations, max representable value = 15.0
# HIF8_224: backward gradients, max representable value = 224.0
HIF8_15_MAX: float = 15.0
HIF8_224_MAX: float = 224.0

# FP32 minimum normal value (2^-126), used to avoid division by zero
_FP32_MIN_NORMAL: float = 2.0**-126

# E8M0 scale maximum exponent
_SCALE_EMAX: float = 2 ** (8.0 - 1.0) - 1  # 127


def _compute_shared_exp(
    tensor_float: torch.Tensor,
    hif8_max: float,
    reduce_dim: int,
) -> torch.Tensor:
    """Compute per-group shared exponent for HiF8 quantization.

    shared_exp = ceil(log2(amax / hif8_max)), clamped to [-127, 127].

    Args:
        tensor_float: Input tensor in float32.
        hif8_max: Maximum representable value for the HiF8 variant.
        reduce_dim: Dimension along which to compute max and share exponent.

    Returns:
        Shared exponent tensor with keepdim=True along reduce_dim.
    """
    amax = torch.amax(torch.abs(tensor_float), dim=reduce_dim, keepdim=True)

    # Avoid log2(0)
    mask = (amax == 0).float()
    safe_amax = amax + _FP32_MIN_NORMAL * mask

    hif8_max_tensor = torch.tensor(hif8_max, dtype=torch.float32, device=tensor_float.device)
    shared_exp = torch.ceil(torch.log2(safe_amax) - torch.log2(hif8_max_tensor))
    shared_exp = torch.where(mask.bool(), torch.zeros_like(shared_exp), shared_exp)

    # Clamp to valid E8M0 range
    shared_exp = torch.clamp(shared_exp, -_SCALE_EMAX, _SCALE_EMAX)

    return shared_exp


def _apply_shared_exp_quantize(
    tensor_float: torch.Tensor,
    shared_exp: torch.Tensor,
    hif8_max: float,
) -> torch.Tensor:
    """Apply HiF8 quantization: scale down → round → clamp → scale up.

    tensor_dequant = clamp(round(tensor / 2^shared_exp), ±hif8_max) * 2^shared_exp

    Args:
        tensor_float: Input tensor in float32.
        shared_exp: Shared exponent tensor (broadcastable to tensor_float).
        hif8_max: Maximum representable value.

    Returns:
        Dequantized tensor in float32.
    """
    tensor_scaled = tensor_float / (2**shared_exp)
    tensor_scaled = torch.clamp(tensor_scaled, -hif8_max, hif8_max)
    tensor_quant = torch.round(tensor_scaled)
    tensor_dequant = tensor_quant * (2**shared_exp)
    return tensor_dequant


def hif8_per_channel_fake_quantize(
    weight: torch.Tensor,
    hif8_max: float = HIF8_15_MAX,
) -> torch.Tensor:
    """Apply per-channel HiF8 fake quantization to a weight tensor.

    Each output channel (row) gets its own shared exponent, computed from
    the maximum absolute value across all input features in that channel.

    This matches the MindSpeed delayed_hif8_pertensor pattern and
    vllm-ascend's per-channel weight quantization (weight_scale shape: (out_features, 1)).

    Args:
        weight: Weight tensor of shape (out_features, in_features).
        hif8_max: Maximum representable value. Default HIF8_15_MAX.

    Returns:
        Dequantized tensor in original dtype, same shape as input.
    """
    original_dtype = weight.dtype
    tensor_float = weight.to(torch.float32)

    # Per-channel shared exponent: one per output channel
    # amax along in_features dim → shared_exp shape: (out_features, 1)
    shared_exp = _compute_shared_exp(tensor_float, hif8_max, reduce_dim=-1)

    tensor_dequant = _apply_shared_exp_quantize(tensor_float, shared_exp, hif8_max)

    return tensor_dequant.to(original_dtype)


def hif8_per_token_fake_quantize(
    activation: torch.Tensor,
    hif8_max: float = HIF8_15_MAX,
) -> torch.Tensor:
    """Apply per-token HiF8 fake quantization to an activation tensor.

    Each token (row in 2D, or last-dim group in higher dims) gets its own
    shared exponent, matching the behavior of torch_npu.npu_dynamic_quant()
    with per-token scaling.

    Args:
        activation: Activation tensor of shape (..., in_features).
        hif8_max: Maximum representable value. Default HIF8_15_MAX.

    Returns:
        Dequantized tensor in original dtype, same shape as input.
    """
    original_dtype = activation.dtype
    original_shape = activation.shape

    # Flatten to 2D for per-token processing
    if activation.dim() == 3:
        x_2d = activation.view(-1, activation.shape[-1])
    else:
        x_2d = activation

    tensor_float = x_2d.to(torch.float32)

    # Per-token shared exponent: one per row
    shared_exp = _compute_shared_exp(tensor_float, hif8_max, reduce_dim=-1)

    tensor_dequant = _apply_shared_exp_quantize(tensor_float, shared_exp, hif8_max)

    return tensor_dequant.view(original_shape).to(original_dtype)


class HIF8FakeQuantFunction(torch.autograd.Function):
    """Straight-Through Estimator for HiF8 fake quantization.

    Supports two modes:
      - "per_channel": for weight quantization (each output channel independent)
      - "per_token": for activation quantization (each token independent)

    Forward: apply HiF8 fake quantization to simulate low-precision.
    Backward: pass gradient through unchanged (STE).
    """

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        tensor: torch.Tensor,
        mode: str = "per_channel",
        hif8_max: float = HIF8_15_MAX,
    ) -> torch.Tensor:
        """Apply HiF8 fake quantization.

        Args:
            tensor: Input tensor to fake-quantize.
            mode: "per_channel" for weights, "per_token" for activations.
            hif8_max: Max value for HiF8 variant.
        """
        if mode == "per_channel":
            return hif8_per_channel_fake_quantize(tensor, hif8_max=hif8_max)
        elif mode == "per_token":
            return hif8_per_token_fake_quantize(tensor, hif8_max=hif8_max)
        else:
            raise ValueError(f"Unknown HiF8 fake quant mode: {mode}. "
                             f"Expected 'per_channel' or 'per_token'.")

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_output: torch.Tensor,
    ) -> tuple:
        """Straight-through estimator: pass gradient through unchanged."""
        return grad_output, None, None
