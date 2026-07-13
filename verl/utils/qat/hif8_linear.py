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

"""W8A8 HiF8 QAT FakeQuantized Linear module for FSDP training.

This module replaces nn.Linear during QAT training to simulate HiF8
quantization during the forward pass while keeping bf16 master weights.

Quantization granularity (aligns with MindSpeed delayed_hif8_pertensor):
  - Weight: per-channel (each output channel gets its own shared exponent)
  - Activation: per-token dynamic (each token independently quantized)

Key design:
  - STE backward: gradients flow through unchanged
  - fake_quant_enabled toggle: for deployment without re-architecting
  - FSDP compatible: uses standard nn.Parameter for weight and bias
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from verl.utils.qat.hif8_fake_quant import HIF8FakeQuantFunction, HIF8_15_MAX

__all__ = ["HIF8QATLinear"]


class HIF8QATLinear(nn.Linear):
    """W8A8 HiF8 FakeQuantized Linear layer with FSDP compatibility.

    Replaces nn.Linear during QAT for HiF8 quantization. In forward:
      1. Fake-quantize weight: per-channel → weight_fq
      2. Fake-quantize activation: per-token → x_fq
      3. Compute matmul in bf16 with dequantized values
      4. STE backward passes gradients to bf16 master weights

    Compatible with FSDP wrapping — the module uses standard nn.Parameter
    for weight and bias, so FSDP sharding works transparently.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        weight_hif8_max: float = HIF8_15_MAX,
        act_hif8_max: float = HIF8_15_MAX,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__(in_features, out_features, bias, device=device, dtype=dtype)

        self.weight_hif8_max = weight_hif8_max
        self.act_hif8_max = act_hif8_max

        # Toggle for switching between QAT and vanilla forward
        self.fake_quant_enabled: bool = True

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        weight_hif8_max: float = HIF8_15_MAX,
        act_hif8_max: float = HIF8_15_MAX,
    ) -> "HIF8QATLinear":
        """Create HIF8QATLinear from an existing nn.Linear, copying weights.

        Args:
            linear: Source nn.Linear module.
            weight_hif8_max: Max value for weight quantization. Default HIF8_15_MAX.
            act_hif8_max: Max value for activation quantization. Default HIF8_15_MAX.

        Returns:
            New HIF8QATLinear with copied parameters.
        """
        has_bias = linear.bias is not None

        new_linear = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            bias=has_bias,
            weight_hif8_max=weight_hif8_max,
            act_hif8_max=act_hif8_max,
            device=linear.weight.device,
            dtype=linear.weight.dtype,
        )

        if linear.weight.device != torch.device("meta"):
            new_linear.weight = nn.Parameter(linear.weight.clone())
            if has_bias:
                new_linear.bias = nn.Parameter(linear.bias.clone())

        return new_linear

    def _fake_quantize_weight(self, weight: torch.Tensor) -> torch.Tensor:
        """Apply per-channel HiF8 fake quantization to weight tensor.

        Each output channel gets its own shared exponent, matching the
        per-channel weight quantization pattern in vllm-ascend.

        Args:
            weight: High-precision weight tensor (out_features, in_features).

        Returns:
            Dequantized weight tensor simulating HiF8 precision.
        """
        return HIF8FakeQuantFunction.apply(
            weight, "per_channel", self.weight_hif8_max
        )

    def _fake_quantize_activation(self, x: torch.Tensor) -> torch.Tensor:
        """Apply per-token HiF8 fake quantization to activation tensor.

        Each token is quantized independently along the feature dimension,
        matching torch_npu.npu_dynamic_quant() per-token behavior.

        Args:
            x: Activation tensor of shape (..., in_features).

        Returns:
            Dequantized activation tensor simulating HiF8 precision.
        """
        return HIF8FakeQuantFunction.apply(
            x, "per_token", self.act_hif8_max
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with HiF8 fake quantization.

        If fake_quant_enabled is False, falls back to standard bf16 matmul.

        Args:
            x: Input activation tensor.

        Returns:
            Output tensor in bf16/fp16.
        """
        if not self.fake_quant_enabled:
            return F.linear(x, self.weight, self.bias)

        # Fake-quantize both weight and activation
        weight_fq = self._fake_quantize_weight(self.weight)

        # W8A8: always quantize activations (unlike W4A16 which skips activation quant)
        x_fq = self._fake_quantize_activation(x)

        # Compute matmul in full precision with dequantized values
        return F.linear(x_fq, weight_fq, self.bias)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, "
            f"fake_quant_enabled={self.fake_quant_enabled}"
        )
