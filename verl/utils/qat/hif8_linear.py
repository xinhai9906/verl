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

"""W8A8 HiF8 QAT FakeQuantized Linear layer (per-element, no external scale).

Replaces nn.Linear during QAT training to simulate HiF8 inference precision.
Both weights and activations are independently quantized per-element:
    x_hif8 = x.to(hifloat8)
    weight_hif8 = weight.to(hifloat8)

Key design:
  - STE backward: gradients flow through unchanged
  - FSDP compatible: uses standard nn.Parameter
  - No external scales, no blocks, no shared exponents
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from verl.utils.qat.hif8_fake_quant import HIF8FakeQuantFunction

__all__ = ["HIF8QATLinear"]


class HIF8QATLinear(nn.Linear):
    """W8A8 HiF8 FakeQuantized Linear — per-element, native quantization.

    Forward pass:
      1. weight_fq = weight.to(hifloat8).to(bf16)  — per-element weight quant
      2. x_fq = x.to(hifloat8).to(bf16)            — per-element activation quant
      3. output = x_fq @ weight_fq^T + bias        — bf16 matmul

    No external scale tensors. Each element is quantized independently
    by the HiF8 float format's native representable precision.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__(in_features, out_features, bias, device=device, dtype=dtype)
        self.fake_quant_enabled: bool = True

    @classmethod
    def from_linear(cls, linear: nn.Linear) -> "HIF8QATLinear":
        """Create HIF8QATLinear from an existing nn.Linear, copying weights."""
        has_bias = linear.bias is not None

        new_linear = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            bias=has_bias,
            device=linear.weight.device,
            dtype=linear.weight.dtype,
        )

        if linear.weight.device != torch.device("meta"):
            new_linear.weight = nn.Parameter(linear.weight.clone())
            if has_bias:
                new_linear.bias = nn.Parameter(linear.bias.clone())

        return new_linear

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.fake_quant_enabled:
            return F.linear(x, self.weight, self.bias)

        # Per-element fake quant: simulate .to(hifloat8) roundtrip
        weight_fq = HIF8FakeQuantFunction.apply(self.weight)
        x_fq = HIF8FakeQuantFunction.apply(x)

        return F.linear(x_fq, weight_fq, self.bias)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, "
            f"fake_quant_enabled={self.fake_quant_enabled}"
        )
