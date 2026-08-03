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

"""QAT FakeQuantized Linear module for NVFP4 (W4A4/W4A16) with FSDP compatibility.

Includes Triton kernels for high-performance FP4 quantization.
"""

from enum import Enum
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["QATLinear", "QATMode", "HIF8QATLinear", "HIF8FakeQuantFunction"]


import triton
import triton.language as tl

_TORCH_TO_TL_DTYPE = {
    torch.float32: tl.float32,
    torch.float16: tl.float16,
    torch.bfloat16: tl.bfloat16,
}
FP4_E2M1_MAX: float = 6.0
FP8_E4M3_MAX: float = 448.0


@triton.jit
def _fp4_fake_quant_kernel(
    x_ptr,
    y_ptr,
    M,
    N,
    global_scale_ptr,
    stride_xm,
    stride_xn,
    stride_ym,
    stride_yn,
    BLOCK_SIZE: tl.constexpr,
    TILE_M: tl.constexpr,
    TILE_N: tl.constexpr,
    NUM_FP4_BLOCKS: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
    FP4_MAX: tl.constexpr,
    FP8_MAX: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    row_start = pid_m * TILE_M
    col_start = pid_n * TILE_N

    x_block_ptr = tl.make_block_ptr(
        base=x_ptr,
        shape=(M, N),
        strides=(stride_xm, stride_xn),
        offsets=(row_start, col_start),
        block_shape=(TILE_M, TILE_N),
        order=(1, 0),
    )
    y_block_ptr = tl.make_block_ptr(
        base=y_ptr,
        shape=(M, N),
        strides=(stride_ym, stride_yn),
        offsets=(row_start, col_start),
        block_shape=(TILE_M, TILE_N),
        order=(1, 0),
    )

    global_scale = tl.load(global_scale_ptr).to(tl.float32)
    global_scale_safe = tl.where(global_scale > 0.0, global_scale, 1e-12)

    tile = tl.load(x_block_ptr, boundary_check=(0, 1), padding_option="zero").to(tl.float32)
    tile_reshaped = tl.reshape(tile, (TILE_M, NUM_FP4_BLOCKS, BLOCK_SIZE))
    x_abs = tl.abs(tile_reshaped)

    block_max = tl.max(x_abs, axis=2, keep_dims=True)
    block_max_scaled = block_max / (FP4_MAX * global_scale_safe)
    block_max_scaled = tl.minimum(block_max_scaled, FP8_MAX)
    block_max_quant = block_max_scaled.to(tl.float8e4nv).to(tl.float32) * global_scale
    block_max_quant = tl.where(block_max_quant >= 1e-5, block_max_quant, 1.0)

    block_max_quant_broadcast = tl.broadcast_to(block_max_quant, (TILE_M, NUM_FP4_BLOCKS, BLOCK_SIZE))
    abs_scaled = x_abs / block_max_quant_broadcast

    q_val = tl.where(
        abs_scaled <= 0.25,
        0.0,
        tl.where(
            abs_scaled < 0.75,
            0.5,
            tl.where(
                abs_scaled <= 1.25,
                1.0,
                tl.where(
                    abs_scaled < 1.75,
                    1.5,
                    tl.where(
                        abs_scaled <= 2.5,
                        2.0,
                        tl.where(abs_scaled < 3.5, 3.0, tl.where(abs_scaled <= 5.0, 4.0, FP4_MAX)),
                    ),
                ),
            ),
        ),
    )

    x_rescaled = q_val * block_max_quant_broadcast
    x_rescaled = tl.where(tile_reshaped >= 0, x_rescaled, -x_rescaled)
    tile_quant = tl.reshape(x_rescaled, (TILE_M, TILE_N))

    tl.store(y_block_ptr, tile_quant.to(OUT_DTYPE), boundary_check=(0, 1))


def fp4_fake_quant_weight(
    weight: torch.Tensor,
    global_amax: torch.Tensor = None,
    block_size: int = 16,
    tile_rows: int = 16,
    tile_cols: int = 64,
) -> torch.Tensor:
    """Apply FP4 fake quantization using Triton kernel."""
    x_shape = weight.shape
    x_dtype = weight.dtype
    x = weight.reshape(-1, x_shape[-1]).contiguous()
    M, N = x.shape
    y = torch.empty_like(x)

    stride_xm, stride_xn = x.stride()
    stride_ym, stride_yn = y.stride()

    tile_cols = max(tile_cols, block_size)
    tile_cols_aligned = ((tile_cols + block_size - 1) // block_size) * block_size
    num_fp4_blocks = tile_cols_aligned // block_size

    if global_amax is None:
        global_amax = weight.abs().max().to(torch.float32)
    global_scale = global_amax.float() / (FP4_E2M1_MAX * FP8_E4M3_MAX)

    grid = (triton.cdiv(M, tile_rows), triton.cdiv(N, tile_cols_aligned))

    _fp4_fake_quant_kernel[grid](
        x,
        y,
        M,
        N,
        global_scale,
        stride_xm,
        stride_xn,
        stride_ym,
        stride_yn,
        BLOCK_SIZE=block_size,
        TILE_M=tile_rows,
        TILE_N=tile_cols_aligned,
        NUM_FP4_BLOCKS=num_fp4_blocks,
        OUT_DTYPE=_TORCH_TO_TL_DTYPE[x_dtype],
        FP4_MAX=FP4_E2M1_MAX,
        FP8_MAX=FP8_E4M3_MAX,
    )
    return y.view(*x_shape)


class STEFP4QuantTriton(torch.autograd.Function):
    """Straight-Through Estimator wrapper for Triton FP4 quantization kernel."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, global_amax: torch.Tensor, block_size: int) -> torch.Tensor:
        return fp4_fake_quant_weight(x, global_amax=global_amax, block_size=block_size)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple:
        return grad_output, None, None


class QATMode(str, Enum):
    """QAT quantization mode."""

    W4A4 = "w4a4"  # Weight 4-bit, Activation 4-bit (dynamic)
    W4A16 = "w4a16"  # Weight 4-bit, Activation 16-bit (weight only)


class QATLinear(nn.Linear):
    """QAT FakeQuantized Linear layer with FSDP compatibility."""

    _UNINITIALIZED_SCALE = -1.0

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        mode: QATMode = QATMode.W4A4,
        group_size: int = 16,
        activation_observer: str = "static_minmax",  # Observer strategy for activation global_scale
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__(in_features, out_features, bias, device=device, dtype=dtype)

        self.mode = mode
        self.group_size = group_size
        self.activation_observer = activation_observer

        self._weight_blockwise_scale: Optional[torch.Tensor] = None
        self._weight_global_scale: Optional[torch.Tensor] = None
        self._cached_weight_amax: Optional[torch.Tensor] = None
        self._fusion_siblings_ref = None

        if mode == QATMode.W4A4:
            self.register_buffer(
                "input_global_scale", torch.tensor([self._UNINITIALIZED_SCALE], dtype=torch.float32), persistent=True
            )

            self.register_buffer(
                "input_amax", torch.tensor([self._UNINITIALIZED_SCALE], dtype=torch.float32), persistent=True
            )

            self._ema_decay: float = 0.01

        self.fake_quant_enabled = True

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        mode: QATMode = QATMode.W4A4,
        group_size: int = 16,
        activation_observer: str = "static_minmax",
    ) -> "QATLinear":
        """Create QATLinear from an existing nn.Linear."""
        has_bias = linear.bias is not None

        new_linear = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            bias=has_bias,
            mode=mode,
            group_size=group_size,
            activation_observer=activation_observer,
            device=linear.weight.device,
            dtype=linear.weight.dtype,
        )

        if linear.weight.device != torch.device("meta"):
            new_linear.weight = nn.Parameter(linear.weight.clone())
            if has_bias:
                new_linear.bias = nn.Parameter(linear.bias.clone())

        return new_linear

    def _is_amax_initialized(self) -> bool:
        """Check if input_amax has been initialized."""
        if not hasattr(self, "input_amax"):
            return False
        return self.input_amax.item() != self._UNINITIALIZED_SCALE

    def _update_input_global_scale(self, x: torch.Tensor):
        """Update static input_global_scale based on observer strategy."""
        assert self.mode == QATMode.W4A4, "_update_input_global_scale should only be called in W4A4 mode"

        current_amax = torch.amax(torch.abs(x)).detach().to(torch.float32)

        if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
            torch.distributed.all_reduce(current_amax, op=torch.distributed.ReduceOp.MAX)

        scale_factor = FP8_E4M3_MAX * FP4_E2M1_MAX

        if self.activation_observer == "memoryless_minmax":
            new_scale = (scale_factor / (current_amax + 1e-12)).view(1)
            self.input_global_scale.copy_(new_scale.to(self.input_global_scale.device))

        elif self.activation_observer == "static_minmax":
            if not self._is_amax_initialized():
                self.input_amax.copy_(current_amax.view(1).to(self.input_amax.device))
            else:
                new_amax = torch.maximum(self.input_amax, current_amax.view(1).to(self.input_amax.device))
                self.input_amax.copy_(new_amax)
            amax_f32 = self.input_amax.to(torch.float32)
            new_scale = (scale_factor / (amax_f32 + 1e-12)).float().view(1)
            self.input_global_scale.copy_(new_scale.to(self.input_global_scale.device))

        elif self.activation_observer == "minmax":
            if not self._is_amax_initialized():
                self.input_amax.copy_(current_amax.view(1).to(self.input_amax.device))
            else:
                new_amax = (1 - self._ema_decay) * self.input_amax + self._ema_decay * current_amax.view(1).to(
                    self.input_amax.device
                )
                self.input_amax.copy_(new_amax)
            amax_f32 = self.input_amax.to(torch.float32)
            new_scale = (scale_factor / (amax_f32 + 1e-12)).float().view(1)
            self.input_global_scale.copy_(new_scale.to(self.input_global_scale.device))

        else:
            raise ValueError(f"Unknown activation_observer: {self.activation_observer}")

    def _fake_quantize_weight(self, weight: torch.Tensor) -> torch.Tensor:
        """Apply fake quantization to weight tensor using Triton kernel."""
        with torch.no_grad():
            if self._cached_weight_amax is not None:
                global_amax = self._cached_weight_amax
            else:
                siblings_ref = getattr(self, "_fusion_siblings_ref", None)

                if siblings_ref is not None:
                    siblings = [ref() for ref in siblings_ref if ref() is not None]
                    siblings = [s for s in siblings if s.weight.device != torch.device("meta")]

                    for sibling in siblings:
                        sibling_amax = getattr(sibling, "_cached_weight_amax", None)
                        if sibling_amax is not None:
                            global_amax = sibling_amax
                            self._cached_weight_amax = global_amax
                            break
                    else:
                        all_modules = [self] + siblings
                        amaxes = [m.weight.abs().max().to(torch.float32) for m in all_modules]
                        global_amax = torch.max(torch.stack(amaxes))

                        self._cached_weight_amax = global_amax
                        for sibling in siblings:
                            sibling._cached_weight_amax = global_amax
                else:
                    global_amax = weight.abs().max().to(torch.float32)
                    self._cached_weight_amax = global_amax

            if self._weight_global_scale is None:
                self._weight_global_scale = global_amax.float() / (FP4_E2M1_MAX * FP8_E4M3_MAX)

        result = STEFP4QuantTriton.apply(weight, global_amax, self.group_size)

        return result

    def _fake_quantize_activation(self, x: torch.Tensor) -> torch.Tensor:
        """Apply fake quantization to activation tensor (W4A4 mode only)."""
        original_shape = x.shape

        if x.dim() == 3:
            x_2d = x.view(-1, x.shape[-1])
        else:
            x_2d = x

        if self.training:
            self._update_input_global_scale(x_2d)

        if self.input_global_scale.item() == self._UNINITIALIZED_SCALE:
            raise RuntimeError("W4A4 input_global_scale uninitialized. Load PTQ model first.")

        global_amax = (FP4_E2M1_MAX * FP8_E4M3_MAX) / self.input_global_scale.to(x.device)
        result = STEFP4QuantTriton.apply(x_2d, global_amax, self.group_size)
        return result.view(original_shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with fake quantization."""
        if not self.fake_quant_enabled:
            return F.linear(x, self.weight, self.bias)

        weight_fq = self._fake_quantize_weight(self.weight)

        if self.mode == QATMode.W4A4:
            x_fq = self._fake_quantize_activation(x)
        else:
            x_fq = x

        return F.linear(x_fq, weight_fq, self.bias)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, mode={self.mode.value}, "
            f"group_size={self.group_size}, fake_quant_enabled={self.fake_quant_enabled}"
        )


# ============================================================================
# HiF8 Quantization
# ============================================================================
# HiF8 is Huawei Ascend's native 8-bit float with tapered precision.
# Fake quant uses pure-software _quant_hif8 that models the Dot-band encoding:
#   |e|<=3→3b, |e|<=7→2b, |e|<=15→1b mantissa.
#
# Granularity modes:
#   per_tensor:  one scale per tensor (weight: scalar, activation: scalar)
#   per_channel: one scale per output channel (weight: (out,1), activation: per-token)
#   per_group:   one scale per group of 32 elements along last dim
# ============================================================================


# HiF8 max: 2^15 × 1.5 = 49152 (Dot=4b, E=4b range [-15,15], M=1b)
HIF8_MAX: float = 49152.0


def _quant_hif8(x: torch.Tensor) -> torch.Tensor:
    """Raw HiFloat8 quantization with tapered precision (per element).
    Models the HiF8 Dot-band encoding in pure software:
      |e| <= 3  → 3-bit mantissa (Dot=2)
      |e| <= 7  → 2-bit mantissa (Dot=3)
      |e| <= 15 → 1-bit mantissa (Dot=4)
    No Python branches — safe for torch.compile / graph capture.
    """
    x_unsigned = x.abs()
    sign = x.sign()
    eps = x_unsigned.amax().clamp(min=1e-30) * 1e-8
    e = torch.floor(torch.log2(x_unsigned + eps))
    abse = e.abs()
    mant_bits = torch.where(abse <= 3, 3.0,
                   torch.where(abse <= 7, 2.0,
                   torch.where(abse <= 15, 1.0, 0.0)))
    q = torch.floor(x_unsigned * 2.0 ** (-e + mant_bits) + 0.5)
    return q * 2.0 ** (e - mant_bits) * sign


def hif8_fake_quant(
    tensor: torch.Tensor, granularity: str = "per_tensor", group_size: int = 32
) -> torch.Tensor:
    """HiF8 fake quant: scale → encode → decode roundtrip.
    granularity='per_tensor':  one scale per tensor  (amax over all elements)
    granularity='per_channel': one scale per row      (amax along last dim)
    granularity='per_group':   one scale per group    (amax per group_size along last dim)
    """
    if granularity == "per_group":
        t = tensor.float()
        dim_size = t.shape[-1]
        pad = (group_size - dim_size % group_size) % group_size
        if pad:
            t = F.pad(t, (0, pad))
        t_blocks = t.unflatten(-1, (-1, group_size))
        amax = t_blocks.abs().amax(dim=-1, keepdim=True)
        scale = (amax / HIF8_MAX).clamp(min=1e-12)
        q_blocks = _quant_hif8(t_blocks / scale) * scale
        result = q_blocks.flatten(-2, -1)
        if pad:
            result = result[..., :dim_size]
        return result.to(tensor.dtype)
    elif granularity == "per_channel":
        amax = tensor.float().abs().amax(dim=-1, keepdim=True)
    else:
        amax = tensor.float().abs().max()
    scale = (amax / HIF8_MAX).clamp(min=1e-12)
    return (_quant_hif8(tensor.float() / scale) * scale).to(tensor.dtype)


class HIF8FakeQuantFunction(torch.autograd.Function):
    """HiF8 QAT: configurable granularity → tapered precision roundtrip.
    Forward:  scale = amax/49152 → _quant_hif8 → ×scale
    Backward: STE (Straight-Through Estimator) — gradient passes through unchanged.
        This is the standard QAT convention: forward sees quantized values so
        the loss captures quant-error; backward uses raw gradients so weight
        updates are full-precision.  Quantizing the gradient would inject noise
        without helping the weights adapt to forward quant-error.
    """

    @staticmethod
    def forward(
        ctx, tensor: torch.Tensor,
        granularity: str = "per_tensor", group_size: int = 32
    ) -> torch.Tensor:
        return hif8_fake_quant(tensor, granularity, group_size).to(tensor.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple:
        return grad_output, None, None


class HIF8QATLinear(nn.Linear):
    """HiF8 FakeQuantized Linear — configurable granularity + activation.
    granularity='per_tensor':  one scale per tensor
    granularity='per_channel': one scale per output channel (weight) / token (act)
    granularity='per_group':   one scale per group of group_size elements
    quantize_activation=False: W8 (weight-only)
    quantize_activation=True:  W8A8 (full)
    FSDP-compatible (standard nn.Parameter, no extra state).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        quantize_activation: bool = False,
        granularity: str = "per_tensor",
        group_size: int = 32,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__(in_features, out_features, bias, device=device, dtype=dtype)
        self.quantize_activation = quantize_activation
        self.granularity = granularity
        self.group_size = group_size

    @classmethod
    def from_linear(
        cls, linear: nn.Linear,
        quantize_activation: bool = False,
        granularity: str = "per_tensor",
        group_size: int = 32,
    ) -> "HIF8QATLinear":
        """Create HIF8QATLinear from an existing nn.Linear, copying weights."""
        has_bias = linear.bias is not None
        new_linear = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            bias=has_bias,
            quantize_activation=quantize_activation,
            granularity=granularity,
            group_size=group_size,
            device=linear.weight.device,
            dtype=linear.weight.dtype,
        )
        if linear.weight.device != torch.device("meta"):
            new_linear.weight = nn.Parameter(linear.weight.clone())
            if has_bias:
                new_linear.bias = nn.Parameter(linear.bias.clone())
        return new_linear

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight_fq = HIF8FakeQuantFunction.apply(
            self.weight, self.granularity, self.group_size)
        if self.quantize_activation:
            x = HIF8FakeQuantFunction.apply(
                x, self.granularity, self.group_size)
        return F.linear(x, weight_fq, self.bias)
