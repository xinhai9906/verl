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

"""
Fast NVFP4 Quantizer for verl FSDP training.

Directly computes scales and quantizes weights using compressed_tensors APIs.
Includes scale computation utilities for weight quantization.
"""

import logging
import os
import re
from typing import Generator, Iterable, Optional

import torch
from compressed_tensors.compressors.quantized_compressors.fp4_quantized import NVFP4PackedCompressor
from compressed_tensors.quantization.quant_args import (
    FP4_E2M1_DATA,
    FP8_E4M3_DATA,
    QuantizationArgs,
    QuantizationStrategy,
    QuantizationType,
)
from compressed_tensors.quantization.utils.helpers import generate_gparam

from verl.utils.device import get_device_name, get_torch_device

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

_LAYER_IDX_RE = re.compile(r"layers\.(\d+)\.")


def compute_blockwise_scale(
    weight: torch.Tensor,
    global_scale: torch.Tensor,
    group_size: int = 16,
) -> torch.Tensor:
    """Compute blockwise scale using pre-computed global_scale (for fusion).
    Returns FP8 E4M3 blockwise scale tensor.
    """
    out_features, in_features = weight.shape
    num_groups = in_features // group_size
    weight_reshaped = weight.view(out_features, num_groups, group_size)
    block_max = torch.amax(torch.abs(weight_reshaped), dim=-1).to(torch.float32)

    local_scale = block_max / FP4_E2M1_DATA.max
    blockwise_scale_f32 = torch.clamp(
        global_scale * local_scale,
        min=-FP8_E4M3_DATA.max,
        max=FP8_E4M3_DATA.max,
    )

    blockwise_scale = blockwise_scale_f32.to(torch.float8_e4m3fn)
    eps = torch.finfo(torch.float8_e4m3fn).eps
    blockwise_scale = torch.where(
        blockwise_scale == 0,
        torch.tensor(eps, dtype=blockwise_scale.dtype, device=weight.device),
        blockwise_scale,
    )

    return blockwise_scale


# Fusion patterns for transformer models
FUSE_PATTERNS = {
    "qkv": ["q_proj", "k_proj", "v_proj"],
    "gate_up": ["gate_proj", "up_proj"],
}


def fuse_global_scales(
    layer_global_scales: dict[str, torch.Tensor],
    strategy: str = "min",
) -> dict[str, torch.Tensor]:
    """Fuse global scales for QKV/GateUp groups (take min across group)."""
    if not layer_global_scales:
        return {}

    # Group by parent module
    parent_to_children: dict[str, dict[str, str]] = {}
    for name in layer_global_scales:
        parent, child = name.rsplit(".", 1) if "." in name else ("", name)
        parent_to_children.setdefault(parent, {})[child] = name

    fused_scales = {}
    processed = set()

    for parent, children in parent_to_children.items():
        for _, patterns in FUSE_PATTERNS.items():
            matched = [children[p] for p in patterns if p in children]
            if len(matched) == len(patterns):
                group_scales = [layer_global_scales[n] for n in matched]
                if strategy == "min":
                    fused_scale = torch.min(torch.cat(group_scales)).reshape([1])
                else:
                    raise ValueError(f"Unknown fuse strategy: {strategy}")
                for layer_name in matched:
                    fused_scales[layer_name] = fused_scale.clone()
                    processed.add(layer_name)

    for name, scale in layer_global_scales.items():
        if name not in processed:
            fused_scales[name] = scale

    return fused_scales


def _encode_hif8(xq: torch.Tensor) -> torch.Tensor:
    """Encode software-quantized HiF8 values into true HiF8 uint8 bit format.

    Exponent is sign-magnitude: E_field = [Es(1b)][Emag(Db)].
    Dot bands and bit layout:
      Dot=4: S(7) | 11(6,5) | Es,Emag2..0(4..1) | M(0)     E=1sign+3mag
      Dot=3: S(7) | 10(6,5) | Es,Emag1..0(4..2) | M(1,0)   E=1sign+2mag
      Dot=2: S(7) | 01(6,5) | Es,Emag0(4,3) | M(2,1,0)     E=1sign+1mag
      Dot=1: S(7) | 001(6,5,4) | Es(3) | M(2,1,0)           E=1sign+0mag
      Dot=0: S(7) | 0001(6,5,4,3) | M(2,1,0)                E=0
    """
    xf = xq.float()
    S = (xf < 0)
    abs_x = xf.abs()
    eps = abs_x.amax().clamp(min=1e-30) * 1e-12
    e_raw = torch.floor(torch.log2(abs_x + eps)).int()
    abse = e_raw.abs().int()
    Es = (e_raw < 0).int()  # exponent sign: 0=pos, 1=neg

    # Dot bands
    d4 = abse >= 8
    d3 = (abse >= 4) & (~d4)
    d2 = (abse >= 2) & (~d4) & (~d3)
    d1 = (abse == 1)
    d0 = (~d4) & (~d3) & (~d2) & (~d1)

    dot_enc = (d4.int() * 0b11 + d3.int() * 0b10 + d2.int() * 0b01
               + d1.int() * 0b001 + d0.int() * 0b0001)
    dot_len = (d4 | d3 | d2).int() * 2 + d1.int() * 3 + d0.int() * 4

    # Exponent magnitude bits (stored, without implicit MSB=1)
    e_mag_bits = d4.int() * 3 + d3.int() * 2 + d2.int() * 1
    e_implicit = d4.int() * 8 + d3.int() * 4 + d2.int() * 2 + d1.int() * 1
    e_mag = (abse - e_implicit).clamp(0, 15).int()

    # E field = [Es @ high bit][Emag @ low bits]
    e_field = (Es << e_mag_bits) | e_mag

    # Mantissa
    m_bits = d4.int() * 1 + d3.int() * 2 + (d2 | d1 | d0).int() * 3
    mant = abs_x / torch.pow(2.0, e_raw.float()).clamp(min=1e-30)
    M = torch.round((mant - 1.0) * torch.pow(2.0, m_bits.float()))
    M = M.clamp(0, torch.pow(2.0, m_bits.float()).int() - 1).int()

    # Pack: S(7) | Dot | E_field | M(0)
    out = (S.int() << 7)
    out |= dot_enc << (7 - dot_len)
    out |= e_field << m_bits
    out |= M
    return out.to(torch.uint8)


class QATQuantizer:
    """Quantizer for QAT-trained weights using compressed_tensors APIs.

    Supports:
      - w4a16 / w4a4: NVFP4 quantization via compressed_tensors
      - w8_hif8: HiF8 per-tensor native + per-tensor scale (tensorwise)
    """

    def __init__(
        self,
        mode: str = "w4a16",
        group_size: int = 16,
        ignore_patterns: Optional[list] = None,
        device: Optional[torch.device] = None,
        param_dtype: Optional[torch.dtype] = None,
    ):
        self.mode = mode.lower()
        self._is_w4a4 = self.mode == "w4a4"  # W4A4 needs input_global_scale
        self._is_hif8 = self.mode == "w8_hif8"  # W8 HiF8 weight-only, per-tensor native
        self.group_size = group_size
        self.ignore_patterns = ignore_patterns or ["lm_head", "embed_tokens", "re:.*mlp.gate$"]
        self.device = device or torch.device(get_device_name())
        self.param_dtype = param_dtype

        # HiF8 mode doesn't use compressed_tensors
        if not self._is_hif8:
            self._compressor = NVFP4PackedCompressor()
            self._quant_args = QuantizationArgs(
                num_bits=4,
                type=QuantizationType.FLOAT,
                symmetric=True,
                strategy=QuantizationStrategy.TENSOR_GROUP,
                group_size=group_size,
                scale_dtype=FP8_E4M3_DATA.dtype,
            )

    def _should_quantize(self, name: str, tensor: torch.Tensor) -> bool:
        """Check if parameter should be quantized."""
        if not name.endswith(".weight"):
            return False
        if tensor.dim() != 2:
            return False

        # HiF8 per-tensor: no block/group dimension constraint
        if not self._is_hif8 and tensor.shape[1] % self.group_size != 0:
            return False

        module_name = name.rsplit(".weight", 1)[0]

        for pattern in self.ignore_patterns:
            if pattern.startswith("re:"):
                # Regex pattern - use re.match like vLLM does
                regex = pattern[3:]
                if re.match(regex, module_name):
                    return False
            else:
                if pattern in module_name:
                    return False
        return True

    @staticmethod
    def _extract_layer_idx(name: str) -> Optional[int]:
        """Extract decoder layer index from parameter name."""
        match = _LAYER_IDX_RE.search(name)
        return int(match.group(1)) if match else None

    def _process_layer_group(
        self,
        layer_idx: Optional[int],
        layer_params: dict[str, torch.Tensor],
        input_global_scales: dict[str, torch.Tensor],
        output_device: torch.device,
    ) -> list[tuple[str, torch.Tensor]]:
        """Quantize one decoder layer's buffered params. Returns list of (name, tensor)."""
        layer_weights = {}
        layer_passthrough = {}

        for name, tensor in layer_params.items():
            if "input_global_scale" in name or "input_amax" in name:
                continue

            if self._should_quantize(name, tensor):
                layer_name = name.rsplit(".weight", 1)[0]
                layer_weights[layer_name] = (name, tensor)
            else:
                layer_passthrough[name] = tensor

        if layer_idx is None and layer_weights:
            raise RuntimeError(
                f"[QAT Quantizer] Unexpected quantizable weights outside decoder layers: "
                f"{list(layer_weights.keys())}. These should be in ignore_patterns."
            )

        if not layer_weights:
            return [(name, tensor.to(output_device)) for name, tensor in layer_passthrough.items()]

        # Move weights to GPU, compute global scales
        weights_on_gpu = {}
        layer_global_scales = {}

        for layer_name, (_, tensor) in layer_weights.items():
            weight_gpu = tensor.to(device=self.device, dtype=self.param_dtype)
            weights_on_gpu[layer_name] = weight_gpu
            amax = torch.amax(torch.abs(weight_gpu)).to(torch.float32)
            layer_global_scales[layer_name] = generate_gparam(
                -amax.unsqueeze(0),
                amax.unsqueeze(0),
                scale_data=FP8_E4M3_DATA,
                quant_data=FP4_E2M1_DATA,
                dtype=torch.float32,
            )

        fused_global_scales = fuse_global_scales(layer_global_scales, strategy="min")

        results = []

        for layer_name, weight_gpu in weights_on_gpu.items():
            fused_global_scale = fused_global_scales[layer_name]
            weight_scale = compute_blockwise_scale(weight_gpu, fused_global_scale, self.group_size)
            weight_packed = self._compressor.compress_weight(
                weight=weight_gpu,
                scale=weight_scale.float(),
                global_scale=fused_global_scale,
                quantization_args=self._quant_args,
            )["weight_packed"]

            results.append((f"{layer_name}.weight_packed", weight_packed.to(output_device)))
            results.append((f"{layer_name}.weight_scale", weight_scale.to(output_device)))
            results.append((f"{layer_name}.weight_global_scale", fused_global_scale.to(output_device)))

            if self._is_w4a4:
                if layer_name in input_global_scales:
                    results.append(
                        (
                            f"{layer_name}.input_global_scale",
                            input_global_scales[layer_name].float().to(output_device),
                        )
                    )
                else:
                    raise ValueError(
                        f"W4A4 mode requires input_global_scale for layer '{layer_name}', "
                        f"but it's not found or uninitialized (-1.0)."
                    )

        del weights_on_gpu, layer_global_scales, fused_global_scales

        for name, tensor in layer_passthrough.items():
            results.append((name, tensor.to(output_device)))

        return results

    def quantize_with_fusion(
        self,
        params: dict[str, torch.Tensor] | Iterable[tuple[str, torch.Tensor]],
        target_device: Optional[torch.device] = None,
    ) -> Generator[tuple[str, torch.Tensor], None, None]:
        """Streaming quantize: consume input layer by layer, yield (name, tensor) pairs."""
        if isinstance(params, dict):
            params = params.items()

        output_device = target_device or torch.device("cpu")

        # Dispatch to HiF8 path for w8_hif8 mode
        if self._is_hif8:
            yield from self._quantize_with_fusion_hif8(params, output_device)
            return

        _sentinel = object()
        current_layer_idx = _sentinel
        layer_buffer: dict[str, torch.Tensor] = {}
        input_global_scales: dict[str, torch.Tensor] = {}
        for name, tensor in params:
            tensor_cpu = tensor.to("cpu") if tensor.is_cuda else tensor
            layer_idx = self._extract_layer_idx(name)

            # Collect input_global_scales for W4A4 as we go
            if self._is_w4a4 and "input_global_scale" in name:
                scale_layer_name = name.replace(".input_global_scale", "")
                if tensor_cpu.numel() == 1 and tensor_cpu.item() == -1.0:
                    logger.warning(f"W4A4: {scale_layer_name} input_global_scale is uninitialized")
                else:
                    input_global_scales[scale_layer_name] = tensor_cpu

            # Layer boundary: flush previous layer
            if layer_idx != current_layer_idx and current_layer_idx is not _sentinel and layer_buffer:
                yield from self._process_layer_group(
                    current_layer_idx, layer_buffer, input_global_scales, output_device
                )
                layer_buffer = {}

            current_layer_idx = layer_idx
            layer_buffer[name] = tensor_cpu

        # Flush last buffered layer
        if layer_buffer:
            yield from self._process_layer_group(current_layer_idx, layer_buffer, input_global_scales, output_device)

        get_torch_device().empty_cache()

    def _quantize_with_fusion_hif8(
        self,
        params: Iterable[tuple[str, torch.Tensor]],
        output_device: torch.device,
    ) -> Generator[tuple[str, torch.Tensor], None, None]:
        """HiF8 per-tensor quantization: bf16 → uint8 + per-tensor scale."""
        _sentinel = object()
        current_layer_idx = _sentinel
        layer_buffer: dict[str, torch.Tensor] = {}

        for name, tensor in params:
            tensor_cpu = tensor.to("cpu") if tensor.is_cuda else tensor
            layer_idx = self._extract_layer_idx(name)

            if layer_idx != current_layer_idx and current_layer_idx is not _sentinel and layer_buffer:
                yield from self._process_layer_hif8(
                    current_layer_idx, layer_buffer, output_device
                )
                layer_buffer = {}

            current_layer_idx = layer_idx
            layer_buffer[name] = tensor_cpu

        if layer_buffer:
            yield from self._process_layer_hif8(
                current_layer_idx, layer_buffer, output_device
            )

        get_torch_device().empty_cache()

    def _process_layer_hif8(
        self,
        layer_idx: Optional[int],
        layer_params: dict[str, torch.Tensor],
        output_device: torch.device,
    ) -> list[tuple[str, torch.Tensor]]:
        """Quantize weights to HiF8: bf16 → uint8 + per-tensor fp32 scale.

        Per-tensor tensorwise (document formula):
          amax = max(|weight|)
          scale = amax / F8max
          weight_scaled = weight / scale
          weight_uint8 = weight_scaled.to(hifloat8).view(uint8)

        Yields (name, weight_uint8) and (name + "_scale", scale_fp32).
        """
        HIF8_MAX = 49152.0  # 2^15 × 1.5 (Dot=4b, E=4b range [-15,15], M=1b)
        layer_weights = {}
        layer_passthrough = {}

        for name, tensor in layer_params.items():
            if "input_global_scale" in name or "input_amax" in name:
                continue

            if self._should_quantize(name, tensor):
                layer_name = name.rsplit(".weight", 1)[0]
                layer_weights[layer_name] = (name, tensor)
            else:
                layer_passthrough[name] = tensor

        if not layer_weights:
            return [(name, tensor.to(output_device)) for name, tensor in layer_passthrough.items()]

        results = []

        for layer_name, (param_name, tensor) in layer_weights.items():
            weight = tensor.to(device=self.device, dtype=torch.float32)

            # Per-tensor tensorwise: scale = amax / HIF8_MAX
            amax = weight.abs().max()
            safe_amax = amax if amax > 0 else torch.tensor(1e-12, device=self.device)
            scale = safe_amax / HIF8_MAX

            # Scale → _quant_hif8 → encode to true HiF8 bytes → uint8
            from verl.utils.qat.linear import _quant_hif8
            weight_scaled = _quant_hif8(weight / scale)
            weight_uint8 = _encode_hif8(weight_scaled)

            results.append((param_name, weight_uint8.to(output_device)))
            results.append((param_name + "_scale", scale.float().reshape(1).to(output_device)))

        for name, tensor in layer_passthrough.items():
            results.append((name, tensor.to(output_device)))

        return results


__all__ = [
    "QATQuantizer",
]
