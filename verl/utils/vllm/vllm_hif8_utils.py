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

"""HiF8 weight utilities for vLLM weight sync.

Provides utilities for quantizing weights to HiF8 format during the
training-to-inference weight transfer, and for loading quantized weights
into the vLLM model.

Quantization granularity (per-channel weight + per-token activation):
  - Weight scale: (out_features, 1) — one shared exponent per output channel
  - Activation: per-token dynamic at runtime via npu_dynamic_quant()

Pattern follows verl/utils/vllm/vllm_fp8_utils.py.
"""

import logging
import re
from typing import Any, Generator

import torch

logger = logging.getLogger(__name__)

# HiF8 constants (matching MindSpeed's FormatEnum)
HIF8_15_MAX: float = 15.0  # Forward quantization max


def is_hif8_model(quant_config: dict[str, Any]) -> bool:
    """Check if the quantization config indicates an HiF8 model.

    Args:
        quant_config: The quantization_config dict from the model or config.

    Returns:
        True if quant_method is "ascend-hif8".
    """
    if quant_config is None:
        return False
    quant_method = quant_config.get("quant_method", None)
    return quant_method == "ascend-hif8"


def _compute_hif8_per_channel_scale(
    weight: torch.Tensor,
) -> torch.Tensor:
    """Compute per-channel HiF8 quantization scales.

    For a weight tensor of shape (out_features, in_features), computes
    one shared exponent per output channel (row).

    Algorithm:
      channel_amax = max(|weight|, dim=-1)  → (out_features, 1)
      shared_exp = ceil(log2(channel_amax / HIF8_15_MAX))
      scale = 2^shared_exp  → (out_features, 1)

    Args:
        weight: Weight tensor of shape (out_features, in_features).

    Returns:
        Scale tensor of shape (out_features, 1) in float32.
    """
    out_features = weight.shape[0]

    # Per-channel max absolute value
    channel_amax = torch.amax(torch.abs(weight.float()), dim=-1, keepdim=True)

    # Compute shared exponent: ceil(log2(amax / HIF8_15_MAX))
    hif8_max_tensor = torch.tensor(HIF8_15_MAX, dtype=torch.float32, device=weight.device)

    # Avoid log2(0)
    safe_amax = torch.where(
        channel_amax > 0.0,
        channel_amax,
        torch.tensor(1e-38, dtype=torch.float32, device=weight.device),
    )

    shared_exp = torch.ceil(torch.log2(safe_amax) - torch.log2(hif8_max_tensor))
    shared_exp = torch.clamp(shared_exp, -127, 127)

    # Scale = 2^shared_exp (positive, broadcastable to weight shape)
    scale = torch.pow(2.0, shared_exp.float())

    # Squeeze to (out_features,)
    return scale.squeeze(-1).float()


def _quantize_weight_to_hif8(
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a weight tensor to HiF8 format (per-channel).

    Args:
        weight: High-precision weight tensor (out_features, in_features).

    Returns:
        Tuple of (quantized_weight, weight_scale):
          - quantized_weight: uint8 tensor of same shape as weight
          - weight_scale: fp32 tensor of shape (out_features,)
    """
    out_features, in_features = weight.shape

    scale = _compute_hif8_per_channel_scale(weight)  # (out_features,)
    scale_expanded = scale.unsqueeze(-1)  # (out_features, 1) for broadcasting

    # Quantize: scale down → round → clamp → offset to uint8
    quantized = torch.round(weight.float() / scale_expanded)

    # Offset by 128 to map signed HiF8 values to uint8 [0, 255]
    quantized_clamped = torch.clamp(quantized + 128, 0, 255)

    return quantized_clamped.to(torch.uint8), scale


def quant_weights_hif8(
    params: Generator[tuple[str, torch.Tensor], None, None],
    model: Any,
    quant_config: dict[str, Any],
    dtype: torch.dtype = torch.bfloat16,
) -> Generator[tuple[str, torch.Tensor], None, None]:
    """Quantize weights to HiF8 during training-to-inference weight sync.

    For each weight:
      - Computes per-channel HiF8 scale
      - Quantizes to uint8
      - Yields (name, quantized_weight_uint8) and (name + "_scale", scale_fp32)

    Non-quantized parameters pass through unchanged.

    Args:
        params: Generator of (name, tensor) pairs from FSDP model.
        model: The vLLM model (used to check which params to quantize).
        quant_config: Quantization config dict with ignore patterns.
        dtype: Target compute dtype (bf16/fp16) for non-quantized params.

    Yields:
        (name, tensor) pairs for all parameters.
    """
    ignore_patterns = quant_config.get("ignore", ["lm_head", "embed_tokens"])

    for name, tensor in params:
        # Only quantize 2D weight tensors not in ignore list
        should_quantize = (
            name.endswith(".weight")
            and tensor.dim() == 2
        )

        if should_quantize:
            # Check if this weight should be ignored
            module_name = name.rsplit(".weight", 1)[0]
            ignored = False
            for pattern in ignore_patterns:
                if pattern.startswith("re:"):
                    regex = pattern[3:]
                    if re.match(regex, module_name):
                        ignored = True
                        break
                elif pattern in module_name:
                    ignored = True
                    break

            if not ignored:
                # Quantize this weight (per-channel)
                weight = tensor.to(torch.float32)
                quantized_weight, weight_scale = _quantize_weight_to_hif8(weight)

                yield (name, quantized_weight)

                # Per-channel scale: (out_features,) → vLLM expects (out_features, 1)
                scale_name = name + "_scale"
                yield (scale_name, weight_scale.unsqueeze(-1))

                logger.debug(
                    f"HiF8 quantized: {name} -> uint8 + scale[{weight_scale.shape[0]}]"
                )
                continue

        # Passthrough: non-quantized params
        if tensor.is_floating_point():
            yield (name, tensor.to(dtype))
        else:
            yield (name, tensor)


def load_quanted_weights_hif8(
    weights: list[tuple[str, torch.Tensor]],
    model_runner: Any,
) -> None:
    """Load HiF8 quantized weights into the vLLM model runner.

    Args:
        weights: List of (name, tensor) pairs from IPC receiver.
        model_runner: The vLLM model runner instance.
    """
    model = model_runner.model

    # Group weight and scale tensors
    weight_dict = {}
    scale_dict = {}
    for name, tensor in weights:
        if name.endswith("_scale"):
            weight_name = name[:-6]  # Remove "_scale" suffix
            scale_dict[weight_name] = tensor
        else:
            weight_dict[name] = tensor

    # Load quantized weights into the model
    weight_pairs = [
        (name, tensor) for name, tensor in weight_dict.items()
    ]
    _, loaded_weights = model.load_weights(weight_pairs)

    # Load scale tensors separately
    if scale_dict:
        scale_pairs = [
            (name + ".weight_scale", scale) for name, scale in scale_dict.items()
        ]
        model.load_weights(scale_pairs)

    logger.info(f"Loaded {len(loaded_weights)} HiF8-quantized weights into vLLM model")
