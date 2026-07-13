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

"""HiF8 weight utilities for vLLM weight sync (per-element, no external scales).

Per-element HiF8 quantization: each weight element is independently converted
to hifloat8 with no shared exponent or block structure. No scale tensors are
needed — the quantized weight tensor carries all the information.

Pattern follows verl/utils/vllm/vllm_fp8_utils.py.
"""

import logging
import re
from typing import Any, Generator

import torch

logger = logging.getLogger(__name__)


def is_hif8_model(quant_config: dict[str, Any]) -> bool:
    """Check if the quantization config indicates an HiF8 model."""
    if quant_config is None:
        return False
    return quant_config.get("quant_method", None) == "ascend-hif8"


def _quantize_weight_to_hif8(weight: torch.Tensor) -> torch.Tensor:
    """Quantize a weight tensor to HiF8 format (per-element).

    Simple dtype conversion: bf16/fp32 → float8_e4m3fn (HiF8 proxy) → uint8.
    No external scale — each element independently quantized by the 8-bit format.

    Args:
        weight: High-precision weight tensor (out_features, in_features).

    Returns:
        uint8 tensor of same shape (byte-level representation of HiF8 values).
    """
    # Use float8_e4m3fn as the closest standard 8-bit float proxy for HiF8
    weight_f8 = weight.float().to(torch.float8_e4m3fn)
    return weight_f8.view(torch.uint8)


def quant_weights_hif8(
    params: Generator[tuple[str, torch.Tensor], None, None],
    model: Any,
    quant_config: dict[str, Any],
    dtype: torch.dtype = torch.bfloat16,
) -> Generator[tuple[str, torch.Tensor], None, None]:
    """Quantize weights to HiF8 during training-to-inference weight sync.

    For each 2D weight not in the ignore list:
      - Convert to float8_e4m3fn (HiF8 proxy)
      - Store as uint8 for IPC compatibility
      - No scale tensor — pure per-element quantization

    Non-quantized parameters pass through unchanged.

    Args:
        params: Generator of (name, tensor) pairs from FSDP model.
        model: The vLLM model (unused, for API consistency).
        quant_config: Quantization config dict with ignore patterns.
        dtype: Target compute dtype for non-quantized params.

    Yields:
        (name, tensor) pairs — weight tensors are uint8 HiF8 bytes.
    """
    ignore_patterns = quant_config.get("ignore", ["lm_head", "embed_tokens"])

    for name, tensor in params:
        should_quantize = (
            name.endswith(".weight")
            and tensor.dim() == 2
        )

        if should_quantize:
            module_name = name.rsplit(".weight", 1)[0]
            ignored = False
            for pattern in ignore_patterns:
                if pattern.startswith("re:"):
                    if re.match(pattern[3:], module_name):
                        ignored = True
                        break
                elif pattern in module_name:
                    ignored = True
                    break

            if not ignored:
                quantized_weight = _quantize_weight_to_hif8(tensor)
                yield (name, quantized_weight)
                logger.debug(f"HiF8 quantized (per-element): {name}")
                continue

        # Passthrough
        if tensor.is_floating_point():
            yield (name, tensor.to(dtype))
        else:
            yield (name, tensor)


def load_quanted_weights_hif8(
    weights: list[tuple[str, torch.Tensor]],
    model_runner: Any,
) -> None:
    """Load HiF8 per-element quantized weights into the vLLM model runner."""
    model = model_runner.model

    weight_pairs = [(name, tensor) for name, tensor in weights]
    _, loaded_weights = model.load_weights(weight_pairs)

    logger.info(f"Loaded {len(loaded_weights)} HiF8-quantized weights into vLLM model")
