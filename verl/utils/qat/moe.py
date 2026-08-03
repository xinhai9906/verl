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

"""HiF8 MoE QAT — wraps MoE block forward methods to pseudo-quantize expert
weights (and optionally activations) via HIF8FakeQuantFunction (STE).

Two weight-storage patterns are handled:
  A) Individual nn.Linear per expert (Qwen3MoE / Qwen3Next)
       → temporarily replace each expert Linear.weight with a quantized Tensor
  B) Stacked nn.Parameter (Qwen3.5MoE / Qwen3VL MoE Experts)
       → temporarily replace gate_up_proj / down_proj with quantized Tensors

After forward, original Parameters are restored. The autograd chain stays intact
because HIF8FakeQuantFunction tracks the original Parameter as input, so
gradients flow: output → GMM → quant_weight → HIF8FakeQuant.backward → raw_param.grad.

When quantize_activation=True (w8a8 mode), the input hidden_states are also
HiF8 pseudo-quantized before being fed to the GMM kernel.
"""

import logging
import types
from typing import Optional

import torch

from verl.utils.qat.linear import HIF8FakeQuantFunction

logger = logging.getLogger(__name__)

_QAT_CONFIG_ATTR = "_hif8_qat_config"


def apply_hif8_qat_to_moe(
    model: torch.nn.Module,
    granularity: str = "per_tensor",
    group_size: int = 32,
    quantize_activation: bool = False,
) -> int:
    """Apply HiF8 QAT to all MoE blocks in the model.

    Detection strategy: walk every submodule, determine the MoE weight layout
    (individual nn.Linear per expert vs. stacked nn.Parameter), then patch the
    forward method to quantize weights before GMM operations.

    When quantize_activation=True (w8a8 mode), the input hidden_states are also
    HiF8 pseudo-quantized before being fed to the GMM kernel.

    Returns:
        int: Number of MoE blocks patched.
    """
    patched_count = 0
    qat_cfg = {
        "granularity": granularity,
        "group_size": group_size,
        "quantize_activation": quantize_activation,
    }

    # Collect all MoE block candidates
    candidates: list[tuple[str, torch.nn.Module, str]] = []
    for name, module in list(model.named_modules()):
        moe_type = _detect_moe_path(module)
        if moe_type is not None:
            candidates.append((name, module, moe_type))

    # Filter out descendants: Qwen3_5MoeSparseMoeBlock contains
    # Qwen3_5MoeExperts, and both match as MoE blocks. Patching both
    # would quantize weights twice. Keep only the outermost block.
    candidate_prefixes = {name for name, _, _ in candidates}
    for name, module, moe_type in candidates:
        # Skip if any ancestor (prefix) is also a candidate
        if any(
            prefix in candidate_prefixes and prefix != name
            for prefix in _ancestor_prefixes(name)
        ):
            logger.debug(
                "[HiF8 MoE QAT] Skipping %s (parent is already patched)", name)
            continue

        setattr(module, _QAT_CONFIG_ATTR, qat_cfg)
        _patch_moe_forward(module, moe_type, qat_cfg)
        patched_count += 1

    logger.info(
        "[HiF8 MoE QAT] Patched %d MoE blocks "
        "(granularity=%s, group_size=%d, quantize_activation=%s)",
        patched_count, granularity, group_size, quantize_activation,
    )
    return patched_count


def _ancestor_prefixes(full_name: str) -> list[str]:
    """Return all ancestor prefixes of a dotted name, e.g.
    'a.b.c' → ['a', 'a.b'].
    """
    parts = full_name.split(".")
    return [".".join(parts[:i]) for i in range(1, len(parts))]


# ---------------------------------------------------------------------------
# MoE block detection
# ---------------------------------------------------------------------------

def _detect_moe_path(module: torch.nn.Module) -> Optional[str]:
    """Classify an MoE module's weight storage layout.

    Returns:
        'qwen3_linear_experts'   — experts submodule with individual nn.Linear
        'stacked_params_experts' — experts submodule with stacked nn.Parameter
        'stacked_params_self'    — the module itself IS stacked params
        None                     — not a recognised MoE block
    """
    experts = getattr(module, "experts", None)

    if experts is not None:
        # individual-expert nn.Linear: first child has proj submodules
        expert_children = list(experts.children())
        if expert_children:
            first = expert_children[0]
            if isinstance(first, torch.nn.Module):
                for _, child in first.named_modules():
                    if isinstance(child, torch.nn.Linear):
                        return "qwen3_linear_experts"

        # stacked nn.Parameter
        if isinstance(
            getattr(experts, "gate_up_proj", None), torch.nn.Parameter
        ):
            return "stacked_params_experts"

    # Module itself is stacked-param experts (e.g. Qwen3_5MoeExperts)
    if isinstance(getattr(module, "gate_up_proj", None), torch.nn.Parameter):
        return "stacked_params_self"

    return None


# ---------------------------------------------------------------------------
# Forward patching
# ---------------------------------------------------------------------------

def _patch_moe_forward(
    moe_block: torch.nn.Module,
    moe_type: str,
    qat_cfg: dict,
):
    """Replace moe_block.forward with a QAT-aware wrapper."""
    orig_forward = moe_block.forward

    if moe_type == "qwen3_linear_experts":
        moe_block.forward = types.MethodType(
            _make_qwen3_linear_qat_forward(orig_forward, qat_cfg), moe_block)
    elif moe_type in ("stacked_params_experts", "stacked_params_self"):
        moe_block.forward = types.MethodType(
            _make_stacked_param_qat_forward(
                orig_forward, qat_cfg, is_self=(moe_type == "stacked_params_self")
            ), moe_block)
    else:
        logger.warning("[HiF8 MoE QAT] Unknown moe_type: %s", moe_type)


# ---------------------------------------------------------------------------
# Qwen3 / Qwen3Next: individual expert nn.Linear
# ---------------------------------------------------------------------------

def _make_qwen3_linear_qat_forward(orig_forward, qat_cfg: dict):
    """Return a QAT-aware forward for SparseMoeBlock where experts have
    individual nn.Linear layers (gate_proj/up_proj/down_proj per expert).

    The underlying NPU forward (_qwen3_sparse_moe_routed_forward_npu) stacks
    expert weights into w1/w2/w3 and passes them to NPUGmmFunction. We intercept
    by temporarily replacing each expert Linear.weight with a quantized version
    *as a regular Tensor* that is part of the autograd graph — the original
    Parameter stays intact.
    """

    granularity = qat_cfg["granularity"]
    group_size = qat_cfg["group_size"]
    quantize_act = qat_cfg.get("quantize_activation", False)

    def qat_forward(self, hidden_states):
        if quantize_act:
            hidden_states = HIF8FakeQuantFunction.apply(
                hidden_states, granularity, group_size
            ).contiguous()

        experts = self.experts
        expert_list = list(experts.children())
        proj_names = ("gate_proj", "up_proj", "down_proj")

        # Build quantized weight dict: for each expert Linear, create a
        # quantized *Tensor* (not Parameter) that carries the HIF8FakeQuant
        # autograd node.  The NPU forward accesses `.weight` and receives
        # the quantized Tensor, so the graph becomes:
        #   raw_weight → HIF8FakeQuant → stacked → GMM → output
        # .contiguous() ensures layout compatibility with NPU GMM kernels.
        quantized_weights: dict[str, torch.Tensor] = {}
        for ei, expert in enumerate(expert_list):
            for pn in proj_names:
                proj = getattr(expert, pn, None)
                if isinstance(proj, torch.nn.Linear):
                    quantized_weights[f"{ei}.{pn}"] = (
                        HIF8FakeQuantFunction.apply(
                            proj.weight, granularity, group_size
                        ).contiguous())

        # Temporarily swap .weight to point to quantized tensors.
        # PyTorch __setattr__ rejects assigning a plain Tensor where an
        # nn.Parameter is registered, so we pop from _parameters first.
        saved: dict[str, torch.nn.Parameter] = {}
        for key, qw in quantized_weights.items():
            ei_str, pn = key.split(".")
            proj = getattr(expert_list[int(ei_str)], pn)
            saved[key] = proj.weight
            proj._parameters.pop("weight", None)
            proj.weight = qw

        try:
            return orig_forward(hidden_states)
        finally:
            for key, orig_param in saved.items():
                ei_str, pn = key.split(".")
                proj = getattr(expert_list[int(ei_str)], pn)
                try:
                    delattr(proj, "weight")
                except AttributeError:
                    pass
                proj.register_parameter("weight", orig_param)

    return qat_forward


# ---------------------------------------------------------------------------
# Stacked nn.Parameter: NPUQwen3VLMoeTextExperts / Qwen3_5MoeExperts
# ---------------------------------------------------------------------------

def _make_stacked_param_qat_forward(
    orig_forward, qat_cfg: dict, is_self: bool = False
):
    """Return QAT-aware forward for experts with stacked gate_up_proj / down_proj
    as nn.Parameter (e.g. NPUQwen3VLMoeTextExperts, Qwen3_5MoeExperts).

    The NPU forward calls NPUGmmFunction or torch.bmm with these parameters.
    We intercept by temporarily replacing them with quantized Tensors.
    """
    granularity = qat_cfg["granularity"]
    group_size = qat_cfg["group_size"]
    quantize_act = qat_cfg.get("quantize_activation", False)

    def qat_forward(self, *args, **kwargs):
        target = self if is_self else self.experts
        saved: dict[str, torch.nn.Parameter] = {}
        for attr in ("gate_up_proj", "down_proj"):
            param = getattr(target, attr, None)
            if isinstance(param, torch.nn.Parameter):
                saved[attr] = param
                # Pop from _parameters so PyTorch allows a plain Tensor
                target._parameters.pop(attr, None)
                setattr(target, attr,
                        HIF8FakeQuantFunction.apply(
                            param, granularity, group_size
                        ).contiguous())

        try:
            if quantize_act and args:
                args = (HIF8FakeQuantFunction.apply(
                    args[0], granularity, group_size
                ).contiguous(),) + args[1:]
            return orig_forward(*args, **kwargs)
        finally:
            for attr, orig_param in saved.items():
                try:
                    delattr(target, attr)
                except AttributeError:
                    pass
                target.register_parameter(attr, orig_param)

    return qat_forward
