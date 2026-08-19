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

When probe_quant_error=True, the patched forward computes per-expert quantisation
error via :func:`_hif8_compute_quant_error` (under ``torch.no_grad()``) and
records it with the shared :class:`QATProbeRecorder`, then runs the original
un-quantized forward — matching the Dense HIF8QATLinear probe behaviour.
"""

import logging
import re
import types
from typing import Optional

import torch
import torch.nn as nn

from verl.utils.qat.linear import HIF8FakeQuantFunction, _hif8_compute_quant_error
from verl.utils.qat.probe import get_qat_probe_recorder
from verl.utils.qat.block_rotation import BlockRotationConfig, apply_block_rotation

logger = logging.getLogger(__name__)

_QAT_CONFIG_ATTR = "_hif8_qat_config"
_MOE_LAYER_IDX_RE = re.compile(r"layers\.(\d+)\.")


def apply_hif8_qat_to_moe(
    model: torch.nn.Module,
    granularity: str = "per_tensor",
    group_size: int = 32,
    quantize_activation: bool = False,
    probe_quant_error: bool = False,
    rotation_enable: bool = False,
    rotation_block_size: int = 32,
    rotation_seed: int = 0,
    probe_layer_names: Optional[dict[int, str]] = None,
    ignore_patterns: Optional[list[str]] = None,
) -> int:
    """Apply HiF8 QAT to all MoE blocks in the model.

    Detection strategy: walk every submodule, determine the MoE weight layout
    (individual nn.Linear per expert vs. stacked nn.Parameter), then patch the
    forward method to quantize weights before GMM operations.

    When quantize_activation=True (w8a8 mode), the input hidden_states are also
    HiF8 pseudo-quantized before being fed to the GMM kernel.

    When probe_quant_error=True, the patched forward runs the original
    un-quantized forward while computing per-weight quantisation MAE via
    ``torch.no_grad()`` and recording it with the shared QATProbeRecorder.

    Args:
        probe_quant_error: Enable per-expert quant-error probing (default False).
        rotation_enable: Apply block Hadamard rotation before quantisation.
        rotation_block_size: Rotation block size (must equal group_size for per_group).
        rotation_seed: Random sign seed for the rotation matrix.
        probe_layer_names: Optional mapping ``{id(module): layer_name}`` for
            populating ``layer_type`` / ``layer_index`` in probe reports.
        ignore_patterns: Module name patterns to exclude from MoE QAT
            (mixed-precision fallback).  Plain patterns are substring-matched,
            ``re:``-prefixed patterns are regex-matched (``re.match``) — same
            semantics as ``QATConfig.ignore_patterns`` for dense linears.  A
            matched MoE block runs completely unquantized, mirroring the
            vLLM-side ``UnquantizedFusedMoEMethod`` fallback.

    Returns:
        int: Number of MoE blocks patched.
    """
    patched_count = 0
    probe_layer_names = probe_layer_names or {}
    ignore_patterns = ignore_patterns or []
    if rotation_enable:
        logger.warning(
            "[HiF8 MoE QAT] Block rotation is NOT supported for MoE blocks and "
            "will be disabled: down_proj weights rotate along the intermediate "
            "dimension whose activations are produced inside the fused GMM "
            "kernel (never rotated), and rotating the MoE input would change "
            "router inputs. Dense layers keep rotation; MoE runs unrotated."
        )
    rotation_config = BlockRotationConfig()  # rotation disabled for MoE
    base_cfg = {
        "granularity": granularity,
        "group_size": group_size,
        "quantize_activation": quantize_activation,
        "probe_quant_error": probe_quant_error,
        "rotation_config": rotation_config,
    }

    def _name_ignored(name: str) -> bool:
        for pattern in ignore_patterns:
            if pattern.startswith("re:"):
                if re.match(pattern[3:], name):
                    return True
            elif pattern in name:
                return True
        return False

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

        # Mixed-precision fallback: ignored blocks run completely unquantized,
        # matching the vLLM-side UnquantizedFusedMoEMethod fallback.
        if _name_ignored(name):
            logger.info(
                "[HiF8 MoE QAT] Ignoring MoE block %s (matches ignore_patterns) — "
                "runs unquantized (BF16 fallback)", name)
            continue

        # Per-block config copy so each block gets its own layer_name
        layer_name = probe_layer_names.get(id(module), name)
        qat_cfg = dict(base_cfg, layer_name=layer_name)
        qat_cfg["layer_type"] = _infer_moe_layer_type(name)
        qat_cfg["layer_index"] = _infer_moe_layer_index(name)

        setattr(module, _QAT_CONFIG_ATTR, qat_cfg)
        _patch_moe_forward(module, moe_type, qat_cfg)
        patched_count += 1

    logger.info(
        "[HiF8 MoE QAT] Patched %d MoE blocks "
        "(granularity=%s, group_size=%d, quantize_activation=%s, probe=%s)",
        patched_count, granularity, group_size, quantize_activation, probe_quant_error,
    )
    return patched_count


def _ancestor_prefixes(full_name: str) -> list[str]:
    """Return all ancestor prefixes of a dotted name, e.g.
    'a.b.c' → ['a', 'a.b'].
    """
    parts = full_name.split(".")
    return [".".join(parts[:i]) for i in range(1, len(parts))]


def _infer_moe_layer_type(name: str) -> Optional[str]:
    """Infer MoE layer type from the dotted module name."""
    if name.endswith(".experts"):
        parent = name.rsplit(".", 2)[0] if name.count(".") >= 2 else name
        return parent.rsplit(".", 1)[-1] if "." in parent else parent
    return None


def _infer_moe_layer_index(name: str) -> Optional[int]:
    """Infer layer index from the dotted module name."""
    match = _MOE_LAYER_IDX_RE.search(name)
    return int(match.group(1)) if match else None


def _record_moe_quant_error(
    qat_cfg: dict, error_type: str, error_sum: float, element_count: int
) -> None:
    """Record one quant-error measurement for an MoE expert weight."""
    recorder = get_qat_probe_recorder()
    recorder.record(
        meta={
            "step": recorder.current_step if recorder.current_step is not None else 0,
            "error_type": error_type,
            "layer_index": qat_cfg.get("layer_index"),
            "layer_type": qat_cfg.get("layer_type"),
            "mode": (
                "w8a8_hif8" if qat_cfg.get("quantize_activation") else "w8a16_hif8"
            ),
            "granularity": qat_cfg.get("granularity"),
            "group_size": qat_cfg.get("group_size"),
            "rank": (
                torch.distributed.get_rank()
                if torch.distributed.is_available() and torch.distributed.is_initialized()
                else 0
            ),
        },
        error_sum=error_sum,
        element_count=element_count,
    )


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

    When probe_quant_error=True, no weights are replaced; the original forward
    runs unmodified while per-expert quantisation MAE is computed under
    ``torch.no_grad()`` and recorded via :func:`_record_moe_quant_error`.
    """

    granularity = qat_cfg["granularity"]
    group_size = qat_cfg["group_size"]
    quantize_act = qat_cfg.get("quantize_activation", False)
    probe_enabled = qat_cfg.get("probe_quant_error", False)
    rotation_config = qat_cfg.get("rotation_config", BlockRotationConfig())

    def qat_forward(self, hidden_states):
        experts = self.experts
        expert_list = list(experts.children())
        proj_names = ("gate_proj", "up_proj", "down_proj")

        # -- probe mode: measure error, run original forward -----------------
        if probe_enabled:
            for ei, expert in enumerate(expert_list):
                for pn in proj_names:
                    proj = getattr(expert, pn, None)
                    if isinstance(proj, nn.Linear):
                        rotated_w = apply_block_rotation(
                            proj.weight, rotation_config
                        )
                        err_sum, err_n = _hif8_compute_quant_error(
                            rotated_w, granularity, group_size
                        )
                        _record_moe_quant_error(
                            qat_cfg,
                            f"weight/{pn}",
                            err_sum,
                            err_n,
                        )
            if quantize_act:
                rotated_a = apply_block_rotation(
                    hidden_states, rotation_config
                )
                a_err, a_n = _hif8_compute_quant_error(
                    rotated_a, granularity, group_size
                )
                _record_moe_quant_error(qat_cfg, "activation", a_err, a_n)
            return orig_forward(hidden_states)

        # -- normal QAT: pseudo-quantize weights via STE ----------------------
        if quantize_act:
            rotated_h = apply_block_rotation(hidden_states, rotation_config)
            hidden_states = HIF8FakeQuantFunction.apply(
                rotated_h, granularity, group_size
            ).contiguous()

        quantized_weights: dict[str, torch.Tensor] = {}
        for ei, expert in enumerate(expert_list):
            for pn in proj_names:
                proj = getattr(expert, pn, None)
                if isinstance(proj, nn.Linear):
                    rotated_w = apply_block_rotation(
                        proj.weight, rotation_config
                    )
                    quantized_weights[f"{ei}.{pn}"] = (
                        HIF8FakeQuantFunction.apply(
                            rotated_w, granularity, group_size
                        ).contiguous())

        # Temporarily swap .weight to point to quantized tensors.
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

    When probe_quant_error=True, no parameters are replaced; the original forward
    runs unmodified while per-expert quantisation MAE is computed (each expert
    slice of the stacked Parameter is measured independently).
    """
    granularity = qat_cfg["granularity"]
    group_size = qat_cfg["group_size"]
    quantize_act = qat_cfg.get("quantize_activation", False)
    probe_enabled = qat_cfg.get("probe_quant_error", False)
    rotation_config = qat_cfg.get("rotation_config", BlockRotationConfig())

    def qat_forward(self, *args, **kwargs):
        target = self if is_self else self.experts

        # -- probe mode: measure error per expert, run original forward --------
        if probe_enabled:
            for attr in ("gate_up_proj", "down_proj"):
                param = getattr(target, attr, None)
                if isinstance(param, nn.Parameter) and param.ndim >= 2:
                    for e in range(param.shape[0]):
                        rotated_w = apply_block_rotation(
                            param[e], rotation_config
                        )
                        err_sum, err_n = _hif8_compute_quant_error(
                            rotated_w, granularity, group_size
                        )
                        _record_moe_quant_error(
                            qat_cfg,
                            f"weight/{attr}/expert_{e}",
                            err_sum,
                            err_n,
                        )
            if quantize_act and args:
                rotated_a = apply_block_rotation(
                    args[0], rotation_config
                )
                a_err, a_n = _hif8_compute_quant_error(
                    rotated_a, granularity, group_size
                )
                _record_moe_quant_error(qat_cfg, "activation", a_err, a_n)
            return orig_forward(*args, **kwargs)

        # -- normal QAT: pseudo-quantize weights via STE -----------------------
        saved: dict[str, torch.nn.Parameter] = {}
        for attr in ("gate_up_proj", "down_proj"):
            param = getattr(target, attr, None)
            if isinstance(param, torch.nn.Parameter):
                saved[attr] = param
                rotated = apply_block_rotation(param, rotation_config)
                target._parameters.pop(attr, None)
                setattr(target, attr,
                        HIF8FakeQuantFunction.apply(
                            rotated, granularity, group_size
                        ).contiguous())

        try:
            if quantize_act and args:
                rotated_h = apply_block_rotation(args[0], rotation_config)
                args = (HIF8FakeQuantFunction.apply(
                    rotated_h, granularity, group_size
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
