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

"""QAT (Quantization-Aware Training) utilities for verl FSDP training."""

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import torch.nn as nn

from verl.base_config import BaseConfig

logger = logging.getLogger(__name__)


@dataclass
class QATConfig(BaseConfig):
    """Unified configuration for QAT (Quantization-Aware Training)."""

    enable: bool = False
    mode: str = "w4a16"  # "w4a16", "w4a4", "w8_hif8", or "w8a8_hif8"
    granularity: str = "per_tensor"  # HiF8 granularity: "per_tensor", "per_channel", "per_group", or "per_group_median"
    group_size: int = 16  # block size for NVFP4; also used by HiF8 per_group/per_group_median mode
    ignore_patterns: list[str] = field(default_factory=lambda: ["lm_head", "embed_tokens", "re:.*mlp.gate$"])
    activation_observer: str = "static_minmax"
    quantization_config_path: Optional[str] = None
    probe_quant_error: bool = False  # Enable per-layer HiF8 quant-error probe (pure measurement, no noise)
    probe_output_path: Optional[str] = None  # JSONL output path for probe reports (None = log only)
    rotation_enable: bool = False  # Apply block Hadamard rotation before quantisation
    rotation_block_size: int = 32  # Rotation block size (must equal group_size for per_group/per_group_median)
    rotation_seed: int = 0  # Random sign seed for the rotation matrix


def load_quantization_config(qat_config: QATConfig) -> dict[str, Any]:
    """Load quantization config JSON file from QATConfig."""
    if not qat_config.quantization_config_path:
        raise ValueError("quantization_config_path is required when QAT is enabled")

    logger.info(f"Loading QAT quantization config from: {qat_config.quantization_config_path}")

    with open(qat_config.quantization_config_path) as f:
        quant_config = json.load(f)

    if qat_config.ignore_patterns:
        original_ignore = quant_config.get("ignore", [])
        quant_config["ignore"] = qat_config.ignore_patterns
        if original_ignore != qat_config.ignore_patterns:
            logger.info(f"Overriding JSON 'ignore' field: {original_ignore} -> {qat_config.ignore_patterns}")

    logger.info("Successfully loaded QAT quantization config")
    return quant_config


def _should_quantize(name: str, module: nn.Module, config: QATConfig) -> bool:
    """Check if a module should be quantized."""
    if not isinstance(module, nn.Linear):
        return False

    for pattern in config.ignore_patterns:
        if pattern.startswith("re:"):
            regex = pattern[3:]
            if re.match(regex, name):
                logger.debug(f"Ignoring {name} due to regex pattern: {regex}")
                return False
        else:
            if pattern in name:
                logger.debug(f"Ignoring {name} due to pattern: {pattern}")
                return False

    if config.mode in ("w8_hif8", "w8a8_hif8"):
        return True

    if module.in_features % config.group_size != 0:
        logger.warning(
            f"Skipping {name}: in_features={module.in_features} not divisible by group_size={config.group_size}"
        )
        return False

    return True


def _replace_modules(
    model: nn.Module,
    config: QATConfig,
    factory: Callable[[nn.Linear, str], nn.Module],
    target_cls: type,
    mode_label: str,
    skip_ids: Optional[set[int]] = None,
) -> int:
    """Find nn.Linear layers and replace them using ``factory(module, name)``.
    Shared logic for both HiF8 and NVFP4 QAT paths — avoids code duplication.
    """
    if skip_ids is None:
        skip_ids = set()
    modules_to_replace = [
        (name, module)
        for name, module in model.named_modules()
        if _should_quantize(name, module, config)
        and not isinstance(module, target_cls)
        and id(module) not in skip_ids
    ]

    logger.info(f"Found {len(modules_to_replace)} Linear layers to convert to {mode_label}")

    for name, module in modules_to_replace:
        _set_module(model, name, factory(module, name))

    logger.info(f"Successfully applied {mode_label} to {len(modules_to_replace)} layers")
    return len(modules_to_replace)


def apply_qat(
    model: nn.Module,
    config: QATConfig | dict[str, Any],
) -> nn.Module:
    """Apply QAT to a model.
    HiF8 modes: replaces nn.Linear with HIF8QATLinear and monkey-patches
    MoE block forward methods for expert weight quantization.
    NVFP4 modes: replaces nn.Linear with QATLinear and sets up fusion siblings.
    """
    if not isinstance(config, QATConfig):
        config = QATConfig(**config)

    if not config.enable:
        logger.info("QAT is disabled, returning original model")
        return model

    if config.mode in ("w8_hif8", "w8a8_hif8"):
        from verl.utils.qat.linear import HIF8QATLinear
        from verl.utils.qat.probe import configure_qat_probe

        quantize_act = (config.mode == "w8a8_hif8")
        granularity = getattr(config, "granularity", "per_tensor")
        group_size = getattr(config, "group_size", 32)
        probe_enabled = getattr(config, "probe_quant_error", False)
        probe_output = getattr(config, "probe_output_path", None)

        if probe_enabled:
            configure_qat_probe(enabled=True, output_path=probe_output)

        logger.info(f"Applying QAT with mode={config.mode} "
                     f"({'W8A8' if quantize_act else 'W8 weight-only'}, "
                     f"granularity={granularity}, group_size={group_size}, "
                     f"probe={probe_enabled}, "
                     f"rotation={config.rotation_enable} "
                     f"(block_size={config.rotation_block_size}, seed={config.rotation_seed}))")

        # Patch MoE blocks FIRST so expert Linear layers under them can be
        # excluded from individual HIF8QATLinear replacement.
        from verl.utils.qat.moe import apply_hif8_qat_to_moe

        moe_patched = apply_hif8_qat_to_moe(
            model, granularity=granularity, group_size=group_size,
            quantize_activation=quantize_act,
            probe_quant_error=probe_enabled,
            rotation_enable=config.rotation_enable,
            rotation_block_size=config.rotation_block_size,
            rotation_seed=config.rotation_seed,
            ignore_patterns=list(config.ignore_patterns))

        # Collect module ids under MoE blocks to skip redundant Linear
        # replacement (MoE forwards bypass Linear.forward anyway).
        if moe_patched > 0:
            moe_child_ids = _collect_moe_child_ids(model)
        else:
            moe_child_ids = set()

        def _hif8_factory(linear: nn.Linear, name: str = "") -> HIF8QATLinear:
            return HIF8QATLinear.from_linear(
                linear, quantize_activation=quantize_act,
                granularity=granularity, group_size=group_size,
                hif8_probe_quant_error=probe_enabled,
                rotation_enable=config.rotation_enable,
                rotation_block_size=config.rotation_block_size,
                rotation_seed=config.rotation_seed,
                layer_name=name)

        _replace_modules(
            model, config,
            factory=_hif8_factory,
            target_cls=HIF8QATLinear,
            mode_label=f"HiF8 QAT ({config.mode}, {granularity})",
            skip_ids=moe_child_ids,
        )

        return model

    # Standard NVFP4 QAT path
    from verl.utils.qat.linear import QATLinear, QATMode

    mode = QATMode(config.mode.lower())
    logger.info(f"Applying QAT with mode={mode.value}, group_size={config.group_size}")

    def _factory(linear: nn.Linear, name: str = "") -> QATLinear:
        return QATLinear.from_linear(
            linear,
            mode=mode,
            group_size=config.group_size,
            activation_observer=config.activation_observer,
        )

    _replace_modules(
        model, config,
        factory=_factory,
        target_cls=QATLinear,
        mode_label=f"QAT ({mode.value})",
    )

    return model


def _collect_moe_child_ids(model: nn.Module) -> set[int]:
    """Collect module ids that are descendants of HiF8-patched MoE blocks.

    After apply_hif8_qat_to_moe() sets ``_hif8_qat_config`` on MoE blocks,
    we traverse named_modules() so any module whose ancestor (determined
    purely from the dotted name prefix) is a patched MoE block is collected.
    This allows _replace_modules to skip redundant HIF8QATLinear wrappers
    for expert Linear layers that the MoE forward bypasses anyway.
    """
    moe_ids: set[int] = set()
    moe_prefixes: list[str] = []
    for name, module in model.named_modules():
        if hasattr(module, "_hif8_qat_config"):
            moe_prefixes.append(name)

    if not moe_prefixes:
        return moe_ids

    # Normalise prefix to "name." or "name" so we can detect children
    for child_name, child_module in model.named_modules():
        for prefix in moe_prefixes:
            if child_name == prefix or child_name.startswith(prefix + "."):
                moe_ids.add(id(child_module))
                break

    return moe_ids


def _set_module(model: nn.Module, name: str, new_module: nn.Module):
    """Set a module in the model by its full name."""
    parts = name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], new_module)


FUSION_PATTERNS = {
    "qkv": ["q_proj", "k_proj", "v_proj"],
    "gate_up": ["gate_proj", "up_proj"],
}


def setup_fusion_siblings(model: nn.Module):
    """Setup fusion siblings for QKV and GateUp layers."""
    import weakref

    from verl.utils.qat.linear import QATLinear

    qat_modules = {name: m for name, m in model.named_modules() if isinstance(m, QATLinear)}

    counts = {}
    for group_name, suffixes in FUSION_PATTERNS.items():
        groups: dict[str, dict[str, nn.Module]] = {}
        for name, module in qat_modules.items():
            for suffix in suffixes:
                if name.endswith(suffix):
                    parent = name.rsplit(".", 1)[0]
                    groups.setdefault(parent, {})[suffix] = module

        count = 0
        for parent, projs in groups.items():
            if len(projs) >= 2:
                modules = list(projs.values())
                for i, m in enumerate(modules):
                    siblings = modules[:i] + modules[i + 1 :]
                    m._fusion_siblings_ref = [weakref.ref(s) for s in siblings]
                count += 1
        counts[group_name] = count

    logger.info(f"[QAT Fuse] Setup fusion siblings: {counts}")
    return counts


def enable_qat_fuse(model: nn.Module):
    """Enable QAT fuse mode: sets up fusion siblings for weight scale fusion."""
    setup_fusion_siblings(model)
    model._qat_fuse_enabled = True
    logger.info("[QAT Fuse] Enabled QAT fuse mode")


def invalidate_all_scales(model: nn.Module):
    """Clear all cached weight scales after optimizer.step()."""
    from verl.utils.qat.linear import QATLinear

    count = 0
    for module in model.modules():
        if isinstance(module, QATLinear):
            module._weight_blockwise_scale = None
            module._weight_global_scale = None
            module._cached_weight_amax = None
            count += 1

    logger.debug(f"[QAT Fuse] Invalidated scales for {count} QATLinear layers")
