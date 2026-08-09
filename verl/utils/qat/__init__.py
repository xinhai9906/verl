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
QAT (Quantization-Aware Training) module for verl.

Supports NVFP4 (W4A4 and W4A16) and Ascend HiF8 (W8/W8A8) QAT modes for
FSDP training, plus a shared quant-error probe for per-layer sensitivity
analysis.

Module Structure:
- core.py: QATConfig, apply_qat, enable_qat_fuse (training setup)
- linear.py: QATLinear (NVFP4) + HIF8QATLinear (HiF8) with Triton kernels
- block_rotation.py: Block Hadamard rotation for reducing quantisation error
- probe.py: QATProbeRecorder — shared quant-error aggregator + JSONL writer
- quantizer.py: QATQuantizer for true quantization + scale computation utilities
- vllm_patch.py: Patches for vLLM dynamic weight loading
- moe.py: MoE-specific QAT helpers

Usage:
    from verl.utils.qat import apply_qat, QATConfig

    config = QATConfig(enable=True, mode="w4a16")
    model = apply_qat(model, config)  # Before FSDP wrapping
"""

from verl.utils.qat.core import (
    QATConfig,
    apply_qat,
    enable_qat_fuse,
    invalidate_all_scales,
    load_quantization_config,
)
from verl.utils.qat.block_rotation import BlockRotationConfig, apply_block_rotation
from verl.utils.qat.linear import HIF8QATLinear
from verl.utils.qat.probe import (
    configure_qat_probe,
    flush_qat_probe,
    get_qat_probe_recorder,
    qat_probe_step_context,
    reset_qat_probe,
    set_qat_probe_step,
)
from verl.utils.qat.vllm_patch import (
    apply_qat_patches,
    manual_process_weights_after_loading,
    prepare_qat_for_load_weights,
)

__all__ = [
    # Core
    "QATConfig",
    "apply_qat",
    "load_quantization_config",
    "enable_qat_fuse",
    "invalidate_all_scales",
    # HiF8 QAT Linear
    "HIF8QATLinear",
    # Block Rotation
    "BlockRotationConfig",
    "apply_block_rotation",
    # QAT Probe
    "get_qat_probe_recorder",
    "configure_qat_probe",
    "flush_qat_probe",
    "reset_qat_probe",
    "set_qat_probe_step",
    "qat_probe_step_context",
    # vLLM Patch
    "apply_qat_patches",
    "manual_process_weights_after_loading",
    "prepare_qat_for_load_weights",
]
