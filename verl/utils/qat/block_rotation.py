"""Block-wise Hadamard rotation for reducing quantisation error.

Repeated application of a single 32×32 orthogonal block matrix Q along the
last dimension.  Q = Hadamard(32) × diag(random_signs), so every element in
a block gets roughly equal magnitude and outlier values are "smeared" across
the block, making per-block or per-channel quantisation scales more uniform.

Used by both HiF8 and MXFP8 QAT paths — the rotation itself is scheme-agnostic.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any

import torch

logger = logging.getLogger(__name__)

__all__ = [
    "BlockRotationConfig",
    "apply_block_rotation",
]

_DEFAULT_BLOCK_SIZE = 32
_DEFAULT_SEED = 0
_MAX_TORCH_SEED = 2**63 - 1
_MATRIX_CACHE: dict[tuple[int, int, str, str], torch.Tensor] = {}
_LOGGED: set[tuple[int, int, str, str]] = set()


@dataclass(frozen=True)
class BlockRotationConfig:
    enable: bool = False
    block_size: int = _DEFAULT_BLOCK_SIZE
    seed: int = _DEFAULT_SEED


def _normalize_seed(seed: int) -> int:
    return int(seed) % _MAX_TORCH_SEED


def _build_hadamard_matrix(size: int) -> torch.Tensor:
    if size <= 0 or size & (size - 1):
        raise ValueError(f"Hadamard size must be a positive power of two, got: {size}")
    matrix = torch.ones((1, 1), dtype=torch.float32)
    while matrix.shape[0] < size:
        top = torch.cat((matrix, matrix), dim=1)
        bottom = torch.cat((matrix, -matrix), dim=1)
        matrix = torch.cat((top, bottom), dim=0)
    return matrix * (1.0 / math.sqrt(size))


def _deterministic_signs(size: int, seed: int) -> list[float]:
    """Deterministic ±1 sequence from a plain Python LCG.

    Replaces torch.Generator/torch.randint, which are not traceable by
    torch.compile (Dynamo graph break, fatal under vLLM VLLM_COMPILE).
    A pure Python LCG is constant-folded by the tracer, produces identical
    results on every rank and on both the training and inference sides, and
    is still seeded by the same ``seed`` config as before.
    """
    x = int(seed)
    signs: list[float] = []
    for _ in range(size):
        x = (x * 1103515245 + 12345) & 0x7FFFFFFF
        signs.append(-1.0 if (x >> 15) & 1 else 1.0)
    return signs


def _build_block_rotation_matrix(
    block_size: int, seed: int, *, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    cache_key = (block_size, _normalize_seed(seed), str(device), str(dtype))
    cached = _MATRIX_CACHE.get(cache_key)
    if cached is not None:
        return cached

    matrix = _build_hadamard_matrix(block_size)
    signs = torch.tensor(
        _deterministic_signs(block_size, _normalize_seed(seed)),
        dtype=torch.float32,
        device="cpu",
    )
    cached = (matrix * signs).to(device=device, dtype=dtype)
    _MATRIX_CACHE[cache_key] = cached
    return cached


def _rotation_work_dtype(dtype: torch.dtype) -> torch.dtype:
    # Rotate in the input dtype: the 32×32 block Hadamard entries (±1/√32)
    # round to ~1e-3 relative error in bf16/fp16 — negligible next to the
    # HiF8 quantisation noise that follows, and both the training and the
    # inference sides apply the same deterministic dtype rotation, so the
    # Q·Qᵀ cancellation residual is a fixed bias the model adapts to.
    # Rotating in fp32 would cost ~4× more on NPU (fp32 is much slower
    # than bf16) with no measurable accuracy benefit.
    return dtype


def apply_block_rotation(
    tensor: torch.Tensor,
    config: BlockRotationConfig,
    *,
    transpose: bool = False,
) -> torch.Tensor:
    """Apply repeated block‑diagonal rotation on the tensor's last dimension.

    For a weight ``[out_features, in_features]`` or activation ``[tokens, hidden]``
    this multiplies each contiguous *block_size*-element segment by the same
    orthogonal matrix Q, i.e. ``tensor_blocked @ Q``.

    Setting ``transpose=True`` applies ``Q.T`` instead (useful for future
    inverse‑rotation experiments).
    """
    if not config.enable:
        return tensor

    block_size = config.block_size
    last_dim = tensor.shape[-1]
    if last_dim % block_size != 0:
        raise ValueError(
            f"Block rotation requires last_dim divisible by block_size={block_size}, "
            f"got shape={tuple(tensor.shape)}"
        )

    # FP8 dtypes are too narrow for the rotation arithmetic — reject early.
    fp8_dtypes = tuple(
        d for d in (
            getattr(torch, "float8_e4m3fn", None),
            getattr(torch, "float8_e5m2", None),
            getattr(torch, "float8_e4m3fnuz", None),
            getattr(torch, "float8_e5m2fnuz", None),
        ) if d is not None
    )
    if tensor.dtype in fp8_dtypes:
        raise ValueError("Block rotation must run before FP8 quantisation")

    original_shape = tensor.shape
    original_dtype = tensor.dtype
    work_dtype = _rotation_work_dtype(original_dtype)
    matrix = _build_block_rotation_matrix(
        block_size, config.seed, device=tensor.device, dtype=work_dtype
    )
    if transpose:
        matrix = matrix.t()

    log_key = (block_size, _normalize_seed(config.seed), str(tensor.device), str(work_dtype))
    # Skip logging while torch.compile is tracing: logger calls and the
    # _LOGGED bookkeeping are not Dynamo-friendly and would graph-break.
    if not torch.compiler.is_compiling() and log_key not in _LOGGED:
        _LOGGED.add(log_key)
        logger.warning(
            "Block rotation applied: block_size=%s, seed=%s, transpose=%s, "
            "device=%s, work_dtype=%s, tensor_shape=%s",
            block_size, config.seed, transpose, tensor.device, work_dtype, tuple(tensor.shape),
        )

    tensor_2d = tensor.reshape(-1, last_dim).to(work_dtype)
    num_blocks = last_dim // block_size
    tensor_blocked = tensor_2d.reshape(tensor_2d.shape[0], num_blocks, block_size)
    rotated = torch.matmul(tensor_blocked, matrix)
    return rotated.reshape(tensor_2d.shape).to(original_dtype).reshape(original_shape)
