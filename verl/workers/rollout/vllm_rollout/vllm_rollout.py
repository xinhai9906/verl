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
The vllm_rollout that can be applied in different backend
When working with FSDP:
- Use DTensor weight loader (recommended) or HF weight loader
- Utilize state_dict from the FSDP to synchronize the weights among tp ranks in vLLM
When working with Megatron:
- Use Megatron weight loader
- During training, only the current pp stage holds the parameters
- Before inference, broadcast the parameters of the current pp rank
  to all other pp ranks (all pp ranks holds all the parameters)
- Bind the parameters to the inference engine
- Do inference in tp. pp is treated as additional dp
- After inference, all the parameters that doesn't belong to this pp rank is freed.
"""

import logging
import os
import time
from typing import Any, Generator, Optional

import ray
import torch
from packaging import version as vs
from torch.distributed.device_mesh import DeviceMesh

from verl import DataProto
from verl.third_party.vllm import VLLM_SLEEP_LEVEL, get_version
from verl.utils.device import get_device_id, is_support_ipc
from verl.workers.config import HFModelConfig, RolloutConfig
from verl.workers.rollout.base import BaseRollout
from verl.workers.rollout.vllm_rollout.bucketed_weight_transfer import BucketedWeightSender
from verl.workers.rollout.vllm_rollout.utils import get_device_uuid

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


def _check_vllm_version_for_sleep_level():
    # https://github.com/vllm-project/vllm/issues/25171
    minver = "0.11.0"
    current_version = get_version("vllm")
    if not current_version:
        logger.warning("Could not determine vLLM version, assuming an older version for sleep_level configuration.")
        return False
    return vs.parse(current_version) >= vs.parse(minver)


_MOE_EXPAND_ARCHITECTURE_WHITELIST = frozenset(
    {
        "Qwen3MoeForCausalLM",
    }
)


def _should_expand_vllm_moe_params(architectures: Optional[list[str]] = None) -> bool:
    current_version = get_version("vllm")
    if not current_version:
        return False

    try:
        version_ok = vs.parse(current_version) <= vs.parse("0.24.0")
    except vs.InvalidVersion:
        return False

    if not version_ok:
        return False

    if architectures is None or len(architectures) == 0:
        return False

    return architectures[0] in _MOE_EXPAND_ARCHITECTURE_WHITELIST


# HiF8 max value: 2^15 × 1.5 = 49152 (Dot=4, E=±15, M=1bit)
_HIF8_MAX: float = 49152.0


def _quant_hif8_inline(x: torch.Tensor) -> torch.Tensor:
    """Per-element HiF8 tapered-precision quant, matching _quant_hif8 in vllm-ascend."""
    x_unsigned = x.abs()
    sign = x.sign()
    eps = x_unsigned.amax().clamp(min=1e-30) * 1e-8
    e = torch.floor(torch.log2(x_unsigned + eps))
    abse = e.abs()
    mant_bits = torch.where(
        abse <= 3, 3.0,
        torch.where(abse <= 7, 2.0,
                    torch.where(abse <= 15, 1.0, 0.0)))
    q = torch.floor(x_unsigned * 2.0 ** (-e + mant_bits) + 0.5)
    return q * 2.0 ** (e - mant_bits) * sign


def _hif8_fake_quant_inline(
    tensor: torch.Tensor, granularity: str = "per_tensor", group_size: int = 32
) -> torch.Tensor:
    """HiF8 fake quant, identical to _hif8_fake_quant in vllm-ascend.

    Inlined here to avoid ``import vllm_ascend`` in the WorkerDict
    (training) process, which can trigger NPU-related side effects.
    """
    if granularity == "per_group":
        t = tensor.float()
        dim_size = t.shape[-1]
        pad = (group_size - dim_size % group_size) % group_size
        if pad:
            t = torch.nn.functional.pad(t, (0, pad))
        t_blocks = t.unflatten(-1, (-1, group_size))
        amax = t_blocks.abs().amax(dim=-1, keepdim=True)
        scale = (amax / _HIF8_MAX).clamp(min=1e-12)
        q_blocks = _quant_hif8_inline(t_blocks / scale) * scale
        result = q_blocks.flatten(-2, -1)
        if pad:
            result = result[..., :dim_size]
        return result.to(tensor.dtype)
    elif granularity == "per_channel":
        amax = tensor.float().abs().amax(dim=-1, keepdim=True)
    else:
        amax = tensor.float().abs().max()
    scale = (amax / _HIF8_MAX).clamp(min=1e-12)
    return (_quant_hif8_inline(tensor.float() / scale) * scale).to(tensor.dtype)


async def _pre_quantize_weights(weights, *, qat_config: dict | None = None):
    """Fake-quantize weight tensors on CPU before sending to vLLM.

    This runs on the verl training worker where weights are on CPU
    (param_offload=True), so the float32 upcast uses system RAM instead
    of NPU HBM — avoiding OOM for large MoE models.

    Only quantizes weights that the vLLM HiF8 scheme wraps (linear + MoE
    expert weights).  Embedding, lm_head, and MoE gate layers are skipped.

    When ``qat_config`` contains ``rotation_enable=True``, block Hadamard-sign
    rotation is applied before fake-quant so that vLLM receives pre-rotated
    weights — matching the QAT training forward where ``apply_block_rotation``
    runs before ``HIF8FakeQuantFunction``.
    """
    import re

    _IGNORE_PATTERNS = [
        re.compile(r".*\.(embed_tokens|lm_head)\.weight$"),
        re.compile(r".*\.mlp\.gate\."),
    ]

    from verl.workers.rollout.utils import ensure_async_iterator

    qat_config = qat_config or {}
    rotation_enable = bool(qat_config.get("rotation_enable", False))
    rotation_block_size = int(qat_config.get("rotation_block_size", 32))
    rotation_seed = int(qat_config.get("rotation_seed", 0))

    if rotation_enable:
        from verl.utils.qat.block_rotation import BlockRotationConfig, apply_block_rotation

        rotation_config = BlockRotationConfig(
            enable=True, block_size=rotation_block_size, seed=rotation_seed
        )
    else:
        rotation_config = None

    async for name, tensor in ensure_async_iterator(weights):
        if any(p.search(name) for p in _IGNORE_PATTERNS):
            yield name, tensor
        else:
            if rotation_config is not None:
                tensor = apply_block_rotation(tensor, rotation_config)
            yield name, _hif8_fake_quant_inline(tensor, "per_tensor", 32).contiguous()



async def _iter_vllm_compatible_moe_params(weights):
    """Expand Transformers 5 packed MoE expert tensors to vLLM checkpoint keys.

    Transformers 5 stores Qwen-style MoE experts as packed 3D parameters:
    ``mlp.experts.gate_up_proj`` with shape
    ``[num_experts, 2 * intermediate_size, hidden_size]`` and
    ``mlp.experts.down_proj`` with shape
    ``[num_experts, hidden_size, intermediate_size]``. vLLM's Qwen MoE reload
    path still accepts the original per-expert checkpoint keys during live
    weight sync, so stream those keys without materializing a full dict.
    """
    from verl.workers.rollout.utils import ensure_async_iterator

    async for name, tensor in ensure_async_iterator(weights):
        if name.endswith(".mlp.experts.gate_up_proj") and tensor.dim() == 3:
            gate, up = tensor.chunk(2, dim=1)
            base = name.removesuffix(".gate_up_proj")
            for expert_id in range(tensor.size(0)):
                yield f"{base}.{expert_id}.gate_proj.weight", gate[expert_id].contiguous()
                yield f"{base}.{expert_id}.up_proj.weight", up[expert_id].contiguous()
            continue

        if name.endswith(".mlp.experts.down_proj") and tensor.dim() == 3:
            base = name.removesuffix(".down_proj")
            for expert_id in range(tensor.size(0)):
                yield f"{base}.{expert_id}.down_proj.weight", tensor[expert_id].contiguous()
            continue

        yield name, tensor


class ServerAdapter(BaseRollout):
    """
    vLLM server adapter used in native async mode, serve as a client to request vLLM server
    to resume/release/update weights and kv_cache.
    """

    def __init__(
        self,
        config: RolloutConfig,
        model_config: HFModelConfig,
        device_mesh: DeviceMesh,
        replica_rank: int = -1,
    ):
        super().__init__(config, model_config, device_mesh)
        self.server_handle: ray.actor.ActorHandle = None

        rank = int(os.environ["RANK"])
        local_world_size = int(os.environ["RAY_LOCAL_WORLD_SIZE"])
        rollout_world_size = (
            self.config.tensor_model_parallel_size
            * self.config.data_parallel_size
            * self.config.pipeline_model_parallel_size
        )
        if replica_rank == -1:
            self.replica_rank = rank // rollout_world_size
        else:
            self.replica_rank = replica_rank
        self.rollout_rank = rank % rollout_world_size
        self.node_rank = self.rollout_rank // local_world_size

        if config.layered_summon or (config.expert_parallel_size > 1 and not _check_vllm_version_for_sleep_level()):
            logger.warning("Setting the sleep level to 1 may cause a memory overflow.")
            self.sleep_level = 1
        else:
            self.sleep_level = VLLM_SLEEP_LEVEL

        self.device_uuid = get_device_uuid(get_device_id())
        # Use replica_rank + node-local rank to form ZMQ handle instead of GPU UUID,
        # because CheckpointEngineWorker and vLLM worker may see different GPU UUIDs
        # when CUDA_VISIBLE_DEVICES differs between processes (common on ROCm/AMD).
        # Must use node-local rank (not rollout_rank) so it matches vLLM worker's
        # local_rank on every node. Include replica_rank to avoid collisions when
        # multiple replicas share a node, and the Ray job id so two independent
        # verl jobs on the same host (or a new run after a crashed one with a
        # stale socket file) cannot collide on the shared /tmp namespace.
        local_rank = self.rollout_rank % local_world_size
        job_id = ray.get_runtime_context().get_job_id()
        self.zmq_handle = f"ipc:///tmp/rl-colocate-zmq-{job_id}-replica-{self.replica_rank}-rank-{local_rank}.sock"

        self.use_shm = not is_support_ipc()
        if self.use_shm:
            logger.warning(
                "IPC is not supported on your devices. Falling back to shared memory for weight transfer, "
                "which may cause performance degradation. If you are using Ascend NPUs, please ensure that "
                "your software and CANN toolkit versions meet the requirements for IPC support. (Ascend HDK version "
                ">= 25.3.rc1 and CANN toolkit version >= 8.3.RC1)"
            )

    def _ensure_server_handle(self) -> bool:
        """Lazy-init server handle. Returns False if this rank should not proceed."""
        if self.rollout_rank != 0:
            return False
        # Lazy init http server adapter because http server is launched after hybrid engine.
        if self.server_handle is None:
            prefix = self._get_server_name_prefix()
            self.server_handle = ray.get_actor(f"{prefix}server_{self.replica_rank}_{self.node_rank}")
        return True

    async def _execute_method(
        self,
        method: str,
        non_block: bool = False,
        timeout: Optional[float] = None,
        args: tuple = (),
        kwargs: Optional[dict] = None,
    ) -> Any:
        """Execute method on inference engine via ray.

        Args:
            method: The method name to execute on the server.
            non_block: If True, execute the method asynchronously and return immediately.
            timeout: Timeout for the collective_rpc call.
            args: Positional arguments for the method.
            kwargs: Keyword arguments for the method.

        Returns:
            The result of the method execution, or None if non_block=True.
        """
        if not self._ensure_server_handle():
            return None

        future = self.server_handle.collective_rpc.remote(method, timeout=timeout, args=args, kwargs=kwargs)
        return future if non_block else await future

    async def resume(self, tags: list[str]):
        """Resume rollout weights or kv cache in GPU memory.

        Args:
            tags: weights or kv_cache.
        """
        if self.config.free_cache_engine and self._ensure_server_handle():
            await self.server_handle.wake_up.remote(tags=tags)

    async def release(self):
        """Release weights and kv cache in GPU memory."""
        if self.config.free_cache_engine and self._ensure_server_handle():
            await self.server_handle.sleep.remote()

    @torch.no_grad()
    async def update_weights(
        self, weights: Generator[tuple[str, torch.Tensor], None, None], global_steps: int = None, **kwargs
    ):
        """Update model weights via CUDA IPC (fallback to shared memory if IPC not supported) to inference workers."""
        start_time = time.time()

        future = await self._execute_method(
            "update_weights_from_ipc",
            non_block=True,
            kwargs={**kwargs, "use_shm": self.use_shm},
        )

        bucket_size_mb = self.config.checkpoint_engine.update_weights_bucket_megabytes
        sender = BucketedWeightSender(
            zmq_handle=self.zmq_handle,
            bucket_size_mb=bucket_size_mb,
            use_shm=self.use_shm,
        )
        if _should_expand_vllm_moe_params(self.model_config.architectures) and not (
            kwargs.get("peft_config") is not None and kwargs.get("base_sync_done", False)
        ):
            weights = _iter_vllm_compatible_moe_params(weights)
        # Pre-quantize weights on CPU before sending to vLLM.
        # This avoids the fp32 memory spike from _hif8_fake_quant on NPU,
        # which OOMs for large MoE models.  vLLM receives already-quantized
        # bf16 weights and stores them directly — no second copy needed.
        qat_config = getattr(self.config, "qat", None) or {}
        weights = _pre_quantize_weights(weights, qat_config=qat_config)
        await sender.async_send_weights(weights)

        if future is not None:
            await future

        # reset prefix cache after updating weights
        if self.rollout_rank == 0:
            await self.server_handle.clear_kv_cache.remote()
            if global_steps is not None:
                await self.server_handle.set_global_steps.remote(global_steps)

        if self.replica_rank == 0 and self.rollout_rank == 0:
            logger.info(f"update_weights done, time cost: {time.time() - start_time:.2f}s")

    def _get_server_name_prefix(self) -> str:
        """Return the Ray actor name prefix matching the rollout type (e.g. 'vllm_')."""
        return f"{self.config.get('name', 'vllm')}_"

    def generate_sequences(self, prompts: DataProto) -> DataProto:
        """Batch generate sequences in sync mode.

        Note: ServerAdapter uses async server mode and does not support synchronous
        generation. Since SPMD mode was retired (PR #4411), the generation workflow
        should use the async server interface instead.

        Raises:
            NotImplementedError: Always raised as sync generation is not supported.
        """
        raise NotImplementedError(
            "ServerAdapter does not support synchronous generate_sequences(). "
            "The vLLM SPMD mode was retired in PR #4411. For batch generation, "
            "please use the async server interface via vLLMReplica and LLMServerClient, "
            "or use HFRollout for synchronous generation. "
            "See https://github.com/verl-project/verl/issues/4682 for more details."
        )
