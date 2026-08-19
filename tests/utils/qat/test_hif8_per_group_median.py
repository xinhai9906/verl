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

"""Tests for the HiF8 per_group_median scale algorithm.

per_group_median: per group of group_size elements, the average of the two
middle sorted |x| values is anchored to 1.0, with amax/HIF8_MAX as the lower
bound so truncation never occurs: scale = max(median_avg, amax / 49152).
"""

import pytest
import torch

from verl.utils.qat.linear import HIF8QATLinear, HIF8_MAX, _quant_hif8, hif8_fake_quant


def _reference_scale(tensor: torch.Tensor, group_size: int) -> torch.Tensor:
    """Independent reference computation of the per_group_median scale."""
    t = tensor.float()
    dim_size = t.shape[-1]
    pad = (group_size - dim_size % group_size) % group_size
    if pad:
        t = torch.nn.functional.pad(t, (0, pad))
    blocks = t.unflatten(-1, (-1, group_size))
    amax = blocks.abs().amax(dim=-1, keepdim=True)
    abs_med = tensor.float().abs()
    if pad:
        abs_med = torch.nn.functional.pad(abs_med, (0, pad), value=float("inf"))
    sorted_vals = abs_med.unflatten(-1, (-1, group_size)).sort(dim=-1).values
    lo = (group_size - 1) // 2
    hi = group_size // 2 + 1
    median_avg = sorted_vals[..., lo:hi].mean(dim=-1, keepdim=True)
    median_avg = torch.nan_to_num(median_avg, nan=0.0, posinf=0.0)
    return torch.maximum(median_avg, amax / HIF8_MAX).clamp(min=1e-12)


class TestPerGroupMedian:
    """per_group_median scale semantics."""

    def test_rotation_block_size_must_equal_group_size(self):
        with pytest.raises(ValueError, match="block_size == group_size"):
            HIF8QATLinear(
                64, 32, granularity="per_group_median", group_size=32,
                rotation_enable=True, rotation_block_size=16,
            )
        # matching sizes must not raise
        HIF8QATLinear(
            64, 32, granularity="per_group_median", group_size=32,
            rotation_enable=True, rotation_block_size=32,
        )

    def test_median_anchors_to_one(self):
        # Two middle sorted values both equal 16 => median_avg = 16 > amax/49152,
        # so scale = 16 and those elements map to exactly 1.0 (exact roundtrip).
        block = torch.arange(32, dtype=torch.float32)
        block[15] = 16.0
        block[16] = 16.0
        out = hif8_fake_quant(block.unsqueeze(0), "per_group_median", 32)
        assert out[0, 15].item() == 16.0
        assert out[0, 16].item() == 16.0

    def test_scale_matches_reference(self):
        torch.manual_seed(0)
        t = torch.randn(4, 64) * 10
        group_size = 32
        ref_scale = _reference_scale(t, group_size)
        expected = (_quant_hif8(t.unflatten(-1, (-1, group_size)) / ref_scale) * ref_scale).flatten(-2, -1)
        out = hif8_fake_quant(t, "per_group_median", group_size)
        torch.testing.assert_close(out.float(), expected, rtol=0.0, atol=0.0)

    def test_no_truncation_invariant(self):
        torch.manual_seed(1)
        t = torch.randn(8, 96) * 100
        t.view(8, -1, 32)[:, :, -1] = torch.rand(8, 3) * 40000
        scale = _reference_scale(t, 32)
        blocks = t.unflatten(-1, (-1, 32))
        assert (blocks / scale).abs().max().item() <= HIF8_MAX + 1e-3

    def test_lower_bound_dominates_with_outlier(self):
        # 31 zeros + one huge outlier: median_avg = 0 => scale = amax/HIF8_MAX,
        # identical to plain per_group mode.
        t = torch.zeros(1, 32)
        t[0, -1] = 40000.0
        out_median = hif8_fake_quant(t, "per_group_median", 32)
        out_amax = hif8_fake_quant(t, "per_group", 32)
        torch.testing.assert_close(out_median, out_amax, rtol=0.0, atol=0.0)

    def test_pad_tail_block(self):
        t = torch.randn(2, 33) * 5
        out = hif8_fake_quant(t, "per_group_median", 32)
        assert out.shape == t.shape
        assert torch.isfinite(out).all()
        ref_scale = _reference_scale(t, 32)
        blocks = torch.nn.functional.pad(t.float(), (0, 31)).unflatten(-1, (-1, 32))
        expected = (_quant_hif8(blocks / ref_scale) * ref_scale).flatten(-2, -1)[..., :33]
        torch.testing.assert_close(out.float(), expected, rtol=0.0, atol=0.0)
