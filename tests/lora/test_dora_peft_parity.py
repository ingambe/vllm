# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.lora.punica_wrapper.punica_cpu import PunicaWrapperCPU

peft = pytest.importorskip("peft")
from peft.tuners.lora.dora import DoraLinearLayer  # type: ignore  # noqa: E402


def _build_linear(weight: torch.Tensor) -> torch.nn.Linear:
    layer = torch.nn.Linear(weight.shape[1], weight.shape[0], bias=False)
    with torch.no_grad():
        layer.weight.copy_(weight)
    return layer


def test_dora_matches_peft_formula():
    """Punica DoRA scaling matches PeFT reference for a tiny layer."""
    torch.manual_seed(0)
    tokens, hidden, rank = 2, 2, 1

    # Base layer with identity weights so base_result == input.
    base_weight = torch.eye(hidden)
    base_layer = _build_linear(base_weight)

    # Input
    x = torch.ones(tokens, hidden)
    base_result = base_layer(x)

    # LoRA A/B chosen so lora_result is all ones.
    lora_a = _build_linear(torch.full((rank, hidden), 0.5))
    lora_b = _build_linear(torch.ones(hidden, rank))

    # DoRA magnitude and base norms
    magnitude = torch.tensor([[[2.0, 3.0]]])  # shape (1, 1, hidden)
    lora_weight = lora_b.weight @ lora_a.weight
    base_norm = (
        torch.linalg.vector_norm(base_weight + lora_weight, dim=1)
        .view(1, 1, -1)
        .contiguous()
    )

    # PeFT reference: result_dora is the delta; add to base_result for final output.
    dora_layer = DoraLinearLayer(fan_in_fan_out=False)
    dora_layer.update_layer(
        base_layer=base_layer,
        lora_A=lora_a.weight,
        lora_B=lora_b.weight,
        scaling=1.0,
        place_on_cpu=False,
    )
    # Override learned magnitude to our test vector.
    dora_layer.weight.data = magnitude.view_as(dora_layer.weight)
    peft_delta = dora_layer(
        x,
        lora_A=lora_a,
        lora_B=lora_b,
        scaling=1.0,
        base_layer=base_layer,
        base_result=base_result,
    )
    peft_output = base_result + peft_delta

    # Punica path
    wrapper = PunicaWrapperCPU(tokens, max_batches=1, device="cpu")
    wrapper.indices_len[0] = tokens
    wrapper._token_lora_indices[:tokens] = 0
    y = base_result.clone()
    buffer = (lora_a(x),)
    lora_b_stacked = (lora_b.weight.view(1, 1, hidden, rank),)
    wrapper.add_expand(
        y,
        buffer,
        lora_b_stacked,
        (hidden,),
        lora_magnitude_stacked=magnitude,
        lora_base_norm_stacked=base_norm,
    )

    assert torch.allclose(y, peft_output, atol=1e-5)


def test_mixed_dora_and_non_dora_end_to_end():
    """Mix DoRA and non-DoRA adapters in one batch without zeroing outputs."""
    tokens, hidden, rank, max_loras = 3, 4, 2, 3
    wrapper = PunicaWrapperCPU(tokens, max_batches=1, device="cpu")
    wrapper.indices_len[0] = tokens
    wrapper._token_lora_indices[:tokens] = torch.tensor([0, 1, 2], device="cpu")

    # Base outputs from a simple linear layer
    base_weight = torch.eye(hidden)
    base_layer = _build_linear(base_weight)
    x = torch.arange(tokens * hidden, dtype=torch.float32).view(tokens, hidden) % 3
    base_output = base_layer(x)

    # shrink outputs (buffer) using a simple A matrix
    lora_a = torch.nn.Linear(hidden, rank, bias=False)
    torch.nn.init.constant_(lora_a.weight, 0.25)
    buffer = (lora_a(x),)

    # B weights: ones for all adapters
    lora_b = (torch.ones(max_loras, 1, hidden, rank),)

    # DoRA magnitudes only for adapter 0; others are plain LoRA
    lora_magnitude = (torch.zeros(max_loras, 1, hidden),)
    lora_magnitude[0][0, 0].fill_(2.0)
    base_norm_vec = torch.linalg.vector_norm(base_weight, dim=1).view(1, 1, -1)
    lora_base_norm = (base_norm_vec.expand(max_loras, -1, -1).contiguous(),)

    y = base_output.clone()
    wrapper.add_expand(
        y,
        buffer,
        lora_b,
        (hidden,),
        lora_magnitude_stacked=lora_magnitude,
        lora_base_norm_stacked=lora_base_norm,
    )

    # Adapter 0 should be scaled; others should still get their LoRA delta.
    assert torch.all(y[0] > base_output[0])
    assert torch.all(y[1] > base_output[1])  # non-DoRA delta added
    assert torch.all(y[2] > base_output[2])  # non-DoRA delta added
