# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import time
from pathlib import Path

import pytest
import torch
from huggingface_hub import snapshot_download
from huggingface_hub.utils import HfHubHTTPError

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.entrypoints.openai.api_server import (
    build_async_engine_client_from_engine_args,
)
from vllm.inputs import TextPrompt
from vllm.lora.punica_wrapper.punica_cpu import PunicaWrapperCPU
from vllm.lora.request import LoRARequest
from vllm.sampling_params import SamplingParams
from vllm.utils.async_utils import merge_async_iterators

DORA_MODEL_PATH = "unsloth/Llama-3.2-1B-Instruct"
DORA_MODULE_ID = "makcedward/Llama-3.2-1B-Instruct-DoRA-Adapter"
DORA_RANK = 16
DEFAULT_MAX_LORAS = 4
DORA_MODULE_PATH: str | None = None


def test_dora_expand_scales_base_and_delta():
    tokens = 2
    hidden = 2
    rank = 1
    max_loras = 2
    punica = PunicaWrapperCPU(tokens, max_batches=1, device="cpu")
    punica.indices_len[0] = tokens
    punica._token_lora_indices[:tokens] = torch.tensor([0, 1], device="cpu")

    y = torch.ones(tokens, hidden)
    buffer = (torch.ones(tokens, rank),)
    lora_b = (torch.zeros(max_loras, 1, hidden, rank),)
    lora_b[0][0, 0].fill_(1.0)
    lora_b[0][1, 0].fill_(1.0)

    lora_magnitude = (torch.tensor([[[2.0, 2.0]], [[3.0, 3.0]]]),)
    base_norm = (torch.tensor([[[1.0, 1.0]], [[2.0, 2.0]]]),)

    punica.add_expand(
        y,
        buffer,
        lora_b,
        (hidden,),
        lora_magnitude_stacked=lora_magnitude,
        lora_base_norm_stacked=base_norm,
    )

    expected = torch.tensor([[4.0, 4.0], [4.5, 4.5]])
    assert torch.allclose(y, expected)


def _prepare_dora_module(tmp_path) -> str:
    try:
        module_path = snapshot_download(repo_id=DORA_MODULE_ID, cache_dir=tmp_path)
    except HfHubHTTPError as e:
        pytest.skip(f"Skipping DoRA integration test (download failed): {e}")

    # Remove tokenizer artifacts to save space / avoid tokenizer loading
    tokenizer_files = [
        "added_tokens.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "tokenizer.model",
    ]
    for tokenizer_file in tokenizer_files:
        Path(module_path, tokenizer_file).unlink(missing_ok=True)
    return module_path


def _get_dora_requests(module_path: str) -> list[LoRARequest]:
    return [
        LoRARequest(lora_name=f"{i}", lora_int_id=i, lora_path=module_path)
        for i in range(1, DEFAULT_MAX_LORAS + 1)
    ]


async def _requests_processing_time(llm, lora_requests: list[LoRARequest]) -> float:
    sampling_params = SamplingParams(
        n=1, temperature=0.0, top_p=1.0, ignore_eos=True, max_tokens=1
    )

    generators = []
    start = time.perf_counter()

    for lora_request in lora_requests:
        lora_int_id = lora_request.lora_int_id
        generator = llm.generate(
            prompt=TextPrompt(prompt=f"hello {lora_int_id}", multi_modal_data=None),  # type: ignore
            sampling_params=sampling_params,
            lora_request=lora_request,
            request_id=f"test{lora_int_id}",
        )
        generators.append(generator)

    all_gens = merge_async_iterators(*generators)
    async for _i, _res in all_gens:
        pass

    end = time.perf_counter()
    return end - start


@pytest.mark.asyncio
@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="DoRA integration test requires GPU to avoid timeouts.",
)
async def test_add_dora(tmp_path):
    """
    Ensures preloading DoRA adapters via add_lora is faster than on-demand load.
    """
    module_path = _prepare_dora_module(tmp_path)
    dora_requests = _get_dora_requests(module_path)

    max_loras = len({lr.lora_int_id for lr in dora_requests})
    engine_args = AsyncEngineArgs(
        model=DORA_MODEL_PATH,
        enable_lora=True,
        max_loras=max_loras,
        max_lora_rank=DORA_RANK,
        max_model_len=128,
        gpu_memory_utilization=0.8,
        enforce_eager=True,
    )

    part_size = len(dora_requests) // 3
    dummy_run_requests = dora_requests[:part_size]
    warmup_run_requests = dora_requests[part_size : part_size * 2]
    cold_run_requests = dora_requests[part_size * 2 :]

    async with build_async_engine_client_from_engine_args(engine_args) as llm:
        await _requests_processing_time(llm, dummy_run_requests)

        add_lora_tasks = [llm.add_lora(lr) for lr in warmup_run_requests]
        add_lora_results = await asyncio.gather(*add_lora_tasks)
        assert all(add_lora_results)

        time_with_add_dora = await _requests_processing_time(llm, warmup_run_requests)
        time_cold_start = await _requests_processing_time(llm, cold_run_requests)

    assert time_with_add_dora < time_cold_start, (
        f"time_with_add_dora={time_with_add_dora}, time_cold_start={time_cold_start}"
    )
