# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import safetensors.torch
import torch

from vllm.lora.models import LoRAModel
from vllm.lora.peft_helper import PEFTHelper
from vllm.model_executor.models.baichuan import BaiChuanBaseForCausalLM
from vllm.model_executor.models.utils import WeightsMapper

lora_lst = ["baichuan7B", "baichuan7B-zero", "baichuan7B-zero-regex", "chatglm3-6b"]
BAICHUAN_LORA_MODULES = [
    "W_pack",
    "o_proj",
    "gate_up_proj",
    "down_proj",
]


@pytest.mark.parametrize("lora_name", lora_lst)
def test_load_checkpoints(
    lora_name,
    baichuan_lora_files,
    baichuan_zero_lora_files,
    baichuan_regex_lora_files,
    chatglm3_lora_files,
):
    packed_modules_mapping = BaiChuanBaseForCausalLM.packed_modules_mapping
    embedding_modules = BaiChuanBaseForCausalLM.embedding_modules
    embed_padding_modules = BaiChuanBaseForCausalLM.embedding_padding_modules
    expected_lora_modules: list[str] = []
    for module in BAICHUAN_LORA_MODULES:
        if module in packed_modules_mapping:
            expected_lora_modules.extend(packed_modules_mapping[module])
        else:
            expected_lora_modules.append(module)
    if lora_name == "baichuan7B":
        peft_helper = PEFTHelper.from_local_dir(
            baichuan_lora_files, max_position_embeddings=4096
        )
        # For the baichuan7B model, load it's LoRA,
        # and the test should pass.
        LoRAModel.from_local_checkpoint(
            baichuan_lora_files,
            expected_lora_modules,
            peft_helper=peft_helper,
            lora_model_id=1,
            device="cpu",
            embedding_modules=embedding_modules,
            embedding_padding_modules=embed_padding_modules,
        )
    elif lora_name == "baichuan7B-zero":
        # Test that the target_modules contain prefix
        # such as "model.layers.0.self_atten.W_pack", and
        # the test should pass.
        peft_helper = PEFTHelper.from_local_dir(
            baichuan_zero_lora_files, max_position_embeddings=4096
        )
        LoRAModel.from_local_checkpoint(
            baichuan_zero_lora_files,
            expected_lora_modules,
            peft_helper=peft_helper,
            lora_model_id=1,
            device="cpu",
            embedding_modules=embedding_modules,
            embedding_padding_modules=embed_padding_modules,
        )
    elif lora_name == "baichuan7B-zero-regex":
        # Test that the `target_modules` in the form of regular expressions,
        # such as `model\\..*(W_pack|o_proj)`, and the test should pass.
        peft_helper = PEFTHelper.from_local_dir(
            baichuan_regex_lora_files, max_position_embeddings=4096
        )
        LoRAModel.from_local_checkpoint(
            baichuan_regex_lora_files,
            expected_lora_modules,
            peft_helper=peft_helper,
            lora_model_id=1,
            device="cpu",
            embedding_modules=embedding_modules,
            embedding_padding_modules=embed_padding_modules,
        )
    else:
        # For the baichuan7B model, load chatglm3-6b's LoRA,
        # and the test should raise the following error.
        expected_error = "Please verify that the loaded LoRA module is correct"  # noqa: E501
        peft_helper = PEFTHelper.from_local_dir(
            chatglm3_lora_files, max_position_embeddings=4096
        )
        with pytest.raises(ValueError, match=expected_error):
            LoRAModel.from_local_checkpoint(
                chatglm3_lora_files,
                expected_lora_modules,
                peft_helper=peft_helper,
                lora_model_id=1,
                device="cpu",
                embedding_modules=embedding_modules,
                embedding_padding_modules=embed_padding_modules,
            )


def test_lora_weights_mapping(baichuan_lora_files):
    packed_modules_mapping = BaiChuanBaseForCausalLM.packed_modules_mapping
    embedding_modules = BaiChuanBaseForCausalLM.embedding_modules
    embed_padding_modules = BaiChuanBaseForCausalLM.embedding_padding_modules
    expected_lora_modules: list[str] = []
    for module in BAICHUAN_LORA_MODULES:
        if module in packed_modules_mapping:
            expected_lora_modules.extend(packed_modules_mapping[module])
        else:
            expected_lora_modules.append(module)

    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "model.": "language_model.model.",
        },
        orig_to_new_substr={
            ".layers.": ".baichuan_layers.",
        },
    )
    peft_helper = PEFTHelper.from_local_dir(
        baichuan_lora_files, max_position_embeddings=4096
    )
    lora_model = LoRAModel.from_local_checkpoint(
        baichuan_lora_files,
        expected_lora_modules,
        peft_helper=peft_helper,
        lora_model_id=1,
        device="cpu",
        embedding_modules=embedding_modules,
        embedding_padding_modules=embed_padding_modules,
        weights_mapper=hf_to_vllm_mapper,
    )
    for name in lora_model.loras:
        assert name.startswith(hf_to_vllm_mapper.orig_to_new_prefix["model."])
        assert ".baichuan_layers." in name


def _write_minimal_dora_adapter(tmp_path) -> str:
    adapter_dir = tmp_path / "dora"
    adapter_dir.mkdir()
    config = {
        "r": 2,
        "lora_alpha": 4,
        "target_modules": ["dense"],
        "use_dora": True,
    }
    (adapter_dir / "adapter_config.json").write_text(__import__("json").dumps(config))
    lora_a = torch.randn(2, 3)
    lora_b = torch.randn(5, 2)
    magnitude = torch.rand(5)
    safetensors.torch.save_file(
        {
            "dense.lora_A.weight": lora_a,
            "dense.lora_B.weight": lora_b,
            "dense.lora_magnitude_vector": magnitude,
        },
        adapter_dir / "adapter_model.safetensors",
    )
    return str(adapter_dir)


def test_load_dora_checkpoint(tmp_path):
    adapter_dir = _write_minimal_dora_adapter(tmp_path)
    peft_helper = PEFTHelper.from_local_dir(adapter_dir, max_position_embeddings=128)
    lora_model = LoRAModel.from_local_checkpoint(
        adapter_dir,
        expected_lora_modules=["dense"],
        peft_helper=peft_helper,
        lora_model_id=1,
        device="cpu",
        embedding_modules={},
        embedding_padding_modules=[],
    )
    dense = lora_model.loras["dense"]
    assert dense.magnitude_vector is not None
    assert dense.magnitude_vector.shape[0] == dense.lora_b.shape[0]


def test_load_dora_checkpoint_missing_magnitude(tmp_path):
    adapter_dir = tmp_path / "dora_missing"
    adapter_dir.mkdir()
    config = {
        "r": 2,
        "lora_alpha": 4,
        "target_modules": ["dense"],
        "use_dora": True,
    }
    (adapter_dir / "adapter_config.json").write_text(__import__("json").dumps(config))
    lora_a = torch.randn(2, 3)
    lora_b = torch.randn(5, 2)
    safetensors.torch.save_file(
        {
            "dense.lora_A.weight": lora_a,
            "dense.lora_B.weight": lora_b,
        },
        adapter_dir / "adapter_model.safetensors",
    )
    peft_helper = PEFTHelper.from_local_dir(adapter_dir, max_position_embeddings=128)
    with pytest.raises(ValueError, match="magnitude_vector"):
        LoRAModel.from_local_checkpoint(
            adapter_dir,
            expected_lora_modules=["dense"],
            peft_helper=peft_helper,
            lora_model_id=1,
            device="cpu",
            embedding_modules={},
            embedding_padding_modules=[],
        )
