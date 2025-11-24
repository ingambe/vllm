# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import torch
from transformers import PretrainedConfig

from vllm.config.lora import LoRAConfig
from vllm.distributed.utils import divide
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    LinearBase,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.platforms import current_platform

from .base import BaseLayerWithLoRA
from .utils import _get_lora_device


class BaseLinearLayerWithLoRA(BaseLayerWithLoRA):
    def __init__(self, base_layer: LinearBase):
        super().__init__()
        self.base_layer = base_layer
        self.input_size = self.base_layer.input_size
        # Ensure tp_size and tp_rank consistency with the base_layer.
        self.tp_size = self.base_layer.tp_size
        self.tp_rank = self.base_layer.tp_rank
        self.device = _get_lora_device(self.base_layer)
        self.output_slices: tuple[int, ...]
        self.output_size: int
        self.n_slices: int
        self.lora_mag_stacked: tuple[torch.Tensor, ...] | None = None
        self.lora_base_norm_stacked: tuple[torch.Tensor, ...] | None = None
        self._base_norm_cache: torch.Tensor | list[torch.Tensor] | None = None
        self._dora_slots: set[int] = set()

    def create_lora_weights(
        self,
        max_loras: int,
        lora_config: LoRAConfig,
        model_config: PretrainedConfig | None = None,
    ) -> None:
        self.lora_config = lora_config
        #
        if isinstance(self.base_layer, ReplicatedLinear):
            lora_a_out_size = lora_config.max_lora_rank
            lora_b_out_size = self.output_size

        elif isinstance(self.base_layer, ColumnParallelLinear):
            lora_a_out_size = (
                lora_config.max_lora_rank
                if not lora_config.fully_sharded_loras
                else divide(lora_config.max_lora_rank, self.tp_size)
            )
            lora_b_out_size = self.output_size

        elif isinstance(self.base_layer, RowParallelLinear):
            lora_a_out_size = lora_config.max_lora_rank
            lora_b_out_size = (
                self.output_size
                if not lora_config.fully_sharded_loras
                else divide(self.output_size, self.tp_size)
            )
        else:
            raise NotImplementedError

        self.lora_a_stacked = tuple(
            torch.zeros(
                max_loras,
                1,
                lora_a_out_size,
                self.input_size,
                dtype=lora_config.lora_dtype,
                device=self.device,
            )
            for _ in range(self.n_slices)
        )
        self.lora_b_stacked = tuple(
            torch.zeros(
                max_loras,
                1,
                lora_b_out_size,
                lora_config.max_lora_rank,
                dtype=lora_config.lora_dtype,
                device=self.device,
            )
            for _ in range(self.n_slices)
        )
        self.output_slices = (self.lora_b_stacked[0].shape[2],)
        self.lora_mag_stacked = tuple(
            torch.zeros(
                max_loras,
                1,
                self.output_slices[i],
                dtype=lora_config.lora_dtype,
                device=self.device,
            )
            for i in range(self.n_slices)
        )
        self.lora_base_norm_stacked = tuple(
            torch.zeros(
                max_loras,
                1,
                self.output_slices[i],
                dtype=lora_config.lora_dtype,
                device=self.device,
            )
            for i in range(self.n_slices)
        )

    def reset_lora(self, index: int):
        for s_index in range(self.n_slices):
            self.lora_a_stacked[s_index][index] = 0
            self.lora_b_stacked[s_index][index] = 0
            if self.lora_mag_stacked:
                self.lora_mag_stacked[s_index][index] = 0
            if self.lora_base_norm_stacked:
                self.lora_base_norm_stacked[s_index][index] = 0
        if index in self._dora_slots:
            self._dora_slots.discard(index)

    def set_lora(
        self,
        index: int,
        lora_a: torch.Tensor | list[torch.Tensor],
        lora_b: torch.Tensor | list[torch.Tensor],
        lora_magnitude: torch.Tensor | list[torch.Tensor] | None = None,
    ):
        # Except for QKVParallelLinearWithLoRA and
        # MergedColumnParallelLinearWithLoRA, all other linear LoRA layers
        # store weights in a tuple of size 1. These two layers will
        # override this function.
        assert isinstance(lora_a, torch.Tensor)
        assert isinstance(lora_b, torch.Tensor)
        assert (
            len(self.lora_a_stacked) == len(self.lora_b_stacked) == self.n_slices == 1
        )

        self.reset_lora(index)
        base_norm = None
        if lora_magnitude is not None:
            base_norm = self._get_base_norm_slices(lora_a=lora_a, lora_b=lora_b)
        if self.tp_size > 1:
            lora_a = self.slice_lora_a(lora_a)
            lora_b = self.slice_lora_b(lora_b)
            if lora_magnitude is not None:
                lora_magnitude = self.slice_lora_magnitude(lora_magnitude)
                if base_norm is not None:
                    base_norm = self.slice_lora_magnitude(base_norm)

        self.lora_a_stacked[0][index, 0, : lora_a.shape[0], : lora_a.shape[1]].copy_(
            lora_a, non_blocking=True
        )
        self.lora_b_stacked[0][index, 0, : lora_b.shape[0], : lora_b.shape[1]].copy_(
            lora_b, non_blocking=True
        )
        if lora_magnitude is not None and self.lora_mag_stacked:
            lora_mag_tensor: torch.Tensor = (
                lora_magnitude[0]
                if isinstance(lora_magnitude, list)
                else lora_magnitude
            )
            self.lora_mag_stacked[0][index, 0, : lora_mag_tensor.shape[0]].copy_(
                lora_mag_tensor, non_blocking=True
            )
            self._dora_slots.add(index)
        else:
            self._dora_slots.discard(index)
        if base_norm is not None and self.lora_base_norm_stacked:
            base_norm_tensor: torch.Tensor = (
                base_norm[0] if isinstance(base_norm, list) else base_norm
            )
            self.lora_base_norm_stacked[0][index, 0, : base_norm_tensor.shape[0]].copy_(
                base_norm_tensor.to(
                    device=self.device, dtype=self.lora_base_norm_stacked[0].dtype
                ),
                non_blocking=True,
            )

    def apply(self, x: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
        output = self.base_layer.quant_method.apply(self.base_layer, x, bias)

        # In Transformers modeling backend, x and output have extra batch dimension like
        # (1, seq_len, hidden_dim), while punica expects (seq_len, hidden_dim),
        # therefore we need to flatten the batch dimensions.
        if x.ndim == 3 and output.ndim == 3:
            output = output.flatten(0, 1)
            x = x.flatten(0, 1)

        lora_output: torch.Tensor | None = self.punica_wrapper.add_lora_linear(
            output,
            x,
            self.lora_a_stacked,
            self.lora_b_stacked,
            1.0,
            self.output_slices,
            lora_magnitude_stacked=(
                self.lora_mag_stacked if self._dora_slots else None
            ),
            lora_base_norm_stacked=(
                self.lora_base_norm_stacked if self._dora_slots else None
            ),
        )
        if not current_platform.can_update_inplace():
            output = lora_output

        return output

    @property
    def weight(self) -> torch.Tensor:
        # unquantizedLinear
        if hasattr(self.base_layer, "weight"):
            return self.base_layer.weight
        # Compressed Tensor
        elif hasattr(self.base_layer, "weight_packed"):
            return self.base_layer.weight_packed
        # GPTQ/AWQ
        elif hasattr(self.base_layer, "qweight"):
            return self.base_layer.qweight
        # marlin
        elif hasattr(self.base_layer, "B"):
            return self.base_layer.B
        # HQQ marlin
        elif hasattr(self.base_layer, "W_q"):
            return self.base_layer.W_q
        else:
            raise ValueError(f"Unsupported base layer: {self.base_layer}")

    @property
    def bias(self) -> torch.Tensor | None:
        if hasattr(self.base_layer, "bias"):
            return self.base_layer.bias
        else:
            return None

    def slice_lora_magnitude(
        self, lora_magnitude: torch.Tensor | list[torch.Tensor]
    ) -> torch.Tensor | list[torch.Tensor]:
        return lora_magnitude

    def _get_base_norm_slices(
        self,
        lora_a: torch.Tensor | list[torch.Tensor] | None = None,
        lora_b: torch.Tensor | list[torch.Tensor] | None = None,
    ) -> torch.Tensor | list[torch.Tensor]:
        weight = self.weight
        if lora_a is not None and lora_b is not None:
            if isinstance(lora_a, list):
                assert isinstance(lora_b, list)
                base_norms: list[torch.Tensor] = []
                for a_i, b_i in zip(lora_a, lora_b):
                    if a_i is None or b_i is None:
                        base_norms.append(
                            torch.linalg.vector_norm(weight.float(), dim=1)
                        )
                        continue
                    lora_weight = b_i @ a_i
                    w_delta = weight + lora_weight
                    base_norms.append(
                        torch.linalg.vector_norm(w_delta.float(), dim=1).contiguous()
                    )
                return base_norms
            else:
                lora_weight = lora_b @ lora_a
                weight = weight + lora_weight
                base_norm = torch.linalg.vector_norm(weight.float(), dim=1)
                if self.n_slices == 1:
                    return base_norm
                splits = torch.split(base_norm, self.output_slices)
                return [s.contiguous() for s in splits]

        if self._base_norm_cache is None:
            base_norm = torch.linalg.vector_norm(weight.float(), dim=1)
            if self.n_slices == 1:
                self._base_norm_cache = base_norm
            else:
                splits = torch.split(base_norm, self.output_slices)
                self._base_norm_cache = [s.contiguous() for s in splits]
        return self._base_norm_cache
