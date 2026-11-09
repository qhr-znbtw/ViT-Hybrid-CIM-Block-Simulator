"""Composite Hybrid CIM implementation of one linear layer."""

from __future__ import annotations

import numpy as np

from ..errors import ComponentNotPreparedError, MatrixShapeError
from ..system import HybridCIMSystem
from ..tensor import TensorPacket
from ..units.base import WeightHandle
from ..weights import HybridLinearWeights


class HybridLinear:
    """Execute ``XW0 + (XA)B + bias`` across RRAM, SRAM, and digital units."""

    def __init__(self, name: str, system: HybridCIMSystem) -> None:
        if not name or not name.strip():
            raise ValueError("component name must be non-empty")
        self.name = name
        self.system = system
        self._weights: HybridLinearWeights | None = None
        self._backbone_handle: WeightHandle | None = None
        self._lora_a_handle: WeightHandle | None = None
        self._lora_b_handle: WeightHandle | None = None
        self._bias: np.ndarray | None = None

    @property
    def is_prepared(self) -> bool:
        return self._weights is not None

    @property
    def input_features(self) -> int:
        self._require_prepared()
        assert self._weights is not None
        return self._weights.input_features

    @property
    def output_features(self) -> int:
        self._require_prepared()
        assert self._weights is not None
        return self._weights.output_features

    def _require_prepared(self) -> None:
        if self._weights is None or self._backbone_handle is None:
            raise ComponentNotPreparedError(
                f"HybridLinear {self.name!r} must be prepared before execution"
            )

    def prepare(self, weights: HybridLinearWeights) -> None:
        """Program/load all layer weights while preserving preparation statistics."""

        self._weights = weights
        self._backbone_handle = self.system.program_rram_weight(
            f"{self.name}.backbone", weights.backbone
        )
        self._lora_a_handle = None
        self._lora_b_handle = None
        self._bias = None
        if weights.has_lora:
            assert weights.lora_a is not None and weights.lora_b is not None
            self._lora_a_handle = self.system.load_sram_weight(
                f"{self.name}.lora_a", weights.lora_a
            )
            self._lora_b_handle = self.system.load_sram_weight(
                f"{self.name}.lora_b", weights.lora_b
            )
        if weights.bias is not None:
            self._bias = self.system.digital.load_parameter(
                f"{self.name}.bias", weights.bias
            )

    def forward(
        self,
        tensor: TensorPacket,
        *,
        output_name: str | None = None,
    ) -> TensorPacket:
        """Run the full data path and return the result at the digital processor."""

        self._require_prepared()
        assert self._weights is not None
        assert self._backbone_handle is not None
        if tensor.data.ndim < 2 or tensor.shape[-1] != self._weights.input_features:
            raise MatrixShapeError(
                f"{self.name} expects input feature dimension "
                f"{self._weights.input_features}; got {tensor.shape}"
            )

        final_name = output_name or f"{self.name}.output"
        if self._weights.has_lora:
            x_rram, x_sram = self.system.data_mover.broadcast(
                tensor,
                (self.system.rram.name, self.system.sram.name),
                transfer_name_prefix=f"{self.name}.input_broadcast",
            )
        else:
            x_rram = self.system.transfer(
                tensor,
                self.system.rram.name,
                transfer_name=f"{self.name}.input_to_rram",
            )
            x_sram = None

        backbone = self.system.rram_matmul_tensor(
            x_rram,
            self._backbone_handle,
            output_name=f"{self.name}.backbone_output",
            op_name=f"{self.name}.backbone_matmul",
        )
        backbone_digital = self.system.transfer(
            backbone,
            self.system.digital.name,
            transfer_name=f"{self.name}.backbone_to_digital",
        )

        merged = backbone_digital
        if self._weights.has_lora:
            assert x_sram is not None
            assert self._lora_a_handle is not None and self._lora_b_handle is not None
            lora_hidden = self.system.sram_resident_matmul_tensor(
                x_sram,
                self._lora_a_handle,
                output_name=f"{self.name}.lora_hidden",
                op_name=f"{self.name}.lora_a_matmul",
            )
            lora_output = self.system.sram_resident_matmul_tensor(
                lora_hidden,
                self._lora_b_handle,
                output_name=f"{self.name}.lora_output",
                op_name=f"{self.name}.lora_b_matmul",
            )
            lora_digital = self.system.transfer(
                lora_output,
                self.system.digital.name,
                transfer_name=f"{self.name}.lora_to_digital",
            )
            merged = self.system.digital.add(
                backbone_digital,
                lora_digital,
                name=f"{self.name}.merged",
                op_name=f"{self.name}.merge",
            )

        if self._bias is not None:
            return self.system.digital.bias_add(
                merged,
                self._bias,
                name=final_name,
                op_name=f"{self.name}.bias",
            )

        if merged.name == final_name:
            return merged
        return TensorPacket(
            final_name,
            np.asarray(merged.data),
            merged.bit_width,
            merged.location,
        )
