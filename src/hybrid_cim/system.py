"""Facade combining RRAM-CIM and SRAM-CIM units."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .data_mover import DataMover
from .digital import DigitalProcessor
from .events import StatsCollector
from .tensor import TensorPacket
from .units.base import WeightHandle
from .units.rram import RRAMCIM
from .units.sram import SRAMCIM


class HybridCIMSystem:
    """Entry point for the current two-unit Hybrid CIM simulation."""

    def __init__(
        self,
        *,
        input_bit_width: int = 8,
        weight_bit_width: int = 8,
        output_bit_width: int = 32,
    ) -> None:
        self.stats = StatsCollector()
        common = {
            "stats": self.stats,
            "input_bit_width": input_bit_width,
            "weight_bit_width": weight_bit_width,
            "output_bit_width": output_bit_width,
        }
        self.rram = RRAMCIM("rram_cim", **common)
        self.sram = SRAMCIM("sram_cim", **common)
        self.digital = DigitalProcessor(
            "digital_processor", self.stats, output_bit_width=output_bit_width
        )
        self.data_mover = DataMover(self.stats)

    def create_tensor(
        self,
        name: str,
        data: ArrayLike,
        *,
        bit_width: int | None = None,
        location: str = "digital_buffer",
    ) -> TensorPacket:
        return TensorPacket(name, data, bit_width or self.rram.input_bit_width, location)

    def transfer(
        self,
        tensor: TensorPacket,
        destination: str,
        *,
        transfer_name: str | None = None,
        output_name: str | None = None,
    ) -> TensorPacket:
        return self.data_mover.transfer(
            tensor,
            destination,
            transfer_name=transfer_name,
            output_name=output_name,
        )

    def program_rram_weight(self, name: str, weights: ArrayLike) -> WeightHandle:
        return self.rram.program_weights(name, weights)

    def load_sram_weight(self, name: str, weights: ArrayLike) -> WeightHandle:
        return self.sram.load_weights(name, weights)

    def rram_matmul(
        self,
        lhs: ArrayLike,
        weight: WeightHandle,
        *,
        op_name: str = "rram_matmul",
    ) -> NDArray[np.generic]:
        return self.rram.matmul(lhs, weight, op_name=op_name)

    def sram_resident_matmul(
        self,
        lhs: ArrayLike,
        weight: WeightHandle,
        *,
        op_name: str = "sram_resident_matmul",
    ) -> NDArray[np.generic]:
        return self.sram.matmul(lhs, weight, op_name=op_name)

    def sram_dynamic_matmul(
        self,
        lhs: ArrayLike,
        rhs: ArrayLike,
        *,
        op_name: str = "sram_dynamic_matmul",
    ) -> NDArray[np.generic]:
        return self.sram.matmul_dynamic(lhs, rhs, op_name=op_name)

    def rram_matmul_tensor(
        self,
        lhs: TensorPacket,
        weight: WeightHandle,
        *,
        output_name: str,
        op_name: str = "rram_matmul",
    ) -> TensorPacket:
        return self.rram.matmul_tensor(
            lhs, weight, output_name=output_name, op_name=op_name
        )

    def sram_resident_matmul_tensor(
        self,
        lhs: TensorPacket,
        weight: WeightHandle,
        *,
        output_name: str,
        op_name: str = "sram_resident_matmul",
    ) -> TensorPacket:
        return self.sram.matmul_tensor(
            lhs, weight, output_name=output_name, op_name=op_name
        )

    def sram_dynamic_matmul_tensor(
        self,
        lhs: TensorPacket,
        rhs: TensorPacket,
        *,
        output_name: str,
        op_name: str = "sram_dynamic_matmul",
    ) -> TensorPacket:
        return self.sram.matmul_dynamic_tensor(
            lhs, rhs, output_name=output_name, op_name=op_name
        )

    def summary(self) -> dict[str, Any]:
        return self.stats.summary()

    def reset_stats(self) -> None:
        self.stats.reset()

