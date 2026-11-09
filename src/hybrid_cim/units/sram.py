"""SRAM-CIM unit for resident and dynamic matrix multiplication."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..errors import TensorLocationError
from ..tensor import TensorPacket
from .base import ResidentMatmulUnit, WeightHandle


class SRAMCIM(ResidentMatmulUnit):
    """Matrix engine supporting resident weights and two dynamic operands."""

    storage_action = "load"

    def load_weights(self, name: str, weights: ArrayLike) -> WeightHandle:
        """Load or replace a matrix in SRAM and return its versioned handle."""

        return self._store_weight(name, weights)

    def matmul(
        self,
        lhs: ArrayLike,
        weight: WeightHandle,
        *,
        op_name: str = "sram_resident_matmul",
    ) -> NDArray[np.generic]:
        """Compute ``lhs @ resident_weight`` and record one compute event."""

        rhs = self._resolve_weight(weight)
        return self._matmul(lhs, rhs, op_name=op_name, mode="resident")

    def matmul_dynamic(
        self,
        lhs: ArrayLike,
        rhs: ArrayLike,
        *,
        op_name: str = "sram_dynamic_matmul",
    ) -> NDArray[np.generic]:
        """Compute a matrix product whose two operands are supplied at runtime."""

        return self._matmul(lhs, rhs, op_name=op_name, mode="dynamic")

    def matmul_tensor(
        self,
        lhs: TensorPacket,
        weight: WeightHandle,
        *,
        output_name: str,
        op_name: str = "sram_resident_matmul",
    ) -> TensorPacket:
        """System-level resident-weight multiplication with location checking."""

        if lhs.location != self.name:
            raise TensorLocationError(
                f"tensor {lhs.name!r} is at {lhs.location!r}; transfer it to {self.name!r}"
            )
        output = self.matmul(lhs.data, weight, op_name=op_name)
        return TensorPacket(output_name, output, self.output_bit_width, self.name)

    def matmul_dynamic_tensor(
        self,
        lhs: TensorPacket,
        rhs: TensorPacket,
        *,
        output_name: str,
        op_name: str = "sram_dynamic_matmul",
    ) -> TensorPacket:
        """System-level dynamic multiplication with location checking."""

        for tensor in (lhs, rhs):
            if tensor.location != self.name:
                raise TensorLocationError(
                    f"tensor {tensor.name!r} is at {tensor.location!r}; "
                    f"transfer it to {self.name!r}"
                )
        output = self.matmul_dynamic(lhs.data, rhs.data, op_name=op_name)
        return TensorPacket(output_name, output, self.output_bit_width, self.name)

