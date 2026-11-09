"""RRAM-CIM unit with explicitly programmed static weights."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..errors import TensorLocationError
from ..tensor import TensorPacket
from .base import ResidentMatmulUnit, WeightHandle


class RRAMCIM(ResidentMatmulUnit):
    """Matrix engine whose right-hand operand must be programmed in advance."""

    storage_action = "program"

    def program_weights(self, name: str, weights: ArrayLike) -> WeightHandle:
        """Program a static matrix and return its versioned resident handle."""

        return self._store_weight(name, weights)

    def matmul(
        self,
        lhs: ArrayLike,
        weight: WeightHandle,
        *,
        op_name: str = "rram_matmul",
    ) -> NDArray[np.generic]:
        """Compute ``lhs @ resident_weight`` and record one compute event."""

        rhs = self._resolve_weight(weight)
        return self._matmul(lhs, rhs, op_name=op_name, mode="resident_static")

    def matmul_tensor(
        self,
        lhs: TensorPacket,
        weight: WeightHandle,
        *,
        output_name: str,
        op_name: str = "rram_matmul",
    ) -> TensorPacket:
        """System-level matrix multiplication with tensor residency checking."""

        if lhs.location != self.name:
            raise TensorLocationError(
                f"tensor {lhs.name!r} is at {lhs.location!r}; transfer it to {self.name!r}"
            )
        output = self.matmul(lhs.data, weight, op_name=op_name)
        return TensorPacket(output_name, output, self.output_bit_width, self.name)

