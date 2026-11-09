"""Tensor metadata used to model logical hardware residency."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True, slots=True)
class TensorPacket:
    """Numerical tensor plus the metadata needed for data-path accounting."""

    name: str
    data: NDArray[np.generic]
    bit_width: int
    location: str

    def __init__(
        self,
        name: str,
        data: ArrayLike,
        bit_width: int,
        location: str,
    ) -> None:
        array = np.asarray(data)
        if not name or not name.strip():
            raise ValueError("tensor name must be non-empty")
        if not location or not location.strip():
            raise ValueError("tensor location must be non-empty")
        if bit_width <= 0:
            raise ValueError("tensor bit_width must be positive")
        if not np.issubdtype(array.dtype, np.number):
            raise TypeError(f"tensor must contain numeric data; got {array.dtype}")

        object.__setattr__(self, "name", name)
        object.__setattr__(self, "data", array)
        object.__setattr__(self, "bit_width", bit_width)
        object.__setattr__(self, "location", location)

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self.data.shape)

    @property
    def elements(self) -> int:
        return int(self.data.size)

    @property
    def nbytes(self) -> int:
        return (self.elements * self.bit_width + 7) // 8

    def at(self, location: str, *, name: str | None = None) -> "TensorPacket":
        """Return a logical view at another location without copying host memory."""

        return TensorPacket(name or self.name, self.data, self.bit_width, location)

