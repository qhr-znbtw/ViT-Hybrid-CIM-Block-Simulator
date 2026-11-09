"""Shared matrix-multiplication behavior for CIM units."""

from __future__ import annotations

from dataclasses import dataclass
from math import prod
from typing import Final

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..errors import MatrixShapeError, StaleWeightHandleError, UnknownWeightError
from ..events import ComputeEvent, StatsCollector, StorageEvent


@dataclass(frozen=True, slots=True)
class WeightHandle:
    """Stable reference to one version of a weight resident on a CIM unit."""

    unit: str
    name: str
    version: int
    shape: tuple[int, ...]


class ResidentMatmulUnit:
    """Base implementation for units with resident matrix weights."""

    storage_action: Final[str] = "store"

    def __init__(
        self,
        name: str,
        stats: StatsCollector,
        *,
        input_bit_width: int = 8,
        weight_bit_width: int = 8,
        output_bit_width: int = 32,
    ) -> None:
        for label, value in (
            ("input_bit_width", input_bit_width),
            ("weight_bit_width", weight_bit_width),
            ("output_bit_width", output_bit_width),
        ):
            if value <= 0:
                raise ValueError(f"{label} must be positive")

        self.name = name
        self.stats = stats
        self.input_bit_width = input_bit_width
        self.weight_bit_width = weight_bit_width
        self.output_bit_width = output_bit_width
        self._weights: dict[str, NDArray[np.generic]] = {}
        self._versions: dict[str, int] = {}

    @staticmethod
    def _as_operand(value: ArrayLike, label: str) -> NDArray[np.generic]:
        array = np.asarray(value)
        if array.ndim < 2:
            raise MatrixShapeError(f"{label} must have at least two dimensions; got {array.shape}")
        if not np.issubdtype(array.dtype, np.number):
            raise TypeError(f"{label} must contain numeric data; got {array.dtype}")
        return array

    @staticmethod
    def _matmul_metadata(
        lhs: NDArray[np.generic], rhs: NDArray[np.generic]
    ) -> tuple[tuple[int, ...], int]:
        if lhs.shape[-1] != rhs.shape[-2]:
            raise MatrixShapeError(
                f"incompatible matmul shapes: {lhs.shape} and {rhs.shape}; "
                f"{lhs.shape[-1]} != {rhs.shape[-2]}"
            )
        try:
            batch_shape = np.broadcast_shapes(lhs.shape[:-2], rhs.shape[:-2])
        except ValueError as exc:
            raise MatrixShapeError(
                f"batch dimensions cannot broadcast: {lhs.shape[:-2]} and {rhs.shape[:-2]}"
            ) from exc

        m, k, n = lhs.shape[-2], lhs.shape[-1], rhs.shape[-1]
        output_shape = tuple(batch_shape) + (m, n)
        macs = prod(batch_shape) * m * k * n
        return output_shape, macs

    def _store_weight(self, name: str, value: ArrayLike) -> WeightHandle:
        if not name or not name.strip():
            raise ValueError("weight name must be non-empty")
        weight = self._as_operand(value, "weight")
        if weight.ndim != 2:
            raise MatrixShapeError(
                f"resident weight must be a two-dimensional matrix; got {weight.shape}"
            )

        version = self._versions.get(name, 0) + 1
        stored = np.array(weight, copy=True)
        self._weights[name] = stored
        self._versions[name] = version
        byte_count = (stored.size * self.weight_bit_width + 7) // 8
        self.stats.record_storage(
            StorageEvent(
                unit=self.name,
                action=self.storage_action,
                weight_name=name,
                version=version,
                shape=tuple(stored.shape),
                elements=int(stored.size),
                bit_width=self.weight_bit_width,
                bytes=byte_count,
            )
        )
        return WeightHandle(self.name, name, version, tuple(stored.shape))

    def _resolve_weight(self, handle: WeightHandle) -> NDArray[np.generic]:
        if handle.unit != self.name:
            raise UnknownWeightError(
                f"weight {handle.name!r} belongs to {handle.unit!r}, not {self.name!r}"
            )
        if handle.name not in self._weights:
            raise UnknownWeightError(f"weight {handle.name!r} is not resident on {self.name}")
        current_version = self._versions[handle.name]
        if handle.version != current_version:
            raise StaleWeightHandleError(
                f"weight {handle.name!r} handle version {handle.version} is stale; "
                f"current version is {current_version}"
            )
        return self._weights[handle.name]

    def _matmul(
        self,
        lhs_value: ArrayLike,
        rhs_value: ArrayLike,
        *,
        op_name: str,
        mode: str,
    ) -> NDArray[np.generic]:
        lhs = self._as_operand(lhs_value, "lhs")
        rhs = self._as_operand(rhs_value, "rhs")
        output_shape, macs = self._matmul_metadata(lhs, rhs)
        output = np.matmul(lhs, rhs)
        self.stats.record_compute(
            ComputeEvent(
                unit=self.name,
                op_name=op_name,
                mode=mode,
                lhs_shape=tuple(lhs.shape),
                rhs_shape=tuple(rhs.shape),
                output_shape=output_shape,
                macs=macs,
                input_bit_width=self.input_bit_width,
                weight_bit_width=self.weight_bit_width,
                output_bit_width=self.output_bit_width,
            )
        )
        return output

