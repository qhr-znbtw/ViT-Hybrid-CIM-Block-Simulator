"""Event records and aggregation for operator-level simulation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class ComputeEvent:
    """One matrix-multiplication operation completed by a CIM unit."""

    unit: str
    op_name: str
    mode: str
    lhs_shape: tuple[int, ...]
    rhs_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    macs: int
    input_bit_width: int
    weight_bit_width: int
    output_bit_width: int


@dataclass(frozen=True, slots=True)
class StorageEvent:
    """One resident-weight programming or loading operation."""

    unit: str
    action: str
    weight_name: str
    version: int
    shape: tuple[int, ...]
    elements: int
    bit_width: int
    bytes: int


@dataclass(frozen=True, slots=True)
class TransferEvent:
    """One logical tensor transfer between two hardware locations."""

    transfer_name: str
    tensor_name: str
    source: str
    destination: str
    shape: tuple[int, ...]
    elements: int
    bit_width: int
    bytes: int


@dataclass(frozen=True, slots=True)
class DigitalComputeEvent:
    """One non-CIM operation executed by the digital processor."""

    unit: str
    op_name: str
    op_type: str
    input_shapes: tuple[tuple[int, ...], ...]
    output_shape: tuple[int, ...]
    output_elements: int
    operation_counts: dict[str, int]


class StatsCollector:
    """Collect compute and resident-weight events from all simulated units."""

    def __init__(self) -> None:
        self._compute_events: list[ComputeEvent] = []
        self._storage_events: list[StorageEvent] = []
        self._transfer_events: list[TransferEvent] = []
        self._digital_events: list[DigitalComputeEvent] = []

    @property
    def compute_events(self) -> tuple[ComputeEvent, ...]:
        return tuple(self._compute_events)

    @property
    def storage_events(self) -> tuple[StorageEvent, ...]:
        return tuple(self._storage_events)

    @property
    def transfer_events(self) -> tuple[TransferEvent, ...]:
        return tuple(self._transfer_events)

    @property
    def digital_events(self) -> tuple[DigitalComputeEvent, ...]:
        return tuple(self._digital_events)

    def record_compute(self, event: ComputeEvent) -> None:
        self._compute_events.append(event)

    def record_storage(self, event: StorageEvent) -> None:
        self._storage_events.append(event)

    def record_transfer(self, event: TransferEvent) -> None:
        self._transfer_events.append(event)

    def record_digital(self, event: DigitalComputeEvent) -> None:
        self._digital_events.append(event)

    def reset(self) -> None:
        self._compute_events.clear()
        self._storage_events.clear()
        self._transfer_events.clear()
        self._digital_events.clear()

    def summary(self) -> dict[str, Any]:
        by_unit: dict[str, dict[str, Any]] = {}

        def unit_entry(unit_name: str) -> dict[str, Any]:
            return by_unit.setdefault(
                unit_name,
                {
                    "matmul_operations": 0,
                    "macs": 0,
                    "weight_events": 0,
                    "weight_bytes": 0,
                    "digital_operations": 0,
                    "digital_output_elements": 0,
                    "digital_primitives": {},
                },
            )

        for event in self._compute_events:
            unit = unit_entry(event.unit)
            unit["matmul_operations"] += 1
            unit["macs"] += event.macs

        for event in self._storage_events:
            unit = unit_entry(event.unit)
            unit["weight_events"] += 1
            unit["weight_bytes"] += event.bytes

        total_digital_primitives: dict[str, int] = {}
        for event in self._digital_events:
            unit = unit_entry(event.unit)
            unit["digital_operations"] += 1
            unit["digital_output_elements"] += event.output_elements
            for primitive, count in event.operation_counts.items():
                unit["digital_primitives"][primitive] = (
                    unit["digital_primitives"].get(primitive, 0) + count
                )
                total_digital_primitives[primitive] = (
                    total_digital_primitives.get(primitive, 0) + count
                )

        transfers_by_path: dict[str, dict[str, int]] = {}
        for event in self._transfer_events:
            path = f"{event.source}->{event.destination}"
            entry = transfers_by_path.setdefault(path, {"transfers": 0, "bytes": 0})
            entry["transfers"] += 1
            entry["bytes"] += event.bytes

        return {
            "totals": {
                "matmul_operations": len(self._compute_events),
                "macs": sum(event.macs for event in self._compute_events),
                "weight_events": len(self._storage_events),
                "weight_bytes": sum(event.bytes for event in self._storage_events),
                "digital_operations": len(self._digital_events),
                "digital_primitives": total_digital_primitives,
                "data_transfers": len(self._transfer_events),
                "transferred_bytes": sum(event.bytes for event in self._transfer_events),
            },
            "by_unit": by_unit,
            "transfers_by_path": transfers_by_path,
            "compute_events": [asdict(event) for event in self._compute_events],
            "storage_events": [asdict(event) for event in self._storage_events],
            "digital_events": [asdict(event) for event in self._digital_events],
            "transfer_events": [asdict(event) for event in self._transfer_events],
        }

