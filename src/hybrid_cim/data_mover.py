"""Explicit logical data transfers for system-level accounting."""

from __future__ import annotations

from collections.abc import Iterable

from .events import StatsCollector, TransferEvent
from .tensor import TensorPacket


class DataMover:
    """Move tensor packets between modeled locations and record traffic."""

    def __init__(self, stats: StatsCollector) -> None:
        self.stats = stats

    def transfer(
        self,
        tensor: TensorPacket,
        destination: str,
        *,
        transfer_name: str | None = None,
        output_name: str | None = None,
    ) -> TensorPacket:
        if not destination or not destination.strip():
            raise ValueError("destination must be non-empty")
        if destination == tensor.location:
            return tensor if output_name is None else tensor.at(destination, name=output_name)

        event_name = transfer_name or f"{tensor.location}_to_{destination}:{tensor.name}"
        self.stats.record_transfer(
            TransferEvent(
                transfer_name=event_name,
                tensor_name=tensor.name,
                source=tensor.location,
                destination=destination,
                shape=tensor.shape,
                elements=tensor.elements,
                bit_width=tensor.bit_width,
                bytes=tensor.nbytes,
            )
        )
        return tensor.at(destination, name=output_name)

    def broadcast(
        self,
        tensor: TensorPacket,
        destinations: Iterable[str],
        *,
        transfer_name_prefix: str | None = None,
    ) -> tuple[TensorPacket, ...]:
        outputs = []
        for destination in destinations:
            transfer_name = None
            if transfer_name_prefix is not None:
                transfer_name = f"{transfer_name_prefix}:{destination}"
            outputs.append(
                self.transfer(tensor, destination, transfer_name=transfer_name)
            )
        return tuple(outputs)

