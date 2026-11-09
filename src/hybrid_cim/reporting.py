"""Structured simulation results and report exporters."""

from __future__ import annotations

import csv
import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .tensor import TensorPacket


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False)
    return value


@dataclass(frozen=True, slots=True)
class SimulationReport:
    phase: str
    data: dict[str, Any]

    def __init__(self, phase: str, data: dict[str, Any]) -> None:
        object.__setattr__(self, "phase", phase)
        object.__setattr__(self, "data", deepcopy(data))

    def summary(self) -> dict[str, Any]:
        return deepcopy(self.data)

    @property
    def totals(self) -> dict[str, Any]:
        return deepcopy(self.data["totals"])

    def export_json(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(
                {"phase": self.phase, **self.data},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return target

    @staticmethod
    def _export_rows(rows: list[dict[str, Any]], path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = sorted({key for row in rows for key in row})
        with path.open("w", encoding="utf-8-sig", newline="") as file:
            if not fieldnames:
                return path
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({key: _csv_value(value) for key, value in row.items()})
        return path

    def export_csv(self, directory: str | Path, *, stem: str) -> dict[str, Path]:
        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        outputs = {}
        for key in (
            "compute_events",
            "digital_events",
            "transfer_events",
            "storage_events",
        ):
            outputs[key] = self._export_rows(
                self.data.get(key, []), target / f"{stem}_{key}.csv"
            )
        return outputs

    def export_markdown(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        lines = [f"# Simulation Report: {self.phase}", "", "## Totals", ""]
        lines.extend(["| Metric | Value |", "|---|---:|"])
        for key, value in self.data["totals"].items():
            rendered = json.dumps(value, ensure_ascii=False) if isinstance(value, dict) else value
            lines.append(f"| {key} | {rendered} |")

        lines.extend(["", "## Compute by Unit", "", "| Unit | MatMuls | MACs | Digital Ops | Weight Bytes |", "|---|---:|---:|---:|---:|"])
        for unit, values in self.data.get("by_unit", {}).items():
            lines.append(
                f"| {unit} | {values['matmul_operations']} | {values['macs']} | "
                f"{values['digital_operations']} | {values['weight_bytes']} |"
            )

        lines.extend(["", "## Transfers by Path", "", "| Path | Transfers | Bytes |", "|---|---:|---:|"])
        for path_name, values in self.data.get("transfers_by_path", {}).items():
            lines.append(f"| {path_name} | {values['transfers']} | {values['bytes']} |")
        target.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return target

    def export(self, directory: str | Path, *, stem: str | None = None) -> dict[str, Path]:
        target = Path(directory)
        report_stem = stem or self.phase
        outputs = {
            "json": self.export_json(target / f"{report_stem}.json"),
            "markdown": self.export_markdown(target / f"{report_stem}.md"),
        }
        outputs.update(self.export_csv(target, stem=report_stem))
        return outputs


@dataclass(frozen=True, slots=True)
class SimulationResult:
    output: np.ndarray
    output_packet: TensorPacket
    report: SimulationReport

