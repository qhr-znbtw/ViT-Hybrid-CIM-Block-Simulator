"""Aggregate raw simulator events into experiment-level statistics."""

from __future__ import annotations

import csv
import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .reporting import SimulationReport


def _share(value: int | float, total: int | float) -> float:
    return 0.0 if total == 0 else float(value) / float(total)


def _stage(name: str) -> str:
    if ".attention." in name:
        return "attention"
    if ".feed_forward." in name:
        return "feed_forward"
    return "block_overhead"


def _compute_role(name: str) -> str:
    if ".backbone_matmul" in name:
        return "frozen_backbone"
    if ".lora_a_matmul" in name or ".lora_b_matmul" in name:
        return "lora"
    if name.endswith(".qk_t") or name.endswith(".probabilities_v"):
        return "dynamic_attention"
    return "other"


COMPONENT_ORDER = (
    "block_input",
    "layer_norm_1",
    "q_projection",
    "k_projection",
    "v_projection",
    "head_preparation",
    "qk_t",
    "scale_softmax",
    "pv",
    "head_merge",
    "attention_output_projection",
    "attention_residual",
    "layer_norm_2",
    "ffn_up_projection",
    "gelu",
    "ffn_down_projection",
    "ffn_residual",
)

COMPONENT_LABELS_ZH = {
    "block_input": "输入进入Digital",
    "layer_norm_1": "LayerNorm 1",
    "q_projection": "Q投影",
    "k_projection": "K投影",
    "v_projection": "V投影",
    "head_preparation": "分头与转置",
    "qk_t": "QK^T",
    "scale_softmax": "缩放与Softmax",
    "pv": "PV",
    "head_merge": "多头合并",
    "attention_output_projection": "Attention输出投影",
    "attention_residual": "Attention残差",
    "layer_norm_2": "LayerNorm 2",
    "ffn_up_projection": "FFN升维",
    "gelu": "GELU",
    "ffn_down_projection": "FFN降维",
    "ffn_residual": "FFN残差",
    "unclassified": "未分类",
}


def _component(name: str) -> str:
    if name.endswith(".input_to_digital"):
        return "block_input"
    if name.endswith(".layer_norm1"):
        return "layer_norm_1"
    if ".attention.q_proj." in name:
        return "q_projection"
    if ".attention.k_proj." in name:
        return "k_projection"
    if ".attention.v_proj." in name:
        return "v_projection"
    if any(
        token in name
        for token in (
            ".attention.reshape_q",
            ".attention.reshape_k",
            ".attention.reshape_v",
            ".attention.transpose_q",
            ".attention.transpose_k",
            ".attention.transpose_v",
            ".attention.transpose_k_t",
        )
    ):
        return "head_preparation"
    if any(
        token in name
        for token in (
            ".attention.qk_t",
            ".attention.q_to_sram",
            ".attention.k_t_to_sram",
            ".attention.scores_to_digital",
        )
    ):
        return "qk_t"
    if name.endswith(".attention.scale_scores") or name.endswith(".attention.softmax"):
        return "scale_softmax"
    if any(
        token in name
        for token in (
            ".attention.probabilities_v",
            ".attention.probabilities_to_sram",
            ".attention.v_to_sram",
            ".attention.context_to_digital",
        )
    ):
        return "pv"
    if name.endswith(".attention.transpose_context") or name.endswith(".attention.merge_heads"):
        return "head_merge"
    if ".attention.out_proj." in name:
        return "attention_output_projection"
    if name.endswith(".attention_residual_add"):
        return "attention_residual"
    if name.endswith(".layer_norm2"):
        return "layer_norm_2"
    if ".feed_forward.up_proj." in name:
        return "ffn_up_projection"
    if name.endswith(".feed_forward.gelu"):
        return "gelu"
    if ".feed_forward.down_proj." in name:
        return "ffn_down_projection"
    if name.endswith(".feed_forward_residual_add"):
        return "ffn_residual"
    return "unclassified"


def _component_stage(component: str) -> str:
    if component in {
        "q_projection",
        "k_projection",
        "v_projection",
        "head_preparation",
        "qk_t",
        "scale_softmax",
        "pv",
        "head_merge",
        "attention_output_projection",
    }:
        return "attention"
    if component in {"ffn_up_projection", "gelu", "ffn_down_projection"}:
        return "feed_forward"
    return "block_overhead"


@dataclass(frozen=True, slots=True)
class ExperimentAnalysis:
    data: dict[str, Any]

    def __init__(self, data: dict[str, Any]) -> None:
        object.__setattr__(self, "data", deepcopy(data))

    def summary(self) -> dict[str, Any]:
        return deepcopy(self.data)

    def export_json(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(self.data, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return target

    @staticmethod
    def _write_csv(path: Path, rows: list[dict[str, Any]]) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = sorted({key for row in rows for key in row})
        with path.open("w", encoding="utf-8-sig", newline="") as file:
            if not fieldnames:
                return path
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        return path

    def export_markdown(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        overview = self.data["overview"]
        lines = [
            "# Experiment Analysis",
            "",
            "## Overview",
            "",
            "| Metric | Value |",
            "|---|---:|",
        ]
        for key, value in overview.items():
            lines.append(f"| {key} | {value} |")

        def table(title: str, rows: list[dict[str, Any]], columns: list[str]) -> None:
            lines.extend(["", f"## {title}", "", "| " + " | ".join(columns) + " |"])
            lines.append("|" + "|".join("---:" for _ in columns) + "|")
            for row in rows:
                lines.append("| " + " | ".join(str(row.get(column, "")) for column in columns) + " |")

        table("Compute by Unit", self.data["compute_by_unit"], ["unit", "macs", "share", "matmul_operations"])
        table("Compute by Stage", self.data["compute_by_stage"], ["stage", "macs", "share"])
        table("Compute by Role", self.data["compute_by_role"], ["role", "macs", "share"])
        table("Traffic by Path", self.data["traffic_by_path"], ["path", "transfers", "bytes", "share"])
        table("Traffic by Stage", self.data["traffic_by_stage"], ["stage", "transfers", "bytes", "share"])
        table(
            "Encoder Block Component Breakdown",
            self.data["component_breakdown"],
            [
                "component_zh",
                "macs",
                "rram_macs",
                "rram_percent",
                "sram_macs",
                "sram_percent",
                "mac_share",
                "transfer_events",
                "transferred_bytes",
                "traffic_share",
            ],
        )

        largest = self.data.get("largest_transfer")
        if largest:
            lines.extend(
                [
                    "",
                    "## Largest Transfer",
                    "",
                    f"- Tensor: `{largest['tensor_name']}`",
                    f"- Path: `{largest['source']} -> {largest['destination']}`",
                    f"- Shape: `{largest['shape']}`",
                    f"- Bytes: {largest['bytes']}",
                ]
            )
        target.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return target

    def export_component_hardware_markdown(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        rows = self.data["component_breakdown"]
        lines = [
            "# Encoder Block逐部件硬件统计",
            "",
            "## RRAM/SRAM计算量",
            "",
            "| 部件 | 总MAC | RRAM MAC | RRAM占部件计算 | SRAM MAC | SRAM占部件计算 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for row in rows:
            rram_percent = "—" if row["macs"] == 0 else f"{row['rram_percent']}%"
            sram_percent = "—" if row["macs"] == 0 else f"{row['sram_percent']}%"
            lines.append(
                f"| {row['component_zh']} | {row['macs']} | {row['rram_macs']} | "
                f"{rram_percent} | {row['sram_macs']} | {sram_percent} |"
            )

        lines.extend(
            [
                "",
                "## RRAM方向的数据传输",
                "",
                "| 部件 | Digital→RRAM 次数 | Digital→RRAM 字节 | RRAM→Digital 次数 | RRAM→Digital 字节 | RRAM总次数 | RRAM总字节 |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in rows:
            lines.append(
                f"| {row['component_zh']} | {row['digital_to_rram_events']} | "
                f"{row['digital_to_rram_bytes']} | {row['rram_to_digital_events']} | "
                f"{row['rram_to_digital_bytes']} | {row['rram_transfer_events']} | "
                f"{row['rram_transfer_bytes']} |"
            )

        lines.extend(
            [
                "",
                "## SRAM方向的数据传输",
                "",
                "| 部件 | Digital→SRAM 次数 | Digital→SRAM 字节 | SRAM→Digital 次数 | SRAM→Digital 字节 | SRAM总次数 | SRAM总字节 |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in rows:
            lines.append(
                f"| {row['component_zh']} | {row['digital_to_sram_events']} | "
                f"{row['digital_to_sram_bytes']} | {row['sram_to_digital_events']} | "
                f"{row['sram_to_digital_bytes']} | {row['sram_transfer_events']} | "
                f"{row['sram_transfer_bytes']} |"
            )

        lines.extend(
            [
                "",
                "## 部件通信总量",
                "",
                "| 部件 | 总传输次数 | 总传输字节 | 全局通信占比 |",
                "|---|---:|---:|---:|",
            ]
        )
        for row in rows:
            lines.append(
                f"| {row['component_zh']} | {row['transfer_events']} | "
                f"{row['transferred_bytes']} | {row['traffic_percent']}% |"
            )
        target.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return target

    def export(self, directory: str | Path) -> dict[str, Path]:
        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        outputs = {
            "json": self.export_json(target / "analysis.json"),
            "markdown": self.export_markdown(target / "analysis.md"),
            "component_hardware_markdown": self.export_component_hardware_markdown(
                target / "component_hardware_breakdown.md"
            ),
        }
        for key in (
            "compute_by_unit",
            "compute_by_stage",
            "compute_by_role",
            "traffic_by_path",
            "traffic_by_stage",
            "component_breakdown",
        ):
            outputs[key] = self._write_csv(target / f"{key}.csv", self.data[key])
        outputs["component_hardware_csv"] = self._write_csv(
            target / "component_hardware_breakdown.csv",
            self.data["component_breakdown"],
        )
        return outputs


def analyze_execution_report(report: SimulationReport) -> ExperimentAnalysis:
    """Convert a raw execution report into comparison-ready statistics."""

    raw = report.summary()
    total_macs = int(raw["totals"]["macs"])
    total_bytes = int(raw["totals"]["transferred_bytes"])

    compute_by_unit = []
    for unit, values in raw.get("by_unit", {}).items():
        macs = int(values["macs"])
        if macs == 0 and values["matmul_operations"] == 0:
            continue
        compute_by_unit.append(
            {
                "unit": unit,
                "macs": macs,
                "share": round(_share(macs, total_macs), 8),
                "matmul_operations": int(values["matmul_operations"]),
            }
        )

    stage_macs: dict[str, int] = {}
    role_macs: dict[str, int] = {}
    for event in raw.get("compute_events", []):
        event_macs = int(event["macs"])
        event_name = event["op_name"]
        stage = _stage(event_name)
        role = _compute_role(event_name)
        stage_macs[stage] = stage_macs.get(stage, 0) + event_macs
        role_macs[role] = role_macs.get(role, 0) + event_macs

    compute_by_stage = [
        {"stage": key, "macs": value, "share": round(_share(value, total_macs), 8)}
        for key, value in sorted(stage_macs.items())
    ]
    compute_by_role = [
        {"role": key, "macs": value, "share": round(_share(value, total_macs), 8)}
        for key, value in sorted(role_macs.items())
    ]

    traffic_by_path = [
        {
            "path": path,
            "transfers": int(values["transfers"]),
            "bytes": int(values["bytes"]),
            "share": round(_share(values["bytes"], total_bytes), 8),
        }
        for path, values in sorted(raw.get("transfers_by_path", {}).items())
    ]

    stage_traffic: dict[str, dict[str, int]] = {}
    transfer_events = raw.get("transfer_events", [])
    for event in transfer_events:
        stage = _stage(event["transfer_name"])
        entry = stage_traffic.setdefault(stage, {"transfers": 0, "bytes": 0})
        entry["transfers"] += 1
        entry["bytes"] += int(event["bytes"])
    traffic_by_stage = [
        {
            "stage": stage,
            "transfers": values["transfers"],
            "bytes": values["bytes"],
            "share": round(_share(values["bytes"], total_bytes), 8),
        }
        for stage, values in sorted(stage_traffic.items())
    ]

    largest_transfer = None
    if transfer_events:
        largest_transfer = deepcopy(max(transfer_events, key=lambda event: event["bytes"]))
        largest_transfer["shape"] = list(largest_transfer["shape"])

    component_data: dict[str, dict[str, Any]] = {
        component: {
            "component": component,
            "stage": _component_stage(component),
            "hardware_set": set(),
            "macs": 0,
            "matmul_operations": 0,
            "digital_operations": 0,
            "digital_primitives": {},
            "transfer_events": 0,
            "transferred_bytes": 0,
            "transfer_paths_set": set(),
            "tensor_shapes_set": set(),
            "macs_by_unit": {},
            "transfers_by_path": {},
        }
        for component in COMPONENT_ORDER
    }
    component_data["unclassified"] = {
        "component": "unclassified",
        "stage": "unclassified",
        "hardware_set": set(),
        "macs": 0,
        "matmul_operations": 0,
        "digital_operations": 0,
        "digital_primitives": {},
        "transfer_events": 0,
        "transferred_bytes": 0,
        "transfer_paths_set": set(),
        "tensor_shapes_set": set(),
        "macs_by_unit": {},
        "transfers_by_path": {},
    }

    for event in raw.get("compute_events", []):
        entry = component_data[_component(event["op_name"])]
        entry["hardware_set"].add(event["unit"])
        entry["macs"] += int(event["macs"])
        entry["macs_by_unit"][event["unit"]] = (
            entry["macs_by_unit"].get(event["unit"], 0) + int(event["macs"])
        )
        entry["matmul_operations"] += 1
        for shape in (event["lhs_shape"], event["rhs_shape"], event["output_shape"]):
            entry["tensor_shapes_set"].add(tuple(shape))

    for event in raw.get("digital_events", []):
        entry = component_data[_component(event["op_name"])]
        entry["hardware_set"].add(event["unit"])
        entry["digital_operations"] += 1
        for primitive, count in event["operation_counts"].items():
            entry["digital_primitives"][primitive] = (
                entry["digital_primitives"].get(primitive, 0) + int(count)
            )
        for shape in event["input_shapes"]:
            entry["tensor_shapes_set"].add(tuple(shape))
        entry["tensor_shapes_set"].add(tuple(event["output_shape"]))

    for event in transfer_events:
        entry = component_data[_component(event["transfer_name"])]
        entry["hardware_set"].add(event["source"])
        entry["hardware_set"].add(event["destination"])
        entry["transfer_events"] += 1
        entry["transferred_bytes"] += int(event["bytes"])
        path = f"{event['source']}->{event['destination']}"
        entry["transfer_paths_set"].add(path)
        path_entry = entry["transfers_by_path"].setdefault(
            path, {"events": 0, "bytes": 0}
        )
        path_entry["events"] += 1
        path_entry["bytes"] += int(event["bytes"])
        entry["tensor_shapes_set"].add(tuple(event["shape"]))

    component_breakdown = []
    for component in (*COMPONENT_ORDER, "unclassified"):
        entry = component_data[component]
        rram_macs = int(entry["macs_by_unit"].get("rram_cim", 0))
        sram_macs = int(entry["macs_by_unit"].get("sram_cim", 0))
        digital_to_rram = entry["transfers_by_path"].get(
            "digital_processor->rram_cim", {"events": 0, "bytes": 0}
        )
        rram_to_digital = entry["transfers_by_path"].get(
            "rram_cim->digital_processor", {"events": 0, "bytes": 0}
        )
        digital_to_sram = entry["transfers_by_path"].get(
            "digital_processor->sram_cim", {"events": 0, "bytes": 0}
        )
        sram_to_digital = entry["transfers_by_path"].get(
            "sram_cim->digital_processor", {"events": 0, "bytes": 0}
        )
        rram_transfer_events = digital_to_rram["events"] + rram_to_digital["events"]
        rram_transfer_bytes = digital_to_rram["bytes"] + rram_to_digital["bytes"]
        sram_transfer_events = digital_to_sram["events"] + sram_to_digital["events"]
        sram_transfer_bytes = digital_to_sram["bytes"] + sram_to_digital["bytes"]
        component_breakdown.append(
            {
                "component": component,
                "component_zh": COMPONENT_LABELS_ZH[component],
                "stage": entry["stage"],
                "hardware": "+".join(sorted(entry["hardware_set"])) or "none",
                "macs": entry["macs"],
                "mac_share": round(_share(entry["macs"], total_macs), 8),
                "rram_macs": rram_macs,
                "rram_percent": round(100.0 * _share(rram_macs, entry["macs"]), 4),
                "sram_macs": sram_macs,
                "sram_percent": round(100.0 * _share(sram_macs, entry["macs"]), 4),
                "matmul_operations": entry["matmul_operations"],
                "digital_operations": entry["digital_operations"],
                "digital_primitives": json.dumps(
                    entry["digital_primitives"], ensure_ascii=False, sort_keys=True
                ),
                "transfer_events": entry["transfer_events"],
                "transferred_bytes": entry["transferred_bytes"],
                "traffic_share": round(_share(entry["transferred_bytes"], total_bytes), 8),
                "traffic_percent": round(
                    100.0 * _share(entry["transferred_bytes"], total_bytes), 4
                ),
                "digital_to_rram_events": digital_to_rram["events"],
                "digital_to_rram_bytes": digital_to_rram["bytes"],
                "rram_to_digital_events": rram_to_digital["events"],
                "rram_to_digital_bytes": rram_to_digital["bytes"],
                "rram_transfer_events": rram_transfer_events,
                "rram_transfer_bytes": rram_transfer_bytes,
                "digital_to_sram_events": digital_to_sram["events"],
                "digital_to_sram_bytes": digital_to_sram["bytes"],
                "sram_to_digital_events": sram_to_digital["events"],
                "sram_to_digital_bytes": sram_to_digital["bytes"],
                "sram_transfer_events": sram_transfer_events,
                "sram_transfer_bytes": sram_transfer_bytes,
                "transfer_paths": ";".join(sorted(entry["transfer_paths_set"])),
                "tensor_shapes": json.dumps(
                    [list(shape) for shape in sorted(entry["tensor_shapes_set"])],
                    ensure_ascii=False,
                ),
            }
        )

    return ExperimentAnalysis(
        {
            "overview": {
                "matmul_operations": int(raw["totals"]["matmul_operations"]),
                "total_macs": total_macs,
                "digital_operations": int(raw["totals"]["digital_operations"]),
                "data_transfers": int(raw["totals"]["data_transfers"]),
                "transferred_bytes": total_bytes,
                "arithmetic_intensity_mac_per_byte": round(_share(total_macs, total_bytes), 8),
                "communication_bytes_per_mac": round(_share(total_bytes, total_macs), 8),
            },
            "compute_by_unit": compute_by_unit,
            "compute_by_stage": compute_by_stage,
            "compute_by_role": compute_by_role,
            "traffic_by_path": traffic_by_path,
            "traffic_by_stage": traffic_by_stage,
            "component_breakdown": component_breakdown,
            "largest_transfer": largest_transfer,
        }
    )
