"""Top-level prepare/run/report interface for one Transformer block."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from .config import SimulatorConfig
from .errors import ComponentNotPreparedError
from .models import TransformerEncoderBlock
from .reporting import SimulationReport, SimulationResult
from .system import HybridCIMSystem
from .tensor import TensorPacket
from .weights import TransformerBlockWeights


class HybridCIMBlockSimulator:
    """Public facade for preparing and running one simulated Encoder Block."""

    def __init__(self, config: SimulatorConfig | None = None) -> None:
        self.config = config or SimulatorConfig()
        precision = self.config.precision
        self.hardware = HybridCIMSystem(
            input_bit_width=precision.activation_bits,
            weight_bit_width=precision.weight_bits,
            output_bit_width=precision.accumulator_bits,
        )
        self.block = TransformerEncoderBlock(
            "transformer_block", self.hardware, self.config.model
        )
        self._prepared = False
        self._preparation_report: SimulationReport | None = None
        self._last_result: SimulationResult | None = None

    @staticmethod
    def _validate_linear_mapping(
        component_name: str,
        mapping: str,
        has_lora: bool,
    ) -> None:
        if mapping == "hybrid" and not has_lora:
            raise ValueError(f"{component_name} is mapped as hybrid but has no LoRA weights")
        if mapping == "rram" and has_lora:
            raise ValueError(f"{component_name} is mapped as rram-only but contains LoRA weights")

    def _validate_mapping(self, weights: TransformerBlockWeights) -> None:
        mapping = self.config.mapping
        attention = weights.attention
        feed_forward = weights.feed_forward
        for name, strategy, linear in (
            ("q_proj", mapping.q_proj, attention.q_proj),
            ("k_proj", mapping.k_proj, attention.k_proj),
            ("v_proj", mapping.v_proj, attention.v_proj),
            ("out_proj", mapping.out_proj, attention.out_proj),
            ("ff_up", mapping.ff_up, feed_forward.up_proj),
            ("ff_down", mapping.ff_down, feed_forward.down_proj),
        ):
            self._validate_linear_mapping(name, strategy, linear.has_lora)

    def prepare(self, weights: TransformerBlockWeights) -> SimulationReport:
        self.hardware.reset_stats()
        self._validate_mapping(weights)
        self.block.prepare(weights)
        self._preparation_report = SimulationReport(
            "preparation", self.hardware.summary()
        )
        self._prepared = True
        self._last_result = None
        self.hardware.reset_stats()
        return self._preparation_report

    def run(
        self,
        input_tensor: ArrayLike | TensorPacket,
        *,
        input_name: str = "block_input",
        output_name: str = "block_output",
        input_location: str = "digital_buffer",
    ) -> SimulationResult:
        if not self._prepared:
            raise ComponentNotPreparedError("simulator.prepare(weights) must be called first")
        self.hardware.reset_stats()
        if isinstance(input_tensor, TensorPacket):
            packet = input_tensor
        else:
            packet = self.hardware.create_tensor(
                input_name,
                input_tensor,
                bit_width=self.config.precision.activation_bits,
                location=input_location,
            )
        output_packet = self.block.forward(packet, output_name=output_name)
        report = SimulationReport("execution", self.hardware.summary())
        self._last_result = SimulationResult(
            output=np.array(output_packet.data, copy=True),
            output_packet=output_packet,
            report=report,
        )
        return self._last_result

    @property
    def preparation_report(self) -> SimulationReport:
        if self._preparation_report is None:
            raise ComponentNotPreparedError("no preparation report is available")
        return self._preparation_report

    def report(self) -> SimulationReport:
        if self._last_result is None:
            raise RuntimeError("no execution report is available; call run() first")
        return self._last_result.report

    def reset_runtime_stats(self) -> None:
        self.hardware.reset_stats()
        self._last_result = None

    def export_report(
        self,
        output_directory: str | Path,
        *,
        include_preparation: bool = True,
    ) -> dict[str, dict[str, Path]]:
        outputs: dict[str, dict[str, Path]] = {}
        if include_preparation:
            outputs["preparation"] = self.preparation_report.export(
                output_directory, stem="preparation"
            )
        outputs["execution"] = self.report().export(
            output_directory, stem="execution"
        )
        return outputs

