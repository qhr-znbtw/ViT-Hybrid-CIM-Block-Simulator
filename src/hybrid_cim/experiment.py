"""Reproducible experiment generation and execution."""

from __future__ import annotations

import json
import platform
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .analysis import ExperimentAnalysis, analyze_execution_report
from .config import ExperimentConfig
from .reporting import SimulationReport, SimulationResult
from .simulator import HybridCIMBlockSimulator
from .weights import (
    FeedForwardWeights,
    HybridLinearWeights,
    MultiHeadAttentionWeights,
    TransformerBlockWeights,
)


def _linear_weights(
    rng: np.random.Generator,
    input_dim: int,
    output_dim: int,
    rank: int,
    mapping: str,
) -> HybridLinearWeights:
    backbone = rng.normal(scale=0.02, size=(input_dim, output_dim)).astype(np.float32)
    bias = np.zeros(output_dim, dtype=np.float32)
    if mapping == "rram":
        return HybridLinearWeights(backbone=backbone, bias=bias)
    return HybridLinearWeights(
        backbone=backbone,
        lora_a=rng.normal(scale=0.01, size=(input_dim, rank)).astype(np.float32),
        lora_b=rng.normal(scale=0.01, size=(rank, output_dim)).astype(np.float32),
        bias=bias,
    )


def generate_block_weights(
    config: ExperimentConfig,
    rng: np.random.Generator,
) -> TransformerBlockWeights:
    """Generate deterministic shape-correct weights for traffic/compute simulation."""

    model = config.simulator.model
    mapping = config.simulator.mapping
    d = model.embed_dim
    ff = model.mlp_dim
    rank = model.lora_rank
    attention = MultiHeadAttentionWeights(
        q_proj=_linear_weights(rng, d, d, rank, mapping.q_proj),
        k_proj=_linear_weights(rng, d, d, rank, mapping.k_proj),
        v_proj=_linear_weights(rng, d, d, rank, mapping.v_proj),
        out_proj=_linear_weights(rng, d, d, rank, mapping.out_proj),
    )
    feed_forward = FeedForwardWeights(
        up_proj=_linear_weights(rng, d, ff, rank, mapping.ff_up),
        down_proj=_linear_weights(rng, ff, d, rank, mapping.ff_down),
    )
    return TransformerBlockWeights(
        ln1_gamma=np.ones(d, dtype=np.float32),
        ln1_beta=np.zeros(d, dtype=np.float32),
        attention=attention,
        ln2_gamma=np.ones(d, dtype=np.float32),
        ln2_beta=np.zeros(d, dtype=np.float32),
        feed_forward=feed_forward,
    )


@dataclass(frozen=True, slots=True)
class ExperimentRunResult:
    output_directory: Path
    preparation_report: SimulationReport
    execution_result: SimulationResult
    analysis: ExperimentAnalysis
    paths: dict[str, Any]


def run_experiment(
    config: ExperimentConfig,
    output_directory: str | Path,
) -> ExperimentRunResult:
    """Run one deterministic block experiment and export all raw/derived results."""

    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(config.seed)
    weights = generate_block_weights(config, rng)
    workload = config.workload
    model = config.simulator.model
    input_tensor = rng.normal(
        size=(workload.batch_size, workload.tokens, model.embed_dim)
    ).astype(np.float32)

    simulator = HybridCIMBlockSimulator(config.simulator)
    preparation = simulator.prepare(weights)
    execution = simulator.run(input_tensor)
    raw_paths = simulator.export_report(output / "raw_reports")
    analysis = analyze_execution_report(execution.report)
    analysis_paths = analysis.export(output / "analysis")

    metadata = {
        "experiment": config.to_dict(),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "output": {
            "shape": list(execution.output.shape),
            "mean": float(execution.output.mean()),
            "std": float(execution.output.std()),
        },
        "preparation_totals": preparation.totals,
        "execution_totals": execution.report.totals,
    }
    metadata_path = output / "experiment_metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    config_path = config.save_json(output / "experiment_config.json")

    paths: dict[str, Any] = {
        "config": config_path,
        "metadata": metadata_path,
        "raw_reports": raw_paths,
        "analysis": analysis_paths,
    }
    return ExperimentRunResult(
        output_directory=output,
        preparation_report=preparation,
        execution_result=execution,
        analysis=analysis,
        paths=paths,
    )
