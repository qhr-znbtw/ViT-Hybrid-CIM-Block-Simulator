"""End-to-end five-layer Transformer Block simulator example."""

from __future__ import annotations

from pathlib import Path
from pprint import pprint

import numpy as np

from hybrid_cim import (
    FeedForwardWeights,
    HybridCIMBlockSimulator,
    HybridLinearWeights,
    MultiHeadAttentionWeights,
    PrecisionConfig,
    SimulatorConfig,
    TransformerBlockWeights,
    TransformerConfig,
)


def linear_weights(
    rng: np.random.Generator,
    input_dim: int,
    output_dim: int,
    rank: int,
) -> HybridLinearWeights:
    return HybridLinearWeights(
        backbone=rng.normal(scale=0.1, size=(input_dim, output_dim)).astype(np.float32),
        lora_a=rng.normal(scale=0.1, size=(input_dim, rank)).astype(np.float32),
        lora_b=rng.normal(scale=0.1, size=(rank, output_dim)).astype(np.float32),
        bias=np.zeros(output_dim, dtype=np.float32),
    )


def main() -> None:
    rng = np.random.default_rng(31)
    model = TransformerConfig(embed_dim=8, num_heads=2, mlp_dim=16, lora_rank=2)
    config = SimulatorConfig(
        model=model,
        precision=PrecisionConfig(activation_bits=8, weight_bits=8, accumulator_bits=32),
    )
    attention = MultiHeadAttentionWeights(
        q_proj=linear_weights(rng, 8, 8, 2),
        k_proj=linear_weights(rng, 8, 8, 2),
        v_proj=linear_weights(rng, 8, 8, 2),
        out_proj=linear_weights(rng, 8, 8, 2),
    )
    feed_forward = FeedForwardWeights(
        up_proj=linear_weights(rng, 8, 16, 2),
        down_proj=linear_weights(rng, 16, 8, 2),
    )
    weights = TransformerBlockWeights(
        ln1_gamma=np.ones(8, dtype=np.float32),
        ln1_beta=np.zeros(8, dtype=np.float32),
        attention=attention,
        ln2_gamma=np.ones(8, dtype=np.float32),
        ln2_beta=np.zeros(8, dtype=np.float32),
        feed_forward=feed_forward,
    )

    simulator = HybridCIMBlockSimulator(config)
    preparation = simulator.prepare(weights)
    x = rng.normal(size=(1, 5, 8)).astype(np.float32)
    result = simulator.run(x)
    output_paths = simulator.export_report(Path("output") / "demo_report")

    print("output shape:", result.output.shape)
    print("preparation totals:")
    pprint(preparation.totals)
    print("execution totals:")
    pprint(result.report.totals)
    print("reports:")
    pprint(output_paths)


if __name__ == "__main__":
    main()

