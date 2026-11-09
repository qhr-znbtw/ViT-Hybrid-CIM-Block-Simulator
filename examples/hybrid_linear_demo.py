"""High-level HybridLinear preparation and execution example."""

from __future__ import annotations

from pprint import pprint

import numpy as np

from hybrid_cim import HybridCIMSystem, HybridLinear, HybridLinearWeights


def main() -> None:
    rng = np.random.default_rng(23)
    system = HybridCIMSystem()
    layer = HybridLinear("q_projection", system)
    weights = HybridLinearWeights(
        backbone=rng.normal(size=(8, 8)).astype(np.float32),
        lora_a=rng.normal(size=(8, 2)).astype(np.float32),
        lora_b=rng.normal(size=(2, 8)).astype(np.float32),
        bias=rng.normal(size=(8,)).astype(np.float32),
    )

    layer.prepare(weights)
    preparation_report = system.summary()

    system.reset_stats()
    x = system.create_tensor(
        "x",
        rng.normal(size=(1, 4, 8)).astype(np.float32),
        location=system.digital.name,
    )
    output = layer.forward(x, output_name="q")
    execution_report = system.summary()

    print("output shape:", output.shape)
    print("preparation totals:")
    pprint(preparation_report["totals"])
    print("execution totals:")
    pprint(execution_report["totals"])


if __name__ == "__main__":
    main()
