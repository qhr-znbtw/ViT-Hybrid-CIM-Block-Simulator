"""Command-line entry point for a configured Transformer Block experiment."""

from __future__ import annotations

import argparse
from pathlib import Path

from hybrid_cim import ExperimentConfig, run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one Hybrid CIM Transformer Encoder Block experiment."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parent / "configs" / "vit_base_block_halora.json",
        help="JSON experiment configuration.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Result directory. Defaults to output/experiments/<experiment-name>.",
    )
    args = parser.parse_args()

    config = ExperimentConfig.load_json(args.config)
    output = args.output or Path("output") / "experiments" / config.name
    result = run_experiment(config, output)
    overview = result.analysis.summary()["overview"]
    print(f"Experiment: {config.name}")
    print(f"Output: {result.output_directory.resolve()}")
    print(f"MACs: {overview['total_macs']}")
    print(f"Transferred bytes: {overview['transferred_bytes']}")
    print(f"Arithmetic intensity: {overview['arithmetic_intensity_mac_per_byte']} MAC/byte")


if __name__ == "__main__":
    main()

