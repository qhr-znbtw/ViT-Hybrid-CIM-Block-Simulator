from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from hybrid_cim import (
    ExperimentConfig,
    MappingConfig,
    PrecisionConfig,
    SimulatorConfig,
    TransformerConfig,
    WorkloadConfig,
    run_experiment,
)


def small_experiment_config() -> ExperimentConfig:
    return ExperimentConfig(
        name="small-halora-block",
        seed=17,
        workload=WorkloadConfig(batch_size=1, tokens=3),
        simulator=SimulatorConfig(
            model=TransformerConfig(
                embed_dim=4,
                num_heads=2,
                mlp_dim=8,
                lora_rank=2,
            ),
            precision=PrecisionConfig(
                activation_bits=8,
                weight_bits=8,
                accumulator_bits=32,
            ),
            mapping=MappingConfig.halora(),
        ),
    )


class ExperimentTests(unittest.TestCase):
    def test_config_json_round_trip(self) -> None:
        config = small_experiment_config()
        with tempfile.TemporaryDirectory() as directory:
            path = config.save_json(Path(directory) / "config.json")
            restored = ExperimentConfig.load_json(path)

        self.assertEqual(restored, config)
        self.assertEqual(restored.simulator.mapping.out_proj, "rram")
        self.assertEqual(restored.simulator.model.head_dim, 2)

    def test_runner_and_analysis_exports(self) -> None:
        config = small_experiment_config()
        with tempfile.TemporaryDirectory() as directory:
            result = run_experiment(config, Path(directory) / "result")
            analysis = result.analysis.summary()

            self.assertEqual(result.execution_result.output.shape, (1, 3, 4))
            self.assertEqual(analysis["overview"]["matmul_operations"], 18)
            self.assertEqual(analysis["overview"]["data_transfers"], 29)
            self.assertGreater(analysis["overview"]["arithmetic_intensity_mac_per_byte"], 0)
            roles = {row["role"] for row in analysis["compute_by_role"]}
            self.assertEqual(
                roles,
                {"dynamic_attention", "frozen_backbone", "lora"},
            )
            components = {
                row["component"]: row for row in analysis["component_breakdown"]
            }
            self.assertEqual(
                sum(row["macs"] for row in analysis["component_breakdown"]),
                analysis["overview"]["total_macs"],
            )
            self.assertEqual(
                sum(row["transferred_bytes"] for row in analysis["component_breakdown"]),
                analysis["overview"]["transferred_bytes"],
            )
            self.assertEqual(components["qk_t"]["macs"], 36)
            self.assertEqual(components["pv"]["macs"], 36)
            self.assertEqual(components["q_projection"]["macs"], 96)
            self.assertEqual(components["q_projection"]["rram_macs"], 48)
            self.assertEqual(components["q_projection"]["sram_macs"], 48)
            self.assertEqual(components["q_projection"]["rram_percent"], 50.0)
            self.assertEqual(components["q_projection"]["sram_percent"], 50.0)
            self.assertEqual(components["q_projection"]["rram_transfer_events"], 2)
            self.assertEqual(components["q_projection"]["rram_transfer_bytes"], 96)
            self.assertEqual(components["q_projection"]["sram_transfer_events"], 2)
            self.assertEqual(components["q_projection"]["sram_transfer_bytes"], 96)
            self.assertEqual(components["unclassified"]["macs"], 0)
            self.assertEqual(components["unclassified"]["transfer_events"], 0)
            self.assertTrue(result.paths["metadata"].exists())
            self.assertTrue(result.paths["analysis"]["json"].exists())
            self.assertTrue(result.paths["analysis"]["markdown"].exists())
            self.assertTrue(result.paths["analysis"]["traffic_by_path"].exists())
            self.assertTrue(result.paths["analysis"]["component_breakdown"].exists())
            self.assertTrue(
                result.paths["analysis"]["component_hardware_markdown"].exists()
            )
            self.assertTrue(result.paths["analysis"]["component_hardware_csv"].exists())

    def test_formal_vit_base_configuration_loads(self) -> None:
        path = (
            Path(__file__).parents[1]
            / "experiments"
            / "configs"
            / "vit_base_block_halora.json"
        )
        config = ExperimentConfig.load_json(path)

        self.assertEqual(config.simulator.model.embed_dim, 768)
        self.assertEqual(config.simulator.model.num_heads, 12)
        self.assertEqual(config.workload.tokens, 197)
        self.assertEqual(config.simulator.mapping.out_proj, "rram")


if __name__ == "__main__":
    unittest.main()
