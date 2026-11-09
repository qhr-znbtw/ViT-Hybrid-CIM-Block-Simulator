from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

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


def make_linear(
    rng: np.random.Generator,
    input_dim: int,
    output_dim: int,
    rank: int,
) -> HybridLinearWeights:
    return HybridLinearWeights(
        rng.normal(scale=0.15, size=(input_dim, output_dim)).astype(np.float32),
        rng.normal(scale=0.1, size=(input_dim, rank)).astype(np.float32),
        rng.normal(scale=0.1, size=(rank, output_dim)).astype(np.float32),
        rng.normal(scale=0.02, size=(output_dim,)).astype(np.float32),
    )


def make_fixture() -> tuple[SimulatorConfig, TransformerBlockWeights, np.ndarray]:
    rng = np.random.default_rng(101)
    model = TransformerConfig(embed_dim=4, num_heads=2, mlp_dim=8, lora_rank=2)
    config = SimulatorConfig(
        model=model,
        precision=PrecisionConfig(activation_bits=8, weight_bits=8, accumulator_bits=32),
    )
    attention = MultiHeadAttentionWeights(
        q_proj=make_linear(rng, 4, 4, 2),
        k_proj=make_linear(rng, 4, 4, 2),
        v_proj=make_linear(rng, 4, 4, 2),
        out_proj=make_linear(rng, 4, 4, 2),
    )
    feed_forward = FeedForwardWeights(
        up_proj=make_linear(rng, 4, 8, 2),
        down_proj=make_linear(rng, 8, 4, 2),
    )
    weights = TransformerBlockWeights(
        ln1_gamma=rng.normal(loc=1.0, scale=0.05, size=4).astype(np.float32),
        ln1_beta=rng.normal(scale=0.02, size=4).astype(np.float32),
        attention=attention,
        ln2_gamma=rng.normal(loc=1.0, scale=0.05, size=4).astype(np.float32),
        ln2_beta=rng.normal(scale=0.02, size=4).astype(np.float32),
        feed_forward=feed_forward,
    )
    x = rng.normal(size=(1, 3, 4)).astype(np.float32)
    return config, weights, x


def reference_linear(x: np.ndarray, weights: HybridLinearWeights) -> np.ndarray:
    result = x @ weights.backbone
    if weights.lora_a is not None and weights.lora_b is not None:
        result = result + (x @ weights.lora_a) @ weights.lora_b
    if weights.bias is not None:
        result = result + weights.bias
    return result


def reference_layer_norm(
    x: np.ndarray,
    gamma: np.ndarray,
    beta: np.ndarray,
    epsilon: float,
) -> np.ndarray:
    mean = x.mean(axis=-1, keepdims=True)
    centered = x - mean
    variance = (centered * centered).mean(axis=-1, keepdims=True)
    return centered / np.sqrt(variance + epsilon) * gamma + beta


def reference_block(
    x: np.ndarray,
    config: TransformerConfig,
    weights: TransformerBlockWeights,
) -> np.ndarray:
    norm1 = reference_layer_norm(
        x, weights.ln1_gamma, weights.ln1_beta, config.layer_norm_epsilon
    )
    attention = weights.attention
    q = reference_linear(norm1, attention.q_proj)
    k = reference_linear(norm1, attention.k_proj)
    v = reference_linear(norm1, attention.v_proj)
    batch, tokens, _ = q.shape
    q = q.reshape(batch, tokens, config.num_heads, config.head_dim).transpose(0, 2, 1, 3)
    k = k.reshape(batch, tokens, config.num_heads, config.head_dim).transpose(0, 2, 1, 3)
    v = v.reshape(batch, tokens, config.num_heads, config.head_dim).transpose(0, 2, 1, 3)
    scores = (q @ k.transpose(0, 1, 3, 2)) / np.sqrt(config.head_dim)
    scores = scores - scores.max(axis=-1, keepdims=True)
    probabilities = np.exp(scores)
    probabilities = probabilities / probabilities.sum(axis=-1, keepdims=True)
    context = probabilities @ v
    context = context.transpose(0, 2, 1, 3).reshape(batch, tokens, config.embed_dim)
    attention_output = reference_linear(context, attention.out_proj)
    residual1 = x + attention_output

    norm2 = reference_layer_norm(
        residual1, weights.ln2_gamma, weights.ln2_beta, config.layer_norm_epsilon
    )
    hidden = reference_linear(norm2, weights.feed_forward.up_proj)
    activated = 0.5 * hidden * (
        1.0
        + np.tanh(
            np.sqrt(2.0 / np.pi) * (hidden + 0.044715 * hidden**3)
        )
    )
    feed_forward_output = reference_linear(
        activated, weights.feed_forward.down_proj
    )
    return residual1 + feed_forward_output


class FullSimulatorTests(unittest.TestCase):
    def test_full_block_matches_numpy_reference(self) -> None:
        config, weights, x = make_fixture()
        simulator = HybridCIMBlockSimulator(config)
        preparation = simulator.prepare(weights)

        result = simulator.run(x)
        expected = reference_block(x, config.model, weights)

        np.testing.assert_allclose(result.output, expected, rtol=1e-5, atol=1e-6)
        self.assertEqual(result.output.shape, x.shape)
        self.assertGreater(preparation.totals["weight_events"], 0)
        self.assertEqual(result.report.totals["weight_events"], 0)
        self.assertEqual(result.report.totals["matmul_operations"], 20)
        self.assertEqual(result.report.totals["data_transfers"], 31)

    def test_report_export(self) -> None:
        config, weights, x = make_fixture()
        simulator = HybridCIMBlockSimulator(config)
        simulator.prepare(weights)
        simulator.run(x)

        with tempfile.TemporaryDirectory() as directory:
            paths = simulator.export_report(directory)
            self.assertTrue(paths["preparation"]["json"].exists())
            self.assertTrue(paths["execution"]["markdown"].exists())
            self.assertTrue(paths["execution"]["transfer_events"].exists())
            json_text = Path(paths["execution"]["json"]).read_text(encoding="utf-8")
            self.assertIn('"phase": "execution"', json_text)


if __name__ == "__main__":
    unittest.main()

