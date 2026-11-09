from __future__ import annotations

import unittest

import numpy as np

from hybrid_cim import HybridCIMSystem, HybridLinear, HybridLinearWeights
from hybrid_cim.errors import ComponentNotPreparedError, MatrixShapeError


class HybridLinearTests(unittest.TestCase):
    def test_forward_with_lora_and_bias(self) -> None:
        system = HybridCIMSystem()
        layer = HybridLinear("projection", system)
        w0 = np.arange(12, dtype=np.float32).reshape(3, 4)
        a = np.arange(6, dtype=np.float32).reshape(3, 2)
        b = np.arange(8, dtype=np.float32).reshape(2, 4)
        bias = np.arange(4, dtype=np.float32)
        layer.prepare(HybridLinearWeights(w0, a, b, bias))
        system.reset_stats()
        x_data = np.arange(6, dtype=np.float32).reshape(1, 2, 3)
        x = system.create_tensor("x", x_data, location=system.digital.name)

        output = layer.forward(x, output_name="y")

        expected = x_data @ w0 + (x_data @ a) @ b + bias
        np.testing.assert_allclose(output.data, expected)
        self.assertEqual(output.name, "y")
        self.assertEqual(output.location, system.digital.name)
        summary = system.summary()
        self.assertEqual(summary["totals"]["matmul_operations"], 3)
        self.assertEqual(summary["totals"]["data_transfers"], 4)
        self.assertEqual(summary["totals"]["digital_operations"], 2)

    def test_forward_without_lora_or_bias(self) -> None:
        system = HybridCIMSystem()
        layer = HybridLinear("projection", system)
        weight = np.arange(12, dtype=np.float32).reshape(3, 4)
        layer.prepare(HybridLinearWeights(weight))
        system.reset_stats()
        x_data = np.ones((2, 3), dtype=np.float32)
        x = system.create_tensor("x", x_data, location=system.digital.name)

        output = layer.forward(x, output_name="y")

        np.testing.assert_allclose(output.data, x_data @ weight)
        summary = system.summary()
        self.assertEqual(summary["totals"]["matmul_operations"], 1)
        self.assertEqual(summary["totals"]["data_transfers"], 2)
        self.assertEqual(summary["totals"]["digital_operations"], 0)

    def test_forward_before_prepare_fails(self) -> None:
        system = HybridCIMSystem()
        layer = HybridLinear("projection", system)
        x = system.create_tensor("x", np.ones((2, 3)))

        with self.assertRaises(ComponentNotPreparedError):
            layer.forward(x)

    def test_input_feature_mismatch_fails(self) -> None:
        system = HybridCIMSystem()
        layer = HybridLinear("projection", system)
        layer.prepare(HybridLinearWeights(np.ones((3, 4))))
        x = system.create_tensor("x", np.ones((2, 5)))

        with self.assertRaises(MatrixShapeError):
            layer.forward(x)

    def test_invalid_lora_shapes_fail_during_weight_construction(self) -> None:
        with self.assertRaises(ValueError):
            HybridLinearWeights(
                np.ones((3, 4)),
                np.ones((3, 2)),
                np.ones((3, 4)),
            )


if __name__ == "__main__":
    unittest.main()
