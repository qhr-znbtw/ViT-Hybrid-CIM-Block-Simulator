from __future__ import annotations

import unittest

import numpy as np

from hybrid_cim import HybridCIMSystem
from hybrid_cim.errors import MatrixShapeError, StaleWeightHandleError
from hybrid_cim.errors import TensorLocationError


class HybridCIMSystemTests(unittest.TestCase):
    def test_rram_resident_matmul_and_mac_count(self) -> None:
        system = HybridCIMSystem()
        lhs = np.arange(6, dtype=np.float32).reshape(2, 3)
        rhs = np.arange(12, dtype=np.float32).reshape(3, 4)
        handle = system.program_rram_weight("weight", rhs)

        actual = system.rram_matmul(lhs, handle, op_name="test_rram")

        np.testing.assert_allclose(actual, lhs @ rhs)
        event = system.stats.compute_events[0]
        self.assertEqual(event.output_shape, (2, 4))
        self.assertEqual(event.macs, 2 * 3 * 4)
        self.assertEqual(event.mode, "resident_static")

    def test_sram_dynamic_batched_matmul_and_mac_count(self) -> None:
        system = HybridCIMSystem()
        lhs = np.ones((2, 5, 3), dtype=np.float32)
        rhs = np.ones((2, 3, 4), dtype=np.float32)

        actual = system.sram_dynamic_matmul(lhs, rhs, op_name="batched")

        self.assertEqual(actual.shape, (2, 5, 4))
        self.assertEqual(system.stats.compute_events[0].macs, 2 * 5 * 3 * 4)
        self.assertEqual(system.stats.compute_events[0].mode, "dynamic")

    def test_reprogramming_invalidates_old_handle(self) -> None:
        system = HybridCIMSystem()
        old_handle = system.program_rram_weight("weight", np.eye(2))
        system.program_rram_weight("weight", np.ones((2, 2)))

        with self.assertRaises(StaleWeightHandleError):
            system.rram_matmul(np.eye(2), old_handle)

    def test_incompatible_shapes_raise_domain_error(self) -> None:
        system = HybridCIMSystem()
        handle = system.program_rram_weight("weight", np.ones((4, 2)))

        with self.assertRaises(MatrixShapeError):
            system.rram_matmul(np.ones((3, 3)), handle)

    def test_summary_separates_rram_and_sram(self) -> None:
        system = HybridCIMSystem(weight_bit_width=8)
        rram_weight = system.program_rram_weight("r", np.ones((3, 2)))
        sram_weight = system.load_sram_weight("s", np.ones((3, 2)))
        lhs = np.ones((4, 3))

        system.rram_matmul(lhs, rram_weight)
        system.sram_resident_matmul(lhs, sram_weight)
        summary = system.summary()

        self.assertEqual(summary["totals"]["matmul_operations"], 2)
        self.assertEqual(summary["totals"]["macs"], 2 * 4 * 3 * 2)
        self.assertEqual(summary["by_unit"]["rram_cim"]["weight_bytes"], 6)
        self.assertEqual(summary["by_unit"]["sram_cim"]["weight_bytes"], 6)

    def test_transfer_records_direction_and_bytes(self) -> None:
        system = HybridCIMSystem()
        tensor = system.create_tensor("x", np.ones((2, 3)), bit_width=8)

        moved = system.transfer(tensor, system.rram.name)

        self.assertEqual(moved.location, "rram_cim")
        event = system.stats.transfer_events[0]
        self.assertEqual(event.source, "digital_buffer")
        self.assertEqual(event.destination, "rram_cim")
        self.assertEqual(event.bytes, 6)

    def test_tensor_matmul_requires_correct_location(self) -> None:
        system = HybridCIMSystem()
        tensor = system.create_tensor("x", np.ones((2, 3)))
        weight = system.program_rram_weight("w", np.ones((3, 4)))

        with self.assertRaises(TensorLocationError):
            system.rram_matmul_tensor(tensor, weight, output_name="y")

    def test_digital_softmax_and_statistics(self) -> None:
        system = HybridCIMSystem()
        tensor = system.create_tensor(
            "scores", np.array([[1.0, 2.0, 3.0]]), location=system.digital.name
        )

        output = system.digital.softmax(tensor, name="probabilities")

        np.testing.assert_allclose(output.data.sum(axis=-1), np.ones(1))
        event = system.stats.digital_events[0]
        self.assertEqual(event.op_type, "softmax")
        self.assertEqual(event.operation_counts["exp"], 3)

    def test_digital_layer_norm(self) -> None:
        system = HybridCIMSystem()
        tensor = system.create_tensor(
            "x",
            np.array([[[1.0, 2.0, 3.0], [2.0, 4.0, 6.0]]]),
            location=system.digital.name,
        )

        output = system.digital.layer_norm(tensor, name="x_norm", epsilon=1e-8)

        np.testing.assert_allclose(output.data.mean(axis=-1), 0.0, atol=1e-7)
        np.testing.assert_allclose(output.data.var(axis=-1), 1.0, atol=1e-7)
        self.assertEqual(system.stats.digital_events[0].op_type, "layer_norm")

    def test_digital_elementwise_and_layout_operations(self) -> None:
        system = HybridCIMSystem()
        tensor = system.create_tensor(
            "x", np.arange(6, dtype=np.float32).reshape(2, 3), location=system.digital.name
        )

        biased = system.digital.bias_add(
            tensor, np.ones(3), name="biased"
        )
        scaled = system.digital.scale(biased, 0.5, name="scaled")
        activated = system.digital.gelu(scaled, name="activated")
        transposed = system.digital.transpose(
            activated, (1, 0), name="transposed"
        )
        reshaped = system.digital.reshape(
            transposed, (1, 3, 2), name="reshaped"
        )
        concatenated = system.digital.concat(
            (reshaped, reshaped), axis=0, name="concatenated"
        )

        self.assertEqual(concatenated.shape, (2, 3, 2))
        self.assertEqual(len(system.stats.digital_events), 6)
        self.assertEqual(system.stats.digital_events[-1].op_type, "concat")

    def test_hybrid_linear_dataflow(self) -> None:
        system = HybridCIMSystem()
        x_data = np.arange(6, dtype=np.float32).reshape(1, 2, 3)
        x = system.create_tensor("x", x_data)
        w0_data = np.arange(12, dtype=np.float32).reshape(3, 4)
        a_data = np.arange(6, dtype=np.float32).reshape(3, 2)
        b_data = np.arange(8, dtype=np.float32).reshape(2, 4)
        w0 = system.program_rram_weight("w0", w0_data)
        a = system.load_sram_weight("a", a_data)
        b = system.load_sram_weight("b", b_data)

        x_rram, x_sram = system.data_mover.broadcast(
            x, (system.rram.name, system.sram.name)
        )
        y0 = system.rram_matmul_tensor(x_rram, w0, output_name="y0")
        hidden = system.sram_resident_matmul_tensor(x_sram, a, output_name="hidden")
        dy = system.sram_resident_matmul_tensor(hidden, b, output_name="dy")
        y0_digital = system.transfer(y0, system.digital.name)
        dy_digital = system.transfer(dy, system.digital.name)
        output = system.digital.add(y0_digital, dy_digital, name="y")

        expected = x_data @ w0_data + (x_data @ a_data) @ b_data
        np.testing.assert_allclose(output.data, expected)
        summary = system.summary()
        self.assertEqual(summary["totals"]["data_transfers"], 4)
        self.assertEqual(summary["totals"]["transferred_bytes"], 76)
        self.assertEqual(summary["totals"]["digital_operations"], 1)
        self.assertEqual(summary["totals"]["digital_primitives"]["add"], 8)


if __name__ == "__main__":
    unittest.main()

