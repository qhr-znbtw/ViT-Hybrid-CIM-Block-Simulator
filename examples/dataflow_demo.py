"""One Hybrid CIM linear layer with explicit data movement."""

from __future__ import annotations

from pprint import pprint

import numpy as np

from hybrid_cim import HybridCIMSystem


def main() -> None:
    rng = np.random.default_rng(11)
    system = HybridCIMSystem(input_bit_width=8, weight_bit_width=8, output_bit_width=32)

    x = system.create_tensor(
        "x",
        rng.normal(size=(1, 4, 8)).astype(np.float32),
        location="digital_buffer",
    )
    w0 = system.program_rram_weight(
        "w0", rng.normal(size=(8, 8)).astype(np.float32)
    )
    a = system.load_sram_weight(
        "lora_a", rng.normal(size=(8, 2)).astype(np.float32)
    )
    b = system.load_sram_weight(
        "lora_b", rng.normal(size=(2, 8)).astype(np.float32)
    )

    x_rram, x_sram = system.data_mover.broadcast(
        x, (system.rram.name, system.sram.name), transfer_name_prefix="broadcast_x"
    )
    y0 = system.rram_matmul_tensor(
        x_rram, w0, output_name="y_backbone", op_name="backbone_linear"
    )
    low_rank = system.sram_resident_matmul_tensor(
        x_sram, a, output_name="lora_hidden", op_name="lora_a"
    )
    delta_y = system.sram_resident_matmul_tensor(
        low_rank, b, output_name="y_lora", op_name="lora_b"
    )

    y0_digital = system.transfer(y0, system.digital.name)
    delta_y_digital = system.transfer(delta_y, system.digital.name)
    y = system.digital.add(
        y0_digital,
        delta_y_digital,
        name="hybrid_linear_output",
        op_name="merge_backbone_and_lora",
    )

    print("output shape:", y.shape)
    pprint(system.summary())


if __name__ == "__main__":
    main()
