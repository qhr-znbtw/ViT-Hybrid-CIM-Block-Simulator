"""Small end-to-end example for the two-unit system."""

from __future__ import annotations

from pprint import pprint

import numpy as np

from hybrid_cim import HybridCIMSystem


def main() -> None:
    rng = np.random.default_rng(7)
    system = HybridCIMSystem()

    x = rng.normal(size=(1, 4, 8)).astype(np.float32)
    w_q = rng.normal(size=(8, 8)).astype(np.float32)
    lora_a = rng.normal(size=(8, 2)).astype(np.float32)
    lora_b = rng.normal(size=(2, 8)).astype(np.float32)

    w_q_handle = system.program_rram_weight("w_q_backbone", w_q)
    lora_a_handle = system.load_sram_weight("q_lora_a", lora_a)
    lora_b_handle = system.load_sram_weight("q_lora_b", lora_b)

    q_backbone = system.rram_matmul(x, w_q_handle, op_name="q_backbone")
    q_lora_hidden = system.sram_resident_matmul(
        x, lora_a_handle, op_name="q_lora_a"
    )
    q_lora = system.sram_resident_matmul(
        q_lora_hidden, lora_b_handle, op_name="q_lora_b"
    )
    q = q_backbone + q_lora

    scores = system.sram_dynamic_matmul(
        q,
        q.swapaxes(-1, -2),
        op_name="attention_qk_t_demo",
    )

    print("q shape:", q.shape)
    print("scores shape:", scores.shape)
    pprint(system.summary())


if __name__ == "__main__":
    main()

