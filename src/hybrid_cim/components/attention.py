"""Multi-head self-attention mapped across Hybrid CIM components."""

from __future__ import annotations

from math import sqrt

from ..config import TransformerConfig
from ..errors import ComponentNotPreparedError, MatrixShapeError
from ..system import HybridCIMSystem
from ..tensor import TensorPacket
from ..weights import MultiHeadAttentionWeights
from .hybrid_linear import HybridLinear


class MultiHeadAttention:
    """ViT self-attention with hybrid projections and SRAM dynamic products."""

    def __init__(
        self,
        name: str,
        system: HybridCIMSystem,
        config: TransformerConfig,
    ) -> None:
        self.name = name
        self.system = system
        self.config = config
        self.q_proj = HybridLinear(f"{name}.q_proj", system)
        self.k_proj = HybridLinear(f"{name}.k_proj", system)
        self.v_proj = HybridLinear(f"{name}.v_proj", system)
        self.out_proj = HybridLinear(f"{name}.out_proj", system)
        self._prepared = False

    def prepare(self, weights: MultiHeadAttentionWeights) -> None:
        d = self.config.embed_dim
        for label, projection in (
            ("q_proj", weights.q_proj),
            ("k_proj", weights.k_proj),
            ("v_proj", weights.v_proj),
            ("out_proj", weights.out_proj),
        ):
            if projection.input_features != d or projection.output_features != d:
                raise ValueError(f"{self.name}.{label} must have shape ({d}, {d})")
        self.q_proj.prepare(weights.q_proj)
        self.k_proj.prepare(weights.k_proj)
        self.v_proj.prepare(weights.v_proj)
        self.out_proj.prepare(weights.out_proj)
        self._prepared = True

    def forward(
        self,
        tensor: TensorPacket,
        *,
        output_name: str | None = None,
    ) -> TensorPacket:
        if not self._prepared:
            raise ComponentNotPreparedError(
                f"MultiHeadAttention {self.name!r} must be prepared before execution"
            )
        if tensor.data.ndim != 3 or tensor.shape[-1] != self.config.embed_dim:
            raise MatrixShapeError(
                f"{self.name} expects [batch, tokens, {self.config.embed_dim}]; "
                f"got {tensor.shape}"
            )

        batch, tokens, _ = tensor.shape
        heads = self.config.num_heads
        head_dim = self.config.head_dim
        q = self.q_proj.forward(tensor, output_name=f"{self.name}.q")
        k = self.k_proj.forward(tensor, output_name=f"{self.name}.k")
        v = self.v_proj.forward(tensor, output_name=f"{self.name}.v")

        q_4d = self.system.digital.reshape(
            q,
            (batch, tokens, heads, head_dim),
            name=f"{self.name}.q_4d",
            op_name=f"{self.name}.reshape_q",
        )
        k_4d = self.system.digital.reshape(
            k,
            (batch, tokens, heads, head_dim),
            name=f"{self.name}.k_4d",
            op_name=f"{self.name}.reshape_k",
        )
        v_4d = self.system.digital.reshape(
            v,
            (batch, tokens, heads, head_dim),
            name=f"{self.name}.v_4d",
            op_name=f"{self.name}.reshape_v",
        )
        q_heads = self.system.digital.transpose(
            q_4d,
            (0, 2, 1, 3),
            name=f"{self.name}.q_heads",
            op_name=f"{self.name}.transpose_q",
        )
        k_heads = self.system.digital.transpose(
            k_4d,
            (0, 2, 1, 3),
            name=f"{self.name}.k_heads",
            op_name=f"{self.name}.transpose_k",
        )
        v_heads = self.system.digital.transpose(
            v_4d,
            (0, 2, 1, 3),
            name=f"{self.name}.v_heads",
            op_name=f"{self.name}.transpose_v",
        )
        k_t = self.system.digital.transpose(
            k_heads,
            (0, 1, 3, 2),
            name=f"{self.name}.k_t",
            op_name=f"{self.name}.transpose_k_t",
        )

        q_sram = self.system.transfer(
            q_heads, self.system.sram.name, transfer_name=f"{self.name}.q_to_sram"
        )
        k_sram = self.system.transfer(
            k_t, self.system.sram.name, transfer_name=f"{self.name}.k_t_to_sram"
        )
        scores_sram = self.system.sram_dynamic_matmul_tensor(
            q_sram,
            k_sram,
            output_name=f"{self.name}.scores_sram",
            op_name=f"{self.name}.qk_t",
        )
        scores = self.system.transfer(
            scores_sram,
            self.system.digital.name,
            transfer_name=f"{self.name}.scores_to_digital",
        )
        scaled_scores = self.system.digital.scale(
            scores,
            1.0 / sqrt(head_dim),
            name=f"{self.name}.scaled_scores",
            op_name=f"{self.name}.scale_scores",
        )
        probabilities = self.system.digital.softmax(
            scaled_scores,
            axis=-1,
            name=f"{self.name}.probabilities",
            op_name=f"{self.name}.softmax",
        )

        probabilities_sram = self.system.transfer(
            probabilities,
            self.system.sram.name,
            transfer_name=f"{self.name}.probabilities_to_sram",
        )
        v_sram = self.system.transfer(
            v_heads,
            self.system.sram.name,
            transfer_name=f"{self.name}.v_to_sram",
        )
        context_sram = self.system.sram_dynamic_matmul_tensor(
            probabilities_sram,
            v_sram,
            output_name=f"{self.name}.context_sram",
            op_name=f"{self.name}.probabilities_v",
        )
        context = self.system.transfer(
            context_sram,
            self.system.digital.name,
            transfer_name=f"{self.name}.context_to_digital",
        )
        context_tokens = self.system.digital.transpose(
            context,
            (0, 2, 1, 3),
            name=f"{self.name}.context_tokens",
            op_name=f"{self.name}.transpose_context",
        )
        merged_heads = self.system.digital.reshape(
            context_tokens,
            (batch, tokens, self.config.embed_dim),
            name=f"{self.name}.merged_heads",
            op_name=f"{self.name}.merge_heads",
        )
        return self.out_proj.forward(
            merged_heads, output_name=output_name or f"{self.name}.output"
        )
