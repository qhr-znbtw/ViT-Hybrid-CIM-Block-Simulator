"""One pre-LayerNorm ViT Transformer Encoder Block."""

from __future__ import annotations

import numpy as np

from ..components import FeedForward, MultiHeadAttention
from ..config import TransformerConfig
from ..errors import ComponentNotPreparedError, MatrixShapeError
from ..system import HybridCIMSystem
from ..tensor import TensorPacket
from ..weights import TransformerBlockWeights


class TransformerEncoderBlock:
    """Pre-LN attention and feed-forward residual sublayers."""

    def __init__(
        self,
        name: str,
        system: HybridCIMSystem,
        config: TransformerConfig,
    ) -> None:
        self.name = name
        self.system = system
        self.config = config
        self.attention = MultiHeadAttention(f"{name}.attention", system, config)
        self.feed_forward = FeedForward(f"{name}.feed_forward", system, config)
        self._ln1_gamma: np.ndarray | None = None
        self._ln1_beta: np.ndarray | None = None
        self._ln2_gamma: np.ndarray | None = None
        self._ln2_beta: np.ndarray | None = None
        self._prepared = False

    def prepare(self, weights: TransformerBlockWeights) -> None:
        expected = (self.config.embed_dim,)
        for label, parameter in (
            ("ln1_gamma", weights.ln1_gamma),
            ("ln1_beta", weights.ln1_beta),
            ("ln2_gamma", weights.ln2_gamma),
            ("ln2_beta", weights.ln2_beta),
        ):
            if parameter.shape != expected:
                raise ValueError(f"{self.name}.{label} must have shape {expected}")

        self._ln1_gamma = self.system.digital.load_parameter(
            f"{self.name}.ln1_gamma", weights.ln1_gamma
        )
        self._ln1_beta = self.system.digital.load_parameter(
            f"{self.name}.ln1_beta", weights.ln1_beta
        )
        self._ln2_gamma = self.system.digital.load_parameter(
            f"{self.name}.ln2_gamma", weights.ln2_gamma
        )
        self._ln2_beta = self.system.digital.load_parameter(
            f"{self.name}.ln2_beta", weights.ln2_beta
        )
        self.attention.prepare(weights.attention)
        self.feed_forward.prepare(weights.feed_forward)
        self._prepared = True

    def forward(
        self,
        tensor: TensorPacket,
        *,
        output_name: str | None = None,
    ) -> TensorPacket:
        if not self._prepared:
            raise ComponentNotPreparedError(
                f"TransformerEncoderBlock {self.name!r} must be prepared before execution"
            )
        if tensor.data.ndim != 3 or tensor.shape[-1] != self.config.embed_dim:
            raise MatrixShapeError(
                f"{self.name} expects [batch, tokens, {self.config.embed_dim}]; "
                f"got {tensor.shape}"
            )
        assert self._ln1_gamma is not None and self._ln1_beta is not None
        assert self._ln2_gamma is not None and self._ln2_beta is not None

        block_input = self.system.transfer(
            tensor,
            self.system.digital.name,
            transfer_name=f"{self.name}.input_to_digital",
            output_name=f"{self.name}.input",
        )
        norm1 = self.system.digital.layer_norm(
            block_input,
            gamma=self._ln1_gamma,
            beta=self._ln1_beta,
            epsilon=self.config.layer_norm_epsilon,
            name=f"{self.name}.norm1",
            op_name=f"{self.name}.layer_norm1",
        )
        attention_output = self.attention.forward(
            norm1, output_name=f"{self.name}.attention_output"
        )
        attention_residual = self.system.digital.residual_add(
            attention_output,
            block_input,
            name=f"{self.name}.attention_residual",
            op_name=f"{self.name}.attention_residual_add",
        )
        norm2 = self.system.digital.layer_norm(
            attention_residual,
            gamma=self._ln2_gamma,
            beta=self._ln2_beta,
            epsilon=self.config.layer_norm_epsilon,
            name=f"{self.name}.norm2",
            op_name=f"{self.name}.layer_norm2",
        )
        feed_forward_output = self.feed_forward.forward(
            norm2, output_name=f"{self.name}.feed_forward_output"
        )
        return self.system.digital.residual_add(
            feed_forward_output,
            attention_residual,
            name=output_name or f"{self.name}.output",
            op_name=f"{self.name}.feed_forward_residual_add",
        )

