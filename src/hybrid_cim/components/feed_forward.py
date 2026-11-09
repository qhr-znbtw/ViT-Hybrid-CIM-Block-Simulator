"""Transformer feed-forward network mapped to HybridLinear components."""

from __future__ import annotations

from ..config import TransformerConfig
from ..errors import ComponentNotPreparedError, MatrixShapeError
from ..system import HybridCIMSystem
from ..tensor import TensorPacket
from ..weights import FeedForwardWeights
from .hybrid_linear import HybridLinear


class FeedForward:
    def __init__(
        self,
        name: str,
        system: HybridCIMSystem,
        config: TransformerConfig,
    ) -> None:
        self.name = name
        self.system = system
        self.config = config
        self.up_proj = HybridLinear(f"{name}.up_proj", system)
        self.down_proj = HybridLinear(f"{name}.down_proj", system)
        self._prepared = False

    def prepare(self, weights: FeedForwardWeights) -> None:
        d = self.config.embed_dim
        ff = self.config.mlp_dim
        if (
            weights.up_proj.input_features != d
            or weights.up_proj.output_features != ff
        ):
            raise ValueError(f"{self.name}.up_proj must have shape ({d}, {ff})")
        if (
            weights.down_proj.input_features != ff
            or weights.down_proj.output_features != d
        ):
            raise ValueError(f"{self.name}.down_proj must have shape ({ff}, {d})")
        self.up_proj.prepare(weights.up_proj)
        self.down_proj.prepare(weights.down_proj)
        self._prepared = True

    def forward(
        self,
        tensor: TensorPacket,
        *,
        output_name: str | None = None,
    ) -> TensorPacket:
        if not self._prepared:
            raise ComponentNotPreparedError(
                f"FeedForward {self.name!r} must be prepared before execution"
            )
        if tensor.data.ndim != 3 or tensor.shape[-1] != self.config.embed_dim:
            raise MatrixShapeError(
                f"{self.name} expects [batch, tokens, {self.config.embed_dim}]; "
                f"got {tensor.shape}"
            )
        hidden = self.up_proj.forward(
            tensor, output_name=f"{self.name}.up_output"
        )
        activated = self.system.digital.gelu(
            hidden,
            name=f"{self.name}.gelu_output",
            op_name=f"{self.name}.gelu",
        )
        return self.down_proj.forward(
            activated, output_name=output_name or f"{self.name}.output"
        )
