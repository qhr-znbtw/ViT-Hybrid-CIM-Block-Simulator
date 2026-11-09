"""Structured model-weight containers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _matrix(value: ArrayLike, name: str) -> NDArray[np.generic]:
    array = np.asarray(value)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional matrix; got {array.shape}")
    if not np.issubdtype(array.dtype, np.number):
        raise TypeError(f"{name} must contain numeric data")
    return np.array(array, copy=True)


@dataclass(frozen=True, slots=True)
class HybridLinearWeights:
    """Backbone, optional LoRA factors, and optional bias for one linear layer."""

    backbone: NDArray[np.generic]
    lora_a: NDArray[np.generic] | None = None
    lora_b: NDArray[np.generic] | None = None
    bias: NDArray[np.generic] | None = None

    def __init__(
        self,
        backbone: ArrayLike,
        lora_a: ArrayLike | None = None,
        lora_b: ArrayLike | None = None,
        bias: ArrayLike | None = None,
    ) -> None:
        backbone_array = _matrix(backbone, "backbone")
        if (lora_a is None) != (lora_b is None):
            raise ValueError("lora_a and lora_b must either both be provided or both be absent")

        a_array = None if lora_a is None else _matrix(lora_a, "lora_a")
        b_array = None if lora_b is None else _matrix(lora_b, "lora_b")
        if a_array is not None and b_array is not None:
            if a_array.shape[0] != backbone_array.shape[0]:
                raise ValueError("lora_a input dimension must match backbone input dimension")
            if a_array.shape[1] != b_array.shape[0]:
                raise ValueError("lora_a rank must match lora_b rank")
            if b_array.shape[1] != backbone_array.shape[1]:
                raise ValueError("lora_b output dimension must match backbone output dimension")

        bias_array = None
        if bias is not None:
            bias_array = np.asarray(bias)
            if bias_array.ndim != 1 or bias_array.shape[0] != backbone_array.shape[1]:
                raise ValueError(
                    "bias must be one-dimensional with length equal to the output dimension"
                )
            if not np.issubdtype(bias_array.dtype, np.number):
                raise TypeError("bias must contain numeric data")
            bias_array = np.array(bias_array, copy=True)

        object.__setattr__(self, "backbone", backbone_array)
        object.__setattr__(self, "lora_a", a_array)
        object.__setattr__(self, "lora_b", b_array)
        object.__setattr__(self, "bias", bias_array)

    @property
    def input_features(self) -> int:
        return int(self.backbone.shape[0])

    @property
    def output_features(self) -> int:
        return int(self.backbone.shape[1])

    @property
    def has_lora(self) -> bool:
        return self.lora_a is not None

    @property
    def lora_rank(self) -> int | None:
        return None if self.lora_a is None else int(self.lora_a.shape[1])


@dataclass(frozen=True, slots=True)
class MultiHeadAttentionWeights:
    q_proj: HybridLinearWeights
    k_proj: HybridLinearWeights
    v_proj: HybridLinearWeights
    out_proj: HybridLinearWeights


@dataclass(frozen=True, slots=True)
class FeedForwardWeights:
    up_proj: HybridLinearWeights
    down_proj: HybridLinearWeights


@dataclass(frozen=True, slots=True)
class TransformerBlockWeights:
    ln1_gamma: NDArray[np.generic]
    ln1_beta: NDArray[np.generic]
    attention: MultiHeadAttentionWeights
    ln2_gamma: NDArray[np.generic]
    ln2_beta: NDArray[np.generic]
    feed_forward: FeedForwardWeights

    def __init__(
        self,
        *,
        ln1_gamma: ArrayLike,
        ln1_beta: ArrayLike,
        attention: MultiHeadAttentionWeights,
        ln2_gamma: ArrayLike,
        ln2_beta: ArrayLike,
        feed_forward: FeedForwardWeights,
    ) -> None:
        def vector(value: ArrayLike, name: str) -> NDArray[np.generic]:
            array = np.asarray(value)
            if array.ndim != 1:
                raise ValueError(f"{name} must be one-dimensional; got {array.shape}")
            if not np.issubdtype(array.dtype, np.number):
                raise TypeError(f"{name} must contain numeric data")
            return np.array(array, copy=True)

        object.__setattr__(self, "ln1_gamma", vector(ln1_gamma, "ln1_gamma"))
        object.__setattr__(self, "ln1_beta", vector(ln1_beta, "ln1_beta"))
        object.__setattr__(self, "attention", attention)
        object.__setattr__(self, "ln2_gamma", vector(ln2_gamma, "ln2_gamma"))
        object.__setattr__(self, "ln2_beta", vector(ln2_beta, "ln2_beta"))
        object.__setattr__(self, "feed_forward", feed_forward)

