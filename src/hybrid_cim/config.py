"""Validated model, mapping, workload, and experiment configuration objects."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class TransformerConfig:
    embed_dim: int = 768
    num_heads: int = 12
    mlp_dim: int = 3072
    lora_rank: int = 4
    layer_norm_epsilon: float = 1e-5

    def __post_init__(self) -> None:
        for name in ("embed_dim", "num_heads", "mlp_dim", "lora_rank"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.embed_dim % self.num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        if self.layer_norm_epsilon <= 0:
            raise ValueError("layer_norm_epsilon must be positive")

    @property
    def head_dim(self) -> int:
        return self.embed_dim // self.num_heads


@dataclass(frozen=True, slots=True)
class PrecisionConfig:
    activation_bits: int = 8
    weight_bits: int = 8
    accumulator_bits: int = 32

    def __post_init__(self) -> None:
        for name in ("activation_bits", "weight_bits", "accumulator_bits"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")


@dataclass(frozen=True, slots=True)
class MappingConfig:
    """Operator-to-hardware mapping for one Transformer Encoder Block."""

    q_proj: str = "hybrid"
    k_proj: str = "hybrid"
    v_proj: str = "hybrid"
    out_proj: str = "hybrid"
    ff_up: str = "hybrid"
    ff_down: str = "hybrid"
    qk_matmul: str = "sram"
    pv_matmul: str = "sram"
    nonlinear: str = "digital"

    def __post_init__(self) -> None:
        for name in ("q_proj", "k_proj", "v_proj", "out_proj", "ff_up", "ff_down"):
            value = getattr(self, name)
            if value not in {"rram", "hybrid"}:
                raise ValueError(f"{name} must be 'rram' or 'hybrid'; got {value!r}")
        if self.qk_matmul != "sram" or self.pv_matmul != "sram":
            raise ValueError("the current simulator supports dynamic attention matmul only on SRAM")
        if self.nonlinear != "digital":
            raise ValueError("the current simulator supports nonlinear operations only on digital")

    @classmethod
    def halora(cls) -> "MappingConfig":
        """HaLoRA-style profile: LoRA on Q/K/V and FFN up/down projections."""

        return cls(
            q_proj="hybrid",
            k_proj="hybrid",
            v_proj="hybrid",
            out_proj="rram",
            ff_up="hybrid",
            ff_down="hybrid",
        )


@dataclass(frozen=True, slots=True)
class SimulatorConfig:
    model: TransformerConfig = field(default_factory=TransformerConfig)
    precision: PrecisionConfig = field(default_factory=PrecisionConfig)
    mapping: MappingConfig = field(default_factory=MappingConfig)


@dataclass(frozen=True, slots=True)
class WorkloadConfig:
    batch_size: int = 1
    tokens: int = 197

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.tokens <= 0:
            raise ValueError("tokens must be positive")


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    name: str = "vit-base-block-halora"
    simulator: SimulatorConfig = field(
        default_factory=lambda: SimulatorConfig(mapping=MappingConfig.halora())
    )
    workload: WorkloadConfig = field(default_factory=WorkloadConfig)
    seed: int = 2026

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            raise ValueError("experiment name must be non-empty")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExperimentConfig":
        simulator_data = data.get("simulator", {})
        simulator = SimulatorConfig(
            model=TransformerConfig(**simulator_data.get("model", {})),
            precision=PrecisionConfig(**simulator_data.get("precision", {})),
            mapping=MappingConfig(**simulator_data.get("mapping", {})),
        )
        return cls(
            name=data.get("name", "vit-base-block-halora"),
            simulator=simulator,
            workload=WorkloadConfig(**data.get("workload", {})),
            seed=data.get("seed", 2026),
        )

    @classmethod
    def load_json(cls, path: str | Path) -> "ExperimentConfig":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(data)

    def save_json(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return target

