"""Public API for the Hybrid CIM simulator."""

from .components import FeedForward, HybridLinear, MultiHeadAttention
from .analysis import ExperimentAnalysis, analyze_execution_report
from .config import (
    ExperimentConfig,
    MappingConfig,
    PrecisionConfig,
    SimulatorConfig,
    TransformerConfig,
    WorkloadConfig,
)
from .data_mover import DataMover
from .digital import DigitalProcessor
from .events import (
    ComputeEvent,
    DigitalComputeEvent,
    StatsCollector,
    StorageEvent,
    TransferEvent,
)
from .experiment import ExperimentRunResult, generate_block_weights, run_experiment
from .models import TransformerEncoderBlock
from .reporting import SimulationReport, SimulationResult
from .simulator import HybridCIMBlockSimulator
from .system import HybridCIMSystem
from .tensor import TensorPacket
from .units.base import WeightHandle
from .units.rram import RRAMCIM
from .units.sram import SRAMCIM
from .weights import (
    FeedForwardWeights,
    HybridLinearWeights,
    MultiHeadAttentionWeights,
    TransformerBlockWeights,
)

__all__ = [
    "ComputeEvent",
    "DataMover",
    "DigitalComputeEvent",
    "DigitalProcessor",
    "ExperimentAnalysis",
    "ExperimentConfig",
    "ExperimentRunResult",
    "FeedForward",
    "FeedForwardWeights",
    "HybridCIMSystem",
    "HybridCIMBlockSimulator",
    "HybridLinear",
    "HybridLinearWeights",
    "MappingConfig",
    "MultiHeadAttention",
    "MultiHeadAttentionWeights",
    "PrecisionConfig",
    "RRAMCIM",
    "SRAMCIM",
    "StatsCollector",
    "StorageEvent",
    "TensorPacket",
    "SimulatorConfig",
    "SimulationReport",
    "SimulationResult",
    "TransformerConfig",
    "TransformerBlockWeights",
    "TransformerEncoderBlock",
    "TransferEvent",
    "WeightHandle",
    "WorkloadConfig",
    "analyze_execution_report",
    "generate_block_weights",
    "run_experiment",
]

