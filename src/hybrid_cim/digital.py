"""Operator-level model of the non-CIM digital processing unit."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike

from .errors import TensorLocationError
from .events import DigitalComputeEvent, StatsCollector, StorageEvent
from .tensor import TensorPacket


class DigitalProcessor:
    """Execute digital tensor operations and record primitive operation counts."""

    def __init__(
        self,
        name: str,
        stats: StatsCollector,
        *,
        output_bit_width: int = 32,
    ) -> None:
        if output_bit_width <= 0:
            raise ValueError("output_bit_width must be positive")
        self.name = name
        self.stats = stats
        self.output_bit_width = output_bit_width
        self._parameter_versions: dict[str, int] = {}

    def load_parameter(
        self,
        name: str,
        value: ArrayLike,
        *,
        bit_width: int | None = None,
    ) -> np.ndarray:
        """Load a small digital parameter and record its one-time setup cost."""

        if not name or not name.strip():
            raise ValueError("parameter name must be non-empty")
        array = np.asarray(value)
        if not np.issubdtype(array.dtype, np.number):
            raise TypeError(f"parameter must contain numeric data; got {array.dtype}")
        stored = np.array(array, copy=True)
        stored_bit_width = bit_width or self.output_bit_width
        if stored_bit_width <= 0:
            raise ValueError("parameter bit width must be positive")
        version = self._parameter_versions.get(name, 0) + 1
        self._parameter_versions[name] = version
        self.stats.record_storage(
            StorageEvent(
                unit=self.name,
                action="load_parameter",
                weight_name=name,
                version=version,
                shape=tuple(stored.shape),
                elements=int(stored.size),
                bit_width=stored_bit_width,
                bytes=(int(stored.size) * stored_bit_width + 7) // 8,
            )
        )
        return stored

    def _require_local(self, *tensors: TensorPacket) -> None:
        for tensor in tensors:
            if tensor.location != self.name:
                raise TensorLocationError(
                    f"tensor {tensor.name!r} is at {tensor.location!r}; "
                    f"transfer it to {self.name!r} before digital execution"
                )

    def _finish(
        self,
        *,
        name: str,
        data: ArrayLike,
        op_name: str,
        op_type: str,
        input_shapes: tuple[tuple[int, ...], ...],
        operation_counts: dict[str, int],
        output_bit_width: int | None,
    ) -> TensorPacket:
        output = TensorPacket(
            name,
            data,
            output_bit_width or self.output_bit_width,
            self.name,
        )
        self.stats.record_digital(
            DigitalComputeEvent(
                unit=self.name,
                op_name=op_name,
                op_type=op_type,
                input_shapes=input_shapes,
                output_shape=output.shape,
                output_elements=output.elements,
                operation_counts=operation_counts,
            )
        )
        return output

    def _add_like(
        self,
        lhs: TensorPacket,
        rhs: TensorPacket,
        *,
        name: str,
        op_name: str,
        op_type: str,
        output_bit_width: int | None,
    ) -> TensorPacket:
        self._require_local(lhs, rhs)
        result = np.add(lhs.data, rhs.data)
        return self._finish(
            name=name,
            data=result,
            op_name=op_name,
            op_type=op_type,
            input_shapes=(lhs.shape, rhs.shape),
            operation_counts={"add": int(result.size)},
            output_bit_width=output_bit_width,
        )

    def add(
        self,
        lhs: TensorPacket,
        rhs: TensorPacket,
        *,
        name: str,
        op_name: str = "add",
        output_bit_width: int | None = None,
    ) -> TensorPacket:
        return self._add_like(
            lhs,
            rhs,
            name=name,
            op_name=op_name,
            op_type="add",
            output_bit_width=output_bit_width,
        )

    def residual_add(
        self,
        main: TensorPacket,
        residual: TensorPacket,
        *,
        name: str,
        op_name: str = "residual_add",
        output_bit_width: int | None = None,
    ) -> TensorPacket:
        return self._add_like(
            main,
            residual,
            name=name,
            op_name=op_name,
            op_type="residual_add",
            output_bit_width=output_bit_width,
        )

    def bias_add(
        self,
        tensor: TensorPacket,
        bias: ArrayLike,
        *,
        name: str,
        op_name: str = "bias_add",
        output_bit_width: int | None = None,
    ) -> TensorPacket:
        self._require_local(tensor)
        bias_array = np.asarray(bias)
        result = np.add(tensor.data, bias_array)
        return self._finish(
            name=name,
            data=result,
            op_name=op_name,
            op_type="bias_add",
            input_shapes=(tensor.shape, tuple(bias_array.shape)),
            operation_counts={"add": int(result.size)},
            output_bit_width=output_bit_width,
        )

    def scale(
        self,
        tensor: TensorPacket,
        factor: float,
        *,
        name: str,
        op_name: str = "scale",
        output_bit_width: int | None = None,
    ) -> TensorPacket:
        self._require_local(tensor)
        result = tensor.data * factor
        return self._finish(
            name=name,
            data=result,
            op_name=op_name,
            op_type="scale",
            input_shapes=(tensor.shape,),
            operation_counts={"multiply": tensor.elements},
            output_bit_width=output_bit_width,
        )

    def softmax(
        self,
        tensor: TensorPacket,
        *,
        axis: int = -1,
        name: str,
        op_name: str = "softmax",
        output_bit_width: int | None = None,
    ) -> TensorPacket:
        self._require_local(tensor)
        axis = axis % tensor.data.ndim
        axis_size = tensor.shape[axis]
        groups = tensor.elements // axis_size
        shifted = tensor.data - np.max(tensor.data, axis=axis, keepdims=True)
        exponentials = np.exp(shifted)
        result = exponentials / np.sum(exponentials, axis=axis, keepdims=True)
        return self._finish(
            name=name,
            data=result,
            op_name=op_name,
            op_type="softmax",
            input_shapes=(tensor.shape,),
            operation_counts={
                "compare": groups * max(axis_size - 1, 0),
                "add_or_subtract": tensor.elements + groups * max(axis_size - 1, 0),
                "exp": tensor.elements,
                "divide": tensor.elements,
            },
            output_bit_width=output_bit_width,
        )

    def layer_norm(
        self,
        tensor: TensorPacket,
        *,
        gamma: ArrayLike | None = None,
        beta: ArrayLike | None = None,
        epsilon: float = 1e-5,
        axis: int = -1,
        name: str,
        op_name: str = "layer_norm",
        output_bit_width: int | None = None,
    ) -> TensorPacket:
        self._require_local(tensor)
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")
        axis = axis % tensor.data.ndim
        axis_size = tensor.shape[axis]
        groups = tensor.elements // axis_size
        mean = np.mean(tensor.data, axis=axis, keepdims=True)
        centered = tensor.data - mean
        variance = np.mean(centered * centered, axis=axis, keepdims=True)
        result = centered / np.sqrt(variance + epsilon)

        counts = {
            "add_or_subtract": (
                groups * max(axis_size - 1, 0)
                + tensor.elements
                + groups * max(axis_size - 1, 0)
                + groups
            ),
            "multiply": tensor.elements,
            "divide": groups * 2 + tensor.elements,
            "sqrt": groups,
        }
        if gamma is not None:
            result = result * np.asarray(gamma)
            counts["multiply"] += tensor.elements
        if beta is not None:
            result = result + np.asarray(beta)
            counts["add_or_subtract"] += tensor.elements

        return self._finish(
            name=name,
            data=result,
            op_name=op_name,
            op_type="layer_norm",
            input_shapes=(tensor.shape,),
            operation_counts=counts,
            output_bit_width=output_bit_width,
        )

    def gelu(
        self,
        tensor: TensorPacket,
        *,
        name: str,
        op_name: str = "gelu",
        output_bit_width: int | None = None,
    ) -> TensorPacket:
        self._require_local(tensor)
        x = tensor.data
        result = 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))
        n = tensor.elements
        return self._finish(
            name=name,
            data=result,
            op_name=op_name,
            op_type="gelu",
            input_shapes=(tensor.shape,),
            operation_counts={"multiply": 6 * n, "add": 2 * n, "tanh": n},
            output_bit_width=output_bit_width,
        )

    def concat(
        self,
        tensors: Sequence[TensorPacket],
        *,
        axis: int,
        name: str,
        op_name: str = "concat",
        output_bit_width: int | None = None,
    ) -> TensorPacket:
        if not tensors:
            raise ValueError("concat requires at least one tensor")
        self._require_local(*tensors)
        result = np.concatenate([tensor.data for tensor in tensors], axis=axis)
        return self._finish(
            name=name,
            data=result,
            op_name=op_name,
            op_type="concat",
            input_shapes=tuple(tensor.shape for tensor in tensors),
            operation_counts={"data_rearrangement": int(result.size)},
            output_bit_width=output_bit_width,
        )

    def transpose(
        self,
        tensor: TensorPacket,
        axes: tuple[int, ...],
        *,
        name: str,
        op_name: str = "transpose",
        output_bit_width: int | None = None,
    ) -> TensorPacket:
        self._require_local(tensor)
        result = np.transpose(tensor.data, axes)
        return self._finish(
            name=name,
            data=result,
            op_name=op_name,
            op_type="transpose",
            input_shapes=(tensor.shape,),
            operation_counts={"data_rearrangement": tensor.elements},
            output_bit_width=output_bit_width,
        )

    def reshape(
        self,
        tensor: TensorPacket,
        shape: tuple[int, ...],
        *,
        name: str,
        op_name: str = "reshape",
        output_bit_width: int | None = None,
    ) -> TensorPacket:
        self._require_local(tensor)
        result = np.reshape(tensor.data, shape)
        return self._finish(
            name=name,
            data=result,
            op_name=op_name,
            op_type="reshape",
            input_shapes=(tensor.shape,),
            operation_counts={"metadata_or_layout": tensor.elements},
            output_bit_width=output_bit_width,
        )
