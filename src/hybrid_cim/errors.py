"""Domain-specific errors raised by the simulator."""


class CIMError(Exception):
    """Base class for simulator errors."""


class MatrixShapeError(CIMError, ValueError):
    """Raised when operands cannot participate in matrix multiplication."""


class UnknownWeightError(CIMError, KeyError):
    """Raised when a resident weight does not exist on a unit."""


class StaleWeightHandleError(CIMError, RuntimeError):
    """Raised when a handle refers to an older version of a resident weight."""


class TensorLocationError(CIMError, RuntimeError):
    """Raised when an operation receives a tensor at the wrong hardware location."""


class ComponentNotPreparedError(CIMError, RuntimeError):
    """Raised when a composite component is executed before weight preparation."""

