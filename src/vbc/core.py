"""Core operations from the published Vector-Based Comparison procedure."""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

Preprocess = Literal["none", "l2", "minmax", "zscore"]


@dataclass(frozen=True)
class VBCResult:
    """Decomposition of one endpoint relative to a primary endpoint."""

    cosine_similarity: float
    angle_radians: float
    parallel: NDArray[np.float64]
    perpendicular: NDArray[np.float64]

    @property
    def angle_degrees(self) -> float:
        return float(np.degrees(self.angle_radians))


def _vector(values: ArrayLike, name: str) -> NDArray[np.float64]:
    result = np.asarray(values, dtype=float)
    if result.ndim != 1 or result.size < 2:
        raise ValueError(f"{name} must be a one-dimensional vector with at least two observations")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} contains a missing or non-finite value")
    return result


def preprocess(values: ArrayLike, method: Preprocess = "none") -> NDArray[np.float64]:
    """Preprocess an endpoint before VBC.

    ``none`` preserves the supplied values and their angle. ``l2`` makes the
    endpoint unit length without changing its direction or angle. ``minmax``
    maps the endpoint to [0, 1] but changes the origin and can alter angles.
    ``zscore`` centers and scales by the sample standard deviation (ddof=1),
    so cosine similarity equals sample Pearson correlation.
    """
    vector = _vector(values, "values")
    if method == "none":
        return vector
    if method == "zscore":
        scale = np.max(np.abs(vector))
        if scale == 0 or np.all(vector == vector[0]):
            raise ValueError("z-score preprocessing requires a non-constant endpoint")
        scaled = vector / scale
        centered = scaled - scaled.mean()
        standard_deviation = np.std(centered, ddof=1)
        if standard_deviation == 0:
            raise ValueError("z-score preprocessing requires a non-constant endpoint")
        return centered / standard_deviation
    if method == "l2":
        scale = np.max(np.abs(vector))
        if scale == 0:
            raise ValueError("L2 preprocessing requires a non-zero endpoint")
        scaled = vector / scale
        return scaled / np.linalg.norm(scaled)
    if method != "minmax":
        raise ValueError(f"unknown preprocessing method: {method!r}")
    # Scaling first avoids overflow in max - min for very large mixed-sign data.
    scale = np.max(np.abs(vector))
    if scale == 0:
        raise ValueError("min-max preprocessing requires a non-constant endpoint")
    scaled = vector / scale
    minimum = scaled.min()
    value_range = scaled.max() - minimum
    if value_range == 0:
        raise ValueError("min-max preprocessing requires a non-constant endpoint")
    return (scaled - minimum) / value_range


def cosine_angle(primary: ArrayLike, endpoint: ArrayLike) -> tuple[float, float]:
    """Return cosine similarity and angle (radians) between two endpoint vectors."""
    a = _vector(primary, "primary")
    b = _vector(endpoint, "endpoint")
    if a.shape != b.shape:
        raise ValueError("primary and endpoint must have the same number of observations")
    if not np.any(a) or not np.any(b):
        raise ValueError("the angle is undefined for a zero-norm vector")
    unit_a = preprocess(a, "l2")
    unit_b = preprocess(b, "l2")
    cosine = float(np.clip(np.dot(unit_a, unit_b), -1.0, 1.0))
    # Half-angle identity retains precision near both parallel and antiparallel vectors.
    angle = 2.0 * np.arctan2(
        np.linalg.norm(unit_a - unit_b), np.linalg.norm(unit_a + unit_b)
    )
    return cosine, float(angle)


def decompose(
    primary: ArrayLike,
    endpoint: ArrayLike,
    *,
    preprocessing: Preprocess = "none",
) -> VBCResult:
    """Orthogonally decompose one endpoint relative to the primary endpoint.

    The parallel component is the vector projection of B onto A. The
    perpendicular component is the residual B - projection_A(B), so its dot
    product with A is zero to numerical precision. Its Euclidean magnitude is
    ``norm(B) * sin(theta)``, consistent with the geometric VBC construction.
    """
    a = preprocess(primary, preprocessing)
    b = preprocess(endpoint, preprocessing)
    cosine, angle = cosine_angle(a, b)
    unit_a = preprocess(a, "l2")
    b_scale = np.max(np.abs(b))
    scaled_b = b / b_scale
    scaled_parallel = float(np.dot(unit_a, scaled_b)) * unit_a
    with np.errstate(over="ignore", invalid="ignore"):
        parallel = scaled_parallel * b_scale
        perpendicular = (scaled_b - scaled_parallel) * b_scale
    if not np.all(np.isfinite(parallel)) or not np.all(np.isfinite(perpendicular)):
        raise ValueError("decomposition exceeds floating-point range; use L2 preprocessing")
    return VBCResult(
        cosine_similarity=cosine,
        angle_radians=angle,
        parallel=parallel,
        perpendicular=perpendicular,
    )


def transform_endpoints(
    endpoints: Mapping[str, ArrayLike],
    primary: str,
    *,
    preprocessing: Preprocess = "none",
) -> dict[str, VBCResult]:
    """Decompose every non-primary endpoint relative to ``primary``."""
    if primary not in endpoints:
        raise KeyError(f"primary endpoint {primary!r} was not provided")
    if len(endpoints) < 2:
        raise ValueError("VBC requires a primary endpoint and at least one other endpoint")
    names: Iterable[str] = (name for name in endpoints if name != primary)
    return {
        name: decompose(endpoints[primary], endpoints[name], preprocessing=preprocessing)
        for name in names
    }
