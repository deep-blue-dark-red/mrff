"""Input validation and RNG utilities for mrff."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

LengthscaleLike = float | int | Sequence[float] | NDArray[np.floating]
SeedLike = int | np.random.SeedSequence | np.random.RandomState | np.random.Generator | None


def resolve_rng(seed: SeedLike) -> np.random.Generator:
    """Return a NumPy Generator from a seed, RandomState, SeedSequence, Generator, or None."""
    if isinstance(seed, np.random.Generator):
        return seed
    if isinstance(seed, np.random.RandomState):
        # Advance the legacy RandomState deterministically to seed a PCG64 Generator.
        entropy = int.from_bytes(seed.bytes(16), "little")
        return np.random.default_rng(entropy)
    return np.random.default_rng(seed)


def validate_positive_int(value: int, name: str) -> int:
    """Validate that a parameter is a strictly positive integer (excluding bool)."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}.")
    ival = int(value)
    if ival <= 0:
        raise ValueError(f"{name} must be positive, got {ival}.")
    return ival


def validate_positive_float(value: float, name: str) -> float:
    """Validate that a parameter is a strictly positive finite float."""
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise TypeError(f"{name} must be a real number, got {type(value).__name__}.")
    fval = float(value)
    if not np.isfinite(fval) or fval <= 0.0:
        raise ValueError(f"{name} must be finite and positive, got {fval}.")
    return fval


def normalize_lengthscale(lengthscale: LengthscaleLike) -> float | tuple[float, ...]:
    """Normalize a scalar or 1D ARD lengthscale into an immutable, validated form."""
    if isinstance(lengthscale, bool):
        raise TypeError("lengthscale must be a positive float or 1D sequence of positive floats.")
    if isinstance(lengthscale, (int, float, np.integer, np.floating)):
        return validate_positive_float(float(lengthscale), "lengthscale")

    arr = np.asarray(lengthscale, dtype=np.float64)
    if arr.ndim == 0:
        return validate_positive_float(float(arr.item()), "lengthscale")
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError(
            "lengthscale must be a positive scalar or non-empty 1D sequence, "
            f"got shape {arr.shape}."
        )
    if not np.all(np.isfinite(arr)) or np.any(arr <= 0.0):
        raise ValueError("All lengthscale entries must be finite and strictly positive.")
    return tuple(float(v) for v in arr)


def resolve_lengthscale_vector(
    lengthscale: LengthscaleLike,
    n_features: int,
) -> NDArray[np.float64]:
    """Expand a scalar or 1D ARD lengthscale to a float64 vector of length n_features."""
    normalized = normalize_lengthscale(lengthscale)
    if isinstance(normalized, (int, float)):
        return np.full(n_features, float(normalized), dtype=np.float64)
    if len(normalized) != n_features:
        raise ValueError(
            f"ARD lengthscale length ({len(normalized)}) does not match "
            f"input feature count ({n_features})."
        )
    return np.asarray(normalized, dtype=np.float64)


def as_2d_float_array(
    X: ArrayLike,
    *,
    name: str = "X",
    expected_features: int | None = None,
    allow_empty_rows: bool = False,
) -> NDArray[np.float64]:
    """Validate and convert an array-like input to a finite 2D float64 ndarray."""
    arr = np.asarray(X, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2D array, got {arr.ndim}D with shape {arr.shape}.")
    if arr.shape[1] == 0:
        raise ValueError(f"{name} must have at least 1 feature column, got shape {arr.shape}.")
    if not allow_empty_rows and arr.shape[0] == 0:
        raise ValueError(f"{name} must have at least 1 sample row, got shape {arr.shape}.")
    if expected_features is not None and arr.shape[1] != expected_features:
        raise ValueError(f"{name} has {arr.shape[1]} features, but expected {expected_features}.")
    if arr.size > 0 and not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN or infinite values.")
    return arr
