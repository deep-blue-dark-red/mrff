"""Orthogonal, structured orthogonal (SORF), and base random matrix operators.

Provides:
- Haar-distributed random orthogonal blocks via sign-corrected QR decomposition
  (Mezzadri, 2007; Yu et al., NeurIPS 2016).
- Zero-mean unit-variance base matrix samplers (``gaussian``, ``rademacher``,
  ``laplace``, ``uniform``) for orthogonal and non-Gaussian projections.
- SIMD-accelerated Fast Walsh-Hadamard Transform (FWHT) via ``mrff._fwht_c``
  with transparent pure-NumPy fallback.
- Fused dense and zero-densification sparse-CSR SORF projections in O(p log p).
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import sparse

from mrff._validation import SeedLike, resolve_rng, validate_positive_int

try:
    from mrff import _fwht_c as _native
except ImportError:  # pragma: no cover - pure-Python fallback when C ext is absent
    _native = None

BaseDistribution = Literal["gaussian", "rademacher", "laplace", "uniform"]


def fwht_backend() -> str:
    """Return the active FWHT backend (e.g. ``'c:neon'``, ``'c:avx2'``, or ``'numpy'``)."""
    if _native is not None:
        return f"c:{_native.simd_backend()}"
    return "numpy"


def next_power_of_two(n: int) -> int:
    """Return the smallest power of two greater than or equal to positive integer n."""
    n = validate_positive_int(n, "n")
    return 1 << (n - 1).bit_length()


def sample_base_matrix(
    distribution: BaseDistribution,
    shape: tuple[int, ...],
    seed: SeedLike = None,
) -> NDArray[np.float64]:
    """Sample an array of i.i.d. zero-mean, unit-variance random entries.

    Supported distributions:
    - ``"gaussian"``: ``N(0, 1)``
    - ``"rademacher"``: ``{-1, +1}`` with equal probability
    - ``"laplace"``: ``Laplace(0, 1 / sqrt(2))`` (variance 1)
    - ``"uniform"``: ``Uniform(-sqrt(3), +sqrt(3))`` (variance 1)
    """
    rng = resolve_rng(seed)
    if distribution == "gaussian":
        return rng.standard_normal(shape)
    if distribution == "rademacher":
        return rng.choice(np.array([-1.0, 1.0], dtype=np.float64), size=shape)
    if distribution == "laplace":
        return rng.laplace(loc=0.0, scale=1.0 / np.sqrt(2.0), size=shape)
    if distribution == "uniform":
        bound = float(np.sqrt(3.0))
        return rng.uniform(-bound, bound, size=shape)
    raise ValueError(
        f"Unknown distribution {distribution!r}. "
        "Supported: 'gaussian', 'rademacher', 'laplace', 'uniform'."
    )


def sample_haar_orthogonal(
    dim: int,
    n_rows: int,
    seed: SeedLike = None,
    *,
    distribution: BaseDistribution = "gaussian",
) -> NDArray[np.float64]:
    """Sample stacked ``d x d`` orthogonal blocks via sign-corrected QR decomposition.

    When ``distribution="gaussian"`` (default), each block follows the exact
    Haar measure on ``O(dim)`` (Mezzadri, 2007). Non-Gaussian base matrices
    (``"rademacher"``, ``"laplace"``, ``"uniform"``) are also supported for
    orthogonalized projections.
    """
    dim = validate_positive_int(dim, "dim")
    n_rows = validate_positive_int(n_rows, "n_rows")
    rng = resolve_rng(seed)

    n_blocks = (n_rows + dim - 1) // dim
    blocks: list[NDArray[np.float64]] = []
    for _ in range(n_blocks):
        g = sample_base_matrix(distribution, (dim, dim), rng)
        q, r = np.linalg.qr(g)
        diag_sign = np.where(np.diag(r) >= 0.0, 1.0, -1.0)
        blocks.append(q * diag_sign)

    return np.vstack(blocks)[:n_rows]


def _fwht_numpy(y: NDArray[np.float64], *, normalized: bool) -> NDArray[np.float64]:
    p = y.shape[-1]
    if p == 1:
        return y
    orig_shape = y.shape
    h = 1
    while h < p:
        view = y.reshape(*orig_shape[:-1], p // (2 * h), 2, h)
        a = view[..., 0, :].copy()
        b = view[..., 1, :]
        view[..., 0, :] = a + b
        view[..., 1, :] = a - b
        h *= 2
    if normalized:
        y /= np.sqrt(float(p))
    return y


def fwht(x: ArrayLike, *, normalized: bool = True, use_native: bool = True) -> NDArray[np.float64]:
    """Compute the Fast Walsh-Hadamard Transform along the last axis.

    Uses the native C/SIMD extension when available (``use_native=True``),
    falling back transparently to pure NumPy.
    """
    y = np.array(x, dtype=np.float64, copy=True, order="C")
    if y.ndim == 0:
        raise ValueError("Input to fwht must be at least 1D.")
    p = y.shape[-1]
    if p < 1 or (p & (p - 1)) != 0:
        raise ValueError(f"Last dimension of fwht input must be a power of two, got {p}.")
    if p == 1 or y.size == 0:
        return y

    if use_native and _native is not None:
        n_rows = y.size // p
        _native.fwht_inplace_f64(y, n_rows, p, bool(normalized))
        return y

    return _fwht_numpy(y, normalized=normalized)


def sample_rademacher_signs(
    n_blocks: int,
    n_stacks: int,
    padded_dim: int,
    seed: SeedLike = None,
    *,
    distribution: BaseDistribution = "rademacher",
) -> NDArray[np.float64]:
    """Sample independent diagonal matrices for SORF blocks."""
    n_blocks = validate_positive_int(n_blocks, "n_blocks")
    n_stacks = validate_positive_int(n_stacks, "n_stacks")
    padded_dim = validate_positive_int(padded_dim, "padded_dim")
    return sample_base_matrix(distribution, (n_blocks, n_stacks, padded_dim), seed)


def apply_sorf_projection(
    X_padded: NDArray[np.float64],
    signs: NDArray[np.float64],
    n_freqs: int,
) -> NDArray[np.float64]:
    """Apply stacked SORF orthogonal transforms ``H D_k ... H D_1`` to ``X_padded``."""
    n_samples, p = X_padded.shape
    n_blocks, n_stacks, sign_p = signs.shape
    if sign_p != p:
        raise ValueError(f"signs last dim ({sign_p}) does not match X_padded dim ({p}).")
    if n_freqs > n_blocks * p:
        raise ValueError(f"n_freqs ({n_freqs}) exceeds available block rows ({n_blocks * p}).")

    out = np.empty((n_samples, n_freqs), dtype=np.float64)
    if n_samples == 0:
        return out

    if _native is not None:
        x_c = np.ascontiguousarray(X_padded, dtype=np.float64)
        inv_ls = np.ones(p, dtype=np.float64)
        signs_c = np.ascontiguousarray(signs, dtype=np.float64)
        radii = np.ones(n_freqs, dtype=np.float64)
        _native.sorf_dense_f64(
            x_c, inv_ls, signs_c, radii, out, n_samples, p, p, n_blocks, n_stacks, n_freqs
        )
        return out

    offset = 0
    for b in range(n_blocks):
        u = X_padded
        for s in range(n_stacks):
            u = fwht(u * signs[b, s], normalized=True)
        take = min(p, n_freqs - offset)
        out[:, offset : offset + take] = u[:, :take]
        offset += take
        if offset >= n_freqs:
            break
    return out


def apply_sorf_scaled(
    X: Any,
    inv_lengthscale: NDArray[np.float64],
    signs: NDArray[np.float64],
    radii: NDArray[np.float64],
    n_freqs: int,
) -> NDArray[np.float64]:
    """Apply fused scaled SORF projection to dense or sparse CSR/CSC input ``X``.

    For sparse matrices, nonzeros are scattered directly into a ``p``-element
    working buffer per row without materializing a dense ``(n_samples, p)``
    matrix.
    """
    n_samples, d = int(X.shape[0]), int(X.shape[1])
    n_blocks, n_stacks, p = signs.shape
    if n_freqs > n_blocks * p:
        raise ValueError(f"n_freqs ({n_freqs}) exceeds available block rows ({n_blocks * p}).")

    out = np.empty((n_samples, n_freqs), dtype=np.float64)
    if n_samples == 0:
        return out

    inv_ls_c = np.ascontiguousarray(inv_lengthscale, dtype=np.float64)
    signs_c = np.ascontiguousarray(signs, dtype=np.float64)
    radii_c = np.ascontiguousarray(radii, dtype=np.float64)

    if sparse.issparse(X):
        csr = X.tocsr()
        data_c = np.ascontiguousarray(csr.data, dtype=np.float64)
        idx_c = np.ascontiguousarray(csr.indices, dtype=np.int64)
        ptr_c = np.ascontiguousarray(csr.indptr, dtype=np.int64)
        if _native is not None:
            _native.sorf_csr_f64(
                data_c,
                idx_c,
                ptr_c,
                inv_ls_c,
                signs_c,
                radii_c,
                out,
                n_samples,
                d,
                p,
                n_blocks,
                n_stacks,
                n_freqs,
            )
            return out

        # Pure-NumPy chunked sparse fallback without full densification
        work = np.empty((1, p), dtype=np.float64)
        for i in range(n_samples):
            start, end = int(ptr_c[i]), int(ptr_c[i + 1])
            cols = idx_c[start:end]
            vals = data_c[start:end] * inv_ls_c[cols]
            offset = 0
            for b in range(n_blocks):
                work.fill(0.0)
                if cols.size > 0:
                    np.add.at(work[0], cols, vals * signs_c[b, 0, cols])
                u = fwht(work, normalized=True, use_native=False)
                for s in range(1, n_stacks):
                    u = fwht(u * signs_c[b, s], normalized=True, use_native=False)
                take = min(p, n_freqs - offset)
                out[i, offset : offset + take] = u[0, :take] * radii_c[offset : offset + take]
                offset += take
                if offset >= n_freqs:
                    break
        return out

    x_c = np.ascontiguousarray(X, dtype=np.float64)
    if _native is not None:
        _native.sorf_dense_f64(
            x_c, inv_ls_c, signs_c, radii_c, out, n_samples, d, p, n_blocks, n_stacks, n_freqs
        )
        return out

    xs = x_c * inv_ls_c[None, :]
    if p > d:
        x_pad = np.zeros((n_samples, p), dtype=np.float64)
        x_pad[:, :d] = xs
    else:
        x_pad = xs
    return apply_sorf_projection(x_pad, signs_c, n_freqs) * radii_c[None, :]
