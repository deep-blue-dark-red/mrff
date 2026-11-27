"""Tests for Haar orthogonal blocks, FWHT, and SORF operators."""

from __future__ import annotations

import numpy as np
import pytest

from mrff.orthogonal import (
    apply_sorf_projection,
    fwht,
    next_power_of_two,
    sample_haar_orthogonal,
    sample_rademacher_signs,
)


def _dense_hadamard(p: int) -> np.ndarray:
    h = np.array([[1.0]], dtype=np.float64)
    while h.shape[0] < p:
        h = np.block([[h, h], [h, -h]])
    return h


@pytest.mark.parametrize(
    ("n", "expected"),
    [(1, 1), (2, 2), (3, 4), (4, 4), (5, 8), (15, 16), (16, 16), (17, 32)],
)
def test_next_power_of_two(n: int, expected: int) -> None:
    assert next_power_of_two(n) == expected


@pytest.mark.parametrize("dim,n_rows", [(1, 1), (1, 5), (6, 4), (6, 6), (6, 15)])
def test_sample_haar_orthogonal_block_orthonormality(dim: int, n_rows: int) -> None:
    q = sample_haar_orthogonal(dim, n_rows, seed=123)
    assert q.shape == (n_rows, dim)

    # Every row is unit norm.
    row_norms = np.linalg.norm(q, axis=1)
    np.testing.assert_allclose(row_norms, 1.0, atol=1e-12)

    # Every complete or partial block of size <= dim has mutually orthonormal rows.
    for start in range(0, n_rows, dim):
        block = q[start : start + dim]
        gram = block @ block.T
        np.testing.assert_allclose(gram, np.eye(block.shape[0]), atol=1e-12)


def test_sample_haar_orthogonal_isotropy_and_det_balance() -> None:
    dim = 5
    n_blocks = 400
    q = sample_haar_orthogonal(dim, dim * n_blocks, seed=7)
    blocks = q.reshape(n_blocks, dim, dim)

    # Haar measure on O(d) puts equal mass on det=+1 and det=-1.
    dets = np.linalg.det(blocks)
    np.testing.assert_allclose(np.abs(dets), 1.0, atol=1e-12)
    pos_frac = float(np.mean(dets > 0.0))
    assert 0.40 < pos_frac < 0.60

    # Marginal mean ~ 0 and covariance ~ I / d across rows.
    np.testing.assert_allclose(q.mean(axis=0), 0.0, atol=0.05)
    cov = (q.T @ q) / q.shape[0]
    np.testing.assert_allclose(cov, np.eye(dim) / dim, atol=1e-12)


def test_sample_haar_orthogonal_deterministic() -> None:
    q1 = sample_haar_orthogonal(8, 20, seed=99)
    q2 = sample_haar_orthogonal(8, 20, seed=99)
    np.testing.assert_array_equal(q1, q2)


@pytest.mark.parametrize("p", [1, 2, 4, 8, 16, 32])
def test_fwht_matches_dense_hadamard(p: int) -> None:
    rng = np.random.default_rng(p)
    x = rng.standard_normal((7, p))
    h = _dense_hadamard(p)

    np.testing.assert_allclose(fwht(x, normalized=False), x @ h.T, atol=1e-12)
    np.testing.assert_allclose(fwht(x, normalized=True), (x @ h.T) / np.sqrt(p), atol=1e-12)

    # Normalized FWHT is self-inverse and preserves Euclidean norms.
    np.testing.assert_allclose(fwht(fwht(x)), x, atol=1e-12)
    np.testing.assert_allclose(
        np.linalg.norm(fwht(x), axis=-1), np.linalg.norm(x, axis=-1), atol=1e-12
    )


def test_fwht_nd_and_invalid_sizes() -> None:
    rng = np.random.default_rng(0)
    x = rng.standard_normal((2, 3, 8))
    np.testing.assert_allclose(fwht(fwht(x)), x, atol=1e-12)

    with pytest.raises(ValueError, match="power of two"):
        fwht(np.ones((3, 6)))
    with pytest.raises(ValueError, match="at least 1D"):
        fwht(np.array(1.0))


def test_sorf_blocks_are_exact_orthogonal_matrices() -> None:
    p = 16
    n_blocks = 3
    signs = sample_rademacher_signs(n_blocks, 3, p, seed=42)

    # Applying SORF to the identity matrix recovers each explicit p x p block Q^T.
    eye = np.eye(p)
    proj = apply_sorf_projection(eye, signs, n_freqs=n_blocks * p)
    assert proj.shape == (p, n_blocks * p)

    for b in range(n_blocks):
        block = proj[:, b * p : (b + 1) * p]
        np.testing.assert_allclose(block.T @ block, np.eye(p), atol=1e-12)
        np.testing.assert_allclose(block @ block.T, np.eye(p), atol=1e-12)

    # Truncation matches slicing full output.
    trunc = apply_sorf_projection(eye, signs, n_freqs=25)
    np.testing.assert_array_equal(trunc, proj[:, :25])
