"""Tests for kernel specifications, exact Gram matrices, and mathematical identities."""

from __future__ import annotations

import math

import numpy as np
import pytest

from mrff.kernels import (
    RBF,
    Cauchy,
    Exponential,
    InverseMultiquadric,
    LaplacianL1,
    Matern,
    RationalQuadratic,
    exact_arccosine_matrix,
    make_kernel,
)

ALL_SHIFT_INVARIANT_KERNELS = [
    RBF(lengthscale=1.3),
    Exponential(lengthscale=1.3),
    Matern(nu=0.5, lengthscale=1.3),
    Matern(nu=1.5, lengthscale=1.3),
    Matern(nu=2.5, lengthscale=1.3),
    Matern(nu=1.0, lengthscale=1.3),  # exercises scipy Bessel branch
    RationalQuadratic(alpha=2.0, lengthscale=1.3),
    Cauchy(lengthscale=1.3),
    InverseMultiquadric(beta=0.5, lengthscale=1.3),
    LaplacianL1(lengthscale=1.3),
]


@pytest.mark.parametrize("kernel", ALL_SHIFT_INVARIANT_KERNELS)
def test_exact_matrix_symmetry_unit_diag_psd(kernel) -> None:
    rng = np.random.default_rng(11)
    X = rng.standard_normal((25, 6))
    Y = rng.standard_normal((10, 6))

    K = kernel.exact_matrix(X)
    assert K.shape == (25, 25)
    np.testing.assert_allclose(K, K.T, atol=1e-13)
    np.testing.assert_allclose(np.diag(K), 1.0, atol=1e-13)

    eigvals = np.linalg.eigvalsh(K)
    assert eigvals.min() >= -1e-11

    K_xy = kernel.exact_matrix(X, Y)
    assert K_xy.shape == (25, 10)
    np.testing.assert_allclose(K_xy, kernel.exact_matrix(Y, X).T, atol=1e-13)


def test_kernel_analytical_identities_and_limits() -> None:
    rng = np.random.default_rng(3)
    X = rng.standard_normal((15, 5))
    ell = 1.6

    # Exponential == Matern(nu=0.5)
    np.testing.assert_allclose(
        Exponential(ell).exact_matrix(X),
        Matern(nu=0.5, lengthscale=ell).exact_matrix(X),
        atol=1e-14,
    )

    # InverseMultiquadric(beta=1.0) == Cauchy
    np.testing.assert_allclose(
        InverseMultiquadric(beta=1.0, lengthscale=ell).exact_matrix(X),
        Cauchy(lengthscale=ell).exact_matrix(X),
        atol=1e-14,
    )

    # RationalQuadratic(alpha=1.0, ell / sqrt(2)) == Cauchy(ell)
    np.testing.assert_allclose(
        RationalQuadratic(alpha=1.0, lengthscale=ell / math.sqrt(2.0)).exact_matrix(X),
        Cauchy(lengthscale=ell).exact_matrix(X),
        atol=1e-14,
    )

    # Matern(nu -> inf) -> RBF
    np.testing.assert_allclose(
        Matern(nu=100.0, lengthscale=ell).exact_matrix(X),
        RBF(lengthscale=ell).exact_matrix(X),
        atol=2e-2,
    )

    # RationalQuadratic(alpha -> inf) -> RBF
    np.testing.assert_allclose(
        RationalQuadratic(alpha=1e5, lengthscale=ell).exact_matrix(X),
        RBF(lengthscale=ell).exact_matrix(X),
        atol=1e-4,
    )


def test_ard_lengthscale_matches_rescaled_inputs() -> None:
    rng = np.random.default_rng(5)
    X = rng.standard_normal((16, 4))
    ls = [0.5, 1.2, 2.5, 0.8]

    for cls in (RBF, Exponential, Cauchy, LaplacianL1):
        k_ard = cls(lengthscale=ls)
        k_unit = cls(lengthscale=1.0)
        np.testing.assert_allclose(
            k_ard.exact_matrix(X),
            k_unit.exact_matrix(X / np.asarray(ls)),
            atol=1e-13,
        )


def test_exact_arccosine_matrix_properties() -> None:
    rng = np.random.default_rng(9)
    X = rng.standard_normal((12, 4))

    K0 = exact_arccosine_matrix(X, order=0)
    np.testing.assert_allclose(K0, K0.T, atol=1e-13)
    np.testing.assert_allclose(np.diag(K0), 1.0, atol=1e-13)
    assert float(np.min(np.linalg.eigvalsh(K0))) >= -1e-11

    K1 = exact_arccosine_matrix(X, order=1)
    np.testing.assert_allclose(K1, K1.T, atol=1e-13)
    np.testing.assert_allclose(np.diag(K1), np.sum(X * X, axis=1), atol=1e-12)
    assert float(np.min(np.linalg.eigvalsh(K1))) >= -1e-11

    # Zero vector handling
    Z = np.zeros((2, 4))
    np.testing.assert_array_equal(exact_arccosine_matrix(Z, order=0), np.zeros((2, 2)))
    np.testing.assert_array_equal(exact_arccosine_matrix(Z, order=1), np.zeros((2, 2)))


def test_make_kernel_aliases_and_validation() -> None:
    assert make_kernel("rbf", lengthscale=2.0) == RBF(2.0)
    assert make_kernel("gaussian", lengthscale=2.0) == RBF(2.0)
    assert make_kernel("exponential", lengthscale=2.0) == Exponential(2.0)
    assert make_kernel("matern12", lengthscale=2.0) == Exponential(2.0)
    assert make_kernel("matern32", lengthscale=2.0) == Matern(nu=1.5, lengthscale=2.0)
    assert make_kernel("matern52", lengthscale=2.0) == Matern(nu=2.5, lengthscale=2.0)
    assert make_kernel("matern", nu=3.5, lengthscale=2.0) == Matern(nu=3.5, lengthscale=2.0)
    assert make_kernel("rq", alpha=3.0, lengthscale=2.0) == RationalQuadratic(3.0, 2.0)
    assert make_kernel("cauchy", lengthscale=2.0) == Cauchy(2.0)
    assert make_kernel("imq", beta=0.75, lengthscale=2.0) == InverseMultiquadric(0.75, 2.0)
    assert make_kernel("laplacian_l1", lengthscale=2.0) == LaplacianL1(2.0)

    k = Matern(nu=2.5, lengthscale=1.5)
    assert make_kernel(k) is k

    with pytest.raises(ValueError, match="Unknown kernel"):
        make_kernel("nonexistent")
    with pytest.raises(ValueError, match="positive"):
        RBF(lengthscale=0.0)
    with pytest.raises(ValueError, match="positive"):
        Matern(nu=-1.0)
    with pytest.raises(ValueError, match="positive"):
        RationalQuadratic(alpha=0.0)
    with pytest.raises(ValueError, match="positive"):
        InverseMultiquadric(beta=-0.5)
