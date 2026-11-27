"""Tests for ORF, SORF, RFF, and ArcCosine feature maps."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse

from mrff import (
    RBF,
    ArcCosineOrthogonalFeatures,
    Cauchy,
    Exponential,
    InverseMultiquadric,
    LaplacianL1,
    Matern,
    NotFittedError,
    OrthogonalRandomFourierFeatures,
    RandomFourierFeatures,
    RationalQuadratic,
    StructuredOrthogonalRandomFourierFeatures,
    exact_arccosine_matrix,
)

ISOTROPIC_KERNELS = [
    RBF(lengthscale=1.7),
    Exponential(lengthscale=1.7),
    Matern(nu=1.5, lengthscale=1.7),
    Matern(nu=2.5, lengthscale=1.7),
    Matern(nu=1.0, lengthscale=1.7),
    RationalQuadratic(alpha=2.0, lengthscale=1.7),
    Cauchy(lengthscale=1.7),
    InverseMultiquadric(beta=0.5, lengthscale=1.7),
]


@pytest.mark.parametrize("kernel", ISOTROPIC_KERNELS)
@pytest.mark.parametrize("use_cos_sin", [False, True])
def test_orf_converges_to_exact_kernel(kernel, use_cos_sin: bool) -> None:
    rng = np.random.default_rng(100)
    X = rng.standard_normal((20, 8))
    Y = rng.standard_normal((12, 8))
    K_xx = kernel.exact_matrix(X)
    K_xy = kernel.exact_matrix(X, Y)

    orf = OrthogonalRandomFourierFeatures(
        n_components=8000,
        kernel=kernel,
        use_cos_sin=use_cos_sin,
        random_state=42,
    )
    Zx = orf.fit_transform(X)
    Zy = orf.transform(Y)
    assert Zx.shape == (20, 8000)
    assert Zy.shape == (12, 8000)

    # Diagonal of cos/sin embedding is identically 1.0 by cos^2 + sin^2 = 1.
    if use_cos_sin:
        np.testing.assert_allclose(np.diag(Zx @ Zx.T), 1.0, atol=1e-12)

    assert np.max(np.abs(Zx @ Zx.T - K_xx)) < 0.05
    assert np.max(np.abs(Zx @ Zy.T - K_xy)) < 0.05


@pytest.mark.parametrize("kernel", ISOTROPIC_KERNELS)
@pytest.mark.parametrize("dim", [12, 16])  # tests both zero-padded (12 -> 16) and power-of-two
def test_sorf_converges_to_exact_kernel(kernel, dim: int) -> None:
    rng = np.random.default_rng(200 + dim)
    X = rng.standard_normal((16, dim))
    K_true = kernel.exact_matrix(X)

    sorf = StructuredOrthogonalRandomFourierFeatures(
        n_components=8000,
        kernel=kernel,
        use_cos_sin=True,
        random_state=7,
    )
    Z = sorf.fit_transform(X)
    assert Z.shape == (16, 8000)
    assert sorf.padded_dim_ == 16
    assert np.max(np.abs(Z @ Z.T - K_true)) < 0.06


def test_rff_supports_laplacian_l1_and_isotropic() -> None:
    rng = np.random.default_rng(300)
    X = rng.standard_normal((15, 5))

    for kernel in (LaplacianL1(lengthscale=2.0), Matern(nu=1.5, lengthscale=1.4)):
        rff = RandomFourierFeatures(
            n_components=12000, kernel=kernel, use_cos_sin=True, random_state=13
        )
        Z = rff.fit_transform(X)
        K_true = kernel.exact_matrix(X)
        assert np.max(np.abs(Z @ Z.T - K_true)) < 0.05


def test_orf_and_sorf_refuse_non_isotropic_laplacian_l1() -> None:
    X = np.ones((4, 5))
    with pytest.raises(TypeError, match="not rotationally invariant"):
        OrthogonalRandomFourierFeatures(n_components=16, kernel=LaplacianL1()).fit(X)
    with pytest.raises(TypeError, match="not rotationally invariant"):
        OrthogonalRandomFourierFeatures(n_components=16, kernel="laplacian_l1").fit(X)
    with pytest.raises(TypeError, match="not rotationally invariant"):
        StructuredOrthogonalRandomFourierFeatures(n_components=16, kernel=LaplacianL1()).fit(X)


def test_ard_lengthscales_with_orf_and_sorf() -> None:
    rng = np.random.default_rng(404)
    d = 10
    X = rng.standard_normal((18, d))
    ls = rng.uniform(0.7, 2.3, size=d)
    kernel = Matern(nu=2.5, lengthscale=ls)
    K_true = kernel.exact_matrix(X)

    for cls in (OrthogonalRandomFourierFeatures, StructuredOrthogonalRandomFourierFeatures):
        feat = cls(n_components=6000, kernel=kernel, use_cos_sin=True, random_state=19)
        Z = feat.fit_transform(X)
        assert np.max(np.abs(Z @ Z.T - K_true)) < 0.06


def test_orf_reduces_variance_relative_to_iid_rff() -> None:
    rng = np.random.default_rng(505)
    d = 16
    # Pairs at moderate distance (~sqrt(d) * 0.4) where ORF variance reduction is clear.
    X = rng.standard_normal((24, d)) * 0.4
    kernel = RBF(lengthscale=1.5)
    K_true = kernel.exact_matrix(X)

    n_trials = 60
    n_components = 64
    orf_mses = []
    rff_mses = []
    for trial in range(n_trials):
        Z_orf = OrthogonalRandomFourierFeatures(
            n_components=n_components, kernel=kernel, use_cos_sin=True, random_state=trial
        ).fit_transform(X)
        Z_rff = RandomFourierFeatures(
            n_components=n_components, kernel=kernel, use_cos_sin=True, random_state=trial
        ).fit_transform(X)
        orf_mses.append(float(np.mean((Z_orf @ Z_orf.T - K_true) ** 2)))
        rff_mses.append(float(np.mean((Z_rff @ Z_rff.T - K_true) ** 2)))

    assert np.mean(orf_mses) < np.mean(rff_mses)


def test_arccosine_orthogonal_features_order_0_and_1() -> None:
    rng = np.random.default_rng(606)
    X = rng.standard_normal((16, 6))

    # Order 0 (step): approximates 1 - theta / pi
    K0 = exact_arccosine_matrix(X, order=0)
    Z0 = ArcCosineOrthogonalFeatures(
        n_components=8000, order=0, zero_order_mode="step", random_state=1
    ).fit_transform(X)
    assert np.max(np.abs(Z0 @ Z0.T - K0)) < 0.05

    # Order 0 (sign): approximates 1 - 2 * theta / pi = 2 * K0 - 1
    K0_sign = 2.0 * K0 - 1.0
    Z0_sign = ArcCosineOrthogonalFeatures(
        n_components=8000, order=0, zero_order_mode="sign", random_state=2
    ).fit_transform(X)
    assert np.max(np.abs(Z0_sign @ Z0_sign.T - K0_sign)) < 0.05

    # Order 1 (ReLU): scales as ||x|| ||y||, so check relative max error
    K1 = exact_arccosine_matrix(X, order=1)
    Z1 = ArcCosineOrthogonalFeatures(n_components=8000, order=1, random_state=3).fit_transform(X)
    assert np.max(np.abs(Z1 @ Z1.T - K1)) / np.max(np.abs(K1)) < 0.04


def test_sparse_inputs_and_float32_preservation() -> None:
    rng = np.random.default_rng(808)
    X64 = rng.standard_normal((12, 8))
    X32 = X64.astype(np.float32)
    X_csr = sparse.csr_matrix(X64)

    for cls in (
        OrthogonalRandomFourierFeatures,
        RandomFourierFeatures,
        ArcCosineOrthogonalFeatures,
    ):
        est_dense = cls(n_components=32, random_state=5).fit(X64)
        est_sparse = cls(n_components=32, random_state=5).fit(X_csr)
        np.testing.assert_allclose(
            est_dense.transform(X64), est_sparse.transform(X_csr), atol=1e-12
        )

        Z32 = cls(n_components=32, random_state=5).fit_transform(X32)
        assert Z32.dtype == np.float32

    Z32_sorf = StructuredOrthogonalRandomFourierFeatures(
        n_components=32, random_state=5
    ).fit_transform(X32)
    assert Z32_sorf.dtype == np.float32


def test_edge_cases_and_validation() -> None:
    rng = np.random.default_rng(707)

    # n_components < d, == d, > d, and d == 1
    for d, m in [(1, 1), (1, 10), (10, 4), (10, 10), (10, 23)]:
        X = rng.standard_normal((5, d))
        Z = OrthogonalRandomFourierFeatures(n_components=m, random_state=0).fit_transform(X)
        assert Z.shape == (5, m)
        Z_s = StructuredOrthogonalRandomFourierFeatures(
            n_components=m, random_state=0
        ).fit_transform(X)
        assert Z_s.shape == (5, m)

    # Determinism with int, RandomState, and Generator
    X = rng.standard_normal((6, 5))
    z1 = OrthogonalRandomFourierFeatures(
        n_components=20, kernel="matern32", random_state=123
    ).fit_transform(X)
    z2 = OrthogonalRandomFourierFeatures(
        n_components=20, kernel="matern32", random_state=123
    ).fit_transform(X)
    np.testing.assert_array_equal(z1, z2)

    z3 = OrthogonalRandomFourierFeatures(
        n_components=20, kernel="matern32", random_state=np.random.RandomState(123)
    ).fit_transform(X)
    z4 = OrthogonalRandomFourierFeatures(
        n_components=20, kernel="matern32", random_state=np.random.RandomState(123)
    ).fit_transform(X)
    np.testing.assert_array_equal(z3, z4)

    # Error handling
    orf = OrthogonalRandomFourierFeatures(n_components=12, random_state=0).fit(np.ones((3, 4)))
    unfitted = OrthogonalRandomFourierFeatures(n_components=8)
    with pytest.raises(NotFittedError):
        unfitted.transform(X)

    with pytest.raises(ValueError, match="features"):
        orf.transform(np.ones((2, 5)))

    with pytest.raises(ValueError, match="even"):
        OrthogonalRandomFourierFeatures(n_components=7, use_cos_sin=True).fit(X)

    with pytest.raises(ValueError, match="NaN"):
        orf.transform(np.array([[1.0, np.nan, 0.0, 0.0]]))

    with pytest.raises(ValueError, match="2D"):
        OrthogonalRandomFourierFeatures(n_components=8).fit(np.ones(5))
