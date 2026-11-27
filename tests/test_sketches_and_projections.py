"""Tests for native C/SIMD FWHT, sparse SORF, projections, Fastfood, sketches, ANOVA, and HKM."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse

from mrff import (
    RBF,
    AnovaRandomFeatures,
    BaseDistribution,
    Cauchy,
    CountSketchFeatures,
    FastfoodFeatures,
    HomogeneousKernelMap,
    Matern,
    OrthogonalRandomProjection,
    RationalQuadratic,
    StructuredOrthogonalRandomFourierFeatures,
    TensorSketchFeatures,
    exact_all_subsets_matrix,
    exact_anova_matrix,
    exact_chi2_matrix,
    exact_hellinger_matrix,
    exact_intersection_matrix,
    exact_polynomial_matrix,
    fwht,
    fwht_backend,
)


@pytest.mark.parametrize("p", [1, 2, 4, 8, 16, 64, 256])
def test_native_c_simd_fwht_matches_numpy(p: int) -> None:
    backend = fwht_backend()
    assert backend.startswith("c:"), f"Expected compiled C/SIMD backend, got {backend!r}"

    rng = np.random.default_rng(p)
    x = rng.standard_normal((13, p))
    np.testing.assert_allclose(
        fwht(x, normalized=True, use_native=True),
        fwht(x, normalized=True, use_native=False),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        fwht(x, normalized=False, use_native=True),
        fwht(x, normalized=False, use_native=False),
        atol=1e-11,
    )


def test_sparse_sorf_matches_dense_sorf_exactly() -> None:
    rng = np.random.default_rng(17)
    X = rng.standard_normal((30, 12))
    X[np.abs(X) < 0.8] = 0.0  # ~60% sparse
    X_csr = sparse.csr_matrix(X)
    X_csc = sparse.csc_matrix(X)

    ls = rng.uniform(0.6, 2.0, size=12)
    sorf = StructuredOrthogonalRandomFourierFeatures(
        n_components=40,
        kernel=Matern(nu=1.5, lengthscale=ls),
        use_cos_sin=True,
        random_state=99,
    ).fit(X_csr)

    Z_dense = sorf.transform(X)
    Z_csr = sorf.transform(X_csr)
    Z_csc = sorf.transform(X_csc)

    np.testing.assert_allclose(Z_csr, Z_dense, atol=1e-12)
    np.testing.assert_allclose(Z_csc, Z_dense, atol=1e-12)


@pytest.mark.parametrize("distribution", ["gaussian", "rademacher", "laplace", "uniform"])
def test_orthogonal_random_projection_exact_multiples_and_unbiased(
    distribution: BaseDistribution,
) -> None:
    rng = np.random.default_rng(21)
    d = 10
    X = rng.standard_normal((16, d))
    K_dot = X @ X.T

    # Exact dot-product preservation when n_components is a multiple of d (unit scaling).
    for m in (d, 2 * d):
        proj = OrthogonalRandomProjection(
            n_components=m,
            mode="orthogonal",
            distribution=distribution,
            scaling="unit",
            random_state=0,
        )
        Z = proj.fit_transform(X)
        np.testing.assert_allclose(Z @ Z.T, K_dot, atol=1e-11)

    # Sparse CSR parity
    X_csr = sparse.csr_matrix(X)
    proj_sp = OrthogonalRandomProjection(
        n_components=12,
        mode="orthogonal",
        distribution=distribution,
        random_state=5,
    ).fit(X_csr)
    np.testing.assert_allclose(proj_sp.transform(X_csr), proj_sp.transform(X), atol=1e-12)


def test_orthogonal_projection_sorf_exact_and_iid_unbiased() -> None:
    rng = np.random.default_rng(33)
    d = 12  # pads to p = 16
    X = rng.standard_normal((14, d))
    K_dot = X @ X.T

    # SORF projection with unit scaling is exact whenever M is a multiple of p = 16.
    sorf_proj = OrthogonalRandomProjection(n_components=32, mode="sorf", random_state=4)
    Z_sorf = sorf_proj.fit_transform(X)
    np.testing.assert_allclose(Z_sorf @ Z_sorf.T, K_dot, atol=1e-11)

    # Sparse CSR parity for SORF projection
    X_csr = sparse.csr_matrix(X)
    np.testing.assert_allclose(sorf_proj.transform(X_csr), Z_sorf, atol=1e-11)

    # Unbiasedness when projecting down (M < d) averaged across trials, and lower variance than iid
    orth_mses, iid_mses = [], []
    orth_sum = np.zeros_like(K_dot)
    n_trials = 250
    for seed in range(n_trials):
        Zo = OrthogonalRandomProjection(
            n_components=6, mode="orthogonal", random_state=seed
        ).fit_transform(X)
        Zi = OrthogonalRandomProjection(
            n_components=6, mode="iid", distribution="rademacher", random_state=seed
        ).fit_transform(X)
        orth_sum += Zo @ Zo.T
        orth_mses.append(float(np.mean((Zo @ Zo.T - K_dot) ** 2)))
        iid_mses.append(float(np.mean((Zi @ Zi.T - K_dot) ** 2)))

    np.testing.assert_allclose(orth_sum / n_trials, K_dot, atol=0.6)
    assert np.mean(orth_mses) < np.mean(iid_mses)


@pytest.mark.parametrize(
    "kernel",
    [
        RBF(lengthscale=1.5),
        Matern(nu=1.5, lengthscale=1.5),
        RationalQuadratic(alpha=2.0, lengthscale=1.5),
        Cauchy(lengthscale=1.5),
    ],
)
def test_fastfood_converges_to_exact_isotropic_kernels(kernel) -> None:
    rng = np.random.default_rng(44)
    X = rng.standard_normal((16, 12))  # pads 12 -> 16
    K_true = kernel.exact_matrix(X)

    ff = FastfoodFeatures(n_components=8000, kernel=kernel, use_cos_sin=True, random_state=7)
    Z = ff.fit_transform(X)
    assert Z.shape == (16, 8000)
    assert np.max(np.abs(Z @ Z.T - K_true)) < 0.06

    # Sparse CSR parity
    X_csr = sparse.csr_matrix(X)
    np.testing.assert_allclose(ff.transform(X_csr), Z, atol=1e-12)


def test_countsketch_and_tensorsketch_unbiased_and_sparse() -> None:
    rng = np.random.default_rng(55)
    X = rng.standard_normal((12, 10)) * 0.4
    Y = rng.standard_normal((8, 10)) * 0.4
    X_csr = sparse.csr_matrix(X)

    # CountSketch: unbiased dot product
    K_dot = X @ Y.T
    cs = CountSketchFeatures(n_components=4096, random_state=1).fit(X)
    Z_x = cs.transform(X)
    Z_y = cs.transform(Y)
    np.testing.assert_allclose(cs.transform(X_csr), Z_x, atol=1e-12)
    assert np.max(np.abs(Z_x @ Z_y.T - K_dot)) < 0.15

    # TensorSketch: unbiased polynomial kernel (gamma * x^T y + coef0)^degree
    for deg, coef0 in [(2, 0.0), (2, 1.0), (3, 0.5)]:
        K_poly = exact_polynomial_matrix(X, Y, degree=deg, gamma=0.8, coef0=coef0)
        ts_single = TensorSketchFeatures(
            n_components=2048, degree=deg, gamma=0.8, coef0=coef0, random_state=3
        ).fit(X)
        np.testing.assert_allclose(ts_single.transform(X_csr), ts_single.transform(X), atol=1e-12)

        approx_sum = np.zeros_like(K_poly)
        n_avg = 20
        for seed in range(n_avg):
            ts = TensorSketchFeatures(
                n_components=2048, degree=deg, gamma=0.8, coef0=coef0, random_state=seed
            ).fit(X)
            approx_sum += ts.transform(X) @ ts.transform(Y).T
        approx = approx_sum / n_avg
        assert float(np.mean(np.abs(approx - K_poly))) < 0.05
        rel_err = np.max(np.abs(approx - K_poly)) / max(1.0, float(np.max(np.abs(K_poly))))
        assert rel_err < 0.08


def test_anova_and_all_subsets_random_features() -> None:
    rng = np.random.default_rng(66)
    X = rng.standard_normal((10, 6)) * 0.35
    Y = rng.standard_normal((7, 6)) * 0.35
    X_csr = sparse.csr_matrix(X)

    for deg in (2, 3):
        K_anova = exact_anova_matrix(X, Y, degree=deg)
        rf = AnovaRandomFeatures(n_components=6000, kernel="anova", degree=deg, random_state=2)
        Zx = rf.fit_transform(X)
        Zy = rf.transform(Y)
        np.testing.assert_allclose(rf.transform(X_csr), Zx, atol=1e-12)
        assert np.max(np.abs(Zx @ Zy.T - K_anova)) < 0.08

    K_all = exact_all_subsets_matrix(X, Y)
    rf_all = AnovaRandomFeatures(n_components=6000, kernel="all_subsets", random_state=4)
    Zx = rf_all.fit_transform(X)
    Zy = rf_all.transform(Y)
    np.testing.assert_allclose(rf_all.transform(X_csr), Zx, atol=1e-12)
    assert np.max(np.abs(Zx @ Zy.T - K_all)) / float(np.max(np.abs(K_all))) < 0.05


def test_homogeneous_kernel_map_chi2_intersection_hellinger() -> None:
    rng = np.random.default_rng(77)
    X = rng.uniform(0.0, 2.0, size=(15, 6))
    X[X < 0.2] = 0.0
    Y = rng.uniform(0.0, 2.0, size=(10, 6))
    X_csr = sparse.csr_matrix(X)

    # Hellinger is exact
    hkm_hel = HomogeneousKernelMap(kernel="hellinger").fit(X)
    Zh_x = hkm_hel.transform(X)
    Zh_y = hkm_hel.transform(Y)
    np.testing.assert_allclose(Zh_x @ Zh_y.T, exact_hellinger_matrix(X, Y), atol=1e-12)
    np.testing.assert_allclose(hkm_hel.transform(X_csr), Zh_x, atol=1e-12)

    # Chi-squared (exponential spectral decay -> very tight with order=3)
    hkm_chi = HomogeneousKernelMap(kernel="chi2", order=3).fit(X)
    Zc_x = hkm_chi.transform(X)
    Zc_y = hkm_chi.transform(Y)
    K_chi = exact_chi2_matrix(X, Y)
    assert np.max(np.abs(Zc_x @ Zc_y.T - K_chi)) / float(np.max(K_chi)) < 0.01
    np.testing.assert_allclose(hkm_chi.transform(X_csr), Zc_x, atol=1e-12)

    # Histogram intersection
    hkm_int = HomogeneousKernelMap(kernel="intersection", order=12).fit(X)
    Zi_x = hkm_int.transform(X)
    Zi_y = hkm_int.transform(Y)
    K_int = exact_intersection_matrix(X, Y)
    assert np.max(np.abs(Zi_x @ Zi_y.T - K_int)) / float(np.max(K_int)) < 0.06
    np.testing.assert_allclose(hkm_int.transform(X_csr), Zi_x, atol=1e-12)

    # Negative inputs rejected
    with pytest.raises(ValueError, match="Negative values in data"):
        HomogeneousKernelMap(kernel="chi2").fit(np.array([[-1.0, 2.0]]))
