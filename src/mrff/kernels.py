"""Kernel specifications, exact Gram matrices, and spectral/radial samplers.

Every isotropic shift-invariant kernel k(x, y) = kappa(||Lambda^{-1}(x - y)||_2)
has a spherically symmetric spectral density p(omega) in unit-lengthscale space,
so its random frequencies decompose as omega = r * q with q ~ Uniform(S^{d-1})
and radial length r = ||omega||_2 independent of q.

Non-isotropic product kernels such as the L1 Laplacian kernel do NOT have
spherically symmetric spectral densities for d > 1 and therefore inherit from
``ShiftInvariantKernel`` rather than ``IsotropicKernel``.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from mrff._validation import (
    LengthscaleLike,
    SeedLike,
    as_2d_float_array,
    normalize_lengthscale,
    resolve_lengthscale_vector,
    resolve_rng,
    validate_positive_float,
    validate_positive_int,
)


def _pairwise_scaled_l2(
    X: ArrayLike,
    Y: ArrayLike | None,
    lengthscale: LengthscaleLike,
) -> NDArray[np.float64]:
    """Compute pairwise scaled Euclidean distances ||Lambda^{-1}(x - y)||_2."""
    X_arr = as_2d_float_array(X, name="X")
    ls = resolve_lengthscale_vector(lengthscale, X_arr.shape[1])
    Xs = X_arr / ls
    if Y is None:
        Ys = Xs
    else:
        Y_arr = as_2d_float_array(Y, name="Y", expected_features=X_arr.shape[1])
        Ys = Y_arr / ls

    # Numerically stable squared distances: ||a||^2 + ||b||^2 - 2 a.b, clipped at 0.
    x_sq = np.einsum("ij,ij->i", Xs, Xs)[:, None]
    y_sq = np.einsum("ij,ij->i", Ys, Ys)[None, :]
    sq_dist = np.maximum(x_sq + y_sq - 2.0 * (Xs @ Ys.T), 0.0)
    if Y is None:
        np.fill_diagonal(sq_dist, 0.0)
    return np.sqrt(sq_dist)


def _pairwise_scaled_l1(
    X: ArrayLike,
    Y: ArrayLike | None,
    lengthscale: LengthscaleLike,
) -> NDArray[np.float64]:
    """Compute pairwise scaled L1 distances ||Lambda^{-1}(x - y)||_1."""
    X_arr = as_2d_float_array(X, name="X")
    ls = resolve_lengthscale_vector(lengthscale, X_arr.shape[1])
    Xs = X_arr / ls
    if Y is None:
        Ys = Xs
    else:
        Y_arr = as_2d_float_array(Y, name="Y", expected_features=X_arr.shape[1])
        Ys = Y_arr / ls
    return np.sum(np.abs(Xs[:, None, :] - Ys[None, :, :]), axis=-1)


class ShiftInvariantKernel(ABC):
    """Base class for shift-invariant positive-definite kernels k(x, y) = k(x - y)."""

    @property
    @abstractmethod
    def lengthscale(self) -> LengthscaleLike:
        """Scalar or per-feature ARD lengthscale."""

    @abstractmethod
    def exact_matrix(
        self,
        X: ArrayLike,
        Y: ArrayLike | None = None,
    ) -> NDArray[np.float64]:
        """Evaluate the exact kernel Gram matrix K(X, Y) (or K(X, X) if Y is None)."""

    @abstractmethod
    def sample_frequencies(
        self,
        dim: int,
        n_freqs: int,
        seed: SeedLike = None,
    ) -> NDArray[np.float64]:
        """Sample i.i.d. spectral frequencies omega ~ p(omega) of shape (n_freqs, dim)."""

    def lengthscale_vector(self, dim: int) -> NDArray[np.float64]:
        """Return the lengthscale expanded to shape ``(dim,)``."""
        return resolve_lengthscale_vector(self.lengthscale, dim)


class IsotropicKernel(ShiftInvariantKernel, ABC):
    """Shift-invariant kernel that is isotropic after ARD lengthscale scaling.

    All subclasses support Orthogonal Random Fourier Features (ORF) and
    Structured Orthogonal Random Fourier Features (SORF).
    """

    @abstractmethod
    def sample_radii(
        self,
        dim: int,
        n_freqs: int,
        seed: SeedLike = None,
    ) -> NDArray[np.float64]:
        """Sample unit-lengthscale radial norms ||omega||_2 of shape ``(n_freqs,)``."""

    @abstractmethod
    def _sample_unit_frequencies(
        self,
        dim: int,
        n_freqs: int,
        rng: np.random.Generator,
    ) -> NDArray[np.float64]:
        """Sample i.i.d. unit-lengthscale spectral frequencies of shape ``(n_freqs, dim)``."""

    def sample_frequencies(
        self,
        dim: int,
        n_freqs: int,
        seed: SeedLike = None,
    ) -> NDArray[np.float64]:
        dim = validate_positive_int(dim, "dim")
        n_freqs = validate_positive_int(n_freqs, "n_freqs")
        rng = resolve_rng(seed)
        ls = self.lengthscale_vector(dim)
        return self._sample_unit_frequencies(dim, n_freqs, rng) / ls[None, :]


@dataclass(frozen=True)
class RBF(IsotropicKernel):
    """Gaussian / Radial Basis Function kernel: ``exp(-0.5 * ||(x - y) / l||_2^2)``."""

    lengthscale: LengthscaleLike = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "lengthscale", normalize_lengthscale(self.lengthscale))

    def exact_matrix(
        self,
        X: ArrayLike,
        Y: ArrayLike | None = None,
    ) -> NDArray[np.float64]:
        r = _pairwise_scaled_l2(X, Y, self.lengthscale)
        return np.exp(-0.5 * (r * r))

    def sample_radii(
        self,
        dim: int,
        n_freqs: int,
        seed: SeedLike = None,
    ) -> NDArray[np.float64]:
        dim = validate_positive_int(dim, "dim")
        n_freqs = validate_positive_int(n_freqs, "n_freqs")
        rng = resolve_rng(seed)
        return np.sqrt(rng.chisquare(df=dim, size=n_freqs))

    def _sample_unit_frequencies(
        self,
        dim: int,
        n_freqs: int,
        rng: np.random.Generator,
    ) -> NDArray[np.float64]:
        return rng.standard_normal((n_freqs, dim))


@dataclass(frozen=True)
class Matern(IsotropicKernel):
    """Matérn kernel of smoothness ``nu > 0``.

    Special cases:
    - ``nu = 0.5``: isotropic exponential kernel ``exp(-||(x - y) / l||_2)``
    - ``nu = 1.5``: once-differentiable Matérn-3/2
    - ``nu = 2.5``: twice-differentiable Matérn-5/2
    - ``nu -> inf``: recovers RBF

    Spectral distribution in unit-lengthscale space is multivariate Student-t
    with ``2 * nu`` degrees of freedom and identity scale matrix.
    """

    nu: float = 1.5
    lengthscale: LengthscaleLike = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "nu", validate_positive_float(self.nu, "nu"))
        object.__setattr__(self, "lengthscale", normalize_lengthscale(self.lengthscale))

    def exact_matrix(
        self,
        X: ArrayLike,
        Y: ArrayLike | None = None,
    ) -> NDArray[np.float64]:
        r = _pairwise_scaled_l2(X, Y, self.lengthscale)
        if math.isclose(self.nu, 0.5, rel_tol=0.0, abs_tol=1e-12):
            return np.exp(-r)
        if math.isclose(self.nu, 1.5, rel_tol=0.0, abs_tol=1e-12):
            s = math.sqrt(3.0) * r
            return (1.0 + s) * np.exp(-s)
        if math.isclose(self.nu, 2.5, rel_tol=0.0, abs_tol=1e-12):
            s = math.sqrt(5.0) * r
            return (1.0 + s + (s * s) / 3.0) * np.exp(-s)

        try:
            from scipy.special import gamma as sp_gamma
            from scipy.special import kv
        except ImportError as exc:
            raise ImportError(
                "Evaluating the exact Matérn Gram matrix for nu not in {0.5, 1.5, 2.5} "
                "requires scipy (install with `pip install mrff[scipy]`). "
                "Random feature sampling works for any nu > 0 with NumPy alone."
            ) from exc

        s = math.sqrt(2.0 * self.nu) * r
        out = np.ones_like(s)
        nz = s > 0.0
        s_nz = s[nz]
        out[nz] = (
            (2.0 ** (1.0 - self.nu) / float(sp_gamma(self.nu)))
            * (s_nz**self.nu)
            * kv(self.nu, s_nz)
        )
        return out

    def sample_radii(
        self,
        dim: int,
        n_freqs: int,
        seed: SeedLike = None,
    ) -> NDArray[np.float64]:
        dim = validate_positive_int(dim, "dim")
        n_freqs = validate_positive_int(n_freqs, "n_freqs")
        rng = resolve_rng(seed)
        chi = np.sqrt(rng.chisquare(df=dim, size=n_freqs))
        g = rng.gamma(shape=self.nu, scale=1.0 / self.nu, size=n_freqs)
        return chi / np.sqrt(g)

    def _sample_unit_frequencies(
        self,
        dim: int,
        n_freqs: int,
        rng: np.random.Generator,
    ) -> NDArray[np.float64]:
        z = rng.standard_normal((n_freqs, dim))
        g = rng.gamma(shape=self.nu, scale=1.0 / self.nu, size=n_freqs)
        return z / np.sqrt(g)[:, None]


@dataclass(frozen=True)
class Exponential(IsotropicKernel):
    """Isotropic (L2) exponential kernel ``exp(-||(x - y) / l||_2)`` (Matérn-1/2)."""

    lengthscale: LengthscaleLike = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "lengthscale", normalize_lengthscale(self.lengthscale))

    def exact_matrix(
        self,
        X: ArrayLike,
        Y: ArrayLike | None = None,
    ) -> NDArray[np.float64]:
        r = _pairwise_scaled_l2(X, Y, self.lengthscale)
        return np.exp(-r)

    def sample_radii(
        self,
        dim: int,
        n_freqs: int,
        seed: SeedLike = None,
    ) -> NDArray[np.float64]:
        dim = validate_positive_int(dim, "dim")
        n_freqs = validate_positive_int(n_freqs, "n_freqs")
        rng = resolve_rng(seed)
        chi = np.sqrt(rng.chisquare(df=dim, size=n_freqs))
        g = rng.gamma(shape=0.5, scale=2.0, size=n_freqs)
        return chi / np.sqrt(g)

    def _sample_unit_frequencies(
        self,
        dim: int,
        n_freqs: int,
        rng: np.random.Generator,
    ) -> NDArray[np.float64]:
        z = rng.standard_normal((n_freqs, dim))
        g = rng.gamma(shape=0.5, scale=2.0, size=n_freqs)
        return z / np.sqrt(g)[:, None]


@dataclass(frozen=True)
class RationalQuadratic(IsotropicKernel):
    """Rational Quadratic kernel: ``(1 + ||(x - y) / l||_2^2 / (2 * alpha))^(-alpha)``.

    Exact scale mixture of RBF kernels over inverse-lengthscale precision
    ``tau ~ Gamma(alpha, scale=1/alpha)``. Recovers RBF as ``alpha -> inf``.
    """

    alpha: float = 1.0
    lengthscale: LengthscaleLike = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "alpha", validate_positive_float(self.alpha, "alpha"))
        object.__setattr__(self, "lengthscale", normalize_lengthscale(self.lengthscale))

    def exact_matrix(
        self,
        X: ArrayLike,
        Y: ArrayLike | None = None,
    ) -> NDArray[np.float64]:
        r = _pairwise_scaled_l2(X, Y, self.lengthscale)
        return np.power(1.0 + (r * r) / (2.0 * self.alpha), -self.alpha)

    def sample_radii(
        self,
        dim: int,
        n_freqs: int,
        seed: SeedLike = None,
    ) -> NDArray[np.float64]:
        dim = validate_positive_int(dim, "dim")
        n_freqs = validate_positive_int(n_freqs, "n_freqs")
        rng = resolve_rng(seed)
        chi = np.sqrt(rng.chisquare(df=dim, size=n_freqs))
        tau = rng.gamma(shape=self.alpha, scale=1.0 / self.alpha, size=n_freqs)
        return chi * np.sqrt(tau)

    def _sample_unit_frequencies(
        self,
        dim: int,
        n_freqs: int,
        rng: np.random.Generator,
    ) -> NDArray[np.float64]:
        z = rng.standard_normal((n_freqs, dim))
        tau = rng.gamma(shape=self.alpha, scale=1.0 / self.alpha, size=n_freqs)
        return z * np.sqrt(tau)[:, None]


@dataclass(frozen=True)
class Cauchy(IsotropicKernel):
    """Isotropic (L2) Cauchy kernel: ``1 / (1 + ||(x - y) / l||_2^2)``."""

    lengthscale: LengthscaleLike = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "lengthscale", normalize_lengthscale(self.lengthscale))

    def exact_matrix(
        self,
        X: ArrayLike,
        Y: ArrayLike | None = None,
    ) -> NDArray[np.float64]:
        r = _pairwise_scaled_l2(X, Y, self.lengthscale)
        return 1.0 / (1.0 + r * r)

    def sample_radii(
        self,
        dim: int,
        n_freqs: int,
        seed: SeedLike = None,
    ) -> NDArray[np.float64]:
        dim = validate_positive_int(dim, "dim")
        n_freqs = validate_positive_int(n_freqs, "n_freqs")
        rng = resolve_rng(seed)
        chi = np.sqrt(rng.chisquare(df=dim, size=n_freqs))
        tau = rng.exponential(scale=1.0, size=n_freqs)
        return chi * np.sqrt(2.0 * tau)

    def _sample_unit_frequencies(
        self,
        dim: int,
        n_freqs: int,
        rng: np.random.Generator,
    ) -> NDArray[np.float64]:
        z = rng.standard_normal((n_freqs, dim))
        tau = rng.exponential(scale=1.0, size=n_freqs)
        return z * np.sqrt(2.0 * tau)[:, None]


@dataclass(frozen=True)
class InverseMultiquadric(IsotropicKernel):
    """Inverse Multiquadric kernel: ``(1 + ||(x - y) / c||_2^2)^(-beta)``.

    Here ``lengthscale`` plays the role of the scale parameter ``c > 0``.
    Special case ``beta = 1.0`` coincides with the isotropic ``Cauchy`` kernel.
    """

    beta: float = 0.5
    lengthscale: LengthscaleLike = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "beta", validate_positive_float(self.beta, "beta"))
        object.__setattr__(self, "lengthscale", normalize_lengthscale(self.lengthscale))

    def exact_matrix(
        self,
        X: ArrayLike,
        Y: ArrayLike | None = None,
    ) -> NDArray[np.float64]:
        r = _pairwise_scaled_l2(X, Y, self.lengthscale)
        return np.power(1.0 + r * r, -self.beta)

    def sample_radii(
        self,
        dim: int,
        n_freqs: int,
        seed: SeedLike = None,
    ) -> NDArray[np.float64]:
        dim = validate_positive_int(dim, "dim")
        n_freqs = validate_positive_int(n_freqs, "n_freqs")
        rng = resolve_rng(seed)
        chi = np.sqrt(rng.chisquare(df=dim, size=n_freqs))
        tau = rng.gamma(shape=self.beta, scale=1.0, size=n_freqs)
        return chi * np.sqrt(2.0 * tau)

    def _sample_unit_frequencies(
        self,
        dim: int,
        n_freqs: int,
        rng: np.random.Generator,
    ) -> NDArray[np.float64]:
        z = rng.standard_normal((n_freqs, dim))
        tau = rng.gamma(shape=self.beta, scale=1.0, size=n_freqs)
        return z * np.sqrt(2.0 * tau)[:, None]


@dataclass(frozen=True)
class LaplacianL1(ShiftInvariantKernel):
    """Non-isotropic L1 Laplacian kernel: ``exp(-||(x - y) / l||_1)``.

    Its spectral density factors as i.i.d. Cauchy along each coordinate and is
    NOT rotationally invariant for ``d > 1``. Use with ``RandomFourierFeatures``
    rather than orthogonal random features.
    """

    lengthscale: LengthscaleLike = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "lengthscale", normalize_lengthscale(self.lengthscale))

    def exact_matrix(
        self,
        X: ArrayLike,
        Y: ArrayLike | None = None,
    ) -> NDArray[np.float64]:
        r1 = _pairwise_scaled_l1(X, Y, self.lengthscale)
        return np.exp(-r1)

    def sample_frequencies(
        self,
        dim: int,
        n_freqs: int,
        seed: SeedLike = None,
    ) -> NDArray[np.float64]:
        dim = validate_positive_int(dim, "dim")
        n_freqs = validate_positive_int(n_freqs, "n_freqs")
        rng = resolve_rng(seed)
        ls = self.lengthscale_vector(dim)
        return rng.standard_cauchy(size=(n_freqs, dim)) / ls[None, :]


def exact_arccosine_matrix(
    X: ArrayLike,
    Y: ArrayLike | None = None,
    *,
    order: int = 1,
) -> NDArray[np.float64]:
    """Compute the exact Arc-Cosine kernel matrix of order 0 or 1.

    - ``order = 0``: ``1 - theta / pi`` (step / Heaviside kernel)
    - ``order = 1``: ``(||x|| ||y|| / pi) * (sin(theta) + (pi - theta) * cos(theta))``
      (ReLU kernel)
    """
    if order not in (0, 1):
        raise ValueError(f"Supported arc-cosine orders are 0 and 1, got {order}.")
    X_arr = as_2d_float_array(X, name="X")
    Y_arr = X_arr if Y is None else as_2d_float_array(Y, name="Y", expected_features=X_arr.shape[1])

    x_norm = np.linalg.norm(X_arr, axis=1, keepdims=True)
    y_norm = x_norm if Y is None else np.linalg.norm(Y_arr, axis=1, keepdims=True)
    denom = x_norm @ y_norm.T

    dot = X_arr @ Y_arr.T
    with np.errstate(divide="ignore", invalid="ignore"):
        cos_theta = np.where(denom > 0.0, np.clip(dot / denom, -1.0, 1.0), 1.0)
    theta = np.arccos(cos_theta)

    if order == 0:
        out = 1.0 - theta / math.pi
        return np.where(denom > 0.0, out, 0.0)

    return (denom / math.pi) * (np.sin(theta) + (math.pi - theta) * cos_theta)


def exact_polynomial_matrix(
    X: ArrayLike,
    Y: ArrayLike | None = None,
    *,
    degree: int = 2,
    gamma: float = 1.0,
    coef0: float = 0.0,
) -> NDArray[np.float64]:
    """Compute the exact polynomial kernel matrix ``(gamma * X @ Y.T + coef0) ** degree``."""
    degree = validate_positive_int(degree, "degree")
    X_arr = as_2d_float_array(X, name="X")
    Y_arr = X_arr if Y is None else as_2d_float_array(Y, name="Y", expected_features=X_arr.shape[1])
    return np.power(float(gamma) * (X_arr @ Y_arr.T) + float(coef0), degree)


def exact_anova_matrix(
    X: ArrayLike,
    Y: ArrayLike | None = None,
    *,
    degree: int = 2,
) -> NDArray[np.float64]:
    """Compute the exact ANOVA kernel matrix of given degree via DP over features."""
    degree = validate_positive_int(degree, "degree")
    X_arr = as_2d_float_array(X, name="X")
    Y_arr = X_arr if Y is None else as_2d_float_array(Y, name="Y", expected_features=X_arr.shape[1])
    n_x, d = X_arr.shape
    n_y = Y_arr.shape[0]
    if degree > d:
        return np.zeros((n_x, n_y), dtype=np.float64)

    dp = np.zeros((degree + 1, n_x, n_y), dtype=np.float64)
    dp[0] = 1.0
    for j in range(d):
        prod = np.outer(X_arr[:, j], Y_arr[:, j])
        for k in range(min(degree, j + 1), 0, -1):
            dp[k] += dp[k - 1] * prod
    return dp[degree]


def exact_all_subsets_matrix(
    X: ArrayLike,
    Y: ArrayLike | None = None,
) -> NDArray[np.float64]:
    """Compute the exact all-subsets kernel ``prod_j (1 + x_j y_j)``."""
    X_arr = as_2d_float_array(X, name="X")
    Y_arr = X_arr if Y is None else as_2d_float_array(Y, name="Y", expected_features=X_arr.shape[1])
    out = np.ones((X_arr.shape[0], Y_arr.shape[0]), dtype=np.float64)
    for j in range(X_arr.shape[1]):
        out *= 1.0 + np.outer(X_arr[:, j], Y_arr[:, j])
    return out


def exact_chi2_matrix(
    X: ArrayLike,
    Y: ArrayLike | None = None,
) -> NDArray[np.float64]:
    """Compute the exact additive chi-squared kernel ``sum_j 2 x_j y_j / (x_j + y_j)``."""
    X_arr = as_2d_float_array(X, name="X")
    Y_arr = X_arr if Y is None else as_2d_float_array(Y, name="Y", expected_features=X_arr.shape[1])
    if np.any(X_arr < 0.0) or np.any(Y_arr < 0.0):
        raise ValueError("Additive chi-squared kernel requires non-negative inputs.")
    num = 2.0 * X_arr[:, None, :] * Y_arr[None, :, :]
    den = X_arr[:, None, :] + Y_arr[None, :, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(den > 0.0, num / den, 0.0)
    return np.sum(ratio, axis=-1)


def exact_intersection_matrix(
    X: ArrayLike,
    Y: ArrayLike | None = None,
) -> NDArray[np.float64]:
    """Compute the exact histogram intersection kernel ``sum_j min(x_j, y_j)``."""
    X_arr = as_2d_float_array(X, name="X")
    Y_arr = X_arr if Y is None else as_2d_float_array(Y, name="Y", expected_features=X_arr.shape[1])
    if np.any(X_arr < 0.0) or np.any(Y_arr < 0.0):
        raise ValueError("Histogram intersection kernel requires non-negative inputs.")
    return np.sum(np.minimum(X_arr[:, None, :], Y_arr[None, :, :]), axis=-1)


def exact_hellinger_matrix(
    X: ArrayLike,
    Y: ArrayLike | None = None,
) -> NDArray[np.float64]:
    """Compute the exact Hellinger kernel ``sum_j sqrt(x_j * y_j)``."""
    X_arr = as_2d_float_array(X, name="X")
    Y_arr = X_arr if Y is None else as_2d_float_array(Y, name="Y", expected_features=X_arr.shape[1])
    if np.any(X_arr < 0.0) or np.any(Y_arr < 0.0):
        raise ValueError("Hellinger kernel requires non-negative inputs.")
    return np.sqrt(X_arr) @ np.sqrt(Y_arr).T


KernelLike = str | ShiftInvariantKernel


def make_kernel(
    kernel: KernelLike = "rbf",
    *,
    lengthscale: LengthscaleLike = 1.0,
    nu: float = 1.5,
    alpha: float = 1.0,
    beta: float = 0.5,
) -> ShiftInvariantKernel:
    """Resolve a kernel name or kernel instance into a ``ShiftInvariantKernel``."""
    if isinstance(kernel, ShiftInvariantKernel):
        return kernel
    if not isinstance(kernel, str):
        raise TypeError(
            f"kernel must be a ShiftInvariantKernel instance or string name, "
            f"got {type(kernel).__name__}."
        )

    key = kernel.strip().lower().replace("-", "_")
    ls = normalize_lengthscale(lengthscale)
    if key in ("rbf", "gaussian"):
        return RBF(lengthscale=ls)
    if key in ("exponential", "matern12", "matern_12", "l2_laplacian"):
        return Exponential(lengthscale=ls)
    if key in ("matern32", "matern_32"):
        return Matern(nu=1.5, lengthscale=ls)
    if key in ("matern52", "matern_52"):
        return Matern(nu=2.5, lengthscale=ls)
    if key == "matern":
        return Matern(nu=nu, lengthscale=ls)
    if key in ("rational_quadratic", "rq"):
        return RationalQuadratic(alpha=alpha, lengthscale=ls)
    if key in ("cauchy", "cauchy_l2"):
        return Cauchy(lengthscale=ls)
    if key in ("imq", "inverse_multiquadric"):
        return InverseMultiquadric(beta=beta, lengthscale=ls)
    if key in ("laplacian_l1", "l1_laplacian", "laplacian"):
        return LaplacianL1(lengthscale=ls)

    raise ValueError(
        f"Unknown kernel name {kernel!r}. Supported names: "
        "'rbf', 'exponential', 'matern', 'matern12', 'matern32', 'matern52', "
        "'rational_quadratic', 'cauchy', 'imq', 'laplacian_l1'."
    )
