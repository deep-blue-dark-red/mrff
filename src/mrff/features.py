"""Scikit-learn-compatible random, orthogonal, and structured orthogonal feature maps.

All transformers inherit from ``ClassNamePrefixFeaturesOutMixin``,
``TransformerMixin``, and ``BaseEstimator``, follow scikit-learn's
``fit``/``transform``/``random_state``/``validate_data`` conventions, and pass
``sklearn.utils.estimator_checks.check_estimator``.
"""

from __future__ import annotations

from typing import Any, Literal, Self

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.base import BaseEstimator, ClassNamePrefixFeaturesOutMixin, TransformerMixin
from sklearn.exceptions import NotFittedError
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.utils.validation import check_is_fitted, check_random_state, validate_data

from mrff._validation import (
    LengthscaleLike,
    SeedLike,
    resolve_rng,
    validate_positive_int,
)
from mrff.kernels import (
    IsotropicKernel,
    KernelLike,
    ShiftInvariantKernel,
    make_kernel,
)
from mrff.orthogonal import (
    BaseDistribution,
    apply_sorf_scaled,
    next_power_of_two,
    sample_base_matrix,
    sample_haar_orthogonal,
    sample_rademacher_signs,
)


def _resolve_random_state(random_state: SeedLike) -> np.random.Generator:
    """Validate and resolve a scikit-learn ``random_state`` into a NumPy Generator."""
    if isinstance(random_state, (np.random.Generator, np.random.SeedSequence)):
        return resolve_rng(random_state)
    rs = check_random_state(random_state)
    return resolve_rng(rs)


def _resolve_n_freqs(n_components: int, use_cos_sin: bool) -> int:
    n_components = validate_positive_int(n_components, "n_components")
    if use_cos_sin:
        if n_components % 2 != 0:
            raise ValueError(
                f"n_components must be even when use_cos_sin=True, got {n_components}."
            )
        return n_components // 2
    return n_components


def _apply_fourier_nonlinearity(
    proj: NDArray[np.floating],
    offset: NDArray[np.floating] | None,
    n_components: int,
) -> NDArray[np.floating]:
    dtype = proj.dtype
    if offset is None:
        n_freqs = proj.shape[1]
        scale = np.asarray(np.sqrt(1.0 / float(n_freqs)), dtype=dtype)
        out = np.empty((proj.shape[0], 2 * n_freqs), dtype=dtype)
        np.cos(proj, out=out[:, :n_freqs])
        np.sin(proj, out=out[:, n_freqs:])
        out *= scale
        return out
    out = np.cos(proj + offset[None, :])
    out *= np.asarray(np.sqrt(2.0 / float(n_components)), dtype=dtype)
    return out


class OrthogonalRandomFourierFeatures(
    ClassNamePrefixFeaturesOutMixin,
    TransformerMixin,
    BaseEstimator,
):
    """Orthogonal Random Fourier Features (ORF) for isotropic and ARD kernels.

    Constructs stacked ``d x d`` Haar-distributed random orthogonal blocks via
    sign-corrected QR decomposition and scales each row by the exact radial
    distribution of the chosen isotropic kernel's spectral density (Yu et al.,
    NeurIPS 2016).
    """

    def __init__(
        self,
        n_components: int = 128,
        *,
        kernel: KernelLike = "rbf",
        lengthscale: LengthscaleLike = 1.0,
        nu: float = 1.5,
        alpha: float = 1.0,
        beta: float = 0.5,
        distribution: BaseDistribution = "gaussian",
        use_cos_sin: bool = False,
        random_state: SeedLike = None,
    ) -> None:
        self.n_components = n_components
        self.kernel = kernel
        self.lengthscale = lengthscale
        self.nu = nu
        self.alpha = alpha
        self.beta = beta
        self.distribution = distribution
        self.use_cos_sin = use_cos_sin
        self.random_state = random_state

    def fit(self, X: ArrayLike, y: Any = None) -> Self:
        """Sample orthogonal spectral frequencies for the input dimension of ``X``."""
        del y
        X_val = validate_data(
            self,
            X,
            accept_sparse=("csr", "csc"),
            dtype=[np.float64, np.float32],
        )
        d = int(X_val.shape[1])
        n_freqs = _resolve_n_freqs(self.n_components, self.use_cos_sin)

        resolved = make_kernel(
            self.kernel,
            lengthscale=self.lengthscale,
            nu=self.nu,
            alpha=self.alpha,
            beta=self.beta,
        )
        if not isinstance(resolved, IsotropicKernel):
            raise TypeError(
                f"{type(resolved).__name__} is not rotationally invariant and cannot be "
                "used with OrthogonalRandomFourierFeatures without introducing bias. "
                "Use Exponential (isotropic L2) or RandomFourierFeatures instead."
            )

        rng = _resolve_random_state(self.random_state)
        q = sample_haar_orthogonal(d, n_freqs, rng, distribution=self.distribution)
        radii = resolved.sample_radii(d, n_freqs, rng)
        ls = resolved.lengthscale_vector(d)

        weights: NDArray[np.floating] = (radii[:, None] * q) / ls[None, :]
        offset: NDArray[np.floating] | None = (
            None if self.use_cos_sin else rng.uniform(0.0, 2.0 * np.pi, size=n_freqs)
        )
        if X_val.dtype == np.float32:
            weights = weights.astype(np.float32, copy=False)
            if offset is not None:
                offset = offset.astype(np.float32, copy=False)

        self.kernel_: IsotropicKernel = resolved
        self.weights_: NDArray[np.floating] = weights
        self.offset_: NDArray[np.floating] | None = offset
        self._n_features_out: int = self.n_components
        return self

    def transform(self, X: ArrayLike) -> NDArray[np.floating]:
        """Map ``X`` into the orthogonal random Fourier feature space."""
        check_is_fitted(self, ["weights_"])
        X_val = validate_data(
            self,
            X,
            accept_sparse=("csr", "csc"),
            dtype=[np.float64, np.float32],
            reset=False,
        )
        weights = self.weights_.astype(X_val.dtype, copy=False)
        offset = None if self.offset_ is None else self.offset_.astype(X_val.dtype, copy=False)
        proj = safe_sparse_dot(X_val, weights.T, dense_output=True)
        return _apply_fourier_nonlinearity(proj, offset, self.n_components)

    def __sklearn_tags__(self) -> Any:
        tags = super().__sklearn_tags__()
        tags.input_tags.sparse = True
        tags.transformer_tags.preserves_dtype = ["float64", "float32"]
        return tags


class StructuredOrthogonalRandomFourierFeatures(
    ClassNamePrefixFeaturesOutMixin,
    TransformerMixin,
    BaseEstimator,
):
    """Structured Orthogonal Random Fourier Features (SORF) in O(p log p) time.

    Uses the native C/SIMD Fast Walsh-Hadamard Transform when compiled (with a
    pure-NumPy fallback) and supports both dense arrays and sparse CSR/CSC
    matrices without densifying sparse inputs.
    """

    def __init__(
        self,
        n_components: int = 128,
        *,
        kernel: KernelLike = "rbf",
        lengthscale: LengthscaleLike = 1.0,
        nu: float = 1.5,
        alpha: float = 1.0,
        beta: float = 0.5,
        n_stacks: int = 3,
        distribution: BaseDistribution = "rademacher",
        use_cos_sin: bool = False,
        random_state: SeedLike = None,
    ) -> None:
        self.n_components = n_components
        self.kernel = kernel
        self.lengthscale = lengthscale
        self.nu = nu
        self.alpha = alpha
        self.beta = beta
        self.n_stacks = n_stacks
        self.distribution = distribution
        self.use_cos_sin = use_cos_sin
        self.random_state = random_state

    def fit(self, X: ArrayLike, y: Any = None) -> Self:
        """Sample diagonal signs and radial scales for SORF projection."""
        del y
        X_val = validate_data(
            self,
            X,
            accept_sparse=("csr", "csc"),
            dtype=[np.float64, np.float32],
        )
        d = int(X_val.shape[1])
        n_freqs = _resolve_n_freqs(self.n_components, self.use_cos_sin)
        n_stacks = validate_positive_int(self.n_stacks, "n_stacks")

        resolved = make_kernel(
            self.kernel,
            lengthscale=self.lengthscale,
            nu=self.nu,
            alpha=self.alpha,
            beta=self.beta,
        )
        if not isinstance(resolved, IsotropicKernel):
            raise TypeError(
                f"{type(resolved).__name__} is not rotationally invariant and cannot be "
                "used with StructuredOrthogonalRandomFourierFeatures. "
                "Use Exponential (isotropic L2) or RandomFourierFeatures instead."
            )

        p = next_power_of_two(d)
        n_blocks = (n_freqs + p - 1) // p
        rng = _resolve_random_state(self.random_state)

        self.kernel_: IsotropicKernel = resolved
        self.padded_dim_: int = p
        self.n_freqs_: int = n_freqs
        self.lengthscale_: NDArray[np.float64] = resolved.lengthscale_vector(d)
        self.inv_lengthscale_: NDArray[np.float64] = 1.0 / self.lengthscale_
        self.signs_: NDArray[np.float64] = sample_rademacher_signs(
            n_blocks, n_stacks, p, rng, distribution=self.distribution
        )
        self.radii_: NDArray[np.float64] = resolved.sample_radii(p, n_freqs, rng)
        self.offset_: NDArray[np.float64] | None = (
            None if self.use_cos_sin else rng.uniform(0.0, 2.0 * np.pi, size=n_freqs)
        )
        self._n_features_out: int = self.n_components
        return self

    def transform(self, X: ArrayLike) -> NDArray[np.floating]:
        """Map dense or sparse ``X`` into the structured orthogonal feature space."""
        check_is_fitted(self, ["signs_", "radii_"])
        X_val = validate_data(
            self,
            X,
            accept_sparse=("csr", "csc"),
            dtype=[np.float64, np.float32],
            reset=False,
        )
        out_dtype = X_val.dtype
        proj = apply_sorf_scaled(
            X_val,
            self.inv_lengthscale_,
            self.signs_,
            self.radii_,
            self.n_freqs_,
        )
        out = _apply_fourier_nonlinearity(proj, self.offset_, self.n_components)
        if out_dtype == np.float32:
            return out.astype(np.float32, copy=False)
        return out

    def __sklearn_tags__(self) -> Any:
        tags = super().__sklearn_tags__()
        tags.input_tags.sparse = True
        tags.transformer_tags.preserves_dtype = ["float64", "float32"]
        return tags


class RandomFourierFeatures(
    ClassNamePrefixFeaturesOutMixin,
    TransformerMixin,
    BaseEstimator,
):
    """Standard i.i.d. Random Fourier Features (Rahimi & Recht, 2007).

    Supports all ``ShiftInvariantKernel`` instances, including non-isotropic
    kernels such as ``LaplacianL1``.
    """

    def __init__(
        self,
        n_components: int = 128,
        *,
        kernel: KernelLike = "rbf",
        lengthscale: LengthscaleLike = 1.0,
        nu: float = 1.5,
        alpha: float = 1.0,
        beta: float = 0.5,
        use_cos_sin: bool = False,
        random_state: SeedLike = None,
    ) -> None:
        self.n_components = n_components
        self.kernel = kernel
        self.lengthscale = lengthscale
        self.nu = nu
        self.alpha = alpha
        self.beta = beta
        self.use_cos_sin = use_cos_sin
        self.random_state = random_state

    def fit(self, X: ArrayLike, y: Any = None) -> Self:
        """Sample i.i.d. spectral frequencies from the kernel's Bochner measure."""
        del y
        X_val = validate_data(
            self,
            X,
            accept_sparse=("csr", "csc"),
            dtype=[np.float64, np.float32],
        )
        d = int(X_val.shape[1])
        n_freqs = _resolve_n_freqs(self.n_components, self.use_cos_sin)

        resolved = make_kernel(
            self.kernel,
            lengthscale=self.lengthscale,
            nu=self.nu,
            alpha=self.alpha,
            beta=self.beta,
        )
        rng = _resolve_random_state(self.random_state)

        weights: NDArray[np.floating] = resolved.sample_frequencies(d, n_freqs, rng)
        offset: NDArray[np.floating] | None = (
            None if self.use_cos_sin else rng.uniform(0.0, 2.0 * np.pi, size=n_freqs)
        )
        if X_val.dtype == np.float32:
            weights = weights.astype(np.float32, copy=False)
            if offset is not None:
                offset = offset.astype(np.float32, copy=False)

        self.kernel_: ShiftInvariantKernel = resolved
        self.weights_: NDArray[np.floating] = weights
        self.offset_: NDArray[np.floating] | None = offset
        self._n_features_out: int = self.n_components
        return self

    def transform(self, X: ArrayLike) -> NDArray[np.floating]:
        """Map ``X`` into the standard random Fourier feature space."""
        check_is_fitted(self, ["weights_"])
        X_val = validate_data(
            self,
            X,
            accept_sparse=("csr", "csc"),
            dtype=[np.float64, np.float32],
            reset=False,
        )
        weights = self.weights_.astype(X_val.dtype, copy=False)
        offset = None if self.offset_ is None else self.offset_.astype(X_val.dtype, copy=False)
        proj = safe_sparse_dot(X_val, weights.T, dense_output=True)
        return _apply_fourier_nonlinearity(proj, offset, self.n_components)

    def __sklearn_tags__(self) -> Any:
        tags = super().__sklearn_tags__()
        tags.input_tags.sparse = True
        tags.transformer_tags.preserves_dtype = ["float64", "float32"]
        return tags


class ArcCosineOrthogonalFeatures(
    ClassNamePrefixFeaturesOutMixin,
    TransformerMixin,
    BaseEstimator,
):
    """Orthogonal Random Features for Arc-Cosine (angular / ReLU) kernels."""

    def __init__(
        self,
        n_components: int = 128,
        *,
        order: Literal[0, 1] = 1,
        zero_order_mode: Literal["step", "sign"] = "step",
        random_state: SeedLike = None,
    ) -> None:
        self.n_components = n_components
        self.order = order
        self.zero_order_mode = zero_order_mode
        self.random_state = random_state

    def fit(self, X: ArrayLike, y: Any = None) -> Self:
        """Sample orthogonal projection weights for the arc-cosine kernel."""
        del y
        X_val = validate_data(
            self,
            X,
            accept_sparse=("csr", "csc"),
            dtype=[np.float64, np.float32],
        )
        d = int(X_val.shape[1])
        n_components = validate_positive_int(self.n_components, "n_components")
        if self.order not in (0, 1):
            raise ValueError(f"order must be 0 or 1, got {self.order!r}.")
        if self.zero_order_mode not in ("step", "sign"):
            raise ValueError(
                f"zero_order_mode must be 'step' or 'sign', got {self.zero_order_mode!r}."
            )

        rng = _resolve_random_state(self.random_state)
        q = sample_haar_orthogonal(d, n_components, rng)
        if self.order == 1:
            radii = np.sqrt(rng.chisquare(df=d, size=n_components))
            weights: NDArray[np.floating] = radii[:, None] * q
        else:
            weights = q

        if X_val.dtype == np.float32:
            weights = weights.astype(np.float32, copy=False)

        self.weights_: NDArray[np.floating] = weights
        self._n_features_out: int = self.n_components
        return self

    def transform(self, X: ArrayLike) -> NDArray[np.floating]:
        """Map ``X`` into the orthogonal arc-cosine feature space."""
        check_is_fitted(self, ["weights_"])
        X_val = validate_data(
            self,
            X,
            accept_sparse=("csr", "csc"),
            dtype=[np.float64, np.float32],
            reset=False,
        )
        dtype = X_val.dtype
        weights = self.weights_.astype(dtype, copy=False)
        proj = safe_sparse_dot(X_val, weights.T, dense_output=True)
        if self.order == 1:
            scale = np.asarray(np.sqrt(2.0 / float(self.n_components)), dtype=dtype)
            return scale * np.maximum(np.asarray(0.0, dtype=dtype), proj)
        if self.zero_order_mode == "step":
            scale = np.asarray(np.sqrt(2.0 / float(self.n_components)), dtype=dtype)
            return scale * (proj > 0.0).astype(dtype)
        inv_sqrt_m = np.asarray(1.0 / np.sqrt(float(self.n_components)), dtype=dtype)
        return np.where(proj >= 0.0, inv_sqrt_m, -inv_sqrt_m)

    def __sklearn_tags__(self) -> Any:
        tags = super().__sklearn_tags__()
        tags.input_tags.sparse = True
        tags.transformer_tags.preserves_dtype = ["float64", "float32"]
        return tags


class OrthogonalRandomProjection(
    ClassNamePrefixFeaturesOutMixin,
    TransformerMixin,
    BaseEstimator,
):
    """Unbiased orthogonal, structured orthogonal, and non-Gaussian linear projections.

    Approximates the linear dot-product kernel ``k(x, y) = x^T y`` (Johnson-
    Lindenstrauss dimensionality reduction or orthogonal lifting) using one of
    three modes:

    - ``mode="orthogonal"`` (default): stacked QR-orthogonalized blocks of
      ``distribution`` (``"gaussian"``, ``"rademacher"``, ``"laplace"``,
      ``"uniform"``). With ``scaling="unit"`` (default), scales by
      ``sqrt(d / M)`` so dot products are **exact** whenever ``M`` is a
      multiple of ``d`` and unbiased with lower variance than i.i.d.
      projections for any ``M``. With ``scaling="chi"``, scales rows by
      ``chi_d / sqrt(M)``.
    - ``mode="sorf"``: O(M log p) structured orthogonal Walsh-Hadamard
      projection ``sqrt(p / M) * H D_k ... H D_1`` (exact whenever ``M`` is a
      multiple of ``p = 2^ceil(log2(d))``), supporting sparse CSR/CSC inputs.
    - ``mode="iid"``: standard i.i.d. random projection ``W / sqrt(M)`` from
      ``distribution`` (zero-mean, unit-variance entries).
    """

    def __init__(
        self,
        n_components: int = 128,
        *,
        mode: Literal["orthogonal", "sorf", "iid"] = "orthogonal",
        distribution: BaseDistribution = "gaussian",
        scaling: Literal["unit", "chi"] = "unit",
        n_stacks: int = 3,
        random_state: SeedLike = None,
    ) -> None:
        self.n_components = n_components
        self.mode = mode
        self.distribution = distribution
        self.scaling = scaling
        self.n_stacks = n_stacks
        self.random_state = random_state

    def fit(self, X: ArrayLike, y: Any = None) -> Self:
        """Sample projection operator for the input dimension of ``X``."""
        del y
        X_val = validate_data(
            self,
            X,
            accept_sparse=("csr", "csc"),
            dtype=[np.float64, np.float32],
        )
        d = int(X_val.shape[1])
        m = validate_positive_int(self.n_components, "n_components")
        if self.mode not in ("orthogonal", "sorf", "iid"):
            raise ValueError(f"mode must be 'orthogonal', 'sorf', or 'iid', got {self.mode!r}.")
        if self.scaling not in ("unit", "chi"):
            raise ValueError(f"scaling must be 'unit' or 'chi', got {self.scaling!r}.")

        rng = _resolve_random_state(self.random_state)

        if self.mode == "sorf":
            n_stacks = validate_positive_int(self.n_stacks, "n_stacks")
            p = next_power_of_two(d)
            n_blocks = (m + p - 1) // p
            # Rademacher is the natural orthogonal diagonal for SORF; allow other base dists too.
            sorf_dist: BaseDistribution = (
                "rademacher" if self.distribution == "gaussian" else self.distribution
            )
            self.padded_dim_: int = p
            self.inv_lengthscale_: NDArray[np.float64] = np.ones(d, dtype=np.float64)
            self.signs_: NDArray[np.float64] = sample_rademacher_signs(
                n_blocks, n_stacks, p, rng, distribution=sorf_dist
            )
            if self.scaling == "chi":
                self.radii_: NDArray[np.float64] = np.sqrt(rng.chisquare(df=p, size=m) / float(m))
            else:
                self.radii_ = np.full(m, np.sqrt(float(p) / float(m)), dtype=np.float64)
            self.weights_: NDArray[np.floating] | None = None
        else:
            if self.mode == "orthogonal":
                q = sample_haar_orthogonal(d, m, rng, distribution=self.distribution)
                if self.scaling == "chi":
                    radii = np.sqrt(rng.chisquare(df=d, size=m) / float(m))
                    weights: NDArray[np.floating] = radii[:, None] * q
                else:
                    weights = q * np.sqrt(float(d) / float(m))
            else:
                base = sample_base_matrix(self.distribution, (m, d), rng)
                weights = base / np.sqrt(float(m))

            if X_val.dtype == np.float32:
                weights = weights.astype(np.float32, copy=False)
            self.weights_ = weights

        self._n_features_out: int = m
        return self

    def transform(self, X: ArrayLike) -> NDArray[np.floating]:
        """Project ``X`` into the ``n_components``-dimensional embedding space."""
        check_is_fitted(self, ["_n_features_out"])
        X_val = validate_data(
            self,
            X,
            accept_sparse=("csr", "csc"),
            dtype=[np.float64, np.float32],
            reset=False,
        )
        if self.mode == "sorf":
            out = apply_sorf_scaled(
                X_val,
                self.inv_lengthscale_,
                self.signs_,
                self.radii_,
                self._n_features_out,
            )
            if X_val.dtype == np.float32:
                return out.astype(np.float32, copy=False)
            return out

        assert self.weights_ is not None
        weights = self.weights_.astype(X_val.dtype, copy=False)
        return safe_sparse_dot(X_val, weights.T, dense_output=True)

    def __sklearn_tags__(self) -> Any:
        tags = super().__sklearn_tags__()
        tags.input_tags.sparse = True
        tags.transformer_tags.preserves_dtype = ["float64", "float32"]
        return tags


__all__ = [
    "ArcCosineOrthogonalFeatures",
    "NotFittedError",
    "OrthogonalRandomFourierFeatures",
    "OrthogonalRandomProjection",
    "RandomFourierFeatures",
    "StructuredOrthogonalRandomFourierFeatures",
]
