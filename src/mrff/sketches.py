"""Additional random feature & kernel sketch families.

Implements scikit-learn-compatible transformers for:
- ``FastfoodFeatures`` (Le, Sarlós, Smola, ICML 2013) — O(M log p) feature map
  for all isotropic shift-invariant kernels (RBF, Matérn, Exponential, RQ,
  Cauchy, IMQ) with scalar or ARD lengthscales.
- ``CountSketchFeatures`` (Charikar et al., 2002) — O(nnz) linear sketch.
- ``TensorSketchFeatures`` (Pham & Pagh, KDD 2013) — O(q (nnz + M log M))
  polynomial kernel approximation via FFT convolution of CountSketches.
- ``AnovaRandomFeatures`` (Kar & Karnick / Atarashi et al.) — ANOVA and
  All-Subsets kernels via vectorized DP.
- ``HomogeneousKernelMap`` (Vedaldi & Zisserman, PAMI 2012) — additive chi2,
  histogram intersection, and Hellinger kernel maps for non-negative inputs.
"""

from __future__ import annotations

import math
from typing import Any, Literal, Self

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import sparse
from sklearn.base import BaseEstimator, ClassNamePrefixFeaturesOutMixin, TransformerMixin
from sklearn.utils.validation import check_is_fitted, check_non_negative, validate_data

from mrff._validation import (
    LengthscaleLike,
    SeedLike,
    validate_positive_float,
    validate_positive_int,
)
from mrff.features import (
    _apply_fourier_nonlinearity,
    _resolve_n_freqs,
    _resolve_random_state,
)
from mrff.kernels import IsotropicKernel, KernelLike, make_kernel
from mrff.orthogonal import (
    BaseDistribution,
    fwht,
    next_power_of_two,
    sample_base_matrix,
)


def _count_sketch_apply(
    X: Any,
    hash_indices: NDArray[np.intp],
    hash_signs: NDArray[np.float64],
    n_components: int,
    *,
    extra_value: float = 0.0,
    extra_index: int = 0,
    extra_sign: float = 1.0,
) -> NDArray[np.float64]:
    """Compute CountSketch of dense or sparse CSR/CSC matrix X in O(nnz(X))."""
    n_samples = int(X.shape[0])
    out = np.zeros((n_samples, n_components), dtype=np.float64)
    if n_samples == 0:
        return out

    if sparse.issparse(X):
        csr = X.tocsr()
        indptr = csr.indptr
        indices = csr.indices
        data = np.asarray(csr.data, dtype=np.float64)
        for i in range(n_samples):
            start, end = int(indptr[i]), int(indptr[i + 1])
            if start < end:
                cols = indices[start:end]
                np.add.at(out[i], hash_indices[cols], data[start:end] * hash_signs[cols])
    else:
        X_f64 = np.asarray(X, dtype=np.float64)
        signed = X_f64 * hash_signs[None, :]
        # Vectorized scatter-add along columns into buckets
        for b in range(n_components):
            mask = hash_indices == b
            if np.any(mask):
                out[:, b] = np.sum(signed[:, mask], axis=1)

    if extra_value != 0.0:
        out[:, extra_index] += extra_value * extra_sign
    return out


class FastfoodFeatures(
    ClassNamePrefixFeaturesOutMixin,
    TransformerMixin,
    BaseEstimator,
):
    """Fastfood O(M log p) feature map for isotropic & ARD shift-invariant kernels.

    Approximates ``V = sqrt(p) * S * H * G * Pi * H * B`` (Le, Sarlós, Smola,
    ICML 2013) using normalized Walsh-Hadamard transforms ``H`` and exact radial
    scaling ``S_i = r_i / ||G||_2`` where ``r_i`` is drawn from the chosen
    ``IsotropicKernel`` in the zero-padded dimension ``p = 2^ceil(log2(d))``.
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
        """Sample Fastfood B, Pi, G, S factors for the input dimension of ``X``."""
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
                "used with FastfoodFeatures. Use RandomFourierFeatures instead."
            )

        p = next_power_of_two(d)
        n_blocks = (n_freqs + p - 1) // p
        rng = _resolve_random_state(self.random_state)

        B = sample_base_matrix("rademacher", (n_blocks, p), rng)
        Pi = np.vstack([rng.permutation(p) for _ in range(n_blocks)])
        G = rng.standard_normal((n_blocks, p))
        g_norms = np.linalg.norm(G, axis=1, keepdims=True)
        # Guard against the measure-zero event ||G|| == 0
        g_norms = np.where(g_norms > 0.0, g_norms, 1.0)
        radii = resolved.sample_radii(p, n_blocks * p, rng).reshape(n_blocks, p)
        S = (math.sqrt(float(p)) * radii) / g_norms

        self.kernel_: IsotropicKernel = resolved
        self.padded_dim_: int = p
        self.n_freqs_: int = n_freqs
        self.inv_lengthscale_: NDArray[np.float64] = 1.0 / resolved.lengthscale_vector(d)
        self.B_: NDArray[np.float64] = B
        self.Pi_: NDArray[np.intp] = Pi
        self.G_: NDArray[np.float64] = G
        self.S_: NDArray[np.float64] = S
        self.offset_: NDArray[np.float64] | None = (
            None if self.use_cos_sin else rng.uniform(0.0, 2.0 * np.pi, size=n_freqs)
        )
        self._n_features_out: int = self.n_components
        return self

    def transform(self, X: ArrayLike) -> NDArray[np.floating]:
        """Map ``X`` through stacked Fastfood Walsh-Hadamard blocks."""
        check_is_fitted(self, ["B_", "Pi_", "G_", "S_"])
        X_val = validate_data(
            self,
            X,
            accept_sparse=("csr", "csc"),
            dtype=[np.float64, np.float32],
            reset=False,
        )
        out_dtype = X_val.dtype
        n_samples, d = int(X_val.shape[0]), int(X_val.shape[1])
        p = self.padded_dim_

        if sparse.issparse(X_val):
            X_f64 = X_val.toarray().astype(np.float64, copy=False)
        else:
            X_f64 = np.asarray(X_val, dtype=np.float64)

        xs = X_f64 * self.inv_lengthscale_[None, :]
        if p > d:
            x_pad = np.zeros((n_samples, p), dtype=np.float64)
            x_pad[:, :d] = xs
        else:
            x_pad = xs

        proj = np.empty((n_samples, self.n_freqs_), dtype=np.float64)
        offset = 0
        for b in range(self.B_.shape[0]):
            u = fwht(x_pad * self.B_[b], normalized=True)
            u = u[:, self.Pi_[b]] * self.G_[b]
            u = fwht(u, normalized=True) * self.S_[b]
            take = min(p, self.n_freqs_ - offset)
            proj[:, offset : offset + take] = u[:, :take]
            offset += take
            if offset >= self.n_freqs_:
                break

        out = _apply_fourier_nonlinearity(proj, self.offset_, self.n_components)
        if out_dtype == np.float32:
            return out.astype(np.float32, copy=False)
        return out

    def __sklearn_tags__(self) -> Any:
        tags = super().__sklearn_tags__()
        tags.input_tags.sparse = True
        tags.transformer_tags.preserves_dtype = ["float64", "float32"]
        return tags


class CountSketchFeatures(
    ClassNamePrefixFeaturesOutMixin,
    TransformerMixin,
    BaseEstimator,
):
    """CountSketch linear projection in O(nnz(X)) time (Charikar et al., 2002).

    Unbiasedly approximates the linear dot-product kernel ``k(x, y) = x^T y``
    by hashing each input coordinate to a random output bucket with a random
    Rademacher sign.
    """

    def __init__(
        self,
        n_components: int = 128,
        *,
        random_state: SeedLike = None,
    ) -> None:
        self.n_components = n_components
        self.random_state = random_state

    def fit(self, X: ArrayLike, y: Any = None) -> Self:
        """Sample hash bucket indices and Rademacher signs."""
        del y
        X_val = validate_data(
            self,
            X,
            accept_sparse=("csr", "csc"),
            dtype=[np.float64, np.float32],
        )
        d = int(X_val.shape[1])
        m = validate_positive_int(self.n_components, "n_components")
        rng = _resolve_random_state(self.random_state)

        self.hash_indices_: NDArray[np.intp] = rng.integers(0, m, size=d, dtype=np.intp)
        self.hash_signs_: NDArray[np.float64] = sample_base_matrix("rademacher", (d,), rng)
        self._n_features_out: int = m
        return self

    def transform(self, X: ArrayLike) -> NDArray[np.floating]:
        """Hash ``X`` into ``n_components`` buckets."""
        check_is_fitted(self, ["hash_indices_", "hash_signs_"])
        X_val = validate_data(
            self,
            X,
            accept_sparse=("csr", "csc"),
            dtype=[np.float64, np.float32],
            reset=False,
        )
        out = _count_sketch_apply(X_val, self.hash_indices_, self.hash_signs_, self._n_features_out)
        if X_val.dtype == np.float32:
            return out.astype(np.float32, copy=False)
        return out

    def __sklearn_tags__(self) -> Any:
        tags = super().__sklearn_tags__()
        tags.input_tags.sparse = True
        tags.transformer_tags.preserves_dtype = ["float64", "float32"]
        return tags


class TensorSketchFeatures(
    ClassNamePrefixFeaturesOutMixin,
    TransformerMixin,
    BaseEstimator,
):
    """TensorSketch polynomial kernel approximation (Pham & Pagh, KDD 2013).

    Unbiasedly approximates ``k(x, y) = (gamma * x^T y + coef0) ** degree`` in
    ``O(degree * (nnz(X) + M log M))`` time via FFT convolution of ``degree``
    independent CountSketches.
    """

    def __init__(
        self,
        n_components: int = 128,
        *,
        degree: int = 2,
        gamma: float = 1.0,
        coef0: float = 0.0,
        random_state: SeedLike = None,
    ) -> None:
        self.n_components = n_components
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.random_state = random_state

    def fit(self, X: ArrayLike, y: Any = None) -> Self:
        """Sample ``degree`` independent CountSketch hash functions."""
        del y
        X_val = validate_data(
            self,
            X,
            accept_sparse=("csr", "csc"),
            dtype=[np.float64, np.float32],
        )
        d = int(X_val.shape[1])
        m = validate_positive_int(self.n_components, "n_components")
        deg = validate_positive_int(self.degree, "degree")
        gamma = validate_positive_float(self.gamma, "gamma")
        if (
            isinstance(self.coef0, bool)
            or not np.isfinite(float(self.coef0))
            or float(self.coef0) < 0.0
        ):
            raise ValueError(f"coef0 must be finite and non-negative, got {self.coef0!r}.")

        rng = _resolve_random_state(self.random_state)
        self.hash_indices_: NDArray[np.intp] = rng.integers(0, m, size=(deg, d), dtype=np.intp)
        self.hash_signs_: NDArray[np.float64] = sample_base_matrix("rademacher", (deg, d), rng)
        self.coef0_indices_: NDArray[np.intp] = rng.integers(0, m, size=deg, dtype=np.intp)
        self.coef0_signs_: NDArray[np.float64] = sample_base_matrix("rademacher", (deg,), rng)
        self._gamma_: float = gamma
        self._sqrt_gamma_: float = math.sqrt(gamma)
        self._sqrt_coef0_: float = math.sqrt(float(self.coef0))
        self._n_features_out: int = m
        return self

    def transform(self, X: ArrayLike) -> NDArray[np.floating]:
        """Compute FFT-convolved polynomial TensorSketch features for ``X``."""
        check_is_fitted(self, ["hash_indices_", "hash_signs_"])
        X_val = validate_data(
            self,
            X,
            accept_sparse=("csr", "csc"),
            dtype=[np.float64, np.float32],
            reset=False,
        )
        m = self._n_features_out
        deg = self.hash_indices_.shape[0]
        X_scaled = X_val * self._sqrt_gamma_ if self._sqrt_gamma_ != 1.0 else X_val

        fft_prod: NDArray[np.complex128] | None = None
        for k in range(deg):
            sketch = _count_sketch_apply(
                X_scaled,
                self.hash_indices_[k],
                self.hash_signs_[k],
                m,
                extra_value=self._sqrt_coef0_,
                extra_index=int(self.coef0_indices_[k]),
                extra_sign=float(self.coef0_signs_[k]),
            )
            spec = np.fft.rfft(sketch, axis=1)
            fft_prod = spec if fft_prod is None else (fft_prod * spec)

        assert fft_prod is not None
        out = np.fft.irfft(fft_prod, n=m, axis=1)
        if X_val.dtype == np.float32:
            return out.astype(np.float32, copy=False)
        return out

    def __sklearn_tags__(self) -> Any:
        tags = super().__sklearn_tags__()
        tags.input_tags.sparse = True
        tags.transformer_tags.preserves_dtype = ["float64", "float32"]
        return tags


class AnovaRandomFeatures(
    ClassNamePrefixFeaturesOutMixin,
    TransformerMixin,
    BaseEstimator,
):
    """Random feature maps for ANOVA and All-Subsets kernels.

    - ``kernel="anova"`` (default): unbiasedly approximates the degree-``q``
      ANOVA kernel ``sum_{j_1 < ... < j_q} x_{j_1} y_{j_1} ... x_{j_q} y_{j_q}``
      using elementary symmetric polynomials computed in ``O(q * d)`` DP time.
    - ``kernel="all_subsets"``: unbiasedly approximates
      ``prod_{j=1}^d (1 + x_j y_j)``.
    """

    def __init__(
        self,
        n_components: int = 128,
        *,
        kernel: Literal["anova", "all_subsets"] = "anova",
        degree: int = 2,
        distribution: BaseDistribution = "rademacher",
        random_state: SeedLike = None,
    ) -> None:
        self.n_components = n_components
        self.kernel = kernel
        self.degree = degree
        self.distribution = distribution
        self.random_state = random_state

    def fit(self, X: ArrayLike, y: Any = None) -> Self:
        """Sample i.i.d. zero-mean unit-variance weights for ANOVA / All-Subsets."""
        del y
        X_val = validate_data(
            self,
            X,
            accept_sparse=("csr", "csc"),
            dtype=[np.float64, np.float32],
        )
        d = int(X_val.shape[1])
        m = validate_positive_int(self.n_components, "n_components")
        if self.kernel not in ("anova", "all_subsets"):
            raise ValueError(f"kernel must be 'anova' or 'all_subsets', got {self.kernel!r}.")
        if self.kernel == "anova":
            validate_positive_int(self.degree, "degree")

        rng = _resolve_random_state(self.random_state)
        self.weights_: NDArray[np.float64] = sample_base_matrix(self.distribution, (d, m), rng)
        self._n_features_out: int = m
        return self

    def transform(self, X: ArrayLike) -> NDArray[np.floating]:
        """Compute ANOVA or All-Subsets random features for ``X``."""
        check_is_fitted(self, ["weights_"])
        X_val = validate_data(
            self,
            X,
            accept_sparse=("csr", "csc"),
            dtype=[np.float64, np.float32],
            reset=False,
        )
        out_dtype = X_val.dtype
        n_samples, d = int(X_val.shape[0]), int(X_val.shape[1])
        m = self._n_features_out
        inv_sqrt_m = 1.0 / math.sqrt(float(m))

        if sparse.issparse(X_val):
            csr = X_val.tocsr()
            out = np.empty((n_samples, m), dtype=np.float64)
            if self.kernel == "all_subsets":
                for i in range(n_samples):
                    start, end = int(csr.indptr[i]), int(csr.indptr[i + 1])
                    acc = np.ones(m, dtype=np.float64)
                    for jj in range(start, end):
                        col = int(csr.indices[jj])
                        acc *= 1.0 + float(csr.data[jj]) * self.weights_[col]
                    out[i] = acc * inv_sqrt_m
            else:
                deg = int(self.degree)
                dp = np.zeros((deg + 1, m), dtype=np.float64)
                for i in range(n_samples):
                    start, end = int(csr.indptr[i]), int(csr.indptr[i + 1])
                    dp.fill(0.0)
                    dp[0] = 1.0
                    for jj in range(start, end):
                        col = int(csr.indices[jj])
                        term = float(csr.data[jj]) * self.weights_[col]
                        for k in range(deg, 0, -1):
                            dp[k] += dp[k - 1] * term
                    out[i] = dp[deg] * inv_sqrt_m
        else:
            X_f64 = np.asarray(X_val, dtype=np.float64)
            if self.kernel == "all_subsets":
                out = np.ones((n_samples, m), dtype=np.float64)
                for j in range(d):
                    out *= 1.0 + X_f64[:, j : j + 1] * self.weights_[j : j + 1, :]
                out *= inv_sqrt_m
            else:
                deg = int(self.degree)
                if deg > d:
                    out = np.zeros((n_samples, m), dtype=np.float64)
                else:
                    dp_b = np.zeros((deg + 1, n_samples, m), dtype=np.float64)
                    dp_b[0] = 1.0
                    for j in range(d):
                        term = X_f64[:, j : j + 1] * self.weights_[j : j + 1, :]
                        for k in range(min(deg, j + 1), 0, -1):
                            dp_b[k] += dp_b[k - 1] * term
                    out = dp_b[deg] * inv_sqrt_m

        if out_dtype == np.float32:
            return out.astype(np.float32, copy=False)
        return out

    def __sklearn_tags__(self) -> Any:
        tags = super().__sklearn_tags__()
        tags.input_tags.sparse = True
        tags.transformer_tags.preserves_dtype = ["float64", "float32"]
        return tags


class HomogeneousKernelMap(
    ClassNamePrefixFeaturesOutMixin,
    TransformerMixin,
    BaseEstimator,
):
    """Vedaldi & Zisserman (PAMI 2012) deterministic homogeneous kernel maps.

    Supports non-negative inputs (histograms, TF-IDF, counts) for:
    - ``kernel="chi2"``: additive chi-squared ``sum_j 2 x_j y_j / (x_j + y_j)``
    - ``kernel="intersection"``: histogram intersection ``sum_j min(x_j, y_j)``
    - ``kernel="hellinger"``: exact Hellinger ``sum_j sqrt(x_j y_j)``

    Output dimension is ``n_features`` for ``"hellinger"`` and
    ``(2 * order + 1) * n_features`` for ``"chi2"`` and ``"intersection"``.
    """

    def __init__(
        self,
        *,
        kernel: Literal["chi2", "intersection", "hellinger"] = "chi2",
        order: int = 2,
        sample_interval: float | None = None,
    ) -> None:
        self.kernel = kernel
        self.order = order
        self.sample_interval = sample_interval

    def fit(self, X: ArrayLike, y: Any = None) -> Self:
        """Validate non-negative inputs and precompute spectral harmonic weights."""
        del y
        X_val = validate_data(
            self,
            X,
            accept_sparse=("csr", "csc"),
            dtype=[np.float64, np.float32],
        )
        self._check_non_negative(X_val)
        d = int(X_val.shape[1])

        if self.kernel not in ("chi2", "intersection", "hellinger"):
            raise ValueError(
                f"kernel must be 'chi2', 'intersection', or 'hellinger', got {self.kernel!r}."
            )

        if self.kernel == "hellinger":
            self.sample_interval_: float = 0.0
            self.freqs_: NDArray[np.float64] = np.empty(0, dtype=np.float64)
            self.zero_weight_: float = 1.0
            self.harmonic_weights_: NDArray[np.float64] = np.empty(0, dtype=np.float64)
            self._n_features_out: int = d
            return self

        order = validate_positive_int(self.order, "order")
        if self.sample_interval is None:
            interval = 0.5 if self.kernel == "chi2" else 1.2 / math.sqrt(float(order))
        else:
            interval = validate_positive_float(self.sample_interval, "sample_interval")

        k = np.arange(1, order + 1, dtype=np.float64)
        freqs = k * interval
        if self.kernel == "chi2":
            kappa0 = 1.0
            kappa = 1.0 / np.cosh(math.pi * freqs)
        else:
            kappa0 = 2.0 / math.pi
            kappa = (2.0 / math.pi) / (1.0 + 4.0 * freqs * freqs)

        self.sample_interval_ = interval
        self.freqs_ = freqs
        self.zero_weight_ = math.sqrt(interval * kappa0)
        self.harmonic_weights_ = np.sqrt(2.0 * interval * kappa)
        self._n_features_out = (2 * order + 1) * d
        return self

    @staticmethod
    def _check_non_negative(X: Any) -> None:
        check_non_negative(X, "HomogeneousKernelMap")

    def transform(self, X: ArrayLike) -> NDArray[np.floating]:
        """Apply the deterministic homogeneous kernel map to non-negative ``X``."""
        check_is_fitted(self, ["zero_weight_", "freqs_"])
        X_val = validate_data(
            self,
            X,
            accept_sparse=("csr", "csc"),
            dtype=[np.float64, np.float32],
            reset=False,
        )
        self._check_non_negative(X_val)
        out_dtype = X_val.dtype

        X_dense = X_val.toarray() if sparse.issparse(X_val) else np.asarray(X_val)
        X_f64 = np.asarray(X_dense, dtype=np.float64)
        sqrt_x = np.sqrt(X_f64)

        if self.kernel == "hellinger":
            return sqrt_x.astype(out_dtype, copy=False)

        n_samples, d = X_f64.shape
        order = self.freqs_.shape[0]
        out = np.empty((n_samples, (2 * order + 1) * d), dtype=np.float64)
        out[:, :d] = self.zero_weight_ * sqrt_x

        pos = X_f64 > 0.0
        log_x = np.zeros_like(X_f64)
        log_x[pos] = np.log(X_f64[pos])

        for idx in range(order):
            angle = self.freqs_[idx] * log_x
            amp = self.harmonic_weights_[idx] * sqrt_x
            c_start = (1 + 2 * idx) * d
            out[:, c_start : c_start + d] = amp * np.cos(angle)
            out[:, c_start + d : c_start + 2 * d] = amp * np.sin(angle)

        if out_dtype == np.float32:
            return out.astype(np.float32, copy=False)
        return out

    def __sklearn_tags__(self) -> Any:
        tags = super().__sklearn_tags__()
        tags.input_tags.sparse = True
        tags.input_tags.positive_only = True
        tags.transformer_tags.preserves_dtype = ["float64", "float32"]
        return tags
