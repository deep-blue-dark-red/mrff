# mrff

**Scikit-learn–compatible Orthogonal, Structured Orthogonal, Fastfood, Sketch, Homogeneous, and Online Random Feature Maps for Python.**

`mrff` implements variance-reduced orthogonal random feature maps (Yu et al., NeurIPS 2016) and complementary kernel approximations in modern Python (3.11–3.13+, NumPy 2.x, scikit-learn 1.6+), accelerated by a portable C11 + SIMD Fast Walsh–Hadamard Transform (ARM NEON, x86 AVX2 / SSE2, scalar fallback) with zero-densification sparse CSR/CSC SORF support.

---

## Table of Contents

- [Why `mrff`?](#why-mrff)
- [Installation](#installation)
- [Supported Kernels & Spectral Formulas](#supported-kernels--spectral-formulas)
- [Transformers & Online Estimators](#transformers--online-estimators)
- [Quick Start & Recipes](#quick-start--recipes)
  - [1. Orthogonal Random Fourier Features (ORF)](#1-orthogonal-random-fourier-features-orf)
  - [2. Structured ORF (SORF) on Sparse Data & ARD Lengthscales](#2-structured-orf-sorf-on-sparse-data--ard-lengthscales)
  - [3. Fastfood, CountSketch, TensorSketch, ANOVA & Homogeneous Maps](#3-fastfood-countsketch-tensorsketch-anova--homogeneous-maps)
  - [4. Orthogonal & Non-Gaussian Linear Projections](#4-orthogonal--non-gaussian-linear-projections)
  - [5. Memory-Free Online Regression & Classification](#5-memory-free-online-regression--classification)
  - [6. Scikit-learn Pipelines & GridSearchCV](#6-scikit-learn-pipelines--gridsearchcv)
- [Native C/SIMD FWHT Backend](#native-csimd-fwht-backend)
- [Comparison & Parity with `neonnnnn/pyrfm`](#comparison--parity-with-neonnnnnpyrfm)
  - [Feature & Engineering Comparison](#feature--engineering-comparison)
  - [Exact Numerical Parity & Measured Accuracy](#exact-numerical-parity--measured-accuracy)
  - [Migrating from `pyrfm` to `mrff`](#migrating-from-pyrfm-to-mrff)
- [Development & Testing](#development--testing)
- [References](#references)

---

## Why `mrff`?

- **Exact multi-kernel ORF, SORF & Fastfood**: Not limited to RBF — works with exact unbiased spectral/radial samplers for **RBF**, **Matérn** (any $\nu > 0$), **Exponential** ($L_2$), **Rational Quadratic**, **Cauchy** ($L_2$), and **Inverse Multiquadric**, all with scalar or per-feature **ARD** lengthscales.
- **First-class scikit-learn estimators**: Every transformer and online model inherits from standard scikit-learn base classes (`BaseEstimator`, `TransformerMixin`, `ClassNamePrefixFeaturesOutMixin`, `RegressorMixin`, `ClassifierMixin`), preserves `float32`/`float64`, supports sparse CSR/CSC inputs, works in `Pipeline`, `GridSearchCV`, `clone`, `pickle`, and `get_feature_names_out`, and passes **100% of `sklearn.utils.estimator_checks.check_estimator`**.
- **Fast portable C11 + SIMD FWHT**: Uses only Python's standard Buffer Protocol (no NumPy C-API ABI coupling, no Cython, no `numpy.distutils`), compiling cleanly on Apple Silicon (ARM NEON), x86_64 (AVX2 / SSE2), and any C11 compiler — with a transparent pure-NumPy fallback.
- **Zero-densification sparse SORF**: Scatters sparse CSR nonzeros directly into an L1-cache working buffer per row before FWHT butterflies.
- **Memory-free online solvers**: `RandomFeatureRegressor` and `RandomFeatureClassifier` compute features on the fly in mini-batches with `adam`, `sgd`, `adagrad`, and $O(N)$-memory `saga`, never materializing the full $N \times M$ transformed feature matrix.
- **Strict typing & verification**: PEP 561 typed (`py.typed`), `ty` + `ruff` clean, and backed by 1,200+ automated tests.

---

## Installation

```bash
# Editable install with dev/test dependencies (uv)
uv sync --extra dev

# Or standard pip install
pip install -e .
```

To skip compiling the optional C/SIMD extension and run in pure-NumPy mode:

```bash
MRFF_PURE_PYTHON=1 pip install -e .
```

**Runtime requirements**: Python >= 3.11, `numpy>=1.26`, `scikit-learn>=1.6`, `scipy>=1.11`.

---

## Supported Kernels & Spectral Formulas

For any isotropic shift-invariant kernel $k(x, y) = \kappa(\lVert\Lambda^{-1}(x - y)\rVert_2)$ with scalar or per-feature ARD lengthscale $\Lambda = \mathrm{diag}(\ell_1, \dots, \ell_d)$, Bochner's theorem yields a spherically symmetric spectral density in unit-lengthscale space:

$$
\omega = r \cdot q, \qquad q \sim \mathrm{Uniform}(S^{d-1}), \quad r = \lVert\omega\rVert_2 \text{ independent of } q.
$$

ORF replaces i.i.d. directions $q$ with rows of stacked $d \times d$ Haar-distributed orthogonal matrices $Q \in O(d)$ (sign-corrected QR per Mezzadri, 2007), preserving the exact marginal distribution of every $\omega_i$ (zero bias) while reducing variance. SORF and Fastfood apply the same radial samplers in the zero-padded dimension $p = 2^{\lceil \log_2 d \rceil}$.

### Shift-Invariant Kernels

| Class | String alias(es) | Kernel $k(x, y)$ (unit $\ell$) | Radial length $r = \lVert\omega\rVert_2$ ($s \sim \chi_d$) | ORF / SORF / Fastfood |
|---|---|---|---|---|
| `RBF` | `"rbf"`, `"gaussian"` | $\exp\left(-\frac{1}{2}\lVert x-y\rVert_2^2\right)$ | $r = s$ | Yes |
| `Matern(nu)` | `"matern"`, `"matern12"`, `"matern32"`, `"matern52"` | $\frac{2^{1-\nu}}{\Gamma(\nu)}(\sqrt{2\nu}\lVert x-y\rVert_2)^\nu K_\nu(\sqrt{2\nu}\lVert x-y\rVert_2)$ | $r = \dfrac{s}{\sqrt{g}}, \quad g \sim \mathrm{Gamma}\left(\nu, \text{scale}=\frac{1}{\nu}\right)$ | Yes |
| `Exponential` | `"exponential"`, `"l2_laplacian"` | $\exp(-\lVert x-y\rVert_2)$ (Matérn $\nu=\frac{1}{2}$) | $r = \dfrac{s}{\sqrt{g}}, \quad g \sim \mathrm{Gamma}\left(\frac{1}{2}, \text{scale}=2\right)$ | Yes |
| `RationalQuadratic(alpha)` | `"rational_quadratic"`, `"rq"` | $\left(1 + \frac{\lVert x-y\rVert_2^2}{2\alpha}\right)^{-\alpha}$ | $r = s\sqrt{\tau}, \quad \tau \sim \mathrm{Gamma}\left(\alpha, \text{scale}=\frac{1}{\alpha}\right)$ | Yes |
| `Cauchy` | `"cauchy"`, `"cauchy_l2"` | $\left(1 + \lVert x-y\rVert_2^2\right)^{-1}$ | $r = s\sqrt{2\tau}, \quad \tau \sim \mathrm{Exp}(1)$ | Yes |
| `InverseMultiquadric(beta)` | `"imq"`, `"inverse_multiquadric"` | $\left(1 + \lVert x-y\rVert_2^2\right)^{-\beta}$ | $r = s\sqrt{2\tau}, \quad \tau \sim \mathrm{Gamma}(\beta, \text{scale}=1)$ | Yes |
| `LaplacianL1` | `"laplacian_l1"`, `"laplacian"` | $\exp(-\lVert x-y\rVert_1)$ | i.i.d. $\omega_j \sim \mathrm{Cauchy}(0, 1)$ *(non-isotropic)* | `RandomFourierFeatures` only |

> **Why `LaplacianL1` is refused by ORF / SORF / Fastfood**: For $d > 1$, the $L_1$ Laplacian spectral density factors across coordinates and is not rotationally invariant. Rotating it with a Haar orthogonal matrix biases the estimator away from $L_1$, so `OrthogonalRandomFourierFeatures`, `StructuredOrthogonalRandomFourierFeatures`, and `FastfoodFeatures` explicitly raise a `TypeError` directing you to `Exponential` ($L_2$) or `RandomFourierFeatures`.

### Non-Shift-Invariant Kernels

| Class | Kernel(s) Approximated | Exact Evaluator |
|---|---|---|
| `ArcCosineOrthogonalFeatures` | Arc-Cosine order 0 ($1 - \theta/\pi$ step or $1 - 2\theta/\pi$ sign) and order 1 (ReLU) | `exact_arccosine_matrix` |
| `OrthogonalRandomProjection` | Linear dot product $x^\top y$ (`mode="orthogonal" \| "sorf" \| "iid"`) | `X @ Y.T` |
| `CountSketchFeatures` | Linear dot product $x^\top y$ in $O(\mathrm{nnz}(X))$ | `X @ Y.T` |
| `TensorSketchFeatures` | Polynomial $(\gamma x^\top y + c)^q$ in $O(q(\mathrm{nnz}(X) + M \log M))$ | `exact_polynomial_matrix` |
| `AnovaRandomFeatures` | Degree $q$ ANOVA and All-Subsets $\prod_j (1 + x_j y_j)$ | `exact_anova_matrix`, `exact_all_subsets_matrix` |
| `HomogeneousKernelMap` | Additive $\chi^2$, Histogram Intersection, Hellinger (non-negative inputs) | `exact_chi2_matrix`, `exact_intersection_matrix`, `exact_hellinger_matrix` |

---

## Transformers & Online Estimators

| Estimator | Complexity per sample | Sparse CSR/CSC | `float32`/`float64` | `check_estimator` |
|---|---|---|---|---|
| `OrthogonalRandomFourierFeatures` | $O(M d)$ | Yes | Preserved | Passing |
| `StructuredOrthogonalRandomFourierFeatures` | $O(M \log p)$ | Yes *(zero-densification)* | Preserved | Passing |
| `FastfoodFeatures` | $O(M \log p)$ | Yes | Preserved | Passing |
| `RandomFourierFeatures` | $O(M d)$ | Yes | Preserved | Passing |
| `ArcCosineOrthogonalFeatures` | $O(M d)$ | Yes | Preserved | Passing |
| `OrthogonalRandomProjection` | $O(M d)$ or $O(M \log p)$ | Yes | Preserved | Passing |
| `CountSketchFeatures` | $O(\mathrm{nnz}(x))$ | Yes | Preserved | Passing |
| `TensorSketchFeatures` | $O(q(\mathrm{nnz}(x) + M \log M))$ | Yes | Preserved | Passing |
| `AnovaRandomFeatures` | $O(q \cdot d \cdot M)$ | Yes | Preserved | Passing |
| `HomogeneousKernelMap` | $O((2k+1) d)$ | Yes | Preserved | Passing |
| `RandomFeatureRegressor` | $O(B \cdot \text{cost}(\phi))$ RAM | Yes | Yes | Passing |
| `RandomFeatureClassifier` | $O(B \cdot \text{cost}(\phi))$ RAM | Yes | Yes | Passing |

---

## Quick Start & Recipes

### 1. Orthogonal Random Fourier Features (ORF)

```python
import numpy as np
from mrff import Matern, OrthogonalRandomFourierFeatures

rng = np.random.default_rng(0)
X = rng.standard_normal((200, 16))

orf = OrthogonalRandomFourierFeatures(
    n_components=512,
    kernel=Matern(nu=1.5, lengthscale=1.5),
    use_cos_sin=True,  # phase-free [cos, sin] embedding (lower variance)
    random_state=42,
)
Z = orf.fit_transform(X)

K_approx = Z @ Z.T
K_exact = orf.kernel_.exact_matrix(X)
print("Max Gram error:", np.max(np.abs(K_approx - K_exact)))
```

### 2. Structured ORF (SORF) on Sparse Data & ARD Lengthscales

```python
import numpy as np
from scipy import sparse
from mrff import RationalQuadratic, StructuredOrthogonalRandomFourierFeatures

rng = np.random.default_rng(1)
X = rng.standard_normal((500, 100))
X[np.abs(X) < 1.5] = 0.0
X_csr = sparse.csr_matrix(X)

# Per-feature ARD lengthscales + zero-densification sparse CSR SORF
ard_ls = rng.uniform(0.8, 2.5, size=100)
sorf = StructuredOrthogonalRandomFourierFeatures(
    n_components=1024,
    kernel=RationalQuadratic(alpha=2.0, lengthscale=ard_ls),
    use_cos_sin=True,
    random_state=0,
)
Z = sorf.fit_transform(X_csr)
```

### 3. Fastfood, CountSketch, TensorSketch, ANOVA & Homogeneous Maps

```python
import numpy as np
from mrff import (
    AnovaRandomFeatures,
    CountSketchFeatures,
    FastfoodFeatures,
    HomogeneousKernelMap,
    TensorSketchFeatures,
)

rng = np.random.default_rng(2)
X = rng.standard_normal((100, 20))
X_pos = np.abs(X)  # non-negative features for homogeneous maps

Z_ff = FastfoodFeatures(n_components=256, kernel="matern52", lengthscale=2.0, random_state=0).fit_transform(X)
Z_cs = CountSketchFeatures(n_components=128, random_state=0).fit_transform(X)
Z_ts = TensorSketchFeatures(n_components=256, degree=3, gamma=0.5, coef0=1.0, random_state=0).fit_transform(X)
Z_anova = AnovaRandomFeatures(n_components=256, kernel="anova", degree=2, random_state=0).fit_transform(X)
Z_chi2 = HomogeneousKernelMap(kernel="chi2", order=2).fit_transform(X_pos)
Z_int = HomogeneousKernelMap(kernel="intersection", order=8).fit_transform(X_pos)
```

### 4. Orthogonal & Non-Gaussian Linear Projections

```python
import numpy as np
from mrff import OrthogonalRandomProjection

X = np.random.default_rng(3).standard_normal((100, 32))

# Exact dot-product preservation when n_components is a multiple of d (or p for SORF)
Z_orth = OrthogonalRandomProjection(
    n_components=32, mode="orthogonal", distribution="rademacher", random_state=0
).fit_transform(X)
assert np.allclose(Z_orth @ Z_orth.T, X @ X.T, atol=1e-11)

# O(M log p) structured orthogonal Johnson-Lindenstrauss projection
Z_sorf = OrthogonalRandomProjection(
    n_components=16, mode="sorf", random_state=0
).fit_transform(X)
```

### 5. Memory-Free Online Regression & Classification

```python
import numpy as np
from mrff import (
    OrthogonalRandomFourierFeatures,
    RandomFeatureClassifier,
    RandomFeatureRegressor,
    StructuredOrthogonalRandomFourierFeatures,
)

rng = np.random.default_rng(4)
X = rng.uniform(-2.0, 2.0, size=(1000, 8))
y_reg = np.sin(X[:, 0]) + np.cos(X[:, 1] * X[:, 2])
y_cls = (np.sum(X[:, :3] ** 2, axis=1) > 3.5).astype(int)

# On-the-fly mini-batch SAGA regression (never stores the 1000 x 512 feature matrix)
reg = RandomFeatureRegressor(
    feature_map=StructuredOrthogonalRandomFourierFeatures(
        n_components=512, kernel="matern52", lengthscale=2.0, use_cos_sin=True
    ),
    optimizer="saga",
    batch_size=64,
    max_iter=50,
    random_state=0,
).fit(X, y_reg)

# Binary / multiclass classification with log_loss, hinge, or squared_hinge
clf = RandomFeatureClassifier(
    feature_map=OrthogonalRandomFourierFeatures(
        n_components=256, kernel="rbf", lengthscale=2.0, use_cos_sin=True
    ),
    loss="log_loss",
    optimizer="adam",
    batch_size=64,
    random_state=0,
).fit(X, y_cls)
probs = clf.predict_proba(X[:5])
```

### 6. Scikit-learn Pipelines & GridSearchCV

```python
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from mrff import OrthogonalRandomFourierFeatures

pipe = Pipeline([
    ("features", OrthogonalRandomFourierFeatures(n_components=256, random_state=0)),
    ("ridge", Ridge(alpha=1e-2)),
])
grid = GridSearchCV(
    pipe,
    param_grid={
        "features__kernel": ["rbf", "matern32", "matern52", "rational_quadratic"],
        "features__lengthscale": [0.5, 1.0, 2.0],
    },
    cv=5,
)
```

---

## Native C/SIMD FWHT Backend

Inspect the active Fast Walsh–Hadamard Transform backend at runtime:

```python
from mrff import fwht, fwht_backend
print(fwht_backend())  # e.g. 'c:neon' on Apple Silicon, 'c:avx2'/'c:sse2' on x86_64, or 'numpy'
```

Measured on Apple Silicon (`arm64-apple-darwin`, `c:neon` backend):

- **`fwht` (`2000 × 256` float64)**: `0.68 ms` (C/NEON) vs `8.73 ms` (pure NumPy) — **12.9× speedup**
- **Sparse CSR SORF (`2000 × 256 → 1024`, 95% sparse)**: `23.96 ms` with zero dense `(2000, 256)` materialization

---

## Comparison & Parity with `neonnnnn/pyrfm`

`mrff` was evaluated directly against [`neonnnnn/pyrfm`](https://github.com/neonnnnn/pyrfm) (latest commit `e88fe8c`, Aug 2020).

### Feature & Engineering Comparison

| Dimension | `mrff` | `neonnnnn/pyrfm` |
|---|---|---|
| **Python / NumPy / sklearn** | Python 3.11–3.13+, NumPy 2.x, scikit-learn 1.6–1.9+, Apple Silicon & x86_64 | Unmaintained since 2020; uses `numpy.distutils` (removed in Python 3.12 / NumPy 2.0), removed `np.complex`, removed `sklearn.utils.testing`, x86-only SSE/AVX FFHT |
| **Shift-invariant kernels (ORF / SORF / Fastfood / RFF)** | `RBF`, `Matern` (any $\nu > 0$), `Exponential` ($L_2$), `RationalQuadratic`, `Cauchy` ($L_2$), `InverseMultiquadric`, `LaplacianL1` (RFF-only) | **RBF only** (plus dot-product when `random_fourier=False`) |
| **ARD lengthscales** | Yes (scalar or per-feature vector on all shift-invariant kernels) | No (single scalar `gamma` only) |
| **Scikit-learn compliance** | Passes 100% of official `check_estimator` across all 12 estimator classes; never mutates `__init__` params; preserves `float32`/`float64` | Mutates `self.n_components` inside `fit()`; no `n_features_in_`, `feature_names_in_`, `get_feature_names_out`, `__sklearn_tags__`, or `float32` preservation |
| **Haar orthogonal QR** | Sign-corrected QR (`sign(diag(R))`, Mezzadri 2007) — exact Haar measure on $O(d)$ | Raw `scipy.linalg.qr_multiply(W, S)` without diagonal sign correction |
| **SORF radial scaling** | Exact radial draw in padded dim $p$ (`chi_p` for RBF; generalizes to all isotropic kernels) | Fixed deterministic $\sqrt{p}$ scaling (RBF-only, larger bias at small $d$) |
| **Non-Fourier / Sketch families** | `ArcCosineOrthogonalFeatures`, `OrthogonalRandomProjection`, `FastfoodFeatures`, `CountSketchFeatures`, `TensorSketchFeatures`, `AnovaRandomFeatures`, `HomogeneousKernelMap` | `FastFood`, `CountSketch`, `TensorSketch`, `RandomKernel`, `SubfeatureRandomKernel`, `RandomMaclaurin`, `SignedCirculantRandomMatrix`, `MB`, `AdditiveChi2Sampler`, `CompactRandomFeature` |
| **Online linear solvers** | `RandomFeatureRegressor` & `RandomFeatureClassifier` (`adam`, `sgd`, `adagrad`, $O(N)$-memory `saga`, `partial_fit`) | Cython `SGD`, `SAGA`, `SDCA`, `AdaGrad`, `Adam`, `DoublySGD`, `SparseMB` |
| **Typing & test suite** | PEP 561 typed (`py.typed`), `ty` + `ruff` clean, 1,208 passing tests | Untyped Python 3.6 style |

### Exact Numerical Parity & Measured Accuracy

Given identical sampled weights/hashes (or for deterministic maps), `mrff` matches `pyrfm`'s feature maps to **machine precision**:

| Feature Map | Max Absolute Feature Diff (`mrff` vs `pyrfm`) |
|---|---|
| `OrthogonalRandomFourierFeatures` (`use_cos_sin=True`, same `W`) | `0.0` |
| `OrthogonalRandomFourierFeatures` (`use_cos_sin=False`, same `W, b`) | `0.0` |
| `CountSketchFeatures` (same hashes) | `0.0` |
| `TensorSketchFeatures` (`degree=2, gamma=1, coef0=0`, same hashes) | `1.78e-15` |
| `AnovaRandomFeatures` (`kernel="anova", degree=2`, same `W`) | `8.88e-16` |
| `HomogeneousKernelMap(kernel="chi2", order=2, sample_interval=0.5)` vs `pyrfm.AdditiveChi2Sampler(sample_steps=3, sample_interval=0.5)` *(deterministic)* | `2.22e-16` (Gram diff `1.78e-15`) |

Measured RBF Gram matrix MSE ($\gamma = 0.5 \iff \ell = 1.0$, $100 \times 16$ Gaussian inputs, $M = 512$, averaged over 30 random trials):

| Method | `pyrfm` Mean Gram MSE | `mrff` Mean Gram MSE |
|---|---|---|
| **ORF** | `0.001442` | `0.001456` *(statistically identical)* |
| **SORF** | `0.002528` | **`0.001464`** *(1.7× lower MSE)* |

At small dimension ($d = 4$, 80-run average), `mrff`'s exact $\chi_p$ radial scaling also reduces SORF bias relative to `pyrfm`'s fixed $\sqrt{p}$ scaling (`max_bias = 0.302` vs `0.417`).

### Migrating from `pyrfm` to `mrff`

1. **Bandwidth (`gamma` $\leftrightarrow$ `lengthscale`)**:

   $$\gamma = \frac{1}{2\ell^2} \iff \ell = \frac{1}{\sqrt{2\gamma}}$$

   For example, `pyrfm`'s default `gamma=0.5` equals `mrff`'s default `lengthscale=1.0`.
2. **Phase embedding (`use_offset` $\leftrightarrow$ `use_cos_sin`)**:
   `pyrfm` defaults to `use_offset=False` (`[cos, sin]`). In `mrff`, pass `use_cos_sin=True` to get the same phase-free `[cos, sin]` feature map (default is `use_cos_sin=False` to match `sklearn.kernel_approximation.RBFSampler` and support `n_components=1`).
3. **Dot-product mode (`random_fourier=False`)**:
   Use `OrthogonalRandomProjection(n_components=..., mode="orthogonal" | "sorf")` instead of `OrthogonalRandomFeature(random_fourier=False)`.
4. **Chi-squared order (`sample_steps` $\leftrightarrow$ `order`)**:
   `pyrfm.AdditiveChi2Sampler(sample_steps=s)` corresponds to `mrff.HomogeneousKernelMap(kernel="chi2", order=s - 1)` — both produce $(2s - 1)d$ columns in the same order.
5. **Exact `n_components`**:
   `mrff` always outputs the exact `n_components` requested without silently rounding up or mutating `self.n_components`.
6. **Random seed stream**:
   `mrff` uses NumPy's `PCG64` (`np.random.Generator`) rather than legacy `MT19937` (`RandomState`) + `scipy.stats.chi.rvs`, so identical integer `random_state` values draw from the same distribution but not the same bit stream.

---

## Development & Testing

```bash
# Build native C/SIMD extension and install dev tools
uv sync --extra dev

# Run full test suite (unit tests + exact kernel identities + sklearn check_estimator)
uv run pytest

# Format, lint, and static typecheck
uv run ruff format --check src tests
uv run ruff check src tests
ty check --python .venv src tests
```

---

## References

- **ORF / SORF**: F. X. Yu, A. T. Suresh, K. Choromanski, D. Holtmann-Rice, S. Kumar, *"Orthogonal Random Features"*, NeurIPS 2016.
- **Haar QR sampling**: F. Mezzadri, *"How to Generate Random Matrices from the Classical Compact Groups"*, Notices of the AMS, 2007.
- **RFF**: A. Rahimi, B. Recht, *"Random Features for Large-Scale Kernel Machines"*, NeurIPS 2007.
- **Fastfood**: Q. Le, T. Sarlós, A. Smola, *"Fastfood — Approximating Kernel Expansions in Loglinear Time"*, ICML 2013.
- **TensorSketch / CountSketch**: N. Pham, R. Pagh, *"Fast and Scalable Polynomial Kernels via Explicit Feature Maps"*, KDD 2013; M. Charikar, K. Chen, M. Farach-Colton, ICALP 2002.
- **ANOVA / All-Subsets Random Features**: K. Atarashi, S. Maji, S. Oyama, *"Random Feature Maps for the Itemset Kernel"*, AAAI 2019.
- **Homogeneous Kernel Maps**: A. Vedaldi, A. Zisserman, *"Efficient Additive Kernels via Explicit Feature Maps"*, IEEE TPAMI 2012.
- **Arc-Cosine Kernels**: Y. Cho, L. K. Saul, *"Kernel Methods for Deep Learning"*, NeurIPS 2009.
- **SAGA**: A. Defazio, F. Bach, S. Lacoste-Julien, *"SAGA: A Fast Incremental Gradient Method With Support for Non-Strongly Convex Composite Objectives"*, NeurIPS 2014.
