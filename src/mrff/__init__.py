"""mrff: Scikit-learn-compatible Orthogonal, Structured, Sketch & Online Random Features."""

from mrff.features import (
    ArcCosineOrthogonalFeatures,
    NotFittedError,
    OrthogonalRandomFourierFeatures,
    OrthogonalRandomProjection,
    RandomFourierFeatures,
    StructuredOrthogonalRandomFourierFeatures,
)
from mrff.kernels import (
    RBF,
    Cauchy,
    Exponential,
    InverseMultiquadric,
    IsotropicKernel,
    KernelLike,
    LaplacianL1,
    Matern,
    RationalQuadratic,
    ShiftInvariantKernel,
    exact_all_subsets_matrix,
    exact_anova_matrix,
    exact_arccosine_matrix,
    exact_chi2_matrix,
    exact_hellinger_matrix,
    exact_intersection_matrix,
    exact_polynomial_matrix,
    make_kernel,
)
from mrff.linear_model import (
    RandomFeatureClassifier,
    RandomFeatureRegressor,
)
from mrff.orthogonal import (
    BaseDistribution,
    apply_sorf_projection,
    apply_sorf_scaled,
    fwht,
    fwht_backend,
    next_power_of_two,
    sample_base_matrix,
    sample_haar_orthogonal,
    sample_rademacher_signs,
)
from mrff.sketches import (
    AnovaRandomFeatures,
    CountSketchFeatures,
    FastfoodFeatures,
    HomogeneousKernelMap,
    TensorSketchFeatures,
)

__version__ = "0.2.0"

__all__ = [
    "__version__",
    # Orthogonal & Fourier transformers
    "OrthogonalRandomFourierFeatures",
    "StructuredOrthogonalRandomFourierFeatures",
    "RandomFourierFeatures",
    "ArcCosineOrthogonalFeatures",
    "OrthogonalRandomProjection",
    "NotFittedError",
    # Additional sketch & kernel map transformers
    "FastfoodFeatures",
    "CountSketchFeatures",
    "TensorSketchFeatures",
    "AnovaRandomFeatures",
    "HomogeneousKernelMap",
    # Online / memory-free linear models
    "RandomFeatureRegressor",
    "RandomFeatureClassifier",
    # Kernels & exact evaluators
    "ShiftInvariantKernel",
    "IsotropicKernel",
    "RBF",
    "Matern",
    "Exponential",
    "RationalQuadratic",
    "Cauchy",
    "InverseMultiquadric",
    "LaplacianL1",
    "KernelLike",
    "make_kernel",
    "exact_arccosine_matrix",
    "exact_polynomial_matrix",
    "exact_anova_matrix",
    "exact_all_subsets_matrix",
    "exact_chi2_matrix",
    "exact_intersection_matrix",
    "exact_hellinger_matrix",
    # Orthogonal & FWHT primitives
    "BaseDistribution",
    "fwht_backend",
    "sample_base_matrix",
    "sample_haar_orthogonal",
    "fwht",
    "next_power_of_two",
    "sample_rademacher_signs",
    "apply_sorf_projection",
    "apply_sorf_scaled",
]
