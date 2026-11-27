"""Full scikit-learn compatibility tests, including official check_estimator."""

from __future__ import annotations

import pickle

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.utils.estimator_checks import parametrize_with_checks

from mrff import (
    AnovaRandomFeatures,
    ArcCosineOrthogonalFeatures,
    CountSketchFeatures,
    FastfoodFeatures,
    HomogeneousKernelMap,
    Matern,
    OrthogonalRandomFourierFeatures,
    OrthogonalRandomProjection,
    RandomFeatureClassifier,
    RandomFeatureRegressor,
    RandomFourierFeatures,
    StructuredOrthogonalRandomFourierFeatures,
    TensorSketchFeatures,
)


@parametrize_with_checks(
    [
        OrthogonalRandomFourierFeatures(),
        OrthogonalRandomFourierFeatures(kernel="matern32"),
        StructuredOrthogonalRandomFourierFeatures(),
        StructuredOrthogonalRandomFourierFeatures(kernel="rq"),
        RandomFourierFeatures(),
        RandomFourierFeatures(kernel="laplacian_l1"),
        ArcCosineOrthogonalFeatures(order=0),
        ArcCosineOrthogonalFeatures(order=1),
        OrthogonalRandomProjection(mode="orthogonal"),
        OrthogonalRandomProjection(mode="sorf"),
        OrthogonalRandomProjection(mode="iid", distribution="rademacher"),
        FastfoodFeatures(),
        CountSketchFeatures(),
        TensorSketchFeatures(),
        AnovaRandomFeatures(kernel="anova"),
        AnovaRandomFeatures(kernel="all_subsets"),
        HomogeneousKernelMap(kernel="chi2"),
        HomogeneousKernelMap(kernel="intersection"),
        HomogeneousKernelMap(kernel="hellinger"),
        RandomFeatureRegressor(optimizer="adam"),
        RandomFeatureRegressor(optimizer="saga"),
        RandomFeatureClassifier(loss="log_loss", optimizer="adam"),
        RandomFeatureClassifier(loss="hinge", optimizer="saga"),
    ]
)
def test_sklearn_check_estimator(estimator, check) -> None:
    check(estimator)


@pytest.mark.parametrize(
    "estimator",
    [
        OrthogonalRandomFourierFeatures(
            n_components=32, kernel="matern32", lengthscale=1.5, random_state=0
        ),
        OrthogonalRandomFourierFeatures(
            n_components=32, kernel=Matern(2.5, 1.2), use_cos_sin=True, random_state=0
        ),
        StructuredOrthogonalRandomFourierFeatures(
            n_components=32, kernel="rq", alpha=2.0, random_state=0
        ),
        RandomFourierFeatures(n_components=32, kernel="laplacian_l1", random_state=0),
        ArcCosineOrthogonalFeatures(n_components=32, order=1, random_state=0),
        OrthogonalRandomProjection(n_components=32, mode="sorf", random_state=0),
        FastfoodFeatures(n_components=32, kernel="matern32", random_state=0),
        CountSketchFeatures(n_components=32, random_state=0),
        TensorSketchFeatures(n_components=32, degree=2, coef0=1.0, random_state=0),
        AnovaRandomFeatures(n_components=32, kernel="anova", degree=2, random_state=0),
    ],
)
def test_clone_pickle_params_and_feature_names(estimator) -> None:
    params = estimator.get_params()
    assert params["n_components"] == 32

    cloned = clone(estimator)
    assert cloned.get_params() == params
    assert cloned is not estimator

    cloned.set_params(n_components=64)
    assert cloned.n_components == 64

    with pytest.raises(ValueError, match="Invalid parameter"):
        cloned.set_params(nonexistent_param=1)

    X = np.random.default_rng(0).standard_normal((10, 6))
    Z = cloned.fit_transform(X)
    names = cloned.get_feature_names_out()
    assert names.shape == (64,)

    restored = pickle.loads(pickle.dumps(cloned))
    np.testing.assert_array_equal(restored.transform(X), Z)


def test_sklearn_pipeline_and_grid_search() -> None:
    rng = np.random.default_rng(42)
    X = rng.uniform(-2.0, 2.0, size=(180, 3))
    y = np.sin(X[:, 0]) + np.cos(X[:, 1] * X[:, 2])

    X_train, X_test = X[:130], X[130:]
    y_train, y_test = y[:130], y[130:]

    for feat in (
        OrthogonalRandomFourierFeatures(
            n_components=256,
            kernel="matern52",
            lengthscale=1.5,
            use_cos_sin=True,
            random_state=0,
        ),
        StructuredOrthogonalRandomFourierFeatures(
            n_components=256,
            kernel="rbf",
            lengthscale=1.5,
            use_cos_sin=True,
            random_state=0,
        ),
        FastfoodFeatures(
            n_components=256,
            kernel="matern32",
            lengthscale=1.5,
            use_cos_sin=True,
            random_state=0,
        ),
    ):
        pipe = make_pipeline(feat, Ridge(alpha=1e-2))
        pipe.fit(X_train, y_train)
        assert pipe.score(X_test, y_test) > 0.90

    search_pipe = Pipeline(
        [
            ("features", OrthogonalRandomFourierFeatures(n_components=128, random_state=0)),
            ("ridge", Ridge(alpha=1e-2)),
        ]
    )
    grid = GridSearchCV(
        search_pipe,
        param_grid={
            "features__kernel": ["rbf", "matern32"],
            "features__lengthscale": [1.0, 2.0],
        },
        cv=3,
    )
    grid.fit(X_train, y_train)
    assert grid.score(X_test, y_test) > 0.85
