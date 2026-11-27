"""Tests for memory-free online RandomFeatureRegressor and RandomFeatureClassifier."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse

from mrff import (
    FastfoodFeatures,
    OrthogonalRandomFourierFeatures,
    RandomFeatureClassifier,
    RandomFeatureRegressor,
    StructuredOrthogonalRandomFourierFeatures,
    TensorSketchFeatures,
)
from mrff.linear_model import ClassificationLoss, OptimizerName


@pytest.mark.parametrize("optimizer", ["adam", "sgd", "adagrad", "saga"])
def test_random_feature_regressor_optimizers(optimizer: OptimizerName) -> None:
    rng = np.random.default_rng(10)
    X = rng.uniform(-2.0, 2.0, size=(240, 3))
    y = np.sin(X[:, 0]) + 0.5 * np.cos(X[:, 1] * X[:, 2])

    reg = RandomFeatureRegressor(
        feature_map=OrthogonalRandomFourierFeatures(
            n_components=128, kernel="matern52", lengthscale=1.5, use_cos_sin=True
        ),
        optimizer=optimizer,
        alpha=1e-3,
        batch_size=32,
        max_iter=80,
        random_state=42,
    )
    reg.fit(X[:180], y[:180])
    assert reg.score(X[180:], y[180:]) > 0.75


def test_random_feature_regressor_huber_sparse_and_partial_fit() -> None:
    rng = np.random.default_rng(11)
    X = rng.uniform(-1.5, 1.5, size=(200, 4))
    y = X[:, 0] ** 2 + np.sin(X[:, 1])

    # Huber loss + SORF on sparse CSR input
    X_csr = sparse.csr_matrix(X)
    reg = RandomFeatureRegressor(
        feature_map=StructuredOrthogonalRandomFourierFeatures(
            n_components=128, kernel="rbf", lengthscale=1.5
        ),
        loss="huber",
        optimizer="adam",
        batch_size=32,
        max_iter=60,
        random_state=0,
    ).fit(X_csr[:150], y[:150])
    assert reg.score(X_csr[150:], y[150:]) > 0.75

    # Streaming partial_fit
    stream = RandomFeatureRegressor(
        feature_map=FastfoodFeatures(n_components=128, kernel="rbf", lengthscale=1.5),
        optimizer="adam",
        batch_size=32,
        random_state=1,
    )
    for _ in range(15):
        for start in range(0, 150, 30):
            stream.partial_fit(X[start : start + 30], y[start : start + 30])
    assert stream.score(X[150:], y[150:]) > 0.75


@pytest.mark.parametrize("optimizer", ["adam", "sgd", "adagrad", "saga"])
@pytest.mark.parametrize("loss", ["log_loss", "hinge", "squared_hinge"])
def test_random_feature_classifier_binary(
    optimizer: OptimizerName, loss: ClassificationLoss
) -> None:
    rng = np.random.default_rng(22)
    X = rng.standard_normal((240, 2))
    # Non-linear circular decision boundary (requires kernel features!)
    y = np.where(np.sum(X * X, axis=1) > 1.4, "outer", "inner")

    clf = RandomFeatureClassifier(
        feature_map=OrthogonalRandomFourierFeatures(
            n_components=128, kernel="rbf", lengthscale=1.2, use_cos_sin=True
        ),
        loss=loss,
        optimizer=optimizer,
        alpha=1e-3,
        batch_size=32,
        max_iter=60,
        random_state=7,
    )
    clf.fit(X[:180], y[:180])
    assert clf.score(X[180:], y[180:]) > 0.85

    if loss == "log_loss":
        probs = clf.predict_proba(X[180:])
        assert probs.shape == (60, 2)
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-12)
    else:
        assert not hasattr(clf, "predict_proba")


def test_random_feature_classifier_multiclass_sparse_and_partial_fit() -> None:
    rng = np.random.default_rng(33)
    centers = np.array([[2.0, 0.0], [-2.0, 0.0], [0.0, 2.2]])
    X_list, y_list = [], []
    for k, c in enumerate(centers):
        X_list.append(rng.standard_normal((80, 2)) * 0.55 + c)
        y_list.append(np.full(80, k))
    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    perm = rng.permutation(len(y))
    X, y = X[perm], y[perm]

    X_csr = sparse.csr_matrix(X)
    clf = RandomFeatureClassifier(
        feature_map=TensorSketchFeatures(n_components=128, degree=2, coef0=1.0),
        loss="log_loss",
        optimizer="saga",
        batch_size=32,
        max_iter=50,
        random_state=5,
    ).fit(X_csr[:180], y[:180])

    assert clf.score(X_csr[180:], y[180:]) > 0.90
    probs = clf.predict_proba(X_csr[180:])
    assert probs.shape == (60, 3)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-12)

    # Streaming partial_fit
    stream = RandomFeatureClassifier(
        feature_map=OrthogonalRandomFourierFeatures(n_components=64, lengthscale=1.5),
        optimizer="adam",
        batch_size=32,
        random_state=0,
    )
    for _ in range(10):
        for start in range(0, 180, 45):
            stream.partial_fit(X[start : start + 45], y[start : start + 45], classes=[0, 1, 2])
    assert stream.score(X[180:], y[180:]) > 0.90
