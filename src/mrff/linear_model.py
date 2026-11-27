"""Memory-free online / mini-batch linear models over random feature maps.

Transforms input data on the fly in mini-batches so the full ``(n_samples, n_components)``
feature matrix is never materialized in memory. Supports ``sgd``, ``adagrad``,
``adam``, and variance-reduced ``saga`` (storing only scalar per-sample loss
derivatives in O(n_samples) memory rather than O(n_samples * n_components)).
"""

from __future__ import annotations

import math
from typing import Any, Literal, Self

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, clone
from sklearn.utils.metaestimators import available_if
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted, validate_data

from mrff._validation import SeedLike, validate_positive_float, validate_positive_int
from mrff.features import OrthogonalRandomFourierFeatures, _resolve_random_state

OptimizerName = Literal["sgd", "adagrad", "adam", "saga"]
RegressionLoss = Literal["squared_error", "huber"]
ClassificationLoss = Literal["log_loss", "hinge", "squared_hinge"]


def _sigmoid(z: NDArray[np.float64]) -> NDArray[np.float64]:
    out = np.empty_like(z, dtype=np.float64)
    pos = z >= 0.0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def _softmax(z: NDArray[np.float64]) -> NDArray[np.float64]:
    shifted = z - np.max(z, axis=1, keepdims=True)
    ez = np.exp(shifted)
    return ez / np.sum(ez, axis=1, keepdims=True)


def _prepare_feature_map(
    feature_map: Any,
    rng: np.random.Generator,
    n_features: int,
) -> Any:
    """Clone ``feature_map`` (or create default ORF) and seed it deterministically."""
    if feature_map is None:
        seed_int = int(rng.integers(0, 2**31 - 1))
        default_ls = max(1.0, math.sqrt(float(n_features)))
        return OrthogonalRandomFourierFeatures(
            n_components=128,
            lengthscale=default_ls,
            random_state=seed_int,
        )
    fmap = clone(feature_map)
    params = fmap.get_params(deep=False)
    if "random_state" in params and params["random_state"] is None:
        fmap.set_params(random_state=int(rng.integers(0, 2**31 - 1)))
    return fmap


def _batched_decision(
    X: Any,
    feature_map: Any,
    coef: NDArray[np.float64],
    intercept: NDArray[np.float64],
    batch_size: int,
) -> NDArray[np.float64]:
    """Compute ``feature_map.transform(X) @ coef.T + intercept`` in mini-batches."""
    n_samples = int(X.shape[0])
    n_outputs = coef.shape[0]
    out = np.empty((n_samples, n_outputs), dtype=np.float64)
    chunk = max(batch_size, 512)
    for start in range(0, n_samples, chunk):
        end = min(start + chunk, n_samples)
        Z = np.asarray(feature_map.transform(X[start:end]), dtype=np.float64)
        out[start:end] = Z @ coef.T + intercept[None, :]
    return out


class _OnlineOptimizerState:
    """Per-parameter optimizer state for SGD, AdaGrad, Adam, and SAGA."""

    def __init__(
        self,
        optimizer: OptimizerName,
        coef_shape: tuple[int, int],
        n_samples: int,
        lr: float,
        alpha: float,
    ) -> None:
        self.optimizer = optimizer
        self.lr = lr
        self.alpha = alpha
        self.step = 0

        k_out, n_feat = coef_shape
        self.m_w = np.zeros((k_out, n_feat), dtype=np.float64)
        self.v_w = np.zeros((k_out, n_feat), dtype=np.float64)
        self.m_b = np.zeros(k_out, dtype=np.float64)
        self.v_b = np.zeros(k_out, dtype=np.float64)

        # SAGA stores only scalar per-sample derivatives (O(N * K_out)), never Z (O(N * M)).
        self.saga_derivs: NDArray[np.float64] | None = (
            np.zeros((n_samples, k_out), dtype=np.float64) if optimizer == "saga" else None
        )
        self.saga_avg_w = np.zeros((k_out, n_feat), dtype=np.float64)
        self.saga_avg_b = np.zeros(k_out, dtype=np.float64)
        self.n_samples = n_samples

    def apply_batch(
        self,
        coef: NDArray[np.float64],
        intercept: NDArray[np.float64],
        Z_b: NDArray[np.float64],
        deriv_b: NDArray[np.float64],
        batch_indices: NDArray[np.intp] | None,
        fit_intercept: bool,
    ) -> float:
        """Update ``coef`` and ``intercept`` in place and return gradient norm."""
        m = float(Z_b.shape[0])
        self.step += 1

        if self.optimizer == "saga" and self.saga_derivs is not None and batch_indices is not None:
            old_d = self.saga_derivs[batch_indices]
            diff = deriv_b - old_d
            grad_w = (diff.T @ Z_b) / m + self.saga_avg_w + self.alpha * coef
            grad_b = (
                np.mean(diff, axis=0) + self.saga_avg_b
                if fit_intercept
                else np.zeros_like(intercept)
            )

            inv_n = 1.0 / float(self.n_samples)
            self.saga_avg_w += (diff.T @ Z_b) * inv_n
            if fit_intercept:
                self.saga_avg_b += np.sum(diff, axis=0) * inv_n
            self.saga_derivs[batch_indices] = deriv_b

            coef -= self.lr * grad_w
            if fit_intercept:
                intercept -= self.lr * grad_b
            return float(np.linalg.norm(grad_w))

        grad_w = (deriv_b.T @ Z_b) / m + self.alpha * coef
        grad_b = np.mean(deriv_b, axis=0) if fit_intercept else np.zeros_like(intercept)

        if self.optimizer == "sgd":
            eta = self.lr / (1.0 + 0.005 * self.step)
            coef -= eta * grad_w
            if fit_intercept:
                intercept -= eta * grad_b
        elif self.optimizer == "adagrad":
            eps = 1e-8
            self.v_w += grad_w * grad_w
            coef -= (self.lr / (np.sqrt(self.v_w) + eps)) * grad_w
            if fit_intercept:
                self.v_b += grad_b * grad_b
                intercept -= (self.lr / (np.sqrt(self.v_b) + eps)) * grad_b
        else:  # adam (or saga fallback under partial_fit)
            b1, b2, eps = 0.9, 0.999, 1e-8
            self.m_w = b1 * self.m_w + (1.0 - b1) * grad_w
            self.v_w = b2 * self.v_w + (1.0 - b2) * (grad_w * grad_w)
            m_hat_w = self.m_w / (1.0 - b1**self.step)
            v_hat_w = self.v_w / (1.0 - b2**self.step)
            coef -= self.lr * m_hat_w / (np.sqrt(v_hat_w) + eps)
            if fit_intercept:
                self.m_b = b1 * self.m_b + (1.0 - b1) * grad_b
                self.v_b = b2 * self.v_b + (1.0 - b2) * (grad_b * grad_b)
                m_hat_b = self.m_b / (1.0 - b1**self.step)
                v_hat_b = self.v_b / (1.0 - b2**self.step)
                intercept -= self.lr * m_hat_b / (np.sqrt(v_hat_b) + eps)

        return float(np.linalg.norm(grad_w))


def _validate_optimizer(optimizer: str) -> OptimizerName:
    if optimizer not in ("sgd", "adagrad", "adam", "saga"):
        raise ValueError(
            f"optimizer must be 'sgd', 'adagrad', 'adam', or 'saga', got {optimizer!r}."
        )
    return optimizer  # type: ignore[return-value]


class RandomFeatureRegressor(RegressorMixin, BaseEstimator):
    """Memory-free mini-batch linear regressor over a random feature map.

    Computes random features on the fly in mini-batches during training and
    prediction, avoiding materializing the full ``(n_samples, n_components)``
    feature matrix.

    Parameters
    ----------
    feature_map : transformer or None, default=None
        Any scikit-learn-compatible feature transformer (e.g.
        ``OrthogonalRandomFourierFeatures``,
        ``StructuredOrthogonalRandomFourierFeatures``, ``FastfoodFeatures``,
        ``TensorSketchFeatures``). If None, defaults to
        ``OrthogonalRandomFourierFeatures(n_components=128)``.
    loss : {"squared_error", "huber"}, default="squared_error"
        Training loss function.
    alpha : float, default=1e-3
        L2 regularization strength.
    optimizer : {"adam", "sgd", "adagrad", "saga"}, default="adam"
        Mini-batch / variance-reduced optimizer.
    learning_rate : float or "auto", default="auto"
        Step size. When ``"auto"``, scales automatically with ``std(y)``.
    batch_size : int, default=64
        Mini-batch size for on-the-fly feature generation.
    max_iter : int, default=50
        Maximum number of passes (epochs) over the training data.
    tol : float, default=1e-4
        Gradient norm tolerance for early stopping.
    epsilon : float, default=1.35
        Threshold for Huber loss.
    fit_intercept : bool, default=True
        Whether to fit an unregularized intercept term.
    random_state : int, RandomState, Generator, or None, default=None
        Controls random feature initialization and mini-batch shuffling.
    """

    def __init__(
        self,
        feature_map: Any = None,
        *,
        loss: RegressionLoss = "squared_error",
        alpha: float = 1e-3,
        optimizer: OptimizerName = "adam",
        learning_rate: float | Literal["auto"] = "auto",
        batch_size: int = 64,
        max_iter: int = 50,
        tol: float = 1e-4,
        epsilon: float = 1.35,
        fit_intercept: bool = True,
        random_state: SeedLike = None,
    ) -> None:
        self.feature_map = feature_map
        self.loss = loss
        self.alpha = alpha
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.tol = tol
        self.epsilon = epsilon
        self.fit_intercept = fit_intercept
        self.random_state = random_state

    def _loss_deriv(
        self, pred: NDArray[np.float64], y_b: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        residual = pred - y_b[:, None]
        if self.loss == "huber":
            return np.clip(residual, -float(self.epsilon), float(self.epsilon))
        return residual

    def _resolve_lr(self, y: NDArray[np.float64]) -> float:
        if self.learning_rate != "auto":
            return validate_positive_float(self.learning_rate, "learning_rate")
        y_scale = max(1.0, float(np.std(y)))
        if self.optimizer == "adam":
            return 0.05 * y_scale
        if self.optimizer == "adagrad":
            return 0.1 * y_scale
        if self.optimizer == "saga":
            return 0.1
        return 0.2

    def fit(self, X: ArrayLike, y: ArrayLike) -> Self:
        """Fit linear weights over on-the-fly random features."""
        X_val, y_val = validate_data(
            self,
            X,
            y,
            accept_sparse=("csr", "csc"),
            dtype=[np.float64, np.float32],
            y_numeric=True,
        )
        if self.loss not in ("squared_error", "huber"):
            raise ValueError(f"loss must be 'squared_error' or 'huber', got {self.loss!r}.")
        opt_name = _validate_optimizer(self.optimizer)
        if (
            isinstance(self.alpha, bool)
            or float(self.alpha) < 0.0
            or not np.isfinite(float(self.alpha))
        ):
            raise ValueError(f"alpha must be finite and non-negative, got {self.alpha!r}.")
        batch_size = validate_positive_int(self.batch_size, "batch_size")
        max_iter = validate_positive_int(self.max_iter, "max_iter")
        if self.loss == "huber":
            validate_positive_float(self.epsilon, "epsilon")

        y_f64 = np.asarray(y_val, dtype=np.float64)
        n_samples = int(X_val.shape[0])
        rng = _resolve_random_state(self.random_state)

        fmap = _prepare_feature_map(self.feature_map, rng, int(X_val.shape[1]))
        fmap.fit(X_val)
        n_rf = int(np.asarray(fmap.transform(X_val[:1])).shape[1])

        coef = np.zeros((1, n_rf), dtype=np.float64)
        intercept = np.array(
            [float(np.mean(y_f64)) if self.fit_intercept else 0.0],
            dtype=np.float64,
        )
        lr = self._resolve_lr(y_f64)
        state = _OnlineOptimizerState(opt_name, (1, n_rf), n_samples, lr, float(self.alpha))

        n_iter = 0
        for epoch in range(max_iter):
            n_iter = epoch + 1
            order = rng.permutation(n_samples)
            epoch_grad_norm = 0.0
            n_batches = 0
            for start in range(0, n_samples, batch_size):
                idx = order[start : start + batch_size]
                Z_b = np.asarray(fmap.transform(X_val[idx]), dtype=np.float64)
                pred = Z_b @ coef.T + intercept[None, :]
                deriv = self._loss_deriv(pred, y_f64[idx])
                epoch_grad_norm += state.apply_batch(
                    coef, intercept, Z_b, deriv, idx, self.fit_intercept
                )
                n_batches += 1
            if n_batches > 0 and (epoch_grad_norm / n_batches) < float(self.tol):
                break

        self.feature_map_ = fmap
        self.coef_: NDArray[np.float64] = coef.ravel()
        self.intercept_: float = float(intercept[0])
        self.n_iter_: int = n_iter
        self._opt_state_ = state
        return self

    def partial_fit(self, X: ArrayLike, y: ArrayLike) -> Self:
        """Perform one online mini-batch update over ``(X, y)``."""
        first_call = not hasattr(self, "coef_")
        X_val, y_val = validate_data(
            self,
            X,
            y,
            accept_sparse=("csr", "csc"),
            dtype=[np.float64, np.float32],
            y_numeric=True,
            reset=first_call,
        )
        y_f64 = np.asarray(y_val, dtype=np.float64)
        opt_name = _validate_optimizer(self.optimizer)
        batch_size = validate_positive_int(self.batch_size, "batch_size")

        if first_call:
            rng = _resolve_random_state(self.random_state)
            fmap = _prepare_feature_map(self.feature_map, rng, int(X_val.shape[1]))
            fmap.fit(X_val)
            n_rf = int(np.asarray(fmap.transform(X_val[:1])).shape[1])
            coef = np.zeros((1, n_rf), dtype=np.float64)
            intercept = np.array(
                [float(np.mean(y_f64)) if self.fit_intercept else 0.0], dtype=np.float64
            )
            lr = self._resolve_lr(y_f64)
            self.feature_map_ = fmap
            self._opt_state_ = _OnlineOptimizerState(
                opt_name, (1, n_rf), int(X_val.shape[0]), lr, float(self.alpha)
            )
            self.n_iter_ = 0
        else:
            coef = self.coef_.reshape(1, -1).copy()
            intercept = np.array([self.intercept_], dtype=np.float64)

        n_samples = int(X_val.shape[0])
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            Z_b = np.asarray(self.feature_map_.transform(X_val[start:end]), dtype=np.float64)
            pred = Z_b @ coef.T + intercept[None, :]
            deriv = self._loss_deriv(pred, y_f64[start:end])
            self._opt_state_.apply_batch(coef, intercept, Z_b, deriv, None, self.fit_intercept)

        self.coef_ = coef.ravel()
        self.intercept_ = float(intercept[0])
        self.n_iter_ += 1
        return self

    def predict(self, X: ArrayLike) -> NDArray[np.float64]:
        """Predict target values in mini-batches without materializing full Z."""
        check_is_fitted(self, ["coef_", "intercept_", "feature_map_"])
        X_val = validate_data(
            self,
            X,
            accept_sparse=("csr", "csc"),
            dtype=[np.float64, np.float32],
            reset=False,
        )
        scores = _batched_decision(
            X_val,
            self.feature_map_,
            self.coef_.reshape(1, -1),
            np.array([self.intercept_], dtype=np.float64),
            self.batch_size,
        )
        return scores.ravel()

    def __sklearn_tags__(self) -> Any:
        tags = super().__sklearn_tags__()
        tags.input_tags.sparse = True
        return tags


def _check_log_loss(estimator: RandomFeatureClassifier) -> bool:
    return estimator.loss == "log_loss"


class RandomFeatureClassifier(ClassifierMixin, BaseEstimator):
    """Memory-free mini-batch linear classifier over a random feature map.

    Supports binary and multiclass classification with ``log_loss`` (logistic /
    multinomial softmax), ``hinge``, and ``squared_hinge`` losses trained via
    ``adam``, ``sgd``, ``adagrad``, or variance-reduced ``saga`` without
    materializing the full ``(n_samples, n_components)`` feature matrix.
    """

    def __init__(
        self,
        feature_map: Any = None,
        *,
        loss: ClassificationLoss = "log_loss",
        alpha: float = 1e-3,
        optimizer: OptimizerName = "adam",
        learning_rate: float | Literal["auto"] = "auto",
        batch_size: int = 64,
        max_iter: int = 50,
        tol: float = 1e-4,
        fit_intercept: bool = True,
        random_state: SeedLike = None,
    ) -> None:
        self.feature_map = feature_map
        self.loss = loss
        self.alpha = alpha
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.random_state = random_state

    def _loss_deriv(
        self,
        scores: NDArray[np.float64],
        y_idx: NDArray[np.intp],
        n_classes: int,
    ) -> NDArray[np.float64]:
        if n_classes == 2:
            if self.loss == "log_loss":
                probs = _sigmoid(scores[:, 0])
                return (probs - y_idx.astype(np.float64))[:, None]
            y_pm = 2.0 * y_idx.astype(np.float64) - 1.0
            margin = y_pm * scores[:, 0]
            if self.loss == "hinge":
                return np.where(margin < 1.0, -y_pm, 0.0)[:, None]
            return (-2.0 * y_pm * np.maximum(0.0, 1.0 - margin))[:, None]

        # Multiclass
        onehot = np.zeros((y_idx.shape[0], n_classes), dtype=np.float64)
        onehot[np.arange(y_idx.shape[0]), y_idx] = 1.0
        if self.loss == "log_loss":
            return _softmax(scores) - onehot
        y_pm = 2.0 * onehot - 1.0
        margin = y_pm * scores
        if self.loss == "hinge":
            return np.where(margin < 1.0, -y_pm, 0.0)
        return -2.0 * y_pm * np.maximum(0.0, 1.0 - margin)

    def _resolve_lr(self) -> float:
        if self.learning_rate != "auto":
            return validate_positive_float(self.learning_rate, "learning_rate")
        if self.optimizer == "adam":
            return 0.05
        if self.optimizer == "adagrad":
            return 0.1
        return 0.1

    def fit(self, X: ArrayLike, y: ArrayLike) -> Self:
        """Fit linear classification weights over on-the-fly random features."""
        X_val, y_val = validate_data(
            self,
            X,
            y,
            accept_sparse=("csr", "csc"),
            dtype=[np.float64, np.float32],
        )
        check_classification_targets(y_val)
        if self.loss not in ("log_loss", "hinge", "squared_hinge"):
            raise ValueError(
                f"loss must be 'log_loss', 'hinge', or 'squared_hinge', got {self.loss!r}."
            )
        opt_name = _validate_optimizer(self.optimizer)
        if (
            isinstance(self.alpha, bool)
            or float(self.alpha) < 0.0
            or not np.isfinite(float(self.alpha))
        ):
            raise ValueError(f"alpha must be finite and non-negative, got {self.alpha!r}.")
        batch_size = validate_positive_int(self.batch_size, "batch_size")
        max_iter = validate_positive_int(self.max_iter, "max_iter")

        classes, y_idx = np.unique(y_val, return_inverse=True)
        n_classes = int(classes.shape[0])
        if n_classes < 2:
            raise ValueError(
                f"The number of classes has to be greater than one; got {n_classes} class."
            )

        k_out = 1 if n_classes == 2 else n_classes
        n_samples = int(X_val.shape[0])
        rng = _resolve_random_state(self.random_state)

        fmap = _prepare_feature_map(self.feature_map, rng, int(X_val.shape[1]))
        fmap.fit(X_val)
        n_rf = int(np.asarray(fmap.transform(X_val[:1])).shape[1])

        coef = np.zeros((k_out, n_rf), dtype=np.float64)
        intercept = np.zeros(k_out, dtype=np.float64)
        lr = self._resolve_lr()
        state = _OnlineOptimizerState(opt_name, (k_out, n_rf), n_samples, lr, float(self.alpha))

        n_iter = 0
        for epoch in range(max_iter):
            n_iter = epoch + 1
            order = rng.permutation(n_samples)
            epoch_grad_norm = 0.0
            n_batches = 0
            for start in range(0, n_samples, batch_size):
                idx = order[start : start + batch_size]
                Z_b = np.asarray(fmap.transform(X_val[idx]), dtype=np.float64)
                scores = Z_b @ coef.T + intercept[None, :]
                deriv = self._loss_deriv(scores, y_idx[idx], n_classes)
                epoch_grad_norm += state.apply_batch(
                    coef, intercept, Z_b, deriv, idx, self.fit_intercept
                )
                n_batches += 1
            if n_batches > 0 and (epoch_grad_norm / n_batches) < float(self.tol):
                break

        self.classes_: NDArray[Any] = classes
        self.feature_map_ = fmap
        self.coef_: NDArray[np.float64] = coef
        self.intercept_: NDArray[np.float64] = intercept
        self.n_iter_: int = n_iter
        self._opt_state_ = state
        return self

    def partial_fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        classes: ArrayLike | None = None,
    ) -> Self:
        """Perform one online mini-batch update over ``(X, y)``."""
        first_call = not hasattr(self, "classes_")
        X_val, y_val = validate_data(
            self,
            X,
            y,
            accept_sparse=("csr", "csc"),
            dtype=[np.float64, np.float32],
            reset=first_call,
        )
        check_classification_targets(y_val)
        opt_name = _validate_optimizer(self.optimizer)
        batch_size = validate_positive_int(self.batch_size, "batch_size")

        if first_call:
            if classes is None:
                raise ValueError("classes must be passed on the first call to partial_fit.")
            cls_arr = np.unique(np.asarray(classes))
            if cls_arr.shape[0] < 2:
                raise ValueError(
                    f"The number of classes has to be greater than one; got {cls_arr.shape[0]}."
                )
            rng = _resolve_random_state(self.random_state)
            fmap = _prepare_feature_map(self.feature_map, rng, int(X_val.shape[1]))
            fmap.fit(X_val)
            n_rf = int(np.asarray(fmap.transform(X_val[:1])).shape[1])
            k_out = 1 if cls_arr.shape[0] == 2 else int(cls_arr.shape[0])
            self.classes_ = cls_arr
            self.feature_map_ = fmap
            self.coef_ = np.zeros((k_out, n_rf), dtype=np.float64)
            self.intercept_ = np.zeros(k_out, dtype=np.float64)
            self._opt_state_ = _OnlineOptimizerState(
                opt_name, (k_out, n_rf), int(X_val.shape[0]), self._resolve_lr(), float(self.alpha)
            )
            self.n_iter_ = 0

        y_idx = np.searchsorted(self.classes_, y_val)
        if np.any(y_idx >= self.classes_.shape[0]) or np.any(self.classes_[y_idx] != y_val):
            raise ValueError("y contains classes not present in classes_.")

        n_classes = int(self.classes_.shape[0])
        n_samples = int(X_val.shape[0])
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            Z_b = np.asarray(self.feature_map_.transform(X_val[start:end]), dtype=np.float64)
            scores = Z_b @ self.coef_.T + self.intercept_[None, :]
            deriv = self._loss_deriv(scores, y_idx[start:end], n_classes)
            self._opt_state_.apply_batch(
                self.coef_, self.intercept_, Z_b, deriv, None, self.fit_intercept
            )
        self.n_iter_ += 1
        return self

    def decision_function(self, X: ArrayLike) -> NDArray[np.float64]:
        """Compute raw decision scores in mini-batches."""
        check_is_fitted(self, ["classes_", "coef_", "intercept_", "feature_map_"])
        X_val = validate_data(
            self,
            X,
            accept_sparse=("csr", "csc"),
            dtype=[np.float64, np.float32],
            reset=False,
        )
        scores = _batched_decision(
            X_val, self.feature_map_, self.coef_, self.intercept_, self.batch_size
        )
        if self.classes_.shape[0] == 2:
            return scores[:, 0]
        return scores

    @available_if(_check_log_loss)
    def predict_proba(self, X: ArrayLike) -> NDArray[np.float64]:
        """Estimate class probabilities (available when ``loss='log_loss'``)."""
        scores = self.decision_function(X)
        if self.classes_.shape[0] == 2:
            p1 = _sigmoid(scores)
            return np.column_stack([1.0 - p1, p1])
        return _softmax(scores)

    def predict(self, X: ArrayLike) -> NDArray[Any]:
        """Predict class labels for samples in ``X``."""
        scores = self.decision_function(X)
        if self.classes_.shape[0] == 2:
            idx = (scores > 0.0).astype(np.intp)
        else:
            idx = np.argmax(scores, axis=1)
        return self.classes_[idx]

    def __sklearn_tags__(self) -> Any:
        tags = super().__sklearn_tags__()
        tags.input_tags.sparse = True
        return tags


__all__ = [
    "RandomFeatureClassifier",
    "RandomFeatureRegressor",
]
